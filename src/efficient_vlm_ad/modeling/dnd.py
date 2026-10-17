from __future__ import annotations

import torch
from torch import nn
import re


class DNDRouter(nn.Module):
    """Router tính điểm độ khó của từng token (d_model -> 1)."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, 1, bias=True)
        nn.init.normal_(self.linear.weight, std = 1e-3)
        nn.init.zeros_(self.linear.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [B, S, D] -> [B, S]
        return torch.sigmoid(self.linear(hidden_states)).squeeze(-1)


class DNDT5Block(nn.Module):
    """
    Wrapper bọc ngoài một T5Block ở Encoder:
    vanilla pass -> router -> top-K -> nested re-forward -> fusion bằng beta.
    Lưu ý: T5Stack gọi block bằng keyword args (attention_mask=..., position_bias=...).
    """

    def __init__(self, original_block: nn.Module, d_model: int,
                 k_target: float = 0.2, beta_init: float = 0.1) -> None:
        super().__init__()
        # [Fix 5] beta = 0 thì gate = 0 -> router không nhận gradient nào
        if beta_init == 0:
            raise ValueError("beta_init phải khác 0, nếu không router sẽ không được học.")
        self.enabled = True
        self.original_block = original_block
        self.router = DNDRouter(d_model)
        self.k_target = k_target
        self.beta = nn.Parameter(torch.tensor(beta_init, dtype=torch.float32))
        self._last_scores: torch.Tensor | None = None
        self._last_valid: torch.Tensor | None = None  # [B, S] bool, False = padding

    # ---------- helpers ----------
    @staticmethod
    def _valid_from_mask(attention_mask, B: int, S: int, device):
        """Trả về (valid [B,S] bool, mask_1d [B,S] additive hoặc None)."""
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                # Hỗ trợ cả (B, 1, 1, S) và (B, 1, S, S) của transformers
                mask_1d = attention_mask[:, 0, 0, :]  # additive: 0 = giữ, rất âm = padding
                valid = mask_1d > -1.0
                return valid, mask_1d
            elif attention_mask.dim() == 2:
                valid = attention_mask.bool()
                neg = torch.finfo(torch.float32).min
                mask_1d = torch.where(valid, 0.0, neg)
                return valid, mask_1d
        return torch.ones(B, S, dtype=torch.bool, device=device), None

    def _gather_position_bias(self, position_bias, mask_1d, idx: torch.Tensor, X: torch.Tensor):
        """
        [Fix 4] Lấy position bias THẬT của K token (hàng + cột theo vị trí gốc)
        thay vì tính lại theo vị trí 0..K-1 hoặc đặt None.
        """
        B, S, _ = X.shape
        K = idx.size(1)
        pb = position_bias
        if pb is None:
            attn = self.original_block.layer[0].SelfAttention
            if not getattr(attn, "has_relative_attention_bias", False):
                return None
            pb = attn.compute_bias(S, S, device=X.device)  # [1, H, S, S]
            if mask_1d is not None:
                # T5Attention chỉ cộng mask khi tự tính bias, nên ta cộng thủ công
                pb = pb + mask_1d[:, None, None, :]
        pb = pb.expand(B, -1, -1, -1)  # [B, H, S, S]
        H = pb.size(1)
        rows = pb.gather(2, idx[:, None, :, None].expand(B, H, K, S))  # [B, H, K, S]
        return rows.gather(3, idx[:, None, None, :].expand(B, H, K, K))  # [B, H, K, K]

    _ARG_NAMES = (
        "attention_mask",
        "position_bias",
        "encoder_hidden_states",
        "encoder_attention_mask",
        "encoder_decoder_position_bias",
        "layer_head_mask",
        "cross_attn_layer_head_mask",
        "past_key_value",
        "use_cache",
        "output_attentions",
        "return_dict",
    )

    # ---------- forward ----------
    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        for name, val in zip(self._ARG_NAMES, args):
            if name not in kwargs:
                kwargs[name] = val

        if not self.enabled:
            self._last_scores = None
            self._last_valid = None
            return self.original_block(hidden_states, **kwargs)

        # 1. Vanilla pass
        vanilla_outputs = self.original_block(hidden_states, **kwargs)
        X_v = vanilla_outputs[0]  # [B, S, D]
        B, S, D = X_v.shape

        attention_mask = kwargs.get("attention_mask")
        valid, mask_1d = self._valid_from_mask(attention_mask, B, S, X_v.device)

        # 2. Router
        scores = self.router(X_v)  # [B, S]
        store =  self.training and torch.is_grad_enabled()  # chỉ lưu khi train, không lưu khi eval
        self._last_scores = scores if store else None
        self._last_valid = valid if store else None

        # 3. Top-K theo ĐỘ DÀI THẬT của từng mẫu
        k = max(1, min(S, int(self.k_target * S)))                              # cận trên, không sync GPU
        n_valid = valid.sum(dim=1)
        k_b = (self.k_target * n_valid.float()).floor().long().clamp_min(1)     # [B]
        topk_idx = torch.topk(scores.masked_fill(~valid, -1.0), k=k, dim=1).indices
        keep = torch.arange(k, device=scores.device)[None, :] < k_b[:, None]    # chỉ giữ k_b token điểm cao nhất
        topk_idx, order = torch.sort(topk_idx, dim=1)                           # về thứ tự vị trí gốc
        keep = keep.gather(1, order)
        sel_valid = valid.gather(1, topk_idx) & keep                            # [B, K]

        # 4. Pack
        gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, D)
        X_packed = torch.gather(X_v, dim=1, index=gather_idx)                   # [B, K, D]

        # 5. Nested pass: token không được giữ thì không được làm key
        neg = torch.finfo(X_v.dtype).min
        pb = self._gather_position_bias(kwargs.get("position_bias"), mask_1d, topk_idx, X_v)
        if pb is not None:
            pb = pb.masked_fill(~sel_valid[:, None, None, :], neg)
        nested_mask = (mask_1d.gather(1, topk_idx) if mask_1d is not None
                       else torch.zeros_like(sel_valid, dtype=X_v.dtype))
        nested_mask = nested_mask.masked_fill(~sel_valid, neg)[:, None, None, :]
        nested_kwargs = dict(kwargs)
        nested_kwargs["position_bias"] = pb
        nested_kwargs["attention_mask"] = nested_mask
        X_d_packed = self.original_block(X_packed, **nested_kwargs)[0]          # [B, K, D]
        
        # 6. Unpack & Fusion
        X_d = torch.zeros_like(X_v).scatter(1, gather_idx, X_d_packed)

        # [Fix 2] chỉ fuse tại token được chọn (và không phải padding); còn lại giữ nguyên X_v
        sel = torch.zeros_like(scores, dtype=torch.bool).scatter(1, topk_idx, sel_valid)  # [B, S]
        # [Fix 5] gradient của router đi qua gate (top-K không khả vi)
        gate = (self.beta * scores * sel.to(scores.dtype)).unsqueeze(-1)  # [B, S, 1]
        X_fused = (1.0 - gate) * X_v + gate * X_d

        return (X_fused,) + tuple(vanilla_outputs[1:])


def collect_dnd_scores(model: nn.Module):
    """Gom score + mask padding từ mọi DNDT5Block để đưa vào loss."""
    scores, masks = [], []
    for m in model.modules():
        if isinstance(m, DNDT5Block) and m._last_scores is not None:
            scores.append(m._last_scores)
            masks.append(m._last_valid)
    return scores, masks


def clear_dnd_scores(model: nn.Module) -> None:
    """Gọi sau backward để giải phóng đồ thị đang giữ trong _last_scores."""
    for m in model.modules():
        if isinstance(m, DNDT5Block):
            m._last_scores = None
            m._last_valid = None


class DNDScoreDispersionLoss(nn.Module):
    """Router loss của paper: lambda_sd * L_sd + lambda_dp * L_dp."""

    def __init__(self, lambda_sd: float = 3e-4, lambda_dp: float = 0.02) -> None:
        super().__init__()
        self.lambda_sd = lambda_sd
        self.lambda_dp = lambda_dp

    def forward(self, layer_scores, layer_masks=None) -> torch.Tensor:
        if not layer_scores:
            return torch.zeros(())
        total = torch.zeros((), device=layer_scores[0].device, dtype=torch.float32)
        for i, scores in enumerate(layer_scores):
            s = scores.float()
            valid = layer_masks[i].to(s.dtype) if layer_masks is not None and layer_masks[i] is not None else torch.ones_like(s)
            n = valid.sum(dim=-1).clamp_min(1.0)
            sv = s * valid
            p = sv / (sv.sum(dim=-1, keepdim=True) + 1e-8)
            l_sd = -(p * torch.log(p + 1e-8)).sum(dim=-1).mean()
            l_dp = ((((s - 0.5) ** 2) * valid).sum(dim=-1) / n).mean()
            total = total + self.lambda_sd * l_sd + self.lambda_dp * l_dp
        return total


def adapt_state_dict_for_dnd(model: nn.Module, state_dict: dict) -> dict:
    """Checkpoint không DND -> model có DND: đổi key block.N.xxx thành block.N.original_block.xxx."""
    encoder = getattr(getattr(model, "text_model", None), "encoder", None)
    block = getattr(encoder, "block", None)
    if block is None:
        return state_dict
    idx = {i for i, b in enumerate(block) if isinstance(b, DNDT5Block)}
    if not idx:
        return state_dict
    pat = re.compile(r"^(.*encoder\.block\.)(\d+)\.(.+)$")
    out = {}
    for key, value in state_dict.items():
        m = pat.match(key)
        if m and int(m.group(2)) in idx and not m.group(3).startswith(("original_block.", "router.", "beta")):
            key = f"{m.group(1)}{m.group(2)}.original_block.{m.group(3)}"
        out[key] = value
    return out


def non_dnd_missing_keys(missing_keys: list[str]) -> list[str]:
    """Các key thiếu KHÔNG thuộc router/beta (nên rỗng)."""
    return [k for k in missing_keys if ".router." not in k and not k.endswith(".beta")]


def exclude_dnd_from_weight_decay(model, groups, weight_decay, router_lr=None):
    skip = {id(p) for n, p in model.named_parameters() if n.endswith(".beta") or ".router." in n}
    out = []
    for g in groups:
        base = {k: v for k, v in g.items() if k not in ("params", "weight_decay")}
        decay = [p for p in g["params"] if id(p) not in skip]
        keep = [p for p in g["params"] if id(p) in skip]
        if decay:
            out.append({**base, "params": decay, "weight_decay": weight_decay})
        if keep:
            g_keep = {**base, "params": keep, "weight_decay": 0.0}
            if router_lr is not None:
                g_keep["lr"] = router_lr
            out.append(g_keep)
    return out