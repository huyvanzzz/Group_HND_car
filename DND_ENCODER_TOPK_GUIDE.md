# HƯỚNG DẪN CHI TIẾT TÍCH HỢP DND (TOP-K) VÀO T5 ENCODER
**Dự án:** Efficient VLM for Autonomous Driving (`Group_HND_car`)  
**Môi trường chạy:** Notebook Full Training có Adapter (`DynamicInstructionAdapter` + `GPA`)

---

## 1. TỔNG QUAN LUỒNG HOẠT ĐỘNG VỚI ADAPTER & DND

Trong notebook của bạn (sử dụng cấu hình như `repvit_t5_efficient_mini_kaggle_2gpu_di_adapter.yaml`), luồng dữ liệu diễn ra như sau:

```
Visual Features [B, 6, 49, 512] (từ 6 camera)
       │
       ▼
[1] DynamicInstructionAdapter (Cross-attention giữa Visual Tokens và Text Tokens)
       │
       ▼
[2] Spatial Position Embeddings (Row + Col embeddings)
       │
       ▼
[3] GatedPoolingAttention (GPA) -> Gộp 6 view thành 1 view [B, 49, 512]
       │
       ▼
[4] MultiModalProjector (Linear 512 -> 384) -> [B, 49, 384]
       │
       ▼
[5] Concat với Text Embeddings -> Sequence chung [B, 49 + S_text, 384]
       │
       ▼
[6] T5 Encoder (8 blocks)
       ├─ Block 0, 1: Giữ nguyên (Trích xuất biểu diễn cơ sở ban đầu)
       ├─ Block 2..5: Bọc bởi DNDT5Block (Router -> Top-K 20% khó nhất -> Re-process -> Fusion)
       └─ Block 6, 7: Giữ nguyên (Ổn định representation trước khi sang Decoder)
       │
       ▼
[7] T5 Decoder -> Sinh chuỗi hành động / câu trả lời
       │
       ▼
[8] Loss = CrossEntropyLoss + λ_sd * DNDScoreDispersionLoss
```

---

## 2. DANH SÁCH CÁC FILE CẦN TẠO MỚI VÀ SỬA ĐỔI

| STT | File cần tác động | Thao tác | Mục đích |
|---|---|---|---|
| **1** | `src/efficient_vlm_ad/modeling/dnd.py` | **Tạo mới** | Định nghĩa `DNDRouter`, `DNDT5Block`, `DNDScoreDispersionLoss` |
| **2** | `src/efficient_vlm_ad/config.py` | **Sửa** | Thêm class `DNDConfig` và đưa vào `ModelConfig`, cập nhật `load_config()` |
| **3** | `src/efficient_vlm_ad/modeling/factory.py` | **Sửa** | Trong `build_text_and_tokenizer()`, bọc các encoder block trung gian bằng `DNDT5Block` |
| **4** | `src/efficient_vlm_ad/modeling/multimodal.py` | **Sửa** | Đảm bảo Router và tham số $\beta$ của DND luôn trainable (đặc biệt khi full train / finetune) |
| **5** | `src/efficient_vlm_ad/pipeline.py` | **Sửa** | Thu thập router scores và cộng `dnd_loss` vào `raw_loss` trong vòng lặp train |
| **6** | `src/efficient_vlm_ad/modeling/__init__.py` | **Sửa** | Export các lớp DND |
| **7** | `configs/..._di_adapter.yaml` | **Sửa** | Thêm khối cấu hình `dnd:` vào file cấu hình chạy notebook |

---

## 3. CHI TIẾT TỪNG BƯỚC THỰC HIỆN

### BƯỚC 1: Tạo file mới `src/efficient_vlm_ad/modeling/dnd.py`

Tạo file mới hoàn toàn chứa toàn bộ module của DND:

```python
from __future__ import annotations

import torch
from torch import nn


class DNDRouter(nn.Module):
    """Router tính điểm độ khó của từng token (d_model -> 1)."""
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, 1, bias=True)
        # Khởi tạo trọng số bằng 0 theo paper để ban đầu điểm số đồng đều
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: [B, S, D] -> scores: [B, S]
        return torch.sigmoid(self.linear(hidden_states)).squeeze(-1)


class DNDT5Block(nn.Module):
    """
    Wrapper bọc ngoài một T5Block ở Encoder.
    Thực hiện:
    1. Vanilla pass qua T5Block gốc
    2. Router chấm điểm độ khó
    3. Chọn Top-K token khó nhất
    4. Nested re-forward K token này qua cùng T5Block
    5. Fused output bằng cổng learnable beta
    """
    def __init__(self, original_block: nn.Module, d_model: int, k_target: float = 0.2, beta_init: float = 0.1) -> None:
        super().__init__()
        self.original_block = original_block
        self.router = DNDRouter(d_model)
        self.k_target = k_target
        self.beta = nn.Parameter(torch.tensor(beta_init, dtype=torch.float32))
        self._last_scores: torch.Tensor | None = None

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        # 1. Vanilla Pass
        vanilla_outputs = self.original_block(hidden_states, *args, **kwargs)
        X_v = vanilla_outputs[0]  # [B, S, D]
        B, S, D = X_v.shape

        # 2. Router Scoring
        scores = self.router(X_v)  # [B, S]
        self._last_scores = scores

        # 3. Top-K Selection
        k = max(1, int(self.k_target * S))
        topk_indices = torch.topk(scores, k=k, dim=1).indices  # [B, K]

        # 4. Pack K tokens
        gather_idx = topk_indices.unsqueeze(-1).expand(-1, -1, D)
        X_packed = torch.gather(X_v, dim=1, index=gather_idx)  # [B, K, D]

        # 5. Nested Pass
        # Khi re-forward chuỗi con K token, đặt position_bias=None để tránh lệch shape
        nested_kwargs = dict(kwargs)
        nested_kwargs["position_bias"] = None
        nested_kwargs["attention_mask"] = None
        nested_outputs = self.original_block(X_packed, *args, **nested_kwargs)
        X_d_packed = nested_outputs[0]  # [B, K, D]

        # 6. Unpack & Fusion
        X_d = torch.zeros_like(X_v)
        X_d.scatter_(dim=1, index=gather_idx, src=X_d_packed)

        # Công thức fusion: X_fused = (1 - beta * p) * X_v + (beta * p) * X_d
        gate = (self.beta * scores).unsqueeze(-1)  # [B, S, 1]
        X_fused = (1.0 - gate) * X_v + gate * X_d

        return (X_fused,) + vanilla_outputs[1:]


class DNDScoreDispersionLoss(nn.Module):
    """Score Dispersion Loss để router phân tán điểm, tăng tính phân biệt."""
    def __init__(self, lambda_sd: float = 3e-4) -> None:
        super().__init__()
        self.lambda_sd = lambda_sd

    def forward(self, layer_scores: list[torch.Tensor]) -> torch.Tensor:
        if not layer_scores:
            return torch.tensor(0.0)
        total_entropy = 0.0
        for scores in layer_scores:
            # Chuẩn hóa scores thành distribution p'
            p_norm = scores / (scores.sum(dim=-1, keepdim=True) + 1e-8)
            # Entropy: minimize H tương đương maximize entropy
            entropy = (p_norm * torch.log(p_norm + 1e-8)).sum(dim=-1).mean()
            total_entropy += entropy
        return self.lambda_sd * total_entropy
```

---

### BƯỚC 2: Cập nhật cấu hình trong `src/efficient_vlm_ad/config.py`

#### 2.1 Thêm Dataclass `DNDConfig` (ngay dưới `AdapterConfig`, khoảng dòng 68):
```python
@dataclass(frozen=True)
class DNDConfig:
    enabled: bool = False
    L_s: int = 2          # Bỏ qua 2 block đầu của Encoder
    L_e: int = -2         # Bỏ qua 2 block cuối (-2 nghĩa là len - 2)
    k_target: float = 0.2 # Chọn top 20% token khó nhất
    beta_init: float = 0.1
    lambda_sd: float = 3e-4
```

#### 2.2 Thêm trường `dnd` vào `ModelConfig`:
```python
@dataclass(frozen=True)
class ModelConfig:
    profile: str
    vision: VisionConfig
    text: TextConfig
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    dnd: DNDConfig = field(default_factory=DNDConfig)  # <--- THÊM DÒNG NÀY
```

#### 2.3 Cập nhật hàm `load_config()` (khoảng dòng 150):
Thêm đọc section `dnd` từ file yaml:
```python
    dnd_raw = model_raw.get("dnd", {})
    dnd_config = DNDConfig(
        enabled=bool(dnd_raw.get("enabled", False)),
        L_s=int(dnd_raw.get("L_s", 2)),
        L_e=int(dnd_raw.get("L_e", -2)),
        k_target=float(dnd_raw.get("k_target", 0.2)),
        beta_init=float(dnd_raw.get("beta_init", 0.1)),
        lambda_sd=float(dnd_raw.get("lambda_sd", 3e-4)),
    )
```
Và truyền `dnd=dnd_config` vào `ModelConfig(...)`.

---

### BƯỚC 3: Bọc DND vào T5 Encoder trong `src/efficient_vlm_ad/modeling/factory.py`

#### 3.1 Thêm import ở đầu file:
```python
from .dnd import DNDT5Block
```

#### 3.2 Sửa hàm `build_text_and_tokenizer(cfg)`:
Thay thế hàm hiện tại bằng:
```python
def build_text_and_tokenizer(cfg):
    if cfg.model.text.model_id == "fake_t5":
        return TinySeq2Seq(cfg.model.text.d_model), SimpleTokenizer()
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text.model_id, revision=cfg.model.text.revision)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model.text.model_id, revision=cfg.model.text.revision)

    # ===== TÍCH HỢP DND VÀO T5 ENCODER =====
    if getattr(getattr(cfg.model, "dnd", None), "enabled", False):
        num_blocks = len(model.encoder.block)
        L_s = cfg.model.dnd.L_s
        L_e = cfg.model.dnd.L_e if cfg.model.dnd.L_e > 0 else (num_blocks + cfg.model.dnd.L_e)
        d_model = cfg.model.text.d_model

        for i in range(L_s, L_e):
            original_block = model.encoder.block[i]
            model.encoder.block[i] = DNDT5Block(
                original_block=original_block,
                d_model=d_model,
                k_target=cfg.model.dnd.k_target,
                beta_init=cfg.model.dnd.beta_init,
            )

    return model, tokenizer
```

---

### BƯỚC 4: Quản lý Trainable Parameters trong `src/efficient_vlm_ad/modeling/multimodal.py`

Khi chạy Notebook Full Train với Adapter (`DynamicInstructionAdapter`), model trải qua 2 stage:
1. `align`: Freeze text model, chỉ train Adapter + GPA + Projector.
2. `finetune`: Full train toàn bộ (Text model + Adapter + GPA + Projector).

Để tham số Router và $\beta$ của DND luôn được cập nhật chính xác, sửa hàm `set_trainable_for_stage`:

```python
def set_trainable_for_stage(modules: nn.ModuleDict, cfg, stage: str) -> None:
    if stage not in {"align", "finetune"}:
        raise ValueError("stage must be 'align' or 'finetune'")

    train_vision = getattr(getattr(cfg, "training", None), "vision_training", "feature_cache") == "end_to_end"
    _set_module_trainable(modules["vision"], train_vision)
    _set_module_trainable(modules["text"], stage == "finetune")

    # ĐẢM BẢO DND PARAMS ĐƯỢC UNFREEZE KHI FINETUNE (HOẶC CẢ ALIGN NẾU MUỐN)
    if getattr(getattr(cfg.model, "dnd", None), "enabled", False):
        text_mod = modules["text"]
        if hasattr(text_mod, "encoder"):
            for block in text_mod.encoder.block:
                if isinstance(block, DNDT5Block) or hasattr(block, "router"):
                    # Ở stage finetune, unfreeze router và beta
                    _set_module_trainable(block.router, stage == "finetune")
                    block.beta.requires_grad = (stage == "finetune")

    for name in ("gpa", "projector", "spatial_pos", "modal_embeddings", "visual_adapter"):
        if name in modules:
            _set_module_trainable(modules[name], True)

    if cfg.model.vision.name.startswith("repvit") and not train_vision:
        modules["vision"].eval()
```

---

### BƯỚC 5: Cập nhật Training Loop trong `src/efficient_vlm_ad/pipeline.py`

#### 5.1 Import Loss Function ở đầu file:
```python
from .modeling.dnd import DNDScoreDispersionLoss
```

#### 5.2 Thêm Helper thu thập điểm router (đặt ngay trước hàm `train_stage`):
```python
def _collect_dnd_scores(model: torch.nn.Module) -> list[torch.Tensor]:
    scores = []
    text_model = getattr(model, "text_model", None)
    if text_model is not None and hasattr(text_model, "encoder"):
        for block in text_model.encoder.block:
            if hasattr(block, "_last_scores") and block._last_scores is not None:
                scores.append(block._last_scores)
    return scores
```

#### 5.3 Cập nhật tính toán Loss trong vòng lặp huấn luyện (`train_stage`, khoảng dòng 811):

**Đoạn code gốc:**
```python
            with accelerator.autocast():
                output = model(**_batch_model_inputs(batch, cfg))
                raw_loss = output.loss
                loss = raw_loss / cfg.training.gradient_accumulation_steps
```

**Sửa thành:**
```python
            with accelerator.autocast():
                output = model(**_batch_model_inputs(batch, cfg))
                raw_loss = output.loss

                # ===== CỘNG DND LOSS =====
                dnd_loss = torch.tensor(0.0, device=raw_loss.device)
                if getattr(getattr(cfg.model, "dnd", None), "enabled", False):
                    unwrapped_m = accelerator.unwrap_model(model)
                    layer_scores = _collect_dnd_scores(unwrapped_m)
                    if layer_scores:
                        dnd_loss_fn = DNDScoreDispersionLoss(lambda_sd=cfg.model.dnd.lambda_sd)
                        dnd_loss = dnd_loss_fn(layer_scores)

                total_raw_loss = raw_loss + dnd_loss
                loss = total_raw_loss / cfg.training.gradient_accumulation_steps
```

---

### BƯỚC 6: Export Module trong `src/efficient_vlm_ad/modeling/__init__.py`

Thêm export vào file:
```python
from .dnd import DNDRouter, DNDT5Block, DNDScoreDispersionLoss

__all__ = [
    # ... các lớp cũ ...
    "DNDRouter",
    "DNDT5Block",
    "DNDScoreDispersionLoss",
]
```

---

### BƯỚC 7: Cập nhật file Config YAML sử dụng trong Notebook

Trong file cấu hình (ví dụ `configs/repvit_t5_efficient_mini_kaggle_2gpu_di_adapter.yaml`), thêm khối `dnd:` bên trong `model:`:

```yaml
model:
  profile: repvit_t5_efficient_mini_kaggle_2gpu_di_adapter
  vision:
    name: repvit_m1_5
    model_id: timm/repvit_m1_5.dist_450e_in1k
    output_dim: 512
    seq_len: 49
    image_size: 224
  text:
    model_id: google/t5-efficient-mini
    d_model: 384
  adapter:
    name: minidrive_di
    num_heads: 8
    dropout: 0.1
    residual_scale: 0.1
  dnd:                      # <--- KHỐI MỚI THÊM
    enabled: true           # Kích hoạt DND
    L_s: 2                  # Bắt đầu từ Encoder Block 2
    L_e: -2                 # Dừng trước 2 block cuối (với T5-mini 8 blocks -> là block 2, 3, 4, 5)
    k_target: 0.2           # Review 20% token khó nhất
    beta_init: 0.1          # Trọng số hòa trộn khởi tạo
    lambda_sd: 0.0003       # Hệ số Router Dispersion Loss
```

---

## 4. TƯƠNG THÍCH KHI CHẠY TRÊN KAGGLE NOTEBOOK

1. **Adapter đã có:** Pipeline của bạn đã có `DynamicInstructionAdapter` đặt trước GPA. DND nằm sau GPA và Projector, hoàn toàn độc lập nên **không gây xung đột tham số hay shape tensor**.
2. **Resume Checkpoint:** 
   - Nếu bạn resume checkpoint từ stage `align` sang `finetune`, T5-model lúc này được khởi tạo thêm các trọng số mới của `router` và `beta`. Hãy dùng chế độ nạp không nghiêm ngặt (`strict=False`) hoặc bắt đầu DND từ stage `finetune`.
3. **Overhead tính toán:**
   - Thêm 4 blocks DND x 385 params = **1,540 tham số** (quá nhỏ so với T5 30M+).
   - Chi phí FLOPs tăng thêm ~6-8% tổng model, hoàn toàn chạy mượt mà trên 2x T4 hoặc 2x P100 của Kaggle.
