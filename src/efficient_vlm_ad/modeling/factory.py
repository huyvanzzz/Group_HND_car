from __future__ import annotations

import torch
from torch import nn

from .adapters import DynamicInstructionAdapter
from .vision import LegacyVitPatchExtractor, RepVitFeatureExtractor
from .vlm import EfficientVLMForAD, EndToEndEfficientVLMForAD


class SimpleTokenizer:
    pad_token_id = 0

    def __init__(self) -> None:
        self.vocab = {"<pad>": 0, "</s>": 1, "<": 2}
        self.inv_vocab = {0: "<pad>", 1: "</s>", 2: "<"}

    def get_vocab(self):
        return dict(self.vocab)

    def add_tokens(self, token):
        tokens = [token] if isinstance(token, str) else list(token)
        added = 0
        for item in tokens:
            if item not in self.vocab:
                idx = len(self.vocab)
                self.vocab[item] = idx
                self.inv_vocab[idx] = item
                added += 1
        return added

    def convert_tokens_to_ids(self, token):
        return self.vocab[token]

    def __len__(self):
        return len(self.vocab)

    def _encode_text(self, text: str) -> list[int]:
        ids = []
        for piece in text.strip().split():
            if piece not in self.vocab:
                idx = len(self.vocab)
                self.vocab[piece] = idx
                self.inv_vocab[idx] = piece
            ids.append(self.vocab[piece])
        return ids or [self.pad_token_id]

    def __call__(self, texts, padding=True, return_tensors="pt"):
        encoded = [self._encode_text(text) for text in texts]
        max_len = max(len(row) for row in encoded)
        ids, masks = [], []
        for row in encoded:
            mask = [1] * len(row)
            row = row + [self.pad_token_id] * (max_len - len(row))
            mask = mask + [0] * (max_len - len(mask))
            ids.append(row)
            masks.append(mask)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }

    def decode(self, ids, skip_special_tokens=True):
        values = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        words = []
        for idx in values:
            if skip_special_tokens and int(idx) in {self.pad_token_id, 1}:
                continue
            words.append(self.inv_vocab.get(int(idx), "road"))
        return " ".join(words).strip() or "road"

    def batch_decode(self, rows, skip_special_tokens=True):
        return [self.decode(row, skip_special_tokens=skip_special_tokens) for row in rows]


class TinySeq2Seq(nn.Module):
    def __init__(self, d_model: int, vocab_size: int = 4096) -> None:
        super().__init__()
        self.config = type("Config", (), {"d_model": d_model, "decoder_start_token_id": 0})()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def get_input_embeddings(self):
        return self.emb

    def resize_token_embeddings(self, size: int):
        if size <= self.emb.num_embeddings:
            return
        new_emb = nn.Embedding(size, self.emb.embedding_dim)
        new_emb.weight.data[: self.emb.num_embeddings] = self.emb.weight.data
        self.emb = new_emb
        self.head = nn.Linear(self.emb.embedding_dim, size)

    def forward(self, *, inputs_embeds, attention_mask, labels=None):
        pooled = inputs_embeds.mean(dim=1)
        logits = self.head(pooled).unsqueeze(1)
        loss = logits.sum() * 0
        if labels is not None:
            target = labels[:, 0].clamp_min(0)
            loss = torch.nn.functional.cross_entropy(logits[:, 0], target)
        return type("Output", (), {"loss": loss, "logits": logits})()

    def generate(
        self,
        *,
        inputs_embeds,
        attention_mask,
        max_new_tokens,
        num_beams=1,
        early_stopping=False,
        length_penalty=1.0,
    ):
        batch = inputs_embeds.shape[0]
        return torch.ones((batch, max(1, min(max_new_tokens, 4))), dtype=torch.long, device=inputs_embeds.device)


class FakeVisionEncoder(nn.Module):
    def __init__(self, seq_len: int, output_dim: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.output_dim = output_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch, views = images.shape[:2]
        base = images.mean(dim=(-1, -2, -3), keepdim=False).view(batch, views, 1, 1)
        return base.expand(batch, views, self.seq_len, self.output_dim).contiguous()

class pruningBlock(nn.Module):
    def __init__(self, fastv_config):
        super().__init__()
        self.fastv_config = fastv_config

    def farway_point_sampling(self, hidden_states, k, select_indices=None):
        def compute_distance_matrix(states):
            normalized = torch.nn.functional.normalize(states, p=2, dim=1)
            cosine_sim = torch.matmul(normalized, normalized.T)
            distances = 1 - cosine_sim
            distances.fill_diagonal_(float('inf'))
            return distances
        
        def find_most_isolated_point(dist_matrix, excluded_mask):
            available_dist = dist_matrix.clone()
            available_dist[:, excluded_mask] = float('-inf')
            row_mins, _ = available_dist.min(dim=1)
            row_mins[excluded_mask] = float('-inf')
            return row_mins.argmax()
        
        def find_farthest_point(dist_matrix, selected_mask, excluded_mask):
            if selected_mask.sum() == 0:
                return find_most_isolated_point(dist_matrix, excluded_mask)
            
            selected_distances = dist_matrix[selected_mask, :]
            min_distances_to_selected, _ = selected_distances.min(dim=0)
            min_distances_to_selected[excluded_mask] = float('-inf')
            return min_distances_to_selected.argmax()
        
        if hidden_states.dim() == 3:
            states_2d = hidden_states[0] 
        else:
            states_2d = hidden_states
        
        distance_matrix = compute_distance_matrix(states_2d)
        n = distance_matrix.shape[0]
        
        selected_mask = torch.zeros(n, dtype=torch.bool, device=hidden_states.device)
        if select_indices is not None:
            selected_mask[select_indices] = True
        
        result = list(select_indices) if select_indices is not None else []
        
        for _ in range(k):
            next_idx = find_farthest_point(distance_matrix, selected_mask, selected_mask)
            selected_mask[next_idx] = True
            result.append(next_idx.item())
        
        return torch.tensor(result[-k:], device=hidden_states.device)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        *args,
        **kwargs
    ):
        batch_size, seq_len, d_model = hidden_states.shape
        
        FASTV_r = self.fastv_config['fastv_r']
        image_start_index = self.fastv_config['image_start_index']
        image_token_length = self.fastv_config['image_token_length']
        num_views = len(image_start_index)
        
        batch_keep_indices = []
        
        # Handle each item in the batch
        for b in range(batch_size):
            selected_images = torch.zeros(seq_len, dtype=torch.long, device=hidden_states.device)
            for i in range(num_views):
                start_i = image_start_index[i]
                selected_images[start_i : start_i + image_token_length] = 1
            
            image_token_indices = torch.where(selected_images == 1)[0]
            gen_attention_mask = torch.ones(seq_len, dtype=torch.bool, device=hidden_states.device)
            
            for i in range(num_views):
                start_idx = i * image_token_length
                end_idx = start_idx + image_token_length
                current_image_token_indices = image_token_indices[start_idx:end_idx]
                
                k_keep = int(image_token_length * FASTV_r[i])
                keep_indices_for_view = self.farway_point_sampling(
                    hidden_states=hidden_states[b:b+1, current_image_token_indices, :],
                    k=k_keep, 
                )
                keep_positions = current_image_token_indices[keep_indices_for_view]
                
                gen_attention_mask[current_image_token_indices] = False
                gen_attention_mask[keep_positions] = True
            
            keep_indexs = torch.where(gen_attention_mask == True)[0].sort().values
            batch_keep_indices.append(keep_indexs)
            
        keep_indices = torch.stack(batch_keep_indices, dim=0)
        self.last_keep_indices = keep_indices # <--- Save this for the decoder cross-attention!
        new_seq_len = keep_indices.shape[1]
        
        # Prune hidden states
        hidden_states = torch.gather(
            hidden_states, 
            dim=1, 
            index=keep_indices.unsqueeze(-1).expand(-1, -1, d_model)
        )
        
        # Prune attention mask
        if attention_mask is not None:
            if attention_mask.size(2) > 1:
                idx2 = keep_indices.view(batch_size, 1, new_seq_len, 1).expand(-1, 1, -1, attention_mask.size(3))
                attention_mask = torch.gather(attention_mask, dim=2, index=idx2)
                idx3 = keep_indices.view(batch_size, 1, 1, new_seq_len).expand(-1, 1, new_seq_len, -1)
                attention_mask = torch.gather(attention_mask, dim=3, index=idx3)
            else:
                idx = keep_indices.view(batch_size, 1, 1, new_seq_len)
                attention_mask = torch.gather(attention_mask, dim=-1, index=idx)
                
        self.pruned_attention_mask = attention_mask
                
        # Prune position bias (Handles T5 relative positions)
        if position_bias is not None:
            num_heads = position_bias.shape[1]
            if position_bias.size(0) == 1 and batch_size > 1:
                position_bias = position_bias.expand(batch_size, -1, -1, -1)
                
            pos_bias = torch.gather(
                position_bias, dim=2, 
                index=keep_indices.view(batch_size, 1, new_seq_len, 1).expand(-1, num_heads, -1, position_bias.size(3))
            )
            position_bias = torch.gather(
                pos_bias, dim=3,
                index=keep_indices.view(batch_size, 1, 1, new_seq_len).expand(-1, num_heads, new_seq_len, -1)
            )
        if kwargs.get("use_cache", False):
            return (hidden_states, None, position_bias)
        return (hidden_states, position_bias, None)


class T5BlockWrapper(nn.Module):
    def __init__(self, original_block, pruning_module):
        super().__init__()
        self.original_block = original_block
        self.pruning_module = pruning_module
        
    def forward(self, *args, **kwargs):
        if hasattr(self.pruning_module, 'pruned_attention_mask'):
            if 'attention_mask' in kwargs:
                kwargs['attention_mask'] = self.pruning_module.pruned_attention_mask
            elif len(args) > 1:
                args = list(args)
                args[1] = self.pruning_module.pruned_attention_mask
                args = tuple(args)
        return self.original_block(*args, **kwargs)


def build_text_and_tokenizer(cfg):
    if cfg.model.text.model_id == "fake_t5":
        return TinySeq2Seq(cfg.model.text.d_model), SimpleTokenizer()
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text.model_id, revision=cfg.model.text.revision)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model.text.model_id, revision=cfg.model.text.revision)

    if cfg.model.fastv is not None:
        fastv_config = {
            'fastv_k': cfg.model.fastv.k,
            'fastv_r': list(cfg.model.fastv.r),
            'image_start_index': list(cfg.model.fastv.image_start_index),
            'image_token_length': cfg.model.fastv.image_token_length
        }
    
        pruning_block = pruningBlock(fastv_config)
        model.encoder.block.insert(fastv_config['fastv_k'], pruning_block)
    
        model.pruning_block = pruning_block
        
        for i in range(fastv_config['fastv_k'] + 1, len(model.encoder.block)):
            model.encoder.block[i] = T5BlockWrapper(model.encoder.block[i], pruning_block)

    return model, tokenizer


def build_vision_encoder(cfg):
    if cfg.model.vision.name == "fake_vision":
        return FakeVisionEncoder(cfg.model.vision.seq_len, cfg.model.vision.output_dim), None
    if cfg.model.vision.name == "repvit_m1_5":
        import timm
        from timm.data import create_transform, resolve_model_data_config

        backbone = timm.create_model(f"hf_hub:{cfg.model.vision.model_id}", pretrained=True)
        backbone.eval()
        transform = create_transform(**resolve_model_data_config(backbone), is_training=False)
        return RepVitFeatureExtractor(backbone), transform
    if cfg.model.vision.name == "legacy_vit_b32_patch":
        from torchvision.models import ViT_B_32_Weights, vit_b_32

        weights = ViT_B_32_Weights.IMAGENET1K_V1
        vit = vit_b_32(weights=weights)
        vit.eval()
        return LegacyVitPatchExtractor(vit), weights.transforms()
    raise ValueError(f"Unknown vision model: {cfg.model.vision.name}")


def build_visual_adapter(cfg):
    if cfg.model.adapter.name == "none":
        return None
    if cfg.model.adapter.name == "minidrive_di":
        return DynamicInstructionAdapter(
            vision_dim=cfg.model.vision.output_dim,
            text_dim=cfg.model.text.d_model,
            num_heads=cfg.model.adapter.num_heads,
            dropout=cfg.model.adapter.dropout,
            residual_scale=cfg.model.adapter.residual_scale,
        )
    raise ValueError(f"Unknown adapter: {cfg.model.adapter.name}")


def build_vlm_model(cfg):
    text_model, tokenizer = build_text_and_tokenizer(cfg)
    model = EfficientVLMForAD(
        text_model=text_model,
        visual_adapter=build_visual_adapter(cfg),
        vision_dim=cfg.model.vision.output_dim,
        d_model=cfg.model.text.d_model,
        seq_len=cfg.model.vision.seq_len,
    )
    return model, tokenizer


def build_end_to_end_vlm_model(cfg, vision_encoder: nn.Module):
    text_model, tokenizer = build_text_and_tokenizer(cfg)
    model = EndToEndEfficientVLMForAD(
        vision_encoder=vision_encoder,
        text_model=text_model,
        visual_adapter=build_visual_adapter(cfg),
        vision_dim=cfg.model.vision.output_dim,
        d_model=cfg.model.text.d_model,
        seq_len=cfg.model.vision.seq_len,
    )
    return model, tokenizer
