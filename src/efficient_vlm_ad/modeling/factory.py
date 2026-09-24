from __future__ import annotations

import torch
from torch import nn

from .vision import LegacyVitPatchExtractor, RepVitFeatureExtractor
from .vlm import EfficientVLMForAD


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

    def generate(self, *, inputs_embeds, attention_mask, max_new_tokens, num_beams=1):
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


def build_text_and_tokenizer(cfg):
    if cfg.model.text.model_id == "fake_t5":
        return TinySeq2Seq(cfg.model.text.d_model), SimpleTokenizer()
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text.model_id, revision=cfg.model.text.revision)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model.text.model_id, revision=cfg.model.text.revision)
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


def build_vlm_model(cfg):
    text_model, tokenizer = build_text_and_tokenizer(cfg)
    model = EfficientVLMForAD(
        text_model=text_model,
        vision_dim=cfg.model.vision.output_dim,
        d_model=cfg.model.text.d_model,
        seq_len=cfg.model.vision.seq_len,
        gpa_hidden_size=cfg.training.gpa_hidden_size,
    )
    return model, tokenizer

