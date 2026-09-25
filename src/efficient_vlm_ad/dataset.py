from __future__ import annotations

from typing import Any

import torch

from .data import format_prompt


class VQABatchCollator:
    def __init__(self, tokenizer: Any, label_pad_id: int = -100) -> None:
        self.tokenizer = tokenizer
        self.label_pad_id = label_pad_id

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        questions = [format_prompt(item["question"]) for item in batch]
        answers = [item["answer"] for item in batch]
        images = torch.stack([item["images"] for item in batch], dim=0)

        encoded = self.tokenizer(questions, padding=True, return_tensors="pt")
        labels_encoded = self.tokenizer(answers, padding=True, return_tensors="pt")
        labels = labels_encoded["input_ids"].clone()
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            labels[labels == pad_id] = self.label_pad_id

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "images": images,
            "labels": labels,
        }

