import torch

from efficient_vlm_ad.dataset import VQABatchCollator


class ToyTokenizer:
    pad_token_id = 0

    def __call__(self, texts, padding=True, return_tensors="pt"):
        max_len = max(len(text.split()) for text in texts)
        ids = []
        mask = []
        for text in texts:
            row = list(range(1, len(text.split()) + 1))
            row_mask = [1] * len(row)
            row += [0] * (max_len - len(row))
            row_mask += [0] * (max_len - len(row_mask))
            ids.append(row)
            mask.append(row_mask)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }


def test_collator_builds_text_masks_and_ignores_label_padding():
    collator = VQABatchCollator(ToyTokenizer())
    batch = [
        {"question": "What now", "answer": "Go straight", "images": torch.zeros(6, 3, 224, 224)},
        {"question": "Why", "answer": "Stop", "images": torch.ones(6, 3, 224, 224)},
    ]

    out = collator(batch)

    assert out["input_ids"].shape == (2, 4)
    assert out["attention_mask"].tolist() == [[1, 1, 1, 1], [1, 1, 1, 0]]
    assert out["labels"].tolist() == [[1, 2], [1, -100]]
    assert out["images"].shape == (2, 6, 3, 224, 224)

