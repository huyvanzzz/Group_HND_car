from types import SimpleNamespace

import torch

from efficient_vlm_ad.config import GenerationConfig
from efficient_vlm_ad.pipeline import _generate_batch


class CapturingModel:
    def __init__(self):
        self.kwargs = None

    def generate_from_features(self, **kwargs):
        self.kwargs = kwargs
        return torch.ones((1, 2), dtype=torch.long)


def test_generate_batch_uses_base_generation_kwargs_only():
    model = CapturingModel()
    cfg = SimpleNamespace(
        generation=GenerationConfig(max_new_tokens=32, num_beams=3, early_stopping=True, length_penalty=0.8),
        training=SimpleNamespace(vision_training="feature_cache"),
    )
    batch = {
        "input_ids": torch.ones((1, 3), dtype=torch.long),
        "attention_mask": torch.ones((1, 3), dtype=torch.long),
        "visual_features": torch.zeros((1, 6, 4, 8)),
    }

    _generate_batch(model, batch, cfg)

    assert model.kwargs["max_new_tokens"] == 32
    assert model.kwargs["num_beams"] == 3
    assert model.kwargs["early_stopping"] is True
    assert model.kwargs["length_penalty"] == 0.8
    removed_keys = ["repetition" + "_penalty", "no_repeat" + "_ngram_size", "renormalize" + "_logits"]
    assert not any(key in model.kwargs for key in removed_keys)
