from types import SimpleNamespace

import torch

from efficient_vlm_ad.modeling.vlm import EfficientVLMForAD


class FakeSeq2Seq(torch.nn.Module):
    def __init__(self, d_model=384, vocab_size=16):
        super().__init__()
        self.config = SimpleNamespace(d_model=d_model, decoder_start_token_id=0)
        self.emb = torch.nn.Embedding(vocab_size, d_model)
        self.last_inputs_embeds = None
        self.last_attention_mask = None

    def get_input_embeddings(self):
        return self.emb

    def forward(self, *, inputs_embeds, attention_mask, labels=None):
        self.last_inputs_embeds = inputs_embeds
        self.last_attention_mask = attention_mask
        return SimpleNamespace(loss=inputs_embeds.sum() * 0, logits=inputs_embeds)


def test_vlm_fuses_visual_tokens_before_text_and_builds_combined_mask():
    text_model = FakeSeq2Seq(d_model=384)
    model = EfficientVLMForAD(
        text_model=text_model,
        vision_dim=512,
        d_model=384,
        seq_len=49,
        gpa_hidden_size=8,
    )
    input_ids = torch.tensor([[1, 2, 0], [3, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    visual_features = torch.randn(2, 6, 49, 512)

    out = model.forward_from_features(
        input_ids=input_ids,
        attention_mask=attention_mask,
        visual_features=visual_features,
        labels=torch.ones(2, 2, dtype=torch.long),
    )

    assert out.loss is not None
    assert text_model.last_inputs_embeds.shape == (2, 52, 384)
    assert text_model.last_attention_mask.shape == (2, 52)
    assert text_model.last_attention_mask[:, :49].sum().item() == 98
    assert text_model.last_attention_mask[:, 49:].tolist() == attention_mask.tolist()
    assert model.last_gpa_weights.shape == (2, 6)


def test_vlm_forward_delegates_to_forward_from_features_for_ddp_wrappers():
    text_model = FakeSeq2Seq(d_model=384)
    model = EfficientVLMForAD(
        text_model=text_model,
        vision_dim=512,
        d_model=384,
        seq_len=49,
        gpa_hidden_size=8,
    )

    out = model(
        input_ids=torch.tensor([[1, 2, 0]]),
        attention_mask=torch.tensor([[1, 1, 0]]),
        visual_features=torch.randn(1, 6, 49, 512),
        labels=torch.ones(1, 2, dtype=torch.long),
    )

    assert out.loss is not None
    assert text_model.last_inputs_embeds.shape == (1, 52, 384)
