import torch

from efficient_vlm_ad.config import GenerationConfig
from efficient_vlm_ad.modeling.factory import build_text_and_tokenizer
from efficient_vlm_ad.tokenization import decode_generated_predictions
from efficient_vlm_ad.tokenization import ensure_special_less_than_token
from efficient_vlm_ad.tokenization import repair_less_than_token_after_checkpoint


class DummyTokenizer:
    def __init__(self):
        self.tokens = {"</s>": 0}
        self.added = []

    def get_vocab(self):
        return dict(self.tokens)

    def add_tokens(self, token):
        if isinstance(token, str):
            token = [token]
        added = 0
        for item in token:
            if item not in self.tokens:
                self.tokens[item] = len(self.tokens)
                self.added.append(item)
                added += 1
        return added

    def convert_tokens_to_ids(self, token):
        return self.tokens[token]

    def __len__(self):
        return len(self.tokens)


class DummyModel:
    def __init__(self, rows):
        self.config = type("Config", (), {})()
        self.emb = torch.nn.Embedding(rows, 4)
        self.resize_to = None

    def get_input_embeddings(self):
        return self.emb

    def resize_token_embeddings(self, size):
        self.resize_to = size
        self.emb = torch.nn.Embedding(size, 4)


def test_less_than_token_adds_without_resize_when_embedding_has_padding_rows():
    tokenizer = DummyTokenizer()
    model = DummyModel(rows=4)

    metadata = ensure_special_less_than_token(tokenizer, model)

    assert tokenizer.added == ["<"]
    assert model.resize_to is None
    assert metadata["token_id"] == 1
    assert metadata["initialized"] is False


def test_less_than_token_resizes_only_when_needed():
    tokenizer = DummyTokenizer()
    model = DummyModel(rows=1)

    ensure_special_less_than_token(tokenizer, model)

    assert model.resize_to == 2


def test_less_than_token_initializes_from_greater_than_when_available():
    tokenizer = DummyTokenizer()
    tokenizer.tokens[">"] = 1
    model = DummyModel(rows=4)
    with torch.no_grad():
        model.emb.weight[1].fill_(3.0)

    metadata = ensure_special_less_than_token(tokenizer, model)

    assert metadata["initialized"] is True
    assert metadata["initialized_from_token_id"] == 1
    assert torch.equal(model.emb.weight[metadata["token_id"]], model.emb.weight[1])


def test_repair_less_than_token_after_old_checkpoint_metadata():
    tokenizer = DummyTokenizer()
    tokenizer.tokens[">"] = 1
    tokenizer.tokens["<"] = 2
    model = DummyModel(rows=4)
    with torch.no_grad():
        model.emb.weight[1].fill_(5.0)
        model.emb.weight[2].fill_(99.0)

    report = repair_less_than_token_after_checkpoint(tokenizer, model, metadata={})

    assert report["repaired"] is True
    assert torch.equal(model.emb.weight[2], model.emb.weight[1])


def test_decode_generated_predictions_keeps_object_tag_open_bracket():
    tokenizer = DummyTokenizer()

    def batch_decode(rows, skip_special_tokens=True):
        assert skip_special_tokens is True
        return ["< c1,CAM_FRONT,100.0,200.0>"]

    tokenizer.batch_decode = batch_decode

    assert decode_generated_predictions(tokenizer, [[1, 2]]) == ["<c1,CAM_FRONT,100.0,200.0>"]


def test_matching_checkpoint_metadata_preserves_learned_less_than_row():
    tokenizer = DummyTokenizer()
    tokenizer.tokens.update({">": 1, "<": 2})
    model = DummyModel(rows=4)
    with torch.no_grad():
        model.emb.weight[2].fill_(99.0)
    report = repair_less_than_token_after_checkpoint(
        tokenizer, model, {"tokenizer": {"less_than_token": {"token_id": 2}}}
    )
    assert report["repaired"] is False
    assert torch.all(model.emb.weight[2] == 99.0)


def test_less_than_initializes_untied_output_embeddings():
    tokenizer = DummyTokenizer()
    tokenizer.tokens[">"] = 1
    model = DummyModel(rows=4)
    output = torch.nn.Linear(4, 4, bias=False)
    model.get_output_embeddings = lambda: output
    with torch.no_grad():
        model.emb.weight[1].fill_(3.0)
        output.weight[1].fill_(7.0)
    ensure_special_less_than_token(tokenizer, model)
    assert torch.all(model.emb.weight[2] == 3.0)
    assert torch.all(output.weight[2] == 7.0)


def test_decode_generated_predictions_keeps_non_tag_less_than_spaced():
    tokenizer = DummyTokenizer()

    def batch_decode(rows, skip_special_tokens=True):
        assert skip_special_tokens is True
        return ["speed< 10 km/h and < c2,CAM_BACK,1,2>"]

    tokenizer.batch_decode = batch_decode

    assert decode_generated_predictions(tokenizer, [[1, 2]]) == ["speed < 10 km/h and <c2,CAM_BACK,1,2>"]


def test_decode_generated_predictions_separates_attached_full_and_partial_tags():
    tokenizer = DummyTokenizer()
    tokenizer.batch_decode = lambda rows, skip_special_tokens=True: [
        "that< c1", "that<c1", "that< c1,CAM_FRONT,100.0,200.0>",
        "that< c12, CAM_BACK,1,2>", "speed< 10 km/h",
    ]

    assert decode_generated_predictions(tokenizer, []) == [
        "that <c1", "that <c1", "that <c1,CAM_FRONT,100.0,200.0>",
        "that <c12, CAM_BACK,1,2>", "speed < 10 km/h",
    ]


def test_build_text_and_tokenizer_adds_less_than_token_for_hf_models(monkeypatch):
    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_id, revision=None):
            return DummyTokenizer()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_id, revision=None):
            return DummyModel(rows=1)

    import sys
    import types

    fake_transformers = types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer, AutoModelForSeq2SeqLM=FakeAutoModel)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    cfg = types.SimpleNamespace(
        model=types.SimpleNamespace(
            text=types.SimpleNamespace(model_id="hf_t5", revision=None, d_model=4),
        ),
        generation=GenerationConfig(),
    )

    model, tokenizer = build_text_and_tokenizer(cfg)

    assert "<" in tokenizer.get_vocab()
    assert model.resize_to == 2

