import torch

from efficient_vlm_ad.tokenization import ensure_special_less_than_token


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


def test_less_than_token_resizes_only_when_needed():
    tokenizer = DummyTokenizer()
    model = DummyModel(rows=1)

    ensure_special_less_than_token(tokenizer, model)

    assert model.resize_to == 2

