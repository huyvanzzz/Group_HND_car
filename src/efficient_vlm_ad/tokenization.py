from __future__ import annotations

from typing import Any


def ensure_special_less_than_token(tokenizer: Any, model: Any) -> dict[str, int | bool]:
    before_len = len(tokenizer)
    if "<" not in tokenizer.get_vocab():
        tokenizer.add_tokens("<")
    token_id = int(tokenizer.convert_tokens_to_ids("<"))
    after_len = len(tokenizer)
    embedding_rows = int(model.get_input_embeddings().weight.shape[0])
    resized = False
    if token_id >= embedding_rows or after_len > embedding_rows:
        model.resize_token_embeddings(after_len)
        embedding_rows = int(model.get_input_embeddings().weight.shape[0])
        resized = True
    return {
        "before_len": before_len,
        "after_len": after_len,
        "token_id": token_id,
        "embedding_rows": embedding_rows,
        "resized": resized,
    }

