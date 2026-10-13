from __future__ import annotations

import re
from typing import Any


def ensure_special_less_than_token(tokenizer: Any, model: Any) -> dict[str, int | bool]:
    before_len = len(tokenizer)
    added = 0
    if "<" not in tokenizer.get_vocab():
        added = int(tokenizer.add_tokens("<"))
    token_id = int(tokenizer.convert_tokens_to_ids("<"))
    after_len = len(tokenizer)
    embedding_rows = int(model.get_input_embeddings().weight.shape[0])
    resized = False
    if token_id >= embedding_rows or after_len > embedding_rows:
        model.resize_token_embeddings(after_len)
        embedding_rows = int(model.get_input_embeddings().weight.shape[0])
        resized = True
    source_token_id = _less_than_init_source_token_id(tokenizer)
    initialized = False
    if added and source_token_id is not None and source_token_id != token_id:
        _copy_embedding_row(model, token_id, source_token_id)
        initialized = True
    return {
        "before_len": before_len,
        "after_len": after_len,
        "token_id": token_id,
        "embedding_rows": embedding_rows,
        "resized": resized,
        "initialized": initialized,
        "initialized_from_token_id": source_token_id if initialized else None,
    }


def less_than_token_metadata(tokenizer: Any) -> dict[str, int | str | bool | None]:
    token_id = int(tokenizer.convert_tokens_to_ids("<")) if "<" in tokenizer.get_vocab() else None
    return {"token": "<", "token_id": token_id, "normal_token": True}


def repair_less_than_token_after_checkpoint(tokenizer: Any, model: Any, metadata: dict[str, Any]) -> dict[str, int | bool | None]:
    current = less_than_token_metadata(tokenizer)
    saved = (metadata.get("tokenizer") or {}).get("less_than_token") if isinstance(metadata, dict) else None
    if saved and saved.get("token_id") == current["token_id"]:
        return {"repaired": False, "token_id": current["token_id"], "source_token_id": None}
    token_id = current["token_id"]
    source_token_id = _less_than_init_source_token_id(tokenizer)
    if token_id is None or source_token_id is None:
        return {"repaired": False, "token_id": token_id, "source_token_id": source_token_id}
    _copy_embedding_row(model, int(token_id), int(source_token_id))
    return {"repaired": True, "token_id": int(token_id), "source_token_id": int(source_token_id)}


def decode_generated_predictions(tokenizer: Any, rows: Any) -> list[str]:
    decoded = tokenizer.batch_decode(rows, skip_special_tokens=True)
    return [_normalize_object_tag_spacing(text) for text in decoded]


def _normalize_object_tag_spacing(text: str) -> str:
    text = re.sub(r"<\s+(c\d+,\s*CAM_[A-Za-z_]+)", r"<\1", text)
    text = re.sub(r"(?<=[A-Za-z0-9])<\s*(?!c\d+,\s*CAM_)", " < ", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def _less_than_init_source_token_id(tokenizer: Any) -> int | None:
    for token in (">", "▁>"):
        try:
            token_id = tokenizer.convert_tokens_to_ids(token)
        except Exception:
            continue
        if isinstance(token_id, int) and token_id >= 0 and token_id != getattr(tokenizer, "unk_token_id", None):
            return token_id
    try:
        encoded = tokenizer(">", add_special_tokens=False).input_ids
        return int(encoded[-1]) if encoded else None
    except Exception:
        return None


def _copy_embedding_row(model: Any, target_id: int, source_id: int) -> None:
    import torch

    with torch.no_grad():
        input_embeddings = model.get_input_embeddings()
        input_embeddings.weight[target_id].copy_(input_embeddings.weight[source_id])
        output_embeddings = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
        if output_embeddings is not None and output_embeddings.weight.shape[0] > max(target_id, source_id):
            output_embeddings.weight[target_id].copy_(output_embeddings.weight[source_id])

