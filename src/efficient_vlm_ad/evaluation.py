from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from typing import Iterable


def write_predictions(path: str | Path, rows: Iterable[dict]) -> None:
    seen: set[int] = set()
    lines: list[str] = []
    for row in rows:
        eval_id = int(row["eval_id"])
        if eval_id in seen:
            raise ValueError(f"Duplicate eval_id in predictions: {eval_id}")
        seen.add(eval_id)
        lines.append(json.dumps(row, ensure_ascii=False))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def generation_diagnostics(rows: list[dict], *, max_new_tokens: int | None = None) -> dict[str, float | int]:
    count = len(rows)
    missing_open_tags = 0
    duplicate_object_tags = 0
    max_new_token_hits = 0
    numeric_repetition_sequences = 0
    repeated_trigram_rates: list[float] = []
    repeated_fourgram_rates: list[float] = []

    for row in rows:
        prediction = str(row.get("prediction", ""))
        missing_open_tags += len(_missing_open_object_tags(prediction))
        tags = _object_tags(prediction)
        duplicate_object_tags += sum(value - 1 for value in Counter(tags).values() if value > 1)
        repeated_trigram_rates.append(_repeated_ngram_rate(prediction, 3))
        repeated_fourgram_rates.append(_repeated_ngram_rate(prediction, 4))
        if _has_numeric_repetition(prediction):
            numeric_repetition_sequences += 1
        if max_new_tokens is not None and int(row.get("generated_token_count", -1)) >= max_new_tokens:
            max_new_token_hits += 1

    return {
        "prediction_count": count,
        "missing_open_object_tag_count": missing_open_tags,
        "duplicate_object_tag_count": duplicate_object_tags,
        "repeated_trigram_rate_mean": _mean(repeated_trigram_rates),
        "repeated_fourgram_rate_mean": _mean(repeated_fourgram_rates),
        "numeric_repetition_sequence_count": numeric_repetition_sequences,
        "max_new_token_hit_count": max_new_token_hits,
    }


def write_generation_diagnostics(path: str | Path, rows: list[dict], *, max_new_tokens: int | None = None) -> dict[str, float | int]:
    diagnostics = generation_diagnostics(rows, max_new_tokens=max_new_tokens)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return diagnostics


def simple_text_metrics(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {"exact_match": 0.0}
    exact = sum(row.get("prediction", "").strip().lower() == row.get("reference", "").strip().lower() for row in rows)
    return {"exact_match": exact / len(rows)}


def _missing_open_object_tags(text: str) -> list[str]:
    return re.findall(r"(?<!<)\bc\d+,CAM_[A-Z_]+(?:,[^>\s,]+){1,4}>", text)


def _object_tags(text: str) -> list[str]:
    return re.findall(r"<?c\d+,CAM_[A-Z_]+(?:,[^>\s,]+){1,4}>", text)


def _tokens(text: str) -> list[str]:
    return re.findall(r"\S+", text.lower())


def _repeated_ngram_rate(text: str, n: int) -> float:
    tokens = _tokens(text)
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]
    duplicates = sum(value - 1 for value in Counter(ngrams).values() if value > 1)
    return duplicates / len(ngrams)


def _has_numeric_repetition(text: str) -> bool:
    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    if len(numbers) < 6:
        return False
    run = 1
    last = None
    for number in numbers:
        run = run + 1 if number == last else 1
        if run >= 4:
            return True
        last = number
    return any(value >= 6 for value in Counter(numbers).values())


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def caption_metrics(rows: list[dict], skip_meteor: bool = False) -> dict[str, float]:
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.rouge.rouge import Rouge

        scorers = [(Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]), (Rouge(), "ROUGE-L"), (Cider(), "CIDEr")]
        if not skip_meteor:
            from pycocoevalcap.meteor.meteor import Meteor

            scorers.insert(1, (Meteor(), "METEOR"))
        refs = {int(row["eval_id"]): [row["reference"]] for row in rows}
        preds = {int(row["eval_id"]): [row["prediction"]] for row in rows}
        scores: dict[str, float] = {}
        for scorer, names in scorers:
            score, _ = scorer.compute_score(refs, preds)
            if isinstance(names, list):
                for name, value in zip(names, score):
                    scores[name] = float(value)
            else:
                scores[names] = float(score)
        return scores
    except Exception:
        return simple_text_metrics(rows)


def metric_display_values(metrics: dict[str, float]) -> dict[str, float]:
    scaled = {"BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr"}
    return {name: round(value * 100, 6) if name in scaled else value for name, value in metrics.items()}
