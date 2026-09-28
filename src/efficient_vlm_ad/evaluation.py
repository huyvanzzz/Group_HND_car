from __future__ import annotations

import json
from pathlib import Path
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


def simple_text_metrics(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {"exact_match": 0.0}
    exact = sum(row.get("prediction", "").strip().lower() == row.get("reference", "").strip().lower() for row in rows)
    return {"exact_match": exact / len(rows)}


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
