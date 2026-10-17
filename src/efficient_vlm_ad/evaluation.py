from __future__ import annotations

import json
import re
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


import re  # <-- Thêm vào đầu file cùng với json, Path, etc.


def clean_prediction_text(text: str) -> str:
    """Hậu xử lý văn bản dự đoán của T5 về đúng định dạng tag của DriveLM.
    
    Xử lý các trường hợp:
    1. Bỏ khoảng trắng sau '<' và trước '>' (ví dụ: '< c1' -> '<c1', '10.0 >' -> '10.0>')
    2. Bổ sung '<' nếu bị thiếu ở đầu tag (ví dụ: 'c2,CAM_FRONT...' -> '<c2,CAM_FRONT...')
    3. Thêm dấu cách trước '<' nếu bị dính liền với từ phía trước (ví dụ: 'that<c1' -> 'that <c1')
    """
    if not text:
        return text

    # 1. Bỏ khoảng trắng sau dấu '<' và trước dấu '>'
    text = re.sub(r'<\s+', '<', text)
    text = re.sub(r'\s+>', '>', text)

    # 2. Bổ sung '<' nếu bị thiếu ở đầu tag kết thúc bằng '>'
    text = re.sub(r'(?<!<)\b(c\d+,[^>\n]+>)', r'<\1', text)

    # 3. Thêm dấu cách trước '<' nếu dính liền với từ phía trước
    # (Không thêm nếu đứng ở đầu câu, hoặc sau khoảng trắng, sau dấu mở ngoặc / nháy kép)
    text = re.sub(r'(?<=[^\s(\[{"\'<])<', r' <', text)

    # 4. Chuẩn hóa khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    return text