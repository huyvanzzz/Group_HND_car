from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import HfApi, snapshot_download

from .config import CAMERA_ORDER, ExperimentConfig
from .debugging import DebugPrinter, path_status, safe_preview
from .progress import progress

NUSCENES_CAMERA_TO_CANONICAL = {
    "CAM_FRONT": "Front",
    "CAM_FRONT_LEFT": "Front-Left",
    "CAM_FRONT_RIGHT": "Front-Right",
    "CAM_BACK": "Back",
    "CAM_BACK_LEFT": "Back-Left",
    "CAM_BACK_RIGHT": "Back-Right",
}


@dataclass(frozen=True)
class DatasetRecord:
    question: str
    answer: str
    camera_paths: dict[str, str]
    sample_id: str
    eval_id: int | None
    split: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _extract_cameras(row: dict[str, Any]) -> dict[str, str]:
    source = row["cameras"] if "cameras" in row and isinstance(row["cameras"], dict) else row
    cameras: dict[str, str] = {}
    for camera in CAMERA_ORDER:
        if camera in source:
            cameras[camera] = str(source[camera])
    for raw_key, canonical_key in NUSCENES_CAMERA_TO_CANONICAL.items():
        if raw_key in source:
            cameras[canonical_key] = str(source[raw_key])
    return cameras


def _record_from_row(row: dict[str, Any], split: str, idx: int) -> DatasetRecord:
    question = row.get("question") or row.get("Q")
    answer = row.get("answer") or row.get("A")
    if question is None or answer is None:
        raise ValueError(f"Record {split}/{idx} must contain question/Q and answer/A")
    cameras = _extract_cameras(row)
    missing = [camera for camera in CAMERA_ORDER if camera not in cameras]
    if missing:
        raise ValueError(f"Record {split}/{idx} missing cameras: {missing}")
    return DatasetRecord(
        question=str(question),
        answer=str(answer),
        camera_paths=cameras,
        sample_id=str(row.get("sample_id") or row.get("id") or f"{split}-{idx}"),
        eval_id=int(row["eval_id"]) if row.get("eval_id") is not None else idx,
        split=split,
    )


def parse_dataset_json(path: str | Path, split: str) -> list[DatasetRecord]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    records: list[DatasetRecord] = []
    for idx, item in enumerate(data):
        if isinstance(item, list) and len(item) == 2 and isinstance(item[0], dict) and isinstance(item[1], dict):
            row = {"Q": item[0].get("Q"), "A": item[0].get("A"), "cameras": item[1]}
        elif isinstance(item, dict):
            row = item
        else:
            raise ValueError(f"Unsupported dataset record at {split}/{idx}")
        records.append(_record_from_row(row, split, idx))
    return records


def split_file_candidates(files: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for split in ("train", "val", "test"):
        candidates = [
            f
            for f in files
            if f.lower().endswith(".json")
            and split in Path(f).name.lower()
            and ("multi_frame" in f.lower() or "frame" in f.lower())
        ]
        if candidates:
            result[split] = sorted(candidates, key=len)[0]
    return result


def validate_split_counts(actual: dict[str, int], expected: dict[str, int], subset: str = "full") -> None:
    if subset != "full":
        return
    mismatches = {split: (actual.get(split), count) for split, count in expected.items() if actual.get(split) != count}
    if mismatches:
        raise ValueError(f"Split count mismatch: {mismatches}")


def inspect_hf_dataset(cfg: ExperimentConfig, token: str | None, debug: DebugPrinter | None = None) -> dict[str, Any]:
    if cfg.data.local_dir:
        files = [str(p.relative_to(cfg.data.local_dir)).replace("\\", "/") for p in Path(cfg.data.local_dir).rglob("*")]
        candidates = split_file_candidates(files)
        report = {
            "hf_repo_id": cfg.data.hf_repo_id,
            "local_dir": cfg.data.local_dir,
            "token_present": bool(token),
            "resolved_revision": "local",
            "file_count": len(files),
            "split_candidates": candidates,
        }
        if debug:
            debug.log("DATASET", report)
        return report
    if not token:
        report = {
            "hf_repo_id": cfg.data.hf_repo_id,
            "token_present": False,
            "resolved_revision": cfg.data.revision,
            "file_count": 0,
            "split_candidates": {},
            "error": "HF_TOKEN is required to inspect the private Hugging Face dataset",
        }
        if debug:
            debug.log("DATASET", report)
        return report
    api = HfApi()
    try:
        info = api.repo_info(cfg.data.hf_repo_id, repo_type="dataset", revision=cfg.data.revision, token=token)
        files = api.list_repo_files(cfg.data.hf_repo_id, repo_type="dataset", revision=cfg.data.revision, token=token)
    except Exception as exc:
        report = {
            "hf_repo_id": cfg.data.hf_repo_id,
            "token_present": True,
            "resolved_revision": cfg.data.revision,
            "file_count": 0,
            "split_candidates": {},
            "error": exc.__class__.__name__,
        }
        if debug:
            debug.log("DATASET", report)
        return report
    report = {
        "hf_repo_id": cfg.data.hf_repo_id,
        "token_present": True,
        "resolved_revision": getattr(info, "sha", None) or cfg.data.revision,
        "file_count": len(files),
        "split_candidates": split_file_candidates(files),
    }
    if debug:
        debug.log("DATASET", {**report, "json_candidates": [f for f in files if f.endswith(".json")][: debug.samples]})
    return report


def prepare_data(
    cfg: ExperimentConfig,
    subset: str = "full",
    debug: DebugPrinter | None = None,
    disable_progress: bool = False,
) -> dict[str, Any]:
    output_dir = Path(cfg.project.output_dir)
    prepared_dir = output_dir / "prepared_data"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    if cfg.data.local_dir:
        root = Path(cfg.data.local_dir).resolve()
        files = [str(p.relative_to(root)).replace("\\", "/") for p in root.rglob("*.json")]
    else:
        token = os.getenv("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is required to download the private Hugging Face dataset")
        root = Path(
            snapshot_download(
                cfg.data.hf_repo_id,
                repo_type="dataset",
                revision=cfg.data.revision,
                token=token,
                allow_patterns=["*.json", "nuscenes/samples/**", "multi_frame/**", "QA_dataset_nus/**"],
            )
        )
        files = [str(p.relative_to(root)).replace("\\", "/") for p in root.rglob("*.json")]

    candidates = split_file_candidates(files)
    all_records: dict[str, list[DatasetRecord]] = {}
    counts: dict[str, int] = {}
    for split, rel_path in progress(candidates.items(), desc="prepare-data", total=len(candidates), disable=disable_progress):
        records = parse_dataset_json(root / rel_path, split)
        if subset != "full":
            records = records[:64]
        all_records[split] = records
        counts[split] = len(records)
        if debug:
            for record in records[: debug.samples]:
                debug.log(
                    "SAMPLE",
                    {
                        "split": split,
                        "sample_id": record.sample_id,
                        "eval_id": record.eval_id,
                        "question": safe_preview(record.question),
                        "answer": safe_preview(record.answer),
                        "camera_order": CAMERA_ORDER,
                        "paths": path_status([(root / record.camera_paths[camera]).resolve() for camera in CAMERA_ORDER]),
                    },
                )
        (prepared_dir / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(record.to_dict()) for record in records) + "\n",
            encoding="utf-8",
        )

    validate_split_counts(counts, cfg.data.expected_counts or {}, subset=subset)
    manifest = {"root": str(root), "counts": counts, "subset": subset, "splits": candidates}
    (prepared_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if debug:
        debug.log("DATASET", manifest)
    return manifest


def load_prepared_records(cfg: ExperimentConfig, split: str) -> list[DatasetRecord]:
    path = Path(cfg.project.output_dir) / "prepared_data" / f"{split}.jsonl"
    if not path.exists():
        prepare_data(cfg, subset="smoke" if cfg.training.max_steps else "full")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [DatasetRecord(**row) for row in rows]
