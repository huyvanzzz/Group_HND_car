from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .benchmarking import summarize_latencies
from .cache import FeatureCacheManifest, create_memmap, validate_manifest
from .checkpointing import load_checkpoint, save_checkpoint
from .config import ExperimentConfig
from .data import format_prompt, normalize_camera_paths
from .debugging import DebugPrinter, default_debug_jsonl, model_param_summary, path_status, safe_preview, tensor_stats
from .evaluation import caption_metrics, metric_display_values, write_predictions
from .hf_data import DatasetRecord, load_prepared_records
from .modeling.factory import build_vision_encoder, build_vlm_model
from .modeling.multimodal import set_trainable_for_stage
from .progress import progress


def resolve_device(cfg: ExperimentConfig) -> torch.device:
    if cfg.runtime.device != "auto":
        return torch.device(cfg.runtime.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_precision(cfg: ExperimentConfig) -> torch.dtype:
    if cfg.runtime.precision == "bf16":
        return torch.bfloat16
    if cfg.runtime.precision == "fp16":
        return torch.float16
    if cfg.runtime.precision == "auto" and torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def _read_prepared_manifest(cfg: ExperimentConfig) -> dict[str, Any]:
    path = Path(cfg.project.output_dir) / "prepared_data" / "manifest.json"
    if not path.exists():
        raise FileNotFoundError("Run prepare-data before this command")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_image(path: Path, image_size: int, transform=None) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if transform is not None:
        return transform(image)
    image = image.resize((image_size, image_size))
    arr = np.asarray(image).astype("float32") / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def _record_key(record: DatasetRecord) -> str:
    return "|".join(record.camera_paths[camera] for camera in sorted(record.camera_paths))


def _debug_printer(cfg: ExperimentConfig, debug: bool = False, debug_samples: int = 3, debug_jsonl: str | None = None) -> DebugPrinter:
    return DebugPrinter(debug, debug_samples, debug_jsonl or default_debug_jsonl(cfg.project.output_dir))


def prepare_features(
    cfg: ExperimentConfig,
    subset: str = "full",
    debug: bool = False,
    debug_samples: int = 3,
    debug_jsonl: str | None = None,
    disable_progress: bool = False,
) -> dict[str, Any]:
    dbg = _debug_printer(cfg, debug, debug_samples, debug_jsonl)
    manifest = _read_prepared_manifest(cfg)
    root = Path(manifest["root"])
    records: list[DatasetRecord] = []
    split_counts: dict[str, int] = {}
    per_split_limit = int(cfg.training.max_steps or 64) if subset != "full" else None
    for split in ("train", "val", "test"):
        split_records = load_prepared_records(cfg, split)
        if per_split_limit is not None:
            split_records = split_records[:per_split_limit]
        split_counts[split] = len(split_records)
        records.extend(split_records)

    unique: dict[str, DatasetRecord] = {}
    for record in records:
        unique.setdefault(_record_key(record), record)

    device = resolve_device(cfg)
    vision, transform = build_vision_encoder(cfg)
    vision.to(device)
    vision.eval()
    dbg.log(
        "FEATURE",
        {
            "device": str(device),
            "precision": str(resolve_precision(cfg)),
            "vision": cfg.model.vision.name,
            "feature_shape": [len(unique), 6, cfg.model.vision.seq_len, cfg.model.vision.output_dim],
            "split_counts": split_counts,
        },
    )

    shape = (len(unique), 6, cfg.model.vision.seq_len, cfg.model.vision.output_dim)
    cache_dir = Path(cfg.cache.dir)
    mmap = create_memmap(cache_dir / "features.float16.memmap", shape=shape, dtype=np.float16)
    index: dict[str, int] = {}

    with torch.no_grad():
        for idx, (key, record) in progress(
            enumerate(unique.items()),
            desc="prepare-features",
            total=len(unique),
            disable=disable_progress,
        ):
            camera_paths = normalize_camera_paths(root, record.camera_paths, cfg.data.view_order)
            images = torch.stack([_load_image(path, cfg.model.vision.image_size, transform) for path in camera_paths])
            if idx < dbg.samples:
                dbg.log("IMAGE", {"sample_id": record.sample_id, "paths": path_status(camera_paths), "tensor": tensor_stats("images", images)})
            features = vision(images.unsqueeze(0).to(device)).detach().cpu().numpy().astype(np.float16)
            if idx < dbg.samples:
                dbg.log("FEATURE", {"sample_id": record.sample_id, "tensor": tensor_stats("features", torch.from_numpy(features))})
            mmap[idx] = features[0]
            index[key] = idx
    mmap.flush()

    (cache_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    cache_manifest = FeatureCacheManifest(
        dataset_revision=str(manifest.get("root")),
        model_id=cfg.model.vision.model_id,
        model_revision=cfg.model.vision.revision,
        preprocessing=f"{cfg.model.vision.name}_{cfg.model.vision.image_size}",
        dtype="float16",
        image_size=cfg.model.vision.image_size,
        view_order=cfg.data.view_order,
        shape=list(shape),
    )
    cache_manifest.write(cache_dir / "manifest.json")
    report = {"feature_count": len(unique), "shape": list(shape), "cache_dir": str(cache_dir)}
    dbg.log("FEATURE", report)
    return report


class CachedVLMDataset(Dataset):
    def __init__(self, cfg: ExperimentConfig, split: str, max_samples: int | None = None) -> None:
        self.cfg = cfg
        self.records = load_prepared_records(cfg, split)
        if max_samples is not None:
            self.records = self.records[:max_samples]
        manifest = validate_manifest(
            Path(cfg.cache.dir) / "manifest.json",
            {
                "model_id": cfg.model.vision.model_id,
                "image_size": cfg.model.vision.image_size,
                "view_order": cfg.data.view_order,
            },
        )
        self.features = np.memmap(
            Path(cfg.cache.dir) / "features.float16.memmap",
            mode="r",
            dtype=np.float16,
            shape=tuple(manifest.shape),
        )
        self.index = json.loads((Path(cfg.cache.dir) / "index.json").read_text(encoding="utf-8"))
        prepared_count = len(self.records)
        self.records = [record for record in self.records if _record_key(record) in self.index]
        self.prepared_count = prepared_count
        self.filtered_count = prepared_count - len(self.records)
        self.cache_count = len(self.index)
        if not self.records:
            raise ValueError(
                f"No {split} records have cached visual features. "
                "Run prepare-features for this split/subset before training or evaluation."
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        feature_idx = self.index[_record_key(record)]
        return {
            "question": record.question,
            "answer": record.answer,
            "visual_features": torch.from_numpy(np.array(self.features[feature_idx], dtype=np.float32)),
            "sample_id": record.sample_id,
            "eval_id": record.eval_id,
        }

    def summary(self) -> dict[str, int]:
        return {
            "prepared_records": self.prepared_count,
            "cached_records": len(self.records),
            "filtered_missing_features": self.filtered_count,
            "feature_index_count": self.cache_count,
        }


class CachedBatchCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = self.tokenizer([format_prompt(item["question"]) for item in batch], padding=True, return_tensors="pt")
        labels = self.tokenizer([item["answer"] for item in batch], padding=True, return_tensors="pt")["input_ids"].clone()
        labels[labels == getattr(self.tokenizer, "pad_token_id", 0)] = -100
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "visual_features": torch.stack([item["visual_features"] for item in batch]),
            "labels": labels,
            "answers": [item["answer"] for item in batch],
            "eval_ids": [item["eval_id"] for item in batch],
        }


def _module_dict_for_freezing(model) -> torch.nn.ModuleDict:
    return torch.nn.ModuleDict(
        {
            "vision": torch.nn.Identity(),
            "text": model.text_model,
            "gpa": model.gpa,
            "projector": model.projector,
            "spatial_pos": torch.nn.ModuleList([m for m in [model.row_embeddings, model.col_embeddings] if m]),
            "modal_embeddings": model.modal_embeddings,
        }
    )


def train_stage(
    cfg: ExperimentConfig,
    stage: str,
    resume: str | None = None,
    max_steps: int | None = None,
    debug: bool = False,
    debug_samples: int = 3,
    debug_jsonl: str | None = None,
    disable_progress: bool = False,
) -> Path:
    dbg = _debug_printer(cfg, debug, debug_samples, debug_jsonl)
    device = resolve_device(cfg)
    model, tokenizer = build_vlm_model(cfg)
    model.to(device)
    set_trainable_for_stage(_module_dict_for_freezing(model), cfg, stage)
    dbg.log("MODEL", {"stage": stage, "params": model_param_summary(model), "device": str(device), "precision": str(resolve_precision(cfg))})

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.training.scheduler_gamma)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and resolve_precision(cfg) == torch.float16)

    start_step = 0
    if resume:
        dbg.log("CHECKPOINT", {"resume": resume})
        metadata = load_checkpoint(resume, model=model, map_location=device)
        if metadata.get("stage") == stage:
            metadata = load_checkpoint(
                resume,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                map_location=device,
            )
            start_step = int(metadata.get("global_step", 0))

    split = "train"
    dataset = CachedVLMDataset(cfg, split)
    dbg.log("FEATURE", {"split": split, **dataset.summary()})
    loader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True, collate_fn=CachedBatchCollator(tokenizer))
    limit = max_steps or cfg.training.max_steps or len(loader)
    global_step = start_step
    model.train()
    loader_iter = iter(loader)
    for _ in progress(range(limit), desc=f"train-{stage}", total=limit, disable=disable_progress):
        try:
            batch = next(loader_iter)
        except StopIteration:
            scheduler.step()
            loader_iter = iter(loader)
            batch = next(loader_iter)
        if debug and global_step == start_step:
            dbg.log(
                "BATCH",
                {
                    "stage": "before_device_transfer",
                    "target_device": str(device),
                    "input_ids": tensor_stats("input_ids", batch["input_ids"]),
                    "attention_mask": tensor_stats("attention_mask", batch["attention_mask"]),
                    "visual_features": tensor_stats("visual_features", batch["visual_features"]),
                    "labels": tensor_stats("labels", batch["labels"].float()),
                    "label_ignore_count": int((batch["labels"] == -100).sum().item()),
                },
            )
        for key in ("input_ids", "attention_mask", "visual_features", "labels"):
            batch[key] = batch[key].to(device)
        with torch.autocast(device_type=device.type, dtype=resolve_precision(cfg), enabled=device.type == "cuda" and resolve_precision(cfg) != torch.float32):
            output = model.forward_from_features(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                visual_features=batch["visual_features"],
                labels=batch["labels"],
            )
            loss = output.loss / cfg.training.gradient_accumulation_steps
        scaler.scale(loss).backward()
        if debug and global_step < start_step + dbg.samples:
            loss_value = float(loss.detach().cpu().item())
            dbg.log(
                "TRAIN",
                {
                    "step": global_step,
                    "loss": loss_value,
                    "finite_loss": bool(torch.isfinite(loss.detach()).item()),
                    "lr": optimizer.param_groups[0]["lr"],
                },
            )
        if (global_step + 1) % cfg.training.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        global_step += 1
    scheduler.step()

    ckpt = Path(cfg.project.output_dir) / "checkpoints" / f"{stage}_latest.pt"
    save_checkpoint(
        ckpt,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        metadata={"stage": stage, "global_step": global_step, "profile": cfg.model.profile},
    )
    dbg.log("CHECKPOINT", {"saved": str(ckpt), "stage": stage, "global_step": global_step})
    return ckpt


def evaluate_checkpoint(
    cfg: ExperimentConfig,
    checkpoint: str,
    max_samples: int | None = None,
    debug: bool = False,
    debug_samples: int = 3,
    debug_jsonl: str | None = None,
    disable_progress: bool = False,
) -> dict[str, float]:
    dbg = _debug_printer(cfg, debug, debug_samples, debug_jsonl)
    device = resolve_device(cfg)
    model, tokenizer = build_vlm_model(cfg)
    model.to(device)
    metadata = load_checkpoint(checkpoint, model=model, map_location=device)
    dbg.log("MODEL", {"device": str(device), "precision": str(resolve_precision(cfg))})
    dbg.log("CHECKPOINT", {"loaded": checkpoint, "metadata": metadata})
    model.eval()
    dataset = CachedVLMDataset(cfg, "test", max_samples=max_samples)
    dbg.log("FEATURE", {"split": "test", **dataset.summary()})
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=CachedBatchCollator(tokenizer))
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for idx, batch in progress(enumerate(loader), desc="evaluate", total=len(loader), disable=disable_progress):
            for key in ("input_ids", "attention_mask", "visual_features"):
                batch[key] = batch[key].to(device)
            ids = model.generate_from_features(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                visual_features=batch["visual_features"],
                max_new_tokens=cfg.generation.max_new_tokens,
                num_beams=cfg.generation.num_beams,
            )
            pred = tokenizer.batch_decode(ids.detach().cpu(), skip_special_tokens=True)[0]
            if idx < dbg.samples:
                dbg.log(
                    "EVAL",
                    {
                        "eval_id": batch["eval_ids"][0],
                        "compute_device": str(device),
                        "generated_tokens": tensor_stats("generated_tokens", ids),
                        "prediction": safe_preview(pred),
                        "reference": safe_preview(batch["answers"][0]),
                    },
                )
            rows.append({"eval_id": batch["eval_ids"][0], "prediction": pred, "reference": batch["answers"][0]})
    output_dir = Path(cfg.project.output_dir)
    write_predictions(output_dir / "predictions.jsonl", rows)
    metrics = caption_metrics(rows)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "metrics_display.json").write_text(json.dumps(metric_display_values(metrics), indent=2), encoding="utf-8")
    dbg.log("EVAL", {"prediction_count": len(rows), "metrics": metrics, "metrics_display": metric_display_values(metrics)})
    return metrics


def benchmark_checkpoint(
    cfg: ExperimentConfig,
    checkpoint: str,
    max_samples: int | None = None,
    debug: bool = False,
    debug_samples: int = 3,
    debug_jsonl: str | None = None,
    disable_progress: bool = False,
) -> dict[str, Any]:
    dbg = _debug_printer(cfg, debug, debug_samples, debug_jsonl)
    device = resolve_device(cfg)
    model, tokenizer = build_vlm_model(cfg)
    model.to(device)
    metadata = load_checkpoint(checkpoint, model=model, map_location=device)
    dbg.log("MODEL", {"device": str(device), "precision": str(resolve_precision(cfg))})
    dbg.log("CHECKPOINT", {"loaded": checkpoint, "metadata": metadata})
    model.eval()
    dataset = CachedVLMDataset(cfg, "test", max_samples=max_samples)
    dbg.log("FEATURE", {"split": "test", **dataset.summary()})
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=CachedBatchCollator(tokenizer))
    e2e, gen, token_counts = [], [], []
    with torch.no_grad():
        for idx, batch in progress(enumerate(loader), desc="benchmark", total=len(loader), disable=disable_progress):
            start = time.perf_counter()
            for key in ("input_ids", "attention_mask", "visual_features"):
                batch[key] = batch[key].to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            gen_start = time.perf_counter()
            ids = model.generate_from_features(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                visual_features=batch["visual_features"],
                max_new_tokens=cfg.generation.max_new_tokens,
                num_beams=cfg.generation.num_beams,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            gen.append(time.perf_counter() - gen_start)
            tokenizer.batch_decode(ids.detach().cpu(), skip_special_tokens=True)
            e2e.append(time.perf_counter() - start)
            token_counts.append(int((ids != getattr(tokenizer, "pad_token_id", 0)).sum().item()))
            if idx < dbg.samples:
                dbg.log("BENCHMARK", {"sample": idx, "e2e_seconds": e2e[-1], "generation_seconds": gen[-1], "output_tokens": token_counts[-1]})
    summary = summarize_latencies(e2e_seconds=e2e, generation_seconds=gen, output_tokens=token_counts)
    summary.update({"device": str(device), "python": platform.python_version(), "torch": torch.__version__})
    if torch.cuda.is_available():
        summary.update({"gpu": torch.cuda.get_device_name(0), "peak_cuda_memory": torch.cuda.max_memory_allocated()})
    output = Path(cfg.project.output_dir) / "benchmark.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    dbg.log("BENCHMARK", summary)
    return summary


def debug_sample(
    cfg: ExperimentConfig,
    split: str,
    index: int,
    debug: bool = True,
    debug_samples: int = 3,
    debug_jsonl: str | None = None,
) -> dict[str, Any]:
    dbg = _debug_printer(cfg, debug, debug_samples, debug_jsonl)
    manifest = _read_prepared_manifest(cfg)
    root = Path(manifest["root"])
    records = load_prepared_records(cfg, split)
    record = records[index]
    dbg.log(
        "SAMPLE",
        {
            "split": split,
            "index": index,
            "sample_id": record.sample_id,
            "eval_id": record.eval_id,
            "question": safe_preview(record.question),
            "answer": safe_preview(record.answer),
            "camera_order": cfg.data.view_order,
        },
    )
    camera_paths = normalize_camera_paths(root, record.camera_paths, cfg.data.view_order)
    dbg.log("SAMPLE", {"paths": path_status(camera_paths)})

    device = resolve_device(cfg)
    vision, transform = build_vision_encoder(cfg)
    vision.to(device)
    vision.eval()
    images = torch.stack([_load_image(path, cfg.model.vision.image_size, transform) for path in camera_paths])
    dbg.log("IMAGE", {"tensor": tensor_stats("images", images)})
    with torch.no_grad():
        features = vision(images.unsqueeze(0).to(device)).detach().cpu()
    dbg.log("FEATURE", {"tensor": tensor_stats("features", features)})

    model, tokenizer = build_vlm_model(cfg)
    model.to(device)
    encoded = tokenizer([format_prompt(record.question)], padding=True, return_tensors="pt")
    labels = tokenizer([record.answer], padding=True, return_tensors="pt")["input_ids"]
    dbg.log("BATCH", {"input_ids": tensor_stats("input_ids", encoded["input_ids"]), "labels": tensor_stats("labels", labels.float())})
    dbg.log("MODEL", {"params": model_param_summary(model)})
    with torch.no_grad():
        output = model.forward_from_features(
            input_ids=encoded["input_ids"].to(device),
            attention_mask=encoded["attention_mask"].to(device),
            visual_features=features.to(device),
            labels=labels.to(device),
        )
    report = {"split": split, "index": index, "loss": float(output.loss.detach().cpu().item())}
    dbg.log("MODEL", report)
    return report
