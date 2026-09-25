from __future__ import annotations

import json
import platform
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .benchmarking import summarize_latencies
from .cache import FeatureCacheManifest, create_memmap, validate_manifest
from .checkpointing import load_checkpoint, read_checkpoint_metadata, save_checkpoint
from .config import ExperimentConfig
from .data import format_prompt, normalize_camera_paths
from .debugging import DebugPrinter, default_debug_jsonl, model_param_summary, path_status, safe_preview, tensor_stats
from .evaluation import caption_metrics, metric_display_values, write_predictions
from .hf_data import DatasetRecord, load_prepared_records
from .modeling.factory import build_vision_encoder, build_vlm_model
from .modeling.multimodal import set_trainable_for_stage
from .progress import progress, progress_bar


class _SingleProcessAccelerator:
    device: torch.device
    num_processes = 1
    process_index = 0
    local_process_index = 0
    distributed_type = "NO"
    mixed_precision = "no"
    is_main_process = True
    sync_gradients = True

    def __init__(self, device: torch.device, mixed_precision: str = "no") -> None:
        self.device = device
        self.mixed_precision = mixed_precision

    def prepare(self, *objects):
        return objects

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def autocast(self):
        dtype = torch.float16 if self.mixed_precision == "fp16" else torch.bfloat16
        enabled = self.device.type == "cuda" and self.mixed_precision in {"fp16", "bf16"}
        return torch.autocast(device_type=self.device.type, dtype=dtype, enabled=enabled)

    def accumulate(self, _model):
        return nullcontext()

    def unwrap_model(self, model):
        return model

    def wait_for_everyone(self) -> None:
        return None

    def clip_grad_norm_(self, parameters, max_norm: float) -> None:
        torch.nn.utils.clip_grad_norm_(parameters, max_norm)


def _accelerate_precision(cfg: ExperimentConfig) -> str:
    if cfg.runtime.precision == "fp16":
        return "fp16"
    if cfg.runtime.precision == "bf16":
        return "bf16"
    if cfg.runtime.precision == "auto" and torch.cuda.is_available():
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    return "no"


def _build_accelerator(cfg: ExperimentConfig):
    mixed_precision = _accelerate_precision(cfg)
    try:
        from accelerate import Accelerator
    except Exception:
        return _SingleProcessAccelerator(resolve_device(cfg), mixed_precision=mixed_precision)
    return Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
    )


def _distributed_debug_payload(accelerator, cfg: ExperimentConfig) -> dict[str, Any]:
    return {
        "num_processes": int(getattr(accelerator, "num_processes", 1)),
        "process_index": int(getattr(accelerator, "process_index", 0)),
        "local_process_index": int(getattr(accelerator, "local_process_index", 0)),
        "distributed_type": str(getattr(accelerator, "distributed_type", "NO")),
        "mixed_precision": str(getattr(accelerator, "mixed_precision", "no")),
        "per_process_batch_size": cfg.training.batch_size,
        "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
        "effective_batch_size": cfg.training.effective_batch_size_for_processes(
            int(getattr(accelerator, "num_processes", 1))
        ),
    }


def _optimizer_lr(optimizer) -> float | None:
    param_groups = getattr(optimizer, "param_groups", None)
    if param_groups is None and hasattr(optimizer, "optimizer"):
        param_groups = getattr(optimizer.optimizer, "param_groups", None)
    if not param_groups:
        return None
    return float(param_groups[0]["lr"])


def _heartbeat(dbg: DebugPrinter, started_at: float, stage: str, **payload: Any) -> None:
    dbg.log(
        "HEARTBEAT",
        {
            "stage": stage,
            "elapsed_seconds": round(time.perf_counter() - started_at, 3),
            **payload,
        },
    )


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    for key in ("input_ids", "attention_mask", "visual_features", "labels"):
        if key in batch:
            batch[key] = batch[key].to(device)
    return batch


def _stage_epochs(cfg: ExperimentConfig, stage: str) -> int:
    return cfg.training.align_epochs if stage == "align" else cfg.training.finetune_epochs


def _stage_step_limit(cfg: ExperimentConfig, stage: str, max_steps: int | None) -> int | None:
    configured_stage_steps = cfg.training.align_max_steps if stage == "align" else cfg.training.finetune_max_steps
    return max_steps or configured_stage_steps or cfg.training.max_steps


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
        self.records = [
            record
            for record in progress(
                self.records,
                desc=f"filter-cache-{split}",
                total=len(self.records),
                disable=len(self.records) < 1000,
            )
            if _record_key(record) in self.index
        ]
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


def validate_stage_loss(
    cfg: ExperimentConfig,
    model: torch.nn.Module,
    tokenizer,
    *,
    stage: str,
    epoch: int,
    device: torch.device,
    debug: DebugPrinter | None = None,
    disable_progress: bool = False,
) -> float:
    dataset = CachedVLMDataset(cfg, "val")
    if debug:
        debug.log("FEATURE", {"split": "val", **dataset.summary()})
    loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=CachedBatchCollator(tokenizer),
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory and device.type == "cuda",
    )
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in progress(
            loader,
            desc=f"validate-{stage} epoch {epoch}",
            total=len(loader),
            disable=disable_progress,
        ):
            batch = _move_batch_to_device(batch, device)
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                visual_features=batch["visual_features"],
                labels=batch["labels"],
            )
            total_loss += float(output.loss.detach().cpu().item())
            total_batches += 1
    if was_training:
        model.train()
    if total_batches == 0:
        raise ValueError("Validation dataloader was empty")
    return total_loss / total_batches


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
    debug_numerics: bool = False,
    disable_progress: bool = False,
) -> Path:
    dbg = _debug_printer(cfg, debug, debug_samples, debug_jsonl)
    started_at = time.perf_counter()
    _heartbeat(dbg, started_at, "train_stage_start", train_stage=stage, resume=bool(resume), max_steps=max_steps)
    _heartbeat(dbg, started_at, "build_accelerator_start")
    accelerator = _build_accelerator(cfg)
    device = accelerator.device
    if accelerator.is_main_process:
        _heartbeat(dbg, started_at, "build_accelerator_done", device=str(device), distributed=_distributed_debug_payload(accelerator, cfg))
        _heartbeat(dbg, started_at, "build_vlm_model_start", text_model=cfg.model.text.model_id)
    model, tokenizer = build_vlm_model(cfg)
    if accelerator.is_main_process:
        _heartbeat(dbg, started_at, "build_vlm_model_done")
        _heartbeat(dbg, started_at, "model_to_device_start", device=str(device))
    model.to(device)
    if accelerator.is_main_process:
        _heartbeat(dbg, started_at, "model_to_device_done", device=str(device))
    set_trainable_for_stage(_module_dict_for_freezing(model), cfg, stage)
    if accelerator.is_main_process:
        dbg.log("DISTRIBUTED", _distributed_debug_payload(accelerator, cfg))
        dbg.log("MODEL", {"stage": stage, "params": model_param_summary(model), "device": str(device), "precision": str(resolve_precision(cfg))})

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.training.scheduler_gamma)

    start_step = 0
    start_epoch = 0
    best_val_loss = float("inf")
    best_epoch = 0
    if resume:
        resume_path = Path(resume)
        if accelerator.is_main_process:
            dbg.log(
                "CHECKPOINT",
                {
                    "resume": resume,
                    "exists": resume_path.exists(),
                    "size_bytes": resume_path.stat().st_size if resume_path.exists() else None,
                },
            )
            _heartbeat(dbg, started_at, "checkpoint_metadata_start", path=resume)
        metadata = read_checkpoint_metadata(resume, map_location="cpu")
        if accelerator.is_main_process:
            _heartbeat(dbg, started_at, "checkpoint_metadata_done", metadata=metadata)
            _heartbeat(dbg, started_at, "checkpoint_load_start", path=resume, same_stage=metadata.get("stage") == stage)
        if metadata.get("stage") == stage:
            metadata = load_checkpoint(
                resume,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                map_location=device,
            )
            start_step = int(metadata.get("global_step", 0))
            start_epoch = int(metadata.get("completed_epochs", metadata.get("epoch", 0)))
            best_val_loss = float(metadata.get("best_val_loss", best_val_loss))
            best_epoch = int(metadata.get("best_epoch", best_epoch))
        else:
            metadata = load_checkpoint(resume, model=model, map_location=device)
        if accelerator.is_main_process:
            _heartbeat(dbg, started_at, "checkpoint_load_done", metadata=metadata)

    split = "train"
    if accelerator.is_main_process:
        _heartbeat(dbg, started_at, "dataset_start", split=split)
    dataset = CachedVLMDataset(cfg, split)
    if accelerator.is_main_process:
        _heartbeat(dbg, started_at, "dataset_done", split=split, **dataset.summary())
        dbg.log("FEATURE", {"split": split, **dataset.summary()})
        _heartbeat(dbg, started_at, "dataloader_start", split=split, batch_size=cfg.training.batch_size, num_workers=cfg.training.num_workers)
    loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=CachedBatchCollator(tokenizer),
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory and device.type == "cuda",
    )
    if accelerator.is_main_process:
        _heartbeat(dbg, started_at, "dataloader_done", split=split, batches=len(loader))
        _heartbeat(dbg, started_at, "accelerator_prepare_start")
    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)
    if accelerator.is_main_process:
        _heartbeat(dbg, started_at, "accelerator_prepare_done")
    step_limit = _stage_step_limit(cfg, stage, max_steps)
    total_epochs = 1 if step_limit is not None else _stage_epochs(cfg, stage)
    global_step = start_step
    model.train()
    latest_ckpt = Path(cfg.project.output_dir) / "checkpoints" / f"{stage}_latest.pt"
    best_ckpt = Path(cfg.project.output_dir) / "checkpoints" / f"{stage}_best.pt"
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, total_epochs):
        epoch_number = epoch + 1
        epoch_loss = 0.0
        epoch_batches = 0
        loader_iter = iter(loader)
        bar_source = range(step_limit) if step_limit is not None else loader
        bar_total = step_limit if step_limit is not None else len(loader)
        if accelerator.is_main_process:
            _heartbeat(dbg, started_at, "train_tqdm_start", epoch=epoch_number, total=bar_total)
        bar = progress_bar(
            bar_source,
            desc=f"train-{stage} epoch {epoch_number}/{total_epochs}",
            total=bar_total,
            disable=disable_progress or not accelerator.is_main_process,
        )
        for item in bar:
            if step_limit is not None:
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(loader)
                    batch = next(loader_iter)
            else:
                batch = item
            if debug and accelerator.is_main_process and global_step == start_step:
                dbg.log(
                    "BATCH",
                    {
                        "stage": "after_accelerator_prepare",
                        "target_device": str(device),
                        "input_ids": tensor_stats("input_ids", batch["input_ids"]),
                        "attention_mask": tensor_stats("attention_mask", batch["attention_mask"]),
                        "visual_features": tensor_stats("visual_features", batch["visual_features"]),
                        "labels": tensor_stats("labels", batch["labels"].float()),
                        "label_ignore_count": int((batch["labels"] == -100).sum().item()),
                    },
                )
            if isinstance(accelerator, _SingleProcessAccelerator):
                batch = _move_batch_to_device(batch, device)
            with accelerator.autocast():
                output = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    visual_features=batch["visual_features"],
                    labels=batch["labels"],
                )
                raw_loss = output.loss
                loss = raw_loss / cfg.training.gradient_accumulation_steps
            if not torch.isfinite(raw_loss.detach()).item():
                if accelerator.is_main_process:
                    numerics = {}
                    debug_model = accelerator.unwrap_model(model)
                    if debug_numerics and hasattr(debug_model, "forward_debug"):
                        with torch.no_grad():
                            numerics = debug_model.forward_debug(
                                input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                visual_features=batch["visual_features"],
                                labels=batch["labels"],
                            )
                    dbg.log(
                        "NUMERICS",
                        {
                            "stage": "non_finite_loss",
                            "step": global_step,
                            "epoch": epoch_number,
                            "loss": float(raw_loss.detach().cpu().item()),
                            **numerics,
                        },
                    )
                raise FloatingPointError(f"Non-finite loss at step {global_step}: {raw_loss.detach().cpu().item()}")
            accelerator.backward(loss)
            loss_value = float(raw_loss.detach().cpu().item())
            epoch_loss += loss_value
            epoch_batches += 1
            if debug and accelerator.is_main_process and global_step < start_step + dbg.samples:
                dbg.log(
                    "TRAIN",
                    {
                        "step": global_step,
                        "epoch": epoch_number,
                        "loss": loss_value,
                        "finite_loss": bool(torch.isfinite(raw_loss.detach()).item()),
                        "lr": _optimizer_lr(optimizer),
                    },
                )
            if (global_step + 1) % cfg.training.gradient_accumulation_steps == 0:
                if cfg.training.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if accelerator.is_main_process and hasattr(bar, "set_postfix"):
                bar.set_postfix(loss=f"{loss_value:.4f}", lr=_optimizer_lr(optimizer), global_step=global_step)
        if epoch_batches == 0:
            raise ValueError("Training dataloader was empty")
        if global_step % cfg.training.gradient_accumulation_steps != 0:
            if cfg.training.max_grad_norm is not None:
                accelerator.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()
        epoch_train_loss = epoch_loss / epoch_batches
        accelerator.wait_for_everyone()
        val_loss = None
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            val_loss = validate_stage_loss(
                cfg,
                unwrapped,
                tokenizer,
                stage=stage,
                epoch=epoch_number,
                device=device,
                debug=dbg if debug else None,
                disable_progress=disable_progress,
            )
            if not np.isfinite(epoch_train_loss) or not np.isfinite(val_loss):
                dbg.log(
                    "NUMERICS",
                    {
                        "stage": "non_finite_epoch_loss",
                        "epoch": epoch_number,
                        "train_loss": epoch_train_loss,
                        "val_loss": val_loss,
                    },
                )
                raise FloatingPointError(f"Non-finite epoch loss at epoch {epoch_number}: train={epoch_train_loss}, val={val_loss}")
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                best_epoch = epoch_number
            metadata = {
                "stage": stage,
                "epoch": epoch_number,
                "completed_epochs": epoch_number,
                "global_step": global_step,
                "profile": cfg.model.profile,
                "train_loss": epoch_train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "learning_rate": _optimizer_lr(optimizer),
                "effective_batch_size": cfg.training.effective_batch_size_for_processes(
                    int(getattr(accelerator, "num_processes", 1))
                ),
                "distributed": _distributed_debug_payload(accelerator, cfg),
                "generation": {
                    "max_new_tokens": cfg.generation.max_new_tokens,
                    "num_beams": cfg.generation.num_beams,
                    "early_stopping": cfg.generation.early_stopping,
                    "length_penalty": cfg.generation.length_penalty,
                },
            }
            save_checkpoint(
                latest_ckpt,
                model=unwrapped,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=None,
                metadata=metadata,
            )
            dbg.log(
                "CHECKPOINT",
                {
                    "saved": str(latest_ckpt),
                    "stage": stage,
                    "epoch": epoch_number,
                    "global_step": global_step,
                    "train_loss": epoch_train_loss,
                    "val_loss": val_loss,
                },
            )
            if is_best:
                save_checkpoint(
                    best_ckpt,
                    model=unwrapped,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=None,
                    metadata=metadata,
                )
                dbg.log("CHECKPOINT", {"saved": str(best_ckpt), "stage": stage, "best_val_loss": best_val_loss})
                if stage == "finetune":
                    shutil.copy2(best_ckpt, Path(cfg.project.output_dir) / "checkpoints" / "best_model.pt")
        accelerator.wait_for_everyone()

    return latest_ckpt


def _batch_debug_summary(batch: dict[str, Any], device: torch.device | None = None) -> dict[str, Any]:
    keys = ("input_ids", "attention_mask", "visual_features", "labels")
    summary = {key: tensor_stats(key, batch[key].float() if key == "labels" else batch[key]) for key in keys if key in batch}
    if device is not None:
        summary["target_device"] = str(device)
    if "labels" in batch:
        summary["label_ignore_count"] = int((batch["labels"] == -100).sum().item())
    return summary


def diagnose_train(
    cfg: ExperimentConfig,
    stage: str,
    resume: str | None = None,
    debug: bool = False,
    debug_samples: int = 3,
    debug_jsonl: str | None = None,
    debug_numerics: bool = False,
    disable_progress: bool = False,
) -> dict[str, Any]:
    dbg = _debug_printer(cfg, debug, debug_samples, debug_jsonl)
    started_at = time.perf_counter()
    durations: dict[str, float] = {}

    def mark(name: str, begin: float) -> None:
        durations[name] = round(time.perf_counter() - begin, 3)
        _heartbeat(dbg, started_at, f"{name}_done", duration_seconds=durations[name])

    _heartbeat(dbg, started_at, "diagnose_start", train_stage=stage, resume=bool(resume))
    begin = time.perf_counter()
    accelerator = _build_accelerator(cfg)
    device = accelerator.device
    mark("build_accelerator", begin)
    if accelerator.is_main_process:
        dbg.log("DISTRIBUTED", _distributed_debug_payload(accelerator, cfg))

    begin = time.perf_counter()
    _heartbeat(dbg, started_at, "build_vlm_model_start", text_model=cfg.model.text.model_id)
    model, tokenizer = build_vlm_model(cfg)
    mark("build_vlm_model", begin)

    begin = time.perf_counter()
    model.to(device)
    set_trainable_for_stage(_module_dict_for_freezing(model), cfg, stage)
    mark("model_to_device", begin)
    if accelerator.is_main_process:
        dbg.log("MODEL", {"stage": stage, "params": model_param_summary(model), "device": str(device), "precision": str(resolve_precision(cfg))})

    checkpoint_metadata = None
    if resume:
        resume_path = Path(resume)
        begin = time.perf_counter()
        checkpoint_metadata = read_checkpoint_metadata(resume, map_location="cpu")
        mark("checkpoint_metadata", begin)
        if accelerator.is_main_process:
            dbg.log(
                "CHECKPOINT",
                {
                    "resume": resume,
                    "exists": resume_path.exists(),
                    "size_bytes": resume_path.stat().st_size if resume_path.exists() else None,
                    "metadata": checkpoint_metadata,
                },
            )

    begin = time.perf_counter()
    dataset = CachedVLMDataset(cfg, "train")
    mark("dataset", begin)
    dataset_summary = dataset.summary()
    if accelerator.is_main_process:
        dbg.log("FEATURE", {"split": "train", **dataset_summary})

    begin = time.perf_counter()
    loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=CachedBatchCollator(tokenizer),
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory and device.type == "cuda",
    )
    mark("dataloader", begin)

    begin = time.perf_counter()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.training.scheduler_gamma)
    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)
    mark("accelerator_prepare", begin)

    begin = time.perf_counter()
    batch = next(iter(loader))
    if isinstance(accelerator, _SingleProcessAccelerator):
        batch = _move_batch_to_device(batch, device)
    mark("first_batch", begin)
    batch_summary = _batch_debug_summary(batch, device)
    if accelerator.is_main_process:
        dbg.log("BATCH", {"stage": "diagnose_first_batch", **batch_summary})
    numerics = None
    debug_model = accelerator.unwrap_model(model)
    if debug_numerics and accelerator.is_main_process and hasattr(debug_model, "forward_debug"):
        with torch.no_grad(), accelerator.autocast():
            numerics = debug_model.forward_debug(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                visual_features=batch["visual_features"],
                labels=batch["labels"],
            )
        dbg.log("NUMERICS", {"stage": "diagnose_forward", **numerics})

    report = {
        "stage": stage,
        "device": str(device),
        "cuda": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "peak_cuda_memory": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        "distributed": _distributed_debug_payload(accelerator, cfg),
        "checkpoint_metadata": checkpoint_metadata,
        "dataset": dataset_summary,
        "feature_cache_shape": list(dataset.features.shape),
        "batch": batch_summary,
        "numerics": numerics,
        "durations": durations,
    }
    _heartbeat(dbg, started_at, "diagnose_done")
    return report


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
                early_stopping=cfg.generation.early_stopping,
                length_penalty=cfg.generation.length_penalty,
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
                early_stopping=cfg.generation.early_stopping,
                length_penalty=cfg.generation.length_penalty,
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
