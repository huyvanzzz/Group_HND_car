from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CAMERA_ORDER = ["Front", "Front-Left", "Front-Right", "Back", "Back-Left", "Back-Right"]


@dataclass(frozen=True)
class ProjectConfig:
    output_dir: str


@dataclass(frozen=True)
class DataConfig:
    hf_repo_id: str
    view_order: list[str]
    revision: str | None = None
    local_dir: str | None = None
    expected_counts: dict[str, int] | None = None


@dataclass(frozen=True)
class CacheConfig:
    dir: str


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "auto"
    precision: str = "auto"


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 64
    num_beams: int = 1
    early_stopping: bool = False
    length_penalty: float = 1.0


@dataclass(frozen=True)
class VisionConfig:
    name: str
    model_id: str
    output_dim: int
    seq_len: int
    image_size: int
    revision: str | None = None


@dataclass(frozen=True)
class TextConfig:
    model_id: str
    d_model: int
    revision: str | None = None


@dataclass(frozen=True)
class ModelConfig:
    profile: str
    vision: VisionConfig
    text: TextConfig


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float = 1e-4
    vision_learning_rate: float | None = None
    text_learning_rate: float | None = None
    head_learning_rate: float | None = None
    weight_decay: float = 0.05
    scheduler_gamma: float = 0.9
    align_epochs: int = 6
    finetune_epochs: int = 6
    gpa_hidden_size: int = 128
    max_steps: int | None = None
    align_max_steps: int | None = None
    finetune_max_steps: int | None = None
    val_every_steps: int | None = None
    save_every_steps: int | None = None
    max_grad_norm: float | None = None
    num_workers: int = 0
    pin_memory: bool = False
    progress_log_every_steps: int = 50
    vision_training: str = "feature_cache"

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps

    def effective_batch_size_for_processes(self, num_processes: int) -> int:
        return self.effective_batch_size * max(1, int(num_processes))


@dataclass(frozen=True)
class EvaluationConfig:
    eval_batch_size: int = 16
    eval_progress_log_every_samples: int = 256
    benchmark_max_samples: int = 200
    benchmark_progress_log_every_samples: int = 20


@dataclass(frozen=True)
class ExperimentConfig:
    project: ProjectConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    cache: CacheConfig
    runtime: RuntimeConfig
    generation: GenerationConfig
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing config key: {key}")
    return mapping[key]


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError("Config must be a YAML mapping")

    data_raw = _require(raw, "data")
    view_order = list(_require(data_raw, "view_order"))
    if view_order != CAMERA_ORDER:
        raise ValueError(f"data.view_order must be canonical view_order: {CAMERA_ORDER}")

    model_raw = _require(raw, "model")
    vision_raw = _require(model_raw, "vision")
    text_raw = _require(model_raw, "text")
    train_raw = _require(raw, "training")
    cache_raw = raw.get("cache", {})
    runtime_raw = raw.get("runtime", {})
    generation_raw = raw.get("generation", {})
    evaluation_raw = raw.get("evaluation", {})
    precision = str(runtime_raw.get("precision", "auto"))
    if precision not in {"auto", "fp32", "fp16", "bf16"}:
        raise ValueError("runtime.precision must be one of: auto, fp32, fp16, bf16")
    vision_training = str(train_raw.get("vision_training", "feature_cache"))
    if vision_training not in {"feature_cache", "end_to_end"}:
        raise ValueError("training.vision_training must be one of: feature_cache, end_to_end")
    expected_counts = data_raw.get("expected_counts") or {"train": 341381, "val": 19785, "test": 16817}

    return ExperimentConfig(
        project=ProjectConfig(output_dir=str(_require(_require(raw, "project"), "output_dir"))),
        data=DataConfig(
            hf_repo_id=str(_require(data_raw, "hf_repo_id")),
            view_order=view_order,
            revision=data_raw.get("revision"),
            local_dir=data_raw.get("local_dir"),
            expected_counts={str(key): int(value) for key, value in expected_counts.items()},
        ),
        model=ModelConfig(
            profile=str(_require(model_raw, "profile")),
            vision=VisionConfig(
                name=str(_require(vision_raw, "name")),
                model_id=str(_require(vision_raw, "model_id")),
                revision=vision_raw.get("revision"),
                output_dim=int(_require(vision_raw, "output_dim")),
                seq_len=int(_require(vision_raw, "seq_len")),
                image_size=int(_require(vision_raw, "image_size")),
            ),
            text=TextConfig(
                model_id=str(_require(text_raw, "model_id")),
                revision=text_raw.get("revision"),
                d_model=int(_require(text_raw, "d_model")),
            ),
        ),
        training=TrainingConfig(
            batch_size=int(_require(train_raw, "batch_size")),
            gradient_accumulation_steps=int(_require(train_raw, "gradient_accumulation_steps")),
            learning_rate=float(train_raw.get("learning_rate", 1e-4)),
            vision_learning_rate=float(train_raw["vision_learning_rate"]) if train_raw.get("vision_learning_rate") is not None else None,
            text_learning_rate=float(train_raw["text_learning_rate"]) if train_raw.get("text_learning_rate") is not None else None,
            head_learning_rate=float(train_raw["head_learning_rate"]) if train_raw.get("head_learning_rate") is not None else None,
            weight_decay=float(train_raw.get("weight_decay", 0.05)),
            scheduler_gamma=float(train_raw.get("scheduler_gamma", 0.9)),
            align_epochs=int(train_raw.get("align_epochs", 6)),
            finetune_epochs=int(train_raw.get("finetune_epochs", 6)),
            gpa_hidden_size=int(train_raw.get("gpa_hidden_size", 128)),
            max_steps=int(train_raw["max_steps"]) if train_raw.get("max_steps") is not None else None,
            align_max_steps=int(train_raw["align_max_steps"]) if train_raw.get("align_max_steps") is not None else None,
            finetune_max_steps=int(train_raw["finetune_max_steps"]) if train_raw.get("finetune_max_steps") is not None else None,
            val_every_steps=int(train_raw["val_every_steps"]) if train_raw.get("val_every_steps") is not None else None,
            save_every_steps=int(train_raw["save_every_steps"]) if train_raw.get("save_every_steps") is not None else None,
            max_grad_norm=float(train_raw["max_grad_norm"]) if train_raw.get("max_grad_norm") is not None else None,
            num_workers=int(train_raw.get("num_workers", 0)),
            pin_memory=bool(train_raw.get("pin_memory", False)),
            progress_log_every_steps=int(train_raw.get("progress_log_every_steps", 50)),
            vision_training=vision_training,
        ),
        cache=CacheConfig(dir=str(cache_raw.get("dir", "outputs/cache"))),
        runtime=RuntimeConfig(device=str(runtime_raw.get("device", "auto")), precision=precision),
        generation=GenerationConfig(
            max_new_tokens=int(generation_raw.get("max_new_tokens", 64)),
            num_beams=int(generation_raw.get("num_beams", 1)),
            early_stopping=bool(generation_raw.get("early_stopping", False)),
            length_penalty=float(generation_raw.get("length_penalty", 1.0)),
        ),
        evaluation=EvaluationConfig(
            eval_batch_size=int(evaluation_raw.get("eval_batch_size", 16)),
            eval_progress_log_every_samples=int(evaluation_raw.get("eval_progress_log_every_samples", 256)),
            benchmark_max_samples=int(evaluation_raw.get("benchmark_max_samples", 200)),
            benchmark_progress_log_every_samples=int(evaluation_raw.get("benchmark_progress_log_every_samples", 20)),
        ),
    )
