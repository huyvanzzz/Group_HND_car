from __future__ import annotations

from dataclasses import dataclass, field
import math
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
class FeatureExtractionConfig:
    output_layer: str = "last_hidden_state"
    prefix_tokens_to_remove: int = 1
    source_grid: list[int] = field(default_factory=lambda: [16, 16])
    pooling: str = "avg_pool2d"
    kernel_size: int = 2
    stride: int = 2
    output_grid: list[int] = field(default_factory=lambda: [8, 8])


@dataclass(frozen=True)
class VisionConfig:
    name: str
    model_id: str
    output_dim: int
    seq_len: int
    image_size: int
    revision: str | None = None
    feature_extraction: FeatureExtractionConfig = field(default_factory=FeatureExtractionConfig)

    def __post_init__(self) -> None:
        if self.name == "dinov2_small_avgpool":
            if (self.output_dim, self.seq_len, self.image_size) != (384, 64, 224):
                raise ValueError("DINOv2 requires output_dim=384, seq_len=64, image_size=224")
            if self.feature_extraction != FeatureExtractionConfig():
                raise ValueError("DINOv2 requires last_hidden_state, one CLS, 16x16 avg_pool2d(2,2) to 8x8")


@dataclass(frozen=True)
class ProjectorConfig:
    type: str = "auto"

    def __post_init__(self) -> None:
        if self.type not in {"auto", "linear"}:
            raise ValueError("model.projector.type must be auto or linear")


@dataclass(frozen=True)
class TextConfig:
    model_id: str
    d_model: int
    revision: str | None = None


@dataclass(frozen=True)
class CameraQueryConfig:
    dim: int = 64
    init_std: float = 1.0
    rms_eps: float = 1e-6
    position_policy: str = "new_sequence"

    def __post_init__(self) -> None:
        if type(self.dim) is not int or self.dim <= 0:
            raise ValueError("fusion.query.dim must be a positive integer")
        for name in ("init_std", "rms_eps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"fusion.query.{name} must be finite and positive")
        if self.position_policy != "new_sequence":
            raise ValueError("fusion.query.position_policy must be new_sequence")


@dataclass(frozen=True)
class FusionConfig:
    name: str = "mvpruner_two_stage"
    eps: float = 1e-8
    stage1_top_k: int = 196
    stage2_top_k: int = 49
    stage2_layer: int = 2
    min_tokens_per_camera: int = 1
    stage1_relevance_mode: str = "shifted_max_cosine"
    beta_start: float = 0.2
    beta_end: float = 0.0
    beta_schedule: str = "linear"
    tau: float = 1.0
    query: CameraQueryConfig | None = None

    def __post_init__(self) -> None:
        if self.name == "mvpruner_camera_query":
            if self.query is None:
                object.__setattr__(self, "query", CameraQueryConfig())
            elif not isinstance(self.query, CameraQueryConfig):
                raise ValueError("fusion.query must be a CameraQueryConfig")
        elif self.query is not None:
            raise ValueError("fusion.query is supported only by mvpruner_camera_query")


@dataclass(frozen=True)
class ModelConfig:
    profile: str
    vision: VisionConfig
    text: TextConfig
    fusion: FusionConfig = field(default_factory=FusionConfig)
    projector: ProjectorConfig = field(default_factory=ProjectorConfig)

    def __post_init__(self) -> None:
        if self.vision.name == "dinov2_small_avgpool" and self.projector.type != "linear":
            raise ValueError("DINOv2 requires model.projector.type=linear")


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
    gradient_checkpointing: bool = False
    entropy_weight: float = 0.0
    seed: int = 42

    def __post_init__(self) -> None:
        if not isinstance(self.gradient_checkpointing, bool):
            raise ValueError("training.gradient_checkpointing must be a boolean")
        if isinstance(self.entropy_weight, bool) or not isinstance(self.entropy_weight, (int, float)) or not math.isfinite(self.entropy_weight) or self.entropy_weight < 0:
            raise ValueError("training.entropy_weight must be finite and nonnegative")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("training.seed must be a nonnegative integer")

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

    def __post_init__(self) -> None:
        if self.model.fusion.name != "mvpruner_camera_query":
            return
        fusion = self.model.fusion
        total = len(self.data.view_order) * self.model.vision.seq_len
        if fusion.min_tokens_per_camera < 1 or not (len(self.data.view_order) * fusion.min_tokens_per_camera <= fusion.stage2_top_k <= fusion.stage1_top_k <= total):
            raise ValueError("camera query budgets must satisfy views*minimum <= stage2_top_k <= stage1_top_k <= total")
        if fusion.stage2_layer != 2 or fusion.stage1_relevance_mode not in {"shifted_max_cosine", "diversity_only"}:
            raise ValueError("camera query requires stage2_layer=2 and a supported Stage 1 relevance mode")
        if not math.isfinite(fusion.eps) or fusion.eps <= 0:
            raise ValueError("camera query eps must be finite and positive")
        if self.runtime.precision != "fp32" or self.training.vision_training != "feature_cache":
            raise ValueError("camera query currently requires fp32 feature_cache training")
        if self.training.gradient_checkpointing:
            raise ValueError("gradient_checkpointing is not supported for camera query")
        if self.training.batch_size < 1 or self.training.gradient_accumulation_steps < 1:
            raise ValueError("camera query batch_size and gradient_accumulation_steps must be positive")


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
    adapter_raw = model_raw.get("adapter", {})
    fusion_raw = model_raw.get("fusion", {})
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
    adapter_name = str(adapter_raw.get("name", "none"))
    # Historical MVPruner YAML may declare an inactive adapter.
    if adapter_name != "none":
        raise ValueError("mvpruner_two_stage requires model.adapter.name=none; adapters are no longer supported")
    fusion_name = str(fusion_raw.get("name", ""))
    if fusion_name not in {"mvpruner_two_stage", "unpruned", "mvpruner_camera_query"}:
        raise ValueError("model.fusion.name must be mvpruner_two_stage, mvpruner_camera_query or unpruned")
    if fusion_name == "unpruned" and vision_training != "feature_cache":
        raise ValueError("unpruned currently requires feature_cache vision training")
    gradient_checkpointing = train_raw.get("gradient_checkpointing", False)
    if not isinstance(gradient_checkpointing, bool):
        raise ValueError("training.gradient_checkpointing must be a boolean")
    if gradient_checkpointing and fusion_name != "unpruned":
        raise ValueError("training.gradient_checkpointing is currently supported only for unpruned")
    stage1_top_k = int(fusion_raw.get("stage1_top_k", len(view_order) * int(vision_raw.get("seq_len", 49))))
    stage2_top_k = int(fusion_raw.get("stage2_top_k", fusion_raw.get("top_k", vision_raw.get("seq_len", 49))))
    minimum_per_camera = int(fusion_raw.get("min_tokens_per_camera", 1))
    relevance_mode = str(fusion_raw.get("stage1_relevance_mode", "shifted_max_cosine"))
    beta_start = float(fusion_raw.get("beta_start", 0.2))
    beta_end = float(fusion_raw.get("beta_end", 0.0))
    beta_schedule = str(fusion_raw.get("beta_schedule", "linear"))
    tau = float(fusion_raw.get("tau", 1.0))
    query_raw = fusion_raw.get("query")
    query = None
    if fusion_name == "mvpruner_camera_query":
        if query_raw is not None and not isinstance(query_raw, dict):
            raise ValueError("fusion.query must be a mapping")
        try:
            query = CameraQueryConfig(**(query_raw or {}))
        except TypeError as exc:
            raise ValueError(f"Invalid fusion.query: {exc}") from exc
    elif query_raw is not None:
        raise ValueError("fusion.query is supported only by mvpruner_camera_query")
    if fusion_name == "mvpruner_two_stage":
        total_visual = len(view_order) * int(_require(vision_raw, "seq_len"))
        if not (len(view_order) * minimum_per_camera <= stage2_top_k <= stage1_top_k <= total_visual):
            raise ValueError("mvpruner budgets must satisfy views*minimum <= stage2_top_k <= stage1_top_k <= total visual tokens")
        if relevance_mode not in {"shifted_max_cosine", "diversity_only"}:
            raise ValueError("stage1_relevance_mode must be shifted_max_cosine or diversity_only")
        if beta_start < beta_end or beta_end < 0:
            raise ValueError("mvpruner beta must satisfy beta_start >= beta_end >= 0")
        if beta_schedule != "linear":
            raise ValueError("mvpruner beta_schedule must be linear")
        if tau <= 0:
            raise ValueError("mvpruner tau must be positive")
        if int(fusion_raw.get("stage2_layer", 2)) != 2:
            raise ValueError("mvpruner stage2_layer must be 2 for the tested four-block T5 adapter")
        if adapter_name != "none":
            raise ValueError("mvpruner_two_stage requires model.adapter.name=none")
        if vision_training != "feature_cache":
            raise ValueError("mvpruner_two_stage currently requires feature_cache vision training")
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
                feature_extraction=FeatureExtractionConfig(**vision_raw.get("feature_extraction", {})),
            ),
            projector=ProjectorConfig(**model_raw.get("projector", {})),
            text=TextConfig(
                model_id=str(_require(text_raw, "model_id")),
                revision=text_raw.get("revision"),
                d_model=int(_require(text_raw, "d_model")),
            ),
            fusion=FusionConfig(
                name=fusion_name,
                eps=float(fusion_raw.get("eps", 1e-8)),
                stage1_top_k=stage1_top_k,
                stage2_top_k=stage2_top_k,
                stage2_layer=int(fusion_raw.get("stage2_layer", 2)),
                min_tokens_per_camera=minimum_per_camera,
                stage1_relevance_mode=relevance_mode,
                beta_start=beta_start,
                beta_end=beta_end,
                beta_schedule=beta_schedule,
                tau=tau,
                query=query,
            ),
        ),
        training=TrainingConfig(
            gradient_checkpointing=gradient_checkpointing,
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
            entropy_weight=train_raw.get("entropy_weight", 0.5 if fusion_name == "mvpruner_camera_query" else 0.0),
            seed=train_raw.get("seed", 42),
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
