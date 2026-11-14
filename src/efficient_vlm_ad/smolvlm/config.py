"""Configuration for the native cached SmolVLM baseline."""
from dataclasses import dataclass, field, fields
from pathlib import Path
import math
import yaml

from ..config import (CAMERA_ORDER, ProjectConfig, DataConfig, CacheConfig,
                      RuntimeConfig, TrainingConfig, GenerationConfig, EvaluationConfig)


def validate_prompt_camera_order(order):
    if (not isinstance(order, (list, tuple)) or len(order) != len(CAMERA_ORDER)
            or any(not isinstance(camera, str) for camera in order)
            or sorted(order) != sorted(CAMERA_ORDER)):
        raise ValueError('model.prompt_camera_order must be an exact permutation of the six canonical cameras')
    return list(order)


@dataclass(frozen=True)
class ModelConfig:
    model_id: str = 'HuggingFaceTB/SmolVLM-256M-Instruct'
    revision: str | None = '7e3e67edbbed1bf9888184d9df282b700a323964'
    prompt_camera_order: list[str] = field(default_factory=lambda: list(CAMERA_ORDER))

    def __post_init__(self):
        validate_prompt_camera_order(self.prompt_camera_order)


@dataclass(frozen=True)
class SmolVLMConfig:
    project: ProjectConfig
    data: DataConfig
    cache: CacheConfig = field(default_factory=lambda: CacheConfig('outputs/smolvlm/cache'))
    runtime: RuntimeConfig = field(default_factory=lambda: RuntimeConfig(precision='fp32'))
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=lambda: TrainingConfig(
        batch_size=8, gradient_accumulation_steps=4, align_epochs=0, finetune_epochs=10,
        gradient_checkpointing=True, max_grad_norm=1.0, num_workers=2,
        pin_memory=True, progress_log_every_steps=500))
    generation: GenerationConfig = field(default_factory=lambda: GenerationConfig(
        max_new_tokens=512, num_beams=3, early_stopping=True, length_penalty=1.0))
    evaluation: EvaluationConfig = field(default_factory=lambda: EvaluationConfig(eval_batch_size=4))

    def __post_init__(self):
        if self.data.view_order != CAMERA_ORDER:
            raise ValueError('data.view_order must match the six canonical cameras')
        if self.runtime.precision not in {'fp32', 'fp16'}:
            raise ValueError('runtime.precision must be fp32 or fp16')
        if self.training.align_epochs != 0 or type(self.training.align_epochs) is not int:
            raise ValueError('native SmolVLM supports finetune only: align_epochs=0')
        if self.training.vision_training != 'feature_cache' or self.training.entropy_weight != 0:
            raise ValueError('native SmolVLM requires feature_cache and native CE only')
        if self.training.beta_decay_updates is not None:
            raise ValueError('beta schedules are unsupported')
        for group in (self.training, self.generation, self.evaluation):
            for f in fields(group):
                value = getattr(group, f.name)
                if f.name in {'pin_memory', 'gradient_checkpointing', 'early_stopping'}:
                    if type(value) is not bool:
                        raise ValueError(f'{f.name} must be boolean')
                elif value is not None and f.name not in {'vision_training'}:
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ValueError(f'{f.name} must be finite numeric')
                    if f.name in {'align_epochs', 'entropy_weight', 'weight_decay', 'seed', 'num_workers'}:
                        if value < 0:
                            raise ValueError(f'{f.name} must be nonnegative')
                    elif value <= 0:
                        raise ValueError(f'{f.name} must be positive')
                    if (f.name.endswith(('steps', 'epochs', 'samples')) or f.name in {
                        'seed', 'num_workers', 'batch_size', 'num_beams', 'max_new_tokens',
                        'eval_batch_size'}) and type(value) is not int:
                        raise ValueError(f'{f.name} must be an integer')
        if self.data.expected_counts is not None and (
            not isinstance(self.data.expected_counts, dict) or any(
                type(v) is not int or v < 0 for v in self.data.expected_counts.values())):
            raise ValueError('expected_counts must contain nonnegative integers')
        for name, value in (('model.revision', self.model.revision), ('data.revision', self.data.revision),
                            ('data.local_dir', self.data.local_dir)):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f'{name} must be null or a nonempty string')
        for name in ('vision_learning_rate', 'text_learning_rate', 'head_learning_rate', 'align_max_steps'):
            if getattr(self.training, name) is not None:
                raise ValueError(f'training.{name} is unsupported by the native baseline')
        for value in (self.project.output_dir, self.cache.dir, self.data.hf_repo_id, self.model.model_id):
            if not isinstance(value, str) or not value.strip():
                raise ValueError('paths and model/dataset ids must be nonempty strings')


def load_config(path: str | Path) -> SmolVLMConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(raw, dict):
        raise ValueError('configuration must be a mapping')
    allowed = {f.name for f in fields(SmolVLMConfig)}
    if set(raw) - allowed:
        raise ValueError(f'Unknown configuration sections: {set(raw) - allowed}')
    defaults = SmolVLMConfig(ProjectConfig('outputs/smolvlm'), DataConfig(
        'minhdang0901/drivelm-nuscenes-v1-0', list(CAMERA_ORDER),
        expected_counts={'train': 341381, 'val': 19785, 'test': 16817}))
    values = {}
    for name in allowed:
        default = getattr(defaults, name)
        section = raw.get(name, {})
        if not isinstance(section, dict):
            raise ValueError(f'{name} must be a mapping')
        forbidden = set(section) - {f.name for f in fields(default)}
        if name == 'training':
            forbidden |= set(section) & {'beta_decay_updates', 'vision_learning_rate',
                                        'text_learning_rate', 'head_learning_rate', 'align_max_steps'}
        if forbidden:
            raise ValueError(f'Unsupported {name} fields: {forbidden}')
        values[name] = type(default)(**{**{f.name: getattr(default, f.name) for f in fields(default)}, **section})
    return SmolVLMConfig(**values)
