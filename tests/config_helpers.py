"""Self-contained configurations for offline tests, independent of example artifacts."""
from dataclasses import asdict

import yaml

from efficient_vlm_ad.config import (
    CAMERA_ORDER, CacheConfig, DataConfig, ExperimentConfig, GenerationConfig,
    ModelConfig, ProjectConfig, RuntimeConfig, TextConfig, TrainingConfig, VisionConfig,
)


def fake_config():
    return ExperimentConfig(
        project=ProjectConfig(output_dir='outputs/test'),
        data=DataConfig(hf_repo_id='local/fake', view_order=CAMERA_ORDER.copy(),
                        expected_counts={'train': 341381, 'val': 19785, 'test': 16817}),
        model=ModelConfig(profile='offline_test',
                          vision=VisionConfig(name='fake_vision', model_id='fake', output_dim=512,
                                              seq_len=49, image_size=224),
                          text=TextConfig(model_id='fake_t5', d_model=256)),
        training=TrainingConfig(batch_size=2, gradient_accumulation_steps=2, max_steps=20),
        cache=CacheConfig(dir='outputs/test/cache'), runtime=RuntimeConfig(),
        generation=GenerationConfig(),
    )


def write_fake_config(tmp_path):
    path = tmp_path / 'fake.yaml'
    path.write_text(yaml.safe_dump(asdict(fake_config())), encoding='utf-8')
    return path
