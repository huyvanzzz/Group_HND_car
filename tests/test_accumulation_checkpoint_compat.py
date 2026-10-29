from dataclasses import replace

import pytest
import torch

from config_helpers import fake_config
from efficient_vlm_ad import query_training
from efficient_vlm_ad import training_accumulation as accumulation
from efficient_vlm_ad.checkpointing import architecture_signature, load_checkpoint, _rng_state


def test_accumulation_recipe_requires_matching_v2_for_training():
    cfg = fake_config()
    cfg = replace(cfg, training=replace(cfg.training, gradient_accumulation_steps=4))
    expected = {'accumulation_recipe': {'version': 2, 'steps': 4,
                'normalization': 'backend_with_tail_correction', 'scheduler': 'epoch'}}
    assert accumulation.accumulation_metadata(cfg) == expected
    accumulation.validate_accumulation_checkpoint(cfg, expected)
    for metadata in ({}, {'accumulation_recipe': {'version': 1}},
                     {'accumulation_recipe': {**expected['accumulation_recipe'], 'steps': 2}},
                     {'accumulation_recipe': {**expected['accumulation_recipe'], 'scheduler': 'batch'}}):
        with pytest.raises(ValueError, match='accumulation'):
            accumulation.validate_accumulation_checkpoint(cfg, metadata)


def test_accumulation_does_not_change_architecture_or_legacy_inference_load(tmp_path):
    cfg = fake_config()
    changed = replace(cfg, training=replace(cfg.training, gradient_accumulation_steps=4))
    assert architecture_signature(cfg) == architecture_signature(changed)
    model = torch.nn.Linear(1, 1)
    path = tmp_path / 'legacy.pt'
    torch.save({'model': model.state_dict(), 'metadata': {'stage': 'finetune'}, 'rng_state': _rng_state()}, path)
    assert load_checkpoint(path, model=model)['stage'] == 'finetune'


@pytest.mark.parametrize('fusion,version', [
    ('mvpruner_camera_query', 'camera_query_manual_accumulation_v2'),
    ('mvpruner_camera_query_qaware', 'qaware_prefix_dp_manual_accumulation_v2')])
def test_query_recipe_versions_record_backend_tail_semantics(fusion, version):
    cfg = fake_config()
    cfg = replace(cfg, model=replace(cfg.model, fusion=replace(cfg.model.fusion, name=fusion)),
                  runtime=replace(cfg.runtime, precision='fp32'))
    recipe = query_training.training_recipe(cfg)
    assert recipe['training_loop_version'] == version
    assert recipe['accumulation_normalization'] == 'backend_with_tail_correction'
    assert recipe['scheduler_cadence'] == 'epoch'


def test_shared_windows_and_tail_loss_preserve_full_window_identity():
    assert query_training.window_size is accumulation.window_size
    assert query_training.window_boundary is accumulation.window_boundary
    assert [accumulation.window_size(i, 5, 2) for i in range(5)] == [2, 2, 2, 2, 1]
    assert [accumulation.window_boundary(i, 5, 2) for i in range(5)] == [False, True, False, True, True]
    raw = torch.tensor(3., requires_grad=True)
    assert accumulation.loss_for_backward(raw, 4, 4) is raw
    corrected = accumulation.loss_for_backward(raw, 4, 2)
    corrected.backward()
    assert corrected.item() == 6.
    assert raw.grad.item() == 2.
