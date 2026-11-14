"""Native runtime contracts for the Kaggle AMP lifecycle."""
from dataclasses import replace
import contextlib

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from efficient_vlm_ad.smolvlm import pipeline
from test_smolvlm_training import prepared


def test_exact_rank_partition_includes_empty_ranks():
    source = DataLoader(TensorDataset(torch.arange(5)), batch_size=2)
    partitions = [pipeline.partition_loader(source, rank, 3) for rank in range(3)]
    actual = [[int(x) for batch in part for x in batch[0]] for part in partitions]
    assert actual == [[0, 3], [1, 4], [2]]
    assert len(pipeline.partition_loader(source, 5, 6)) == 0


def test_validation_restores_training_and_vision_stays_frozen(tmp_path):
    from efficient_vlm_ad.smolvlm.model import build_model_and_processor
    cfg = prepared(tmp_path)
    model, processor = build_model_and_processor(cfg)
    model.train()
    value = pipeline.validation(model, pipeline.loader(cfg, model, processor, 'val'), torch.device('cpu'))
    assert value > 0
    assert model.training and model.native.model.text_model.training
    assert not model.native.model.vision_model.training


def test_loader_persistent_workers_are_conditional(tmp_path):
    from efficient_vlm_ad.smolvlm.model import build_model_and_processor
    cfg = prepared(tmp_path)
    model, processor = build_model_and_processor(cfg)
    assert pipeline.loader(cfg, model, processor, 'train', train=True).persistent_workers is False
    variant = replace(cfg, training=replace(cfg.training, num_workers=2))
    batches = pipeline.loader(variant, model, processor, 'train', train=True)
    assert batches.persistent_workers is True


def test_no_sync_encloses_forward_and_backward_only_before_boundary():
    trace = []
    class Backend:
        @contextlib.contextmanager
        def no_sync(self, model):
            trace.append('enter')
            yield
            trace.append('exit')
    for boundary in (False, True):
        with pipeline.accumulation_context(Backend(), object(), boundary):
            trace.extend(['forward', 'backward'])
    assert trace == ['enter', 'forward', 'backward', 'exit', 'forward', 'backward']


def test_prediction_cleanup_only_removes_outer_whitespace():
    assert pipeline.clean_predictions(['  A  road.\n', '\t', 'A\nB']) == ['A  road.', '', 'A\nB']


def test_stress_uses_distinct_longest_indices_and_rank_slices():
    from efficient_vlm_ad.smolvlm.stress import longest_indices
    assert longest_indices([4, 10, 7, 10, 1], 4) == [1, 3, 2, 0]
    with pytest.raises(ValueError, match='distinct'):
        longest_indices([1, 2], 3)


def test_stress_offline_real_train_tail_validation_and_generation(tmp_path):
    from efficient_vlm_ad.smolvlm.stress import run_stress
    cfg = prepared(tmp_path, rows=5)
    cache = tmp_path / 'output/cache'
    initial = {path.name: path.read_bytes() for path in cache.iterdir()}
    train = run_stress(cfg, phase='train', disable_progress=True)
    rank = train['ranks'][0]
    assert rank['gradient_checkpointing_active'] is True
    assert rank['successful_updates'] == 3
    assert [window['window_size'] for window in rank['optimizer_windows']] == [2, 2, 1]
    assert len(set(rank['train_selection']['source_indices'])) == 5
    assert rank['validation_loss'] > 0
    generated = run_stress(cfg, phase='evaluate', disable_progress=True)
    assert generated['ranks'][0]['generated_samples'] == 2
    assert generated['ranks'][0]['precision'] == 'fp32'
    assert not (tmp_path / 'output/checkpoints').exists()
    assert initial == {path.name: path.read_bytes() for path in cache.iterdir()}


def test_native_training_skips_overflow_update_and_keeps_tail_counter(tmp_path, monkeypatch):
    """Exercise real GradScaler/AdamW skip through the production train loop on CPU."""
    import json
    cfg = prepared(tmp_path, rows=3)
    backend = pipeline.accelerator_for(cfg)
    backend.scaler = torch.amp.GradScaler('cpu', init_scale=256)
    def unscale(optimizer=None):
        for wrapped in backend._optimizers if optimizer is None else [optimizer]:
            raw = wrapped.optimizer if hasattr(wrapped, 'optimizer') else wrapped
            backend.scaler.unscale_(raw)
    monkeypatch.setattr(backend, 'unscale_gradients', unscale)
    monkeypatch.setattr(pipeline, 'accelerator_for', lambda config: backend)
    original_setup = pipeline.setup
    def setup(*args, **kwargs):
        model, processor = original_setup(*args, **kwargs)
        first = next(p for p in model.parameters() if p.requires_grad)
        calls = [0]
        def overflow_once(gradient):
            calls[0] += 1
            return torch.full_like(gradient, float('inf')) if calls[0] == 1 else gradient
        first.register_hook(overflow_once)
        return model, processor
    monkeypatch.setattr(pipeline, 'setup', setup)
    result = pipeline.train(cfg, disable_progress=True)
    assert result['global_step'] == 3 and result['optimizer_step'] == 1
    rows = [json.loads(line) for line in (tmp_path / 'output/debug/debug_events.jsonl').read_text().splitlines()]
    steps = [row['payload'] for row in rows if row['section'] == 'TRAIN_STEP']
    assert [row['optimizer_skipped'] for row in steps] == [True, False]
    assert [row['window_size'] for row in steps] == [2, 1]
    assert steps[0]['gradient_norm_before_clipping'] is None
    assert [row['grad_scaler_scale'] for row in steps] == [128., 128.]
