import json
from pathlib import Path

import pytest
import torch

from efficient_vlm_ad.accumulation_debug import compare_accumulation_gradients
from efficient_vlm_ad.pipeline import _SingleProcessAccelerator


@pytest.mark.parametrize('steps', [1, 2, 4])
def test_debug_compares_full_and_tail_without_updating_weights_or_rng(steps):
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(.3), torch.nn.Linear(4, 1))
    batches = [torch.randn(2, 3) + i for i in range(steps)]
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    backend = _SingleProcessAccelerator(torch.device('cpu'), gradient_accumulation_steps=steps)
    report = compare_accumulation_gradients(model, batches, backend,
                                           lambda batch: model(batch).square().mean())
    assert report['ok'] and report['optimizer_steps'] == 0
    assert report['scope'] == 'single_process_gradient_comparison'
    assert report['cases'][0]['window_size'] == steps
    for case in report['cases']:
        assert case['ok'] and case['checked_tensors'] > 0
        assert case['gradient_norm_ratio'] == pytest.approx(1.)
        assert case['max_abs_error'] < 1e-6
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())


def test_debug_detects_double_division():
    model = torch.nn.Linear(2, 1)
    backend = _SingleProcessAccelerator(torch.device('cpu'), gradient_accumulation_steps=2)
    backward = backend.backward
    backend.backward = lambda loss: backward(loss / 2)
    batches = [torch.ones(2, 2), torch.full((2, 2), 3.)]
    report = compare_accumulation_gradients(model, batches, backend, lambda b: model(b).square().mean())
    assert not report['ok']
    assert all(case['gradient_norm_ratio'] == pytest.approx(.5) for case in report['cases'])


def test_debug_handles_missing_and_nonfinite_gradients():
    model = torch.nn.Linear(2, 1)
    backend = _SingleProcessAccelerator(torch.device('cpu'), gradient_accumulation_steps=2)
    report = compare_accumulation_gradients(model, [torch.ones(1, 2)] * 2, backend,
                                           lambda b: model(b).sum() * torch.tensor(float('nan')))
    assert not report['ok']
    assert all(not case['finite'] for case in report['cases'])


def test_mvpruner_notebook_has_numeric_debug_before_training():
    notebook = json.loads(Path('notebooks/kaggle_train_dinov2_mvpruner_finetune.ipynb').read_text(encoding='utf8'))
    cells = [''.join(c['source']) for c in notebook['cells']]
    debug = next(i for i, s in enumerate(cells) if '-m efficient_vlm_ad.accumulation_debug' in s)
    train = next(i for i, s in enumerate(cells) if 'accelerate.commands.launch' in s)
    assert debug < train
    assert '--config "$CONFIG"' in cells[debug]


def test_actual_t5_mvpruner_probe_writes_json_and_debug_event(tmp_path, monkeypatch):
    from dataclasses import replace
    from config_helpers import fake_config
    from efficient_vlm_ad import pipeline
    from efficient_vlm_ad.accumulation_debug import diagnose_accumulation
    cfg = fake_config()
    cfg = replace(cfg, project=replace(cfg.project, output_dir=str(tmp_path)),
                  training=replace(cfg.training, batch_size=1, gradient_accumulation_steps=2),
                  runtime=replace(cfg.runtime, device='cpu', precision='fp32'))

    class Samples(torch.utils.data.Dataset):
        def __len__(self): return 2
        def __getitem__(self, index):
            return {'question': f'question {index}', 'answer': 'road',
                    'visual_features': torch.full((6, cfg.model.vision.seq_len, cfg.model.vision.output_dim),
                                                  float(index + 1)), 'eval_id': str(index)}
    monkeypatch.setattr(pipeline, '_make_dataset_and_collator',
                        lambda cfg, split, tokenizer, **kwargs: (Samples(), pipeline.CachedBatchCollator(tokenizer)))
    build = pipeline._build_train_model
    def with_dropout(cfg):
        model, tokenizer, transform = build(cfg)
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout): module.p = .2
        return model, tokenizer, transform
    monkeypatch.setattr(pipeline, '_build_train_model', with_dropout)
    report = diagnose_accumulation(cfg)
    assert report['ok'] and report['weights_and_buffers_unchanged']
    assert [c['window_size'] for c in report['cases']] == [2, 1]
    assert all(c['gradient_norm_ratio'] == pytest.approx(1.) for c in report['cases'])
    saved = json.loads(Path(report['report_path']).read_text(encoding='utf8'))
    assert saved == report
    from efficient_vlm_ad.debugging import default_debug_jsonl
    events = [json.loads(s) for s in Path(default_debug_jsonl(str(tmp_path))).read_text().splitlines()]
    assert events[-1]['section'] == 'ACCUMULATION_VERIFY'
    assert not (tmp_path / 'checkpoints').exists()


def test_debug_detects_incorrect_training_loss_helper(monkeypatch):
    import efficient_vlm_ad.accumulation_debug as module
    original = module.loss_for_backward
    monkeypatch.setattr(module, 'loss_for_backward', lambda raw, a, k: original(raw, a, k) / a)
    model = torch.nn.Linear(2, 1)
    backend = _SingleProcessAccelerator(torch.device('cpu'), gradient_accumulation_steps=2)
    report = module.compare_accumulation_gradients(model, [torch.ones(1, 2)] * 2, backend,
                                                   lambda b: model(b).square().sum())
    assert not report['ok']
    assert all(c['gradient_norm_ratio'] == pytest.approx(.5) for c in report['cases'])


def test_debug_command_exits_with_failure_if_comparison_fails(monkeypatch, capsys):
    import efficient_vlm_ad.accumulation_debug as module
    from efficient_vlm_ad import config
    monkeypatch.setattr(config, 'load_config', lambda path: object())
    monkeypatch.setattr(module, 'diagnose_accumulation', lambda cfg: {
        'ok': False, 'report_path': 'debug/gradient_accumulation_verify.json',
        'cases': [{'ok': False, 'name': 'full_window', 'window_size': 2,
                   'gradient_norm_ratio': .5, 'max_abs_error': .1}],
    })
    with pytest.raises(SystemExit, match='do not start full training'):
        module.main(['--config', 'runtime.yaml'])
    assert 'FAIL full_window' in capsys.readouterr().out
