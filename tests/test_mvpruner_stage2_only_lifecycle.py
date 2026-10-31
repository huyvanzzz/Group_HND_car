"""Offline integration coverage for Stage-2-only native T5 attention pruning."""
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from config_helpers import fake_config
from test_accumulation_v2_lifecycle import _dataset
from efficient_vlm_ad import pipeline
from efficient_vlm_ad.checkpointing import architecture_signature, validate_checkpoint_signature
from efficient_vlm_ad.modeling.factory import build_vlm_model
from efficient_vlm_ad.training_accumulation import accumulation_metadata


def _config(path, epochs=1):
    cfg = fake_config()
    return replace(cfg, project=replace(cfg.project, output_dir=str(path)),
        model=replace(cfg.model, vision=replace(cfg.model.vision, output_dim=8, seq_len=4),
            projector=replace(cfg.model.projector, type='linear'),
            fusion=replace(cfg.model.fusion, name='mvpruner_stage2_only', stage1_top_k=24,
                stage2_top_k=8, min_tokens_per_camera=1, stage2_scoring=None)),
        training=replace(cfg.training, seed=42, batch_size=1, gradient_accumulation_steps=2,
            max_steps=None, finetune_epochs=epochs),
        runtime=replace(cfg.runtime, device='cpu', precision='fp32'),
        generation=replace(cfg.generation, max_new_tokens=2))


def _batch(tokenizer, seed=42):
    generator = torch.Generator().manual_seed(seed)
    encoded = tokenizer(['what is ahead', 'where is traffic'], padding=True, return_tensors='pt')
    return {**encoded, 'visual_features': torch.randn(2, 6, 4, 8, generator=generator),
            'labels': tokenizer(['road', 'car'], padding=True, return_tensors='pt')['input_ids']}


def _fallback(monkeypatch):
    monkeypatch.setattr(pipeline, '_build_accelerator', lambda cfg:
        pipeline._SingleProcessAccelerator(torch.device('cpu'),
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps))


def test_real_t5_answer_ce_reaches_native_attention_and_trainable_components(tmp_path):
    torch.manual_seed(42)
    model, tokenizer = build_vlm_model(_config(tmp_path))
    model.train()
    batch = _batch(tokenizer)
    output = model.forward_from_features(**batch)
    expected = torch.nn.functional.cross_entropy(output.logits.flatten(0, 1), batch['labels'].flatten())
    torch.testing.assert_close(output.loss, expected)
    assert getattr(output, 'entropy_loss', None) is None
    output.loss.backward()
    assert not hasattr(model.visual_fusion, 'question_scorer')
    assert not [name for name, _ in model.visual_fusion.named_parameters()
                if not name.startswith('camera_embeddings.')]
    attention = [(name, parameter) for name, parameter in model.text_model.named_parameters()
                 if name.startswith('encoder.') and 'SelfAttention' in name and ('.q.' in name or '.k.' in name)]
    assert attention
    for name, parameter in attention:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    for component in (model.projector, model.modal_embeddings, model.row_embeddings,
                      model.col_embeddings, model.visual_fusion.camera_embeddings, model.text_model):
        gradients = [p.grad for p in component.parameters() if p.requires_grad]
        assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum() for g in gradients) > 0


def test_accumulation_probe_preserves_state_and_handles_tail(tmp_path):
    from efficient_vlm_ad.accumulation_debug import compare_accumulation_gradients
    torch.manual_seed(42)
    model, tokenizer = build_vlm_model(_config(tmp_path))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    report = compare_accumulation_gradients(model, [_batch(tokenizer, seed) for seed in (3, 7)],
        pipeline._SingleProcessAccelerator(torch.device('cpu'), gradient_accumulation_steps=2),
        lambda batch: model.forward_from_features(**batch).loss)
    assert report['ok'] and report['optimizer_steps'] == 0
    assert [case['window_size'] for case in report['cases']] == [2, 1]
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())


def test_unique_signature_and_strict_checkpoint_round_trip(tmp_path):
    cfg = _config(tmp_path)
    signature = architecture_signature(cfg)
    assert signature['fusion.name'] == 'mvpruner_stage2_only'
    assert signature['fusion.stage1_enabled'] is False
    assert not any(key in signature for key in ('mvpruner.stage1_algorithm_version',
                   'fusion.stage1_top_k', 'fusion.stage1_relevance_mode'))
    assert signature['fusion.stage2_scoring'] == 't5_pre_dropout_attention'
    assert not any('question_scoring' in key or key.startswith('fusion.query') for key in signature)
    for profile in ('mvpruner_two_stage', 'mvpruner_qaware', 'mvpruner_camera_query_qaware', 'unpruned'):
        other = replace(cfg, model=replace(cfg.model, fusion=replace(cfg.model.fusion, name=profile, stage2_scoring=None)))
        with pytest.raises(ValueError, match='architecture_signature'):
            validate_checkpoint_signature({'architecture_signature': signature}, cfg=other)
        with pytest.raises(ValueError, match='architecture_signature'):
            validate_checkpoint_signature({'architecture_signature': architecture_signature(other)}, cfg=cfg)
    model, _ = build_vlm_model(cfg)
    path = tmp_path / 'state.pt'
    torch.save(model.state_dict(), path)
    rebuilt, _ = build_vlm_model(cfg)
    rebuilt.load_state_dict(torch.load(path, weights_only=True), strict=True)
    assert all(torch.equal(value, rebuilt.state_dict()[key]) for key, value in model.state_dict().items())
    stale = dict(model.state_dict(), stale_query_weight=torch.ones(1))
    with pytest.raises(RuntimeError, match='Unexpected key'):
        rebuilt.load_state_dict(stale, strict=True)
    missing = dict(model.state_dict())
    del missing[next(iter(missing))]
    with pytest.raises(RuntimeError, match='Missing key'):
        rebuilt.load_state_dict(missing, strict=True)


def test_train_diagnose_resume_eval_benchmark_lifecycle(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    _dataset(monkeypatch, count=3)
    _fallback(monkeypatch)
    probe = pipeline.diagnose_train(cfg, 'finetune', disable_progress=True)
    assert probe['optimizer_step_performed'] is False
    assert probe['mvpruner_gradients']['stage1_enabled'] is False
    assert probe['mvpruner_gradients']['attention_path']['ok']
    assert probe['mvpruner_gradients']['attention_path']['nonzero_ratio'] > 0
    with monkeypatch.context() as fresh:
        def no_warmstart(*args, **kwargs):
            pytest.fail('fresh finetune attempted checkpoint loading')
        fresh.setattr(pipeline, 'load_checkpoint', no_warmstart)
        fresh.setattr(pipeline, '_strict_load_model_from_checkpoint', no_warmstart)
        checkpoint = Path(pipeline.train_stage(cfg, 'finetune', disable_progress=True))
    payload = torch.load(checkpoint, weights_only=False)
    assert payload['metadata']['optimizer_step'] == 2
    assert payload['metadata']['global_step'] == 3
    assert payload['metadata']['accumulation_recipe'] == accumulation_metadata(cfg)['accumulation_recipe']
    beta = {key: value for key, value in payload['model'].items() if key.endswith('beta_step')}
    assert beta and all(value.item() == 2 for value in beta.values())
    assert all(value.item() == 2 for key, value in payload['model'].items() if key.endswith('beta_target_steps'))
    assert all(int(state['step'].item()) == 2 for state in payload['optimizer']['state'].values())
    files = [checkpoint, *Path(str(checkpoint) + '.rng').iterdir()]
    originals = {path: path.read_bytes() for path in files}
    pipeline.verify_resume_checkpoint(cfg, str(checkpoint), stage='finetune', disable_progress=True)
    assert all(path.read_bytes() == value for path, value in originals.items())
    resumed = pipeline.train_stage(replace(cfg, training=replace(cfg.training, finetune_epochs=2)),
        'finetune', resume=str(checkpoint), disable_progress=True)
    resumed_payload = torch.load(resumed, weights_only=False)
    assert resumed_payload['metadata']['optimizer_step'] == 4
    assert all(value.item() == 4 for key, value in resumed_payload['model'].items() if key.endswith('beta_target_steps'))
    # Evaluation accepts a checkpoint predating accumulation metadata and never rewrites it.
    legacy = tmp_path / 'legacy_eval.pt'
    payload['metadata'].pop('accumulation_recipe', None)
    torch.save(payload, legacy)
    original = legacy.read_bytes()
    monkeypatch.setattr(pipeline, 'caption_metrics', lambda rows: {'BLEU_4': 0.})
    pipeline.evaluate_checkpoint(cfg, str(legacy), disable_progress=True)
    pipeline.benchmark_checkpoint(cfg, str(legacy), disable_progress=True)
    assert legacy.read_bytes() == original


def test_align_rejected_before_model_build(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, '_build_train_model', lambda cfg: pytest.fail('align built model'))
    with pytest.raises(ValueError, match='finetune'):
        pipeline.train_stage(_config(tmp_path), 'align', disable_progress=True)


def test_original_pre_prepare_beta_horizon_is_preserved(tmp_path, monkeypatch):
    from accelerate.data_loader import prepare_data_loader
    cfg = _config(tmp_path, epochs=2)
    _dataset(monkeypatch, count=5)

    class Sharded(pipeline._SingleProcessAccelerator):
        def prepare(self, model, optimizer, loader, scheduler):
            assert len(loader) == 5
            loader = prepare_data_loader(loader, device=torch.device('cpu'), num_processes=2, process_index=0)
            assert len(loader) == 3
            return model, optimizer, loader, scheduler

    monkeypatch.setattr(pipeline, '_build_accelerator', lambda cfg:
        Sharded(torch.device('cpu'), gradient_accumulation_steps=2))
    checkpoint = pipeline.train_stage(cfg, 'finetune', disable_progress=True)
    metadata = torch.load(checkpoint, weights_only=False)['metadata']
    assert metadata['optimizer_step'] == 4
    assert metadata['mvpruner']['beta_step'] == 4
    assert metadata['mvpruner']['beta_target_steps'] == 6


def test_new_training_debug_identifies_scorer_and_ce(tmp_path):
    model, tokenizer = build_vlm_model(_config(tmp_path))
    model(**_batch(tokenizer))
    report = pipeline._mvpruner_train_payload(model)['mvpruner']
    assert report['stage1_enabled'] is False
    assert report['stage2_method'] == 't5_attention_direct_camera_v1'
    assert report['loss_type'] == 'ce'


@pytest.mark.parametrize('entrypoint', ['train_stage', 'diagnose_train', 'verify_resume_checkpoint'])
def test_same_profile_align_checkpoint_cannot_initialize_finetune(tmp_path, monkeypatch, entrypoint):
    from efficient_vlm_ad.checkpointing import architecture_signature
    cfg = _config(tmp_path)
    path = tmp_path / 'align.pt'
    metadata = {'stage': 'align', 'architecture_signature': architecture_signature(cfg),
                **accumulation_metadata(cfg)}
    torch.save({'metadata': metadata, 'model': {}, 'optimizer': {}}, path)
    _fallback(monkeypatch)
    monkeypatch.setattr(torch.nn.Module, 'load_state_dict',
                        lambda *a, **kw: pytest.fail('align checkpoint reached weight loading'))
    with pytest.raises(ValueError, match='finetune'):
        if entrypoint == 'verify_resume_checkpoint':
            pipeline.verify_resume_checkpoint(cfg, str(path), 'finetune', disable_progress=True)
        else:
            getattr(pipeline, entrypoint)(cfg, 'finetune', resume=str(path), disable_progress=True)
