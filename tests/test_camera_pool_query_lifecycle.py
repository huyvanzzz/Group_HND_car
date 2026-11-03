"""Real tiny-T5 integration for camera-ranked spatial pooled queries."""
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from config_helpers import fake_config
from efficient_vlm_ad import pipeline
from efficient_vlm_ad.modeling.factory import build_vlm_model
from efficient_vlm_ad.query_training import is_camera_query, is_question_aware_camera_query, training_recipe


def _config(path, epochs=1):
    base = fake_config()
    return replace(base, project=replace(base.project, output_dir=str(path)),
        cache=replace(base.cache, dir=str(path / 'cache')),
        model=replace(base.model, vision=replace(base.model.vision, output_dim=8, seq_len=64),
            projector=replace(base.model.projector, type='linear'),
            fusion=replace(base.model.fusion, name='mvpruner_camera_pool_qaware',
                           stage1_top_k=384, stage2_top_k=46)),
        training=replace(base.training, batch_size=1, gradient_accumulation_steps=2,
            max_steps=None, finetune_epochs=epochs, entropy_weight=0., seed=42, num_workers=0),
        runtime=replace(base.runtime, device='cpu', precision='fp32'),
        generation=replace(base.generation, max_new_tokens=2, num_beams=1))


def _batch(tokenizer, seed=42):
    generator = torch.Generator().manual_seed(seed)
    text = tokenizer(['what is ahead', 'traffic'], padding=True, return_tensors='pt')
    labels = tokenizer(['road', 'car'], padding=True, return_tensors='pt')['input_ids']
    return {**text, 'labels': labels,
            'visual_features': torch.randn(2, 6, 64, 8, generator=generator)}


def _dataset(monkeypatch):
    class Samples(torch.utils.data.Dataset):
        def __len__(self): return 3
        def summary(self): return {'cached_records': 3}
        def __getitem__(self, index):
            return {'question': f'question {index}', 'answer': 'road', 'eval_id': str(index),
                    'visual_features': torch.randn(6, 64, 8, generator=torch.Generator().manual_seed(index))}
    monkeypatch.setattr(pipeline, '_make_dataset_and_collator',
        lambda cfg, split, tokenizer, **kw: (Samples(), pipeline.CachedBatchCollator(tokenizer)))
    monkeypatch.setattr(pipeline, 'validate_stage_loss', lambda *args, **kwargs: 1.)


def test_pool_recipe_has_no_query_bank_or_beta_dependencies(tmp_path):
    cfg = _config(tmp_path)
    assert is_camera_query(cfg)
    assert not is_question_aware_camera_query(cfg)
    recipe = training_recipe(cfg)
    assert recipe['entropy_weight'] == 0.
    assert 'spatial_pool' in recipe['training_loop_version']
    assert 'beta' not in str(recipe)


def test_benchmark_metadata_identifies_spatial_pooling_not_old_query_bank(tmp_path):
    from efficient_vlm_ad.prefill_benchmarking import prefill_measurement_metadata
    cfg = _config(tmp_path)
    model, _ = build_vlm_model(cfg)
    report = prefill_measurement_metadata(model, cfg.generation, warmup_samples=3)
    reduction = report['token_reduction']
    assert reduction['name'] == 'mvpruner_camera_pool_qaware'
    assert reduction['stage1_enabled'] is False
    assert reduction['input_visual_tokens'] == 384
    assert reduction['visual_tokens'] == 46
    assert reduction['rank_quotas'] == [16, 9, 9, 4, 4, 4]
    assert 'spatial pooling' in report['scope']
    assert 'Stage 1' not in report['scope'] and 'DP' not in report['scope']


def test_real_t5_lengths_masks_gradients_and_ce_only(tmp_path, monkeypatch):
    from efficient_vlm_ad.modeling.mvpruner import MVPrunerTwoStage
    from efficient_vlm_ad.modeling.t5_split_encoder import T5SplitEncoder
    monkeypatch.setattr(MVPrunerTwoStage, 'stage1', lambda *a, **kw: pytest.fail('Stage 1 called'))
    monkeypatch.setattr(T5SplitEncoder, '_pre_dropout_attention',
                        lambda *a, **kw: pytest.fail('native attention scorer called'))
    model, tokenizer = build_vlm_model(_config(tmp_path))
    batch = _batch(tokenizer)
    lengths = []
    handles = [block.register_forward_pre_hook(lambda module, args: lengths.append(args[0].shape[1]))
               for block in model.text_model.encoder.block]
    output = model(**batch)
    for handle in handles: handle.remove()
    prompt_len = batch['input_ids'].shape[1]
    assert lengths == [384 + prompt_len] * 2 + [46 + prompt_len] * 2
    assert output.loss is output.answer_loss
    assert torch.isfinite(output.entropy_loss)
    output.loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    fusion = model.visual_fusion
    assert not any('query_bank' in name for name, _ in fusion.named_parameters())
    assert fusion.question_scorer.fc2.weight.grad.abs().sum() > 0
    debug = model.forward_debug(**batch)
    assert debug['mvpruner']['stage1_enabled'] is False
    assert debug['weighted_entropy_loss'] == 0.
    assert 'stage1_indices' not in debug['mvpruner']
    assert 'beta' not in debug['mvpruner']
    model.eval()
    generated = model.generate_from_features(**{k: v for k, v in batch.items() if k != 'labels'},
                                             max_new_tokens=2, num_beams=1)
    assert generated.shape[0] == 2


@pytest.mark.parametrize('fallback', [True, False])
def test_train_diagnose_resume_evaluate_benchmark(tmp_path, monkeypatch, fallback):
    cfg = _config(tmp_path)
    _dataset(monkeypatch)
    if fallback:
        monkeypatch.setattr(pipeline, '_build_accelerator', lambda cfg:
            pipeline._SingleProcessAccelerator(torch.device('cpu'), gradient_accumulation_steps=2))
    probe = pipeline.diagnose_train(cfg, 'finetune', debug_numerics=True, disable_progress=True)
    assert probe['mvpruner_gradients']['ok']
    assert probe['optimizer_step_performed'] is False
    checkpoint = Path(pipeline.train_stage(cfg, 'finetune', disable_progress=True))
    saved = torch.load(checkpoint, weights_only=False)
    assert saved['metadata']['optimizer_step'] == 2
    assert saved['metadata']['entropy_weight'] == 0.
    assert saved['metadata']['epoch_boundary']
    assert 'beta_step' not in saved['metadata'] and 'mvpruner' not in saved['metadata']
    assert all(value.item() == 0 for name, value in saved['model'].items() if name.endswith('beta_step'))
    pipeline.verify_resume_checkpoint(cfg, str(checkpoint), stage='finetune', disable_progress=True)
    resumed = pipeline.train_stage(replace(cfg, training=replace(cfg.training, finetune_epochs=2)),
        'finetune', resume=str(checkpoint), disable_progress=True)
    assert torch.load(resumed, weights_only=False)['metadata']['optimizer_step'] == 4
    monkeypatch.setattr(pipeline, 'caption_metrics', lambda rows: {'BLEU_4': 0.})
    pipeline.evaluate_checkpoint(cfg, str(resumed), max_samples=2, disable_progress=True)
    pipeline.benchmark_checkpoint(cfg, str(resumed), disable_progress=True)


def test_align_rejected_before_build(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, '_build_train_model', lambda cfg: pytest.fail('align built model'))
    with pytest.raises(ValueError, match='finetune'):
        pipeline.train_stage(_config(tmp_path), 'align', disable_progress=True)


def test_prompt_preservation_reset_bias_and_padding_invariance(tmp_path, monkeypatch):
    model, tokenizer = build_vlm_model(_config(tmp_path))
    model.eval()
    batch = _batch(tokenizer)
    split = model.t5_split_encoder
    original = split._run_block
    captured = {}

    def observe(block, hidden, mask, bias, cache):
        if block is model.text_model.encoder.block[2]:
            captured.update(hidden=hidden.detach().clone(), mask=mask.detach().clone(),
                            bias=bias.detach().clone(), cache=cache.detach().clone())
        result = original(block, hidden, mask, bias, cache)
        if block is model.text_model.encoder.block[1]:
            captured['before'] = result[0].detach().clone()
        return result

    monkeypatch.setattr(split, '_run_block', observe)
    with torch.no_grad():
        output = model(**batch)
    torch.testing.assert_close(captured['hidden'][:, 46:], captured['before'][:, 384:])
    expected_mask = torch.cat([torch.ones(2, 46, dtype=batch['attention_mask'].dtype),
                               batch['attention_mask']], dim=1)
    expected_extended = model.text_model.encoder.get_extended_attention_mask(expected_mask, expected_mask.shape)
    torch.testing.assert_close(captured['mask'], expected_extended)
    length = expected_mask.shape[1]
    expected_bias = model.text_model.encoder.block[0].layer[0].SelfAttention.compute_bias(
        length, length, device=torch.device('cpu'), cache_position=torch.arange(length))
    torch.testing.assert_close(captured['bias'], expected_bias + expected_extended)
    changed = dict(batch, input_ids=batch['input_ids'].clone())
    changed['input_ids'][~batch['attention_mask'].bool()] = batch['input_ids'][0, 0]
    with torch.no_grad():
        perturbed = model(**changed)
    torch.testing.assert_close(output.logits, perturbed.logits, rtol=1e-5, atol=1e-6)
    debug = model.forward_debug(**{k: v for k, v in batch.items() if k != 'labels'})
    assert debug['answer_loss'] is None


@pytest.mark.parametrize('fallback', [True, False])
def test_dropout_epoch_resume_matches_continuous_without_beta(tmp_path, monkeypatch, fallback):
    from efficient_vlm_ad.modeling.mvpruner import MVPrunerTwoStage
    import numpy as np
    cfg = _config(tmp_path / 'full', epochs=2)
    _dataset(monkeypatch)
    sampler_restores = []
    restore_sampler = pipeline._restore_sampler_state

    def record_sampler_restore(loader, state):
        sampler_restores.append(state)
        return restore_sampler(loader, state)

    monkeypatch.setattr(pipeline, '_restore_sampler_state', record_sampler_restore)
    for name in ('configure_beta', 'step_beta', 'current_beta'):
        monkeypatch.setattr(MVPrunerTwoStage, name,
                            lambda *a, **kw: pytest.fail('inactive beta invoked'))
    build = pipeline._build_train_model

    def with_dropout(cfg):
        result = build(cfg)
        result[1](['question 0', 'question 1', 'question 2', 'road'], padding=True, return_tensors='pt')
        for module in result[0].modules():
            if isinstance(module, torch.nn.Dropout): module.p = .2
        return result

    monkeypatch.setattr(pipeline, '_build_train_model', with_dropout)
    if fallback:
        monkeypatch.setattr(pipeline, '_build_accelerator', lambda cfg:
            pipeline._SingleProcessAccelerator(torch.device('cpu'), gradient_accumulation_steps=2))
    full = Path(pipeline.train_stage(cfg, 'finetune', disable_progress=True))
    first_cfg = replace(cfg, project=replace(cfg.project, output_dir=str(tmp_path / 'split')),
                        training=replace(cfg.training, finetune_epochs=1))
    first = Path(pipeline.train_stage(first_cfg, 'finetune', disable_progress=True))
    pipeline.verify_resume_checkpoint(first_cfg, str(first), stage='finetune', disable_progress=True)
    resumed = Path(pipeline.train_stage(replace(first_cfg, training=replace(first_cfg.training, finetune_epochs=2)),
                                       'finetune', resume=str(first), disable_progress=True))
    assert len(sampler_restores) == 1, 'saved sampler state was not passed to prepared loader'
    a = torch.load(full, weights_only=False)
    b = torch.load(resumed, weights_only=False)

    def equal(left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left: equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            assert len(left) == len(right)
            for lvalue, rvalue in zip(left, right): equal(lvalue, rvalue)
        else:
            assert left == right

    for key in ('model', 'optimizer', 'scheduler'): equal(a[key], b[key])
    assert a['metadata']['optimizer_step'] == b['metadata']['optimizer_step'] == 4
    ra = torch.load(Path(str(full) + '.rng') / 'rank_00000.pt', weights_only=False)['rng_state']
    rb = torch.load(Path(str(resumed) + '.rng') / 'rank_00000.pt', weights_only=False)['rng_state']
    assert ra['python'] == rb['python']
    assert np.array_equal(ra['numpy'][1], rb['numpy'][1])
    assert ra['numpy'][2:] == rb['numpy'][2:]
    equal(ra['torch'], rb['torch'])
