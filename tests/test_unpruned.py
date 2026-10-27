"""Full-token baseline must execute native T5 without pruning work."""
from dataclasses import replace

import pytest
import torch

from config_helpers import fake_config
from efficient_vlm_ad.config import load_config
from efficient_vlm_ad.modeling.factory import build_vlm_model
from efficient_vlm_ad.modeling.mvpruner import MVPrunerTwoStage
from efficient_vlm_ad.tokenization import decode_generated_predictions
from efficient_vlm_ad.checkpointing import architecture_signature


def baseline_config():
    cfg = fake_config()
    return replace(cfg, model=replace(cfg.model,
        fusion=replace(cfg.model.fusion, name="unpruned"),
        vision=replace(cfg.model.vision, output_dim=12, seq_len=4),
        text=replace(cfg.model.text, d_model=8)))


def test_unpruned_keeps_every_token_in_every_encoder_block(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("baseline must not execute MVPruner")
    monkeypatch.setattr(MVPrunerTwoStage, "stage1", forbidden)
    monkeypatch.setattr(MVPrunerTwoStage, "stage2_from_scores", forbidden)
    cfg = baseline_config()
    model, tokenizer = build_vlm_model(cfg)
    assert not any(isinstance(m, MVPrunerTwoStage) for m in model.modules())
    assert getattr(model, "t5_split_encoder", None) is None
    assert model.visual_token_len == 24
    inputs = dict(input_ids=torch.tensor([[2,3,0],[2,4,5]]),
                  attention_mask=torch.tensor([[1,1,0],[1,1,1]]),
                  visual_features=torch.randn(2,6,4,12), labels=torch.tensor([[3,1],[4,1]]))
    seen = []
    handles = [block.register_forward_pre_hook(lambda _, args: seen.append(args[0].shape[1]))
               for block in model.text_model.encoder.block]
    out = model(**inputs)
    for handle in handles:
        handle.remove()
    assert seen == [27]*4
    out.loss.backward()
    assert model.projector.proj.weight.grad.abs().sum() > 0
    assert model.visual_fusion.camera_embeddings.weight.grad.abs().sum() > 0
    model.eval()
    report = model.forward_debug(**inputs)
    assert report["inputs_embeds"]["shape"] == [2,27,8]
    assert report["combined_attention_mask"]["shape"] == [2,27]
    ids = model.generate_from_features(**{k:v for k,v in inputs.items() if k!='labels'},
                                       max_new_tokens=2, num_beams=2)
    assert ids.shape[0] == 2
    assert len(decode_generated_predictions(tokenizer, ids)) == 2


def test_baseline_checkpoint_cannot_be_used_as_mvpruner(tmp_path):
    from efficient_vlm_ad.checkpointing import save_checkpoint, load_checkpoint
    cfg = baseline_config()
    model, _ = build_vlm_model(cfg)
    path = tmp_path / "baseline.pt"
    save_checkpoint(path, model=model, optimizer=torch.optim.AdamW(model.parameters()),
                    scheduler=None, scaler=None, metadata={"stage":"finetune"})
    mv_cfg = replace(cfg, model=replace(cfg.model,
        fusion=replace(cfg.model.fusion, name="mvpruner_two_stage",stage1_top_k=12,stage2_top_k=6)))
    mv, _ = build_vlm_model(mv_cfg)
    with pytest.raises(ValueError, match="signature"):
        load_checkpoint(path, model=mv, cfg=mv_cfg)


def test_baseline_profile_keeps_main_experiment_settings():
    a = load_config("configs/dinov2_t5_efficient_mini_kaggle_2gpu_mvpruner.yaml")
    b = load_config("configs/dinov2_t5_efficient_mini_kaggle_2gpu_unpruned.yaml")
    assert a.model.vision == b.model.vision
    assert a.model.text == b.model.text
    assert a.model.projector == b.model.projector
    assert a.generation == b.generation and a.evaluation == b.evaluation
    assert a.training.effective_batch_size_for_processes(2) == b.training.effective_batch_size_for_processes(2)
    assert b.project.output_dir != a.project.output_dir
    assert b.cache.dir != a.cache.dir
    model, _ = build_vlm_model(baseline_config())
    assert model.visual_token_len == 24


def test_checkpointing_defaults_and_baseline_signature():
    cfg = baseline_config()
    assert cfg.training.gradient_checkpointing is False
    enabled = replace(cfg, training=replace(cfg.training, gradient_checkpointing=True))
    assert architecture_signature(cfg) == architecture_signature(enabled)
    assert {k:v for k,v in architecture_signature(cfg).items() if k.startswith('fusion.')} == {
        'fusion.name': 'unpruned'}
    model, _ = build_vlm_model(enabled)
    assert model.text_model.encoder.gradient_checkpointing
    assert model.text_model.decoder.gradient_checkpointing
    assert model.text_model.config.use_cache is True


@pytest.mark.parametrize('value', ['false', 0, None])
def test_checkpointing_requires_boolean(tmp_path, value):
    from dataclasses import asdict
    import yaml
    raw = asdict(baseline_config())
    raw['training']['gradient_checkpointing'] = value
    path = tmp_path / 'bad.yaml'
    path.write_text(yaml.safe_dump(raw), encoding='utf-8')
    with pytest.raises(ValueError, match='gradient_checkpointing'):
        load_config(path)


def test_mvpruner_checkpointing_is_not_silently_enabled():
    cfg = fake_config()
    cfg = replace(cfg, training=replace(cfg.training, gradient_checkpointing=True))
    with pytest.raises(ValueError, match='unpruned'):
        build_vlm_model(cfg)


def inputs():
    return dict(input_ids=torch.tensor([[2,3,0],[2,4,5]]),
                attention_mask=torch.tensor([[1,1,0],[1,1,1]]),
                visual_features=torch.randn(2,6,4,12), labels=torch.tensor([[3,1],[4,1]]))


def test_checkpointing_preserves_gradients_and_generation_cache():
    cfg = baseline_config()
    plain, _ = build_vlm_model(cfg)
    checked, _ = build_vlm_model(replace(cfg, training=replace(cfg.training, gradient_checkpointing=True)))
    checked.load_state_dict(plain.state_dict())
    batch = inputs()
    a = plain(**batch)
    observed = []
    handle = checked.text_model.register_forward_pre_hook(
        lambda m,args,kwargs: observed.append(kwargs.get('use_cache')), with_kwargs=True)
    b = checked(**batch)
    assert observed == [False]
    torch.testing.assert_close(a.logits,b.logits,rtol=1e-5,atol=1e-6)
    a.loss.backward(); b.loss.backward()
    for (n,p),(n2,p2) in zip(plain.named_parameters(),checked.named_parameters()):
        assert n == n2 and p.grad is not None and p2.grad is not None
        assert torch.isfinite(p2.grad).all()
        torch.testing.assert_close(p.grad,p2.grad,rtol=1e-5,atol=1e-6)
    checked.eval()
    checked.generate_from_features(**{k:v for k,v in batch.items() if k!='labels'},max_new_tokens=2)
    handle.remove()
    assert all(v is True for v in observed[1:])


def test_native_baseline_matches_mvpruner_identity():
    cfg = baseline_config()
    native, _ = build_vlm_model(cfg)
    mv_cfg = replace(cfg, model=replace(cfg.model,
        fusion=replace(cfg.model.fusion,name='mvpruner_two_stage',stage1_top_k=24,stage2_top_k=24)))
    identity, _ = build_vlm_model(mv_cfg)
    state = identity.state_dict()
    state.update(native.state_dict())
    identity.load_state_dict(state)
    batch = inputs()
    native.eval(); identity.eval()
    with torch.no_grad():
        a = native(**batch); b = identity(**batch)
    torch.testing.assert_close(a.logits,b.logits,rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(a.loss,b.loss,rtol=1e-5,atol=1e-6)


def test_baseline_full_384_visual_tokens():
    cfg = baseline_config()
    cfg = replace(cfg, model=replace(cfg.model,vision=replace(cfg.model.vision,seq_len=64)))
    model, _ = build_vlm_model(cfg)
    batch = inputs(); batch['visual_features'] = torch.randn(2,6,64,12)
    seen = []
    def capture(module,args,kwargs):
        mask = kwargs['attention_mask']
        assert mask[:,:384].eq(1).all()
        torch.testing.assert_close(mask[:,384:],batch['attention_mask'])
    handle = model.text_model.register_forward_pre_hook(capture,with_kwargs=True)
    handles = [block.register_forward_pre_hook(lambda m,a: seen.append(a[0].shape[1]))
               for block in model.text_model.encoder.block]
    model(**batch).loss.backward()
    handle.remove()
    for h in handles: h.remove()
    assert seen == [387]*4


def test_baseline_finetune_lifecycle_and_checkpointing_resume(tmp_path, monkeypatch):
    import json
    from test_verify_align_checkpoint import _write_fake_config
    from efficient_vlm_ad import pipeline
    from efficient_vlm_ad.hf_data import prepare_data
    cfg = load_config(_write_fake_config(tmp_path))
    cfg = replace(cfg, model=replace(cfg.model,fusion=replace(cfg.model.fusion,name='unpruned')),
                  training=replace(cfg.training,max_steps=1),
                  generation=replace(cfg.generation,max_new_tokens=2))
    monkeypatch.setattr(pipeline,'_build_accelerator',
        lambda _: pipeline._SingleProcessAccelerator(torch.device('cpu')))
    with pytest.raises(ValueError,match='finetune'):
        pipeline.train_stage(cfg,'align',disable_progress=True)
    prepare_data(cfg,subset='smoke',disable_progress=True)
    pipeline.prepare_features(cfg,subset='smoke',disable_progress=True)
    ckpt = pipeline.train_stage(cfg,'finetune',disable_progress=True)
    old = torch.load(ckpt,weights_only=False)
    assert old['metadata']['unpruned'] == {'visual_tokens':24,'gradient_checkpointing':False}
    assert not any('beta_' in key for key in old['model'])
    enabled = replace(cfg,training=replace(cfg.training,gradient_checkpointing=True,
                                         max_steps=None,finetune_epochs=2))
    pipeline.diagnose_train(enabled,'finetune',resume=str(ckpt),disable_progress=True)
    pipeline.verify_resume_checkpoint(enabled,str(ckpt),'finetune',disable_progress=True)
    resumed = pipeline.train_stage(enabled,'finetune',resume=str(ckpt),disable_progress=True)
    new = torch.load(resumed,weights_only=False)
    assert old['metadata']['architecture_signature'] == new['metadata']['architecture_signature']
    assert new['metadata']['completed_epochs'] == 2
    assert new['metadata']['unpruned']['gradient_checkpointing'] is True
    assert (resumed.parent/(resumed.name+'.rng')/'rank_00000.pt').exists()
    pipeline.evaluate_checkpoint(enabled,str(resumed),max_samples=1,disable_progress=True)
    report = pipeline.benchmark_checkpoint(enabled,str(resumed),max_samples=1,disable_progress=True)
    assert report['unpruned']['visual_tokens'] == 24
    assert report['unpruned']['training_gradient_checkpointing'] is True
    assert json.loads((tmp_path/'outputs'/'metrics.json').read_text())

