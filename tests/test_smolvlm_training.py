"""Native training window, validation units and checkpoint identity contracts."""
from dataclasses import replace
import json
import pytest
import torch

from smolvlm_helpers import create_local_run
from efficient_vlm_ad.smolvlm.config import load_config
from efficient_vlm_ad.smolvlm import pipeline, checkpoint
from efficient_vlm_ad.smolvlm.model import build_model_and_processor
from efficient_vlm_ad.smolvlm.cache import prepare_features
from efficient_vlm_ad.hf_data import prepare_data


def prepared(tmp_path, rows=3, dropout=0.0):
    cfg = load_config(create_local_run(tmp_path, rows=rows,dropout=dropout))
    prepare_data(cfg, disable_progress=True)
    model, processor = build_model_and_processor(cfg)
    prepare_features(cfg, model, processor, disable_progress=True)
    return cfg


@pytest.mark.parametrize('rows,A,limit,expected_sizes', [
    (3,1,None,[1,1,1]), (3,2,None,[2,1]), (1,4,None,[1]),
    (3,2,1,[1]), (1,2,3,[2,1])])
def test_real_training_full_tail_short_and_debug_repeat(tmp_path, rows,A,limit,expected_sizes):
    cfg = prepared(tmp_path, rows)
    cfg = replace(cfg, training=replace(cfg.training, gradient_accumulation_steps=A,
                                       max_steps=limit))
    state = pipeline.train(cfg, debug=True, debug_samples=100)
    assert state['optimizer_step'] == len(expected_sizes)
    assert state['global_step'] == (limit or rows)
    events = [json.loads(x) for x in (tmp_path/'output/debug/debug_events.jsonl').read_text().splitlines()]
    steps = [x['payload'] for x in events if x['section']=='TRAIN_STEP']
    assert [x['window_size'] for x in steps] == expected_sizes
    payload = torch.load(tmp_path/'output/checkpoints/finetune_latest.pt', weights_only=False)
    expected_completed = 0 if limit and limit < rows else 1
    assert payload['metadata']['completed_epochs'] == expected_completed
    assert payload['scheduler_state_dict']['last_epoch'] == expected_completed


def test_token_weighted_validation_is_batch_partition_invariant(tmp_path):
    cfg = prepared(tmp_path)
    model,processor = build_model_and_processor(cfg)
    losses=[]
    for size in (1,2,3):
        variant = replace(cfg,evaluation=replace(cfg.evaluation,eval_batch_size=size))
        losses.append(pipeline.validation(model,pipeline.loader(variant,model,processor,'val'),torch.device('cpu')))
    torch.testing.assert_close(torch.tensor(losses),torch.full((3,),losses[0]),rtol=1e-6,atol=1e-6)


def test_checkpoint_identity_recipe_and_rng_strict_and_inference_independent(tmp_path):
    cfg = prepared(tmp_path)
    pipeline.train(cfg)
    path = tmp_path/'output/checkpoints/finetune_latest.pt'
    model,_ = build_model_and_processor(cfg)
    original=torch.load(path,weights_only=False)
    changed=tmp_path/'bad.pt'
    for signature in (None, {'backend':'t5'}):
        payload={**original,'architecture_signature':signature}
        torch.save(payload,changed)
        with pytest.raises(ValueError,match='identity'):
            checkpoint.read(changed,model)
    with pytest.raises(ValueError,match='world size'):
        checkpoint.read(path,model,cfg,training=True,world_size=2)
    incompatible=replace(cfg,training=replace(cfg.training,batch_size=2))
    with pytest.raises(ValueError,match='recipe'):
        checkpoint.read(path,model,incompatible,training=True)
    sidecar=path.parent/(path.name+'.rng')/'rank_00000.pt'
    original_side=torch.load(sidecar,weights_only=False)
    torch.save({**original_side,'rng_state':{}},sidecar)
    with pytest.raises(ValueError,match='RNG'):
        checkpoint.read(path,model,cfg,training=True)
    torch.save({**original_side,'sampler_state':{'seed':999,'epoch':1,'batch_cursor':0}},sidecar)
    with pytest.raises(ValueError,match='sampler'):
        checkpoint.read(path,model,cfg,training=True)
    torch.save({**original,'metadata':{**original['metadata'],'cuda_rng_state_sizes':[200]}},path)
    for invalid in ([],[torch.zeros(1,dtype=torch.uint8)]):
        torch.save({**original_side,'rng_state':{**original_side['rng_state'],'cuda':invalid}},sidecar)
        with pytest.raises(ValueError,match='RNG'):
            checkpoint.read(path,model,cfg,training=True)
    torch.save(original,path)
    sidecar.unlink()
    assert checkpoint.read(path,model)['metadata']['optimizer_step']==2
    with pytest.raises(FileNotFoundError):
        checkpoint.read(path,model,cfg,training=True)
    with pytest.raises(ValueError,match='max_samples'):
        pipeline.inference(cfg,path,max_samples=0)


def test_mid_epoch_resume_matches_continuous_at_update_boundary(tmp_path):
    cfg=prepared(tmp_path,rows=3,dropout=.1)
    limited=replace(cfg,training=replace(cfg.training,max_steps=2))
    first=pipeline.train(limited)
    assert first['epoch']==0 and first['batch_cursor']==2
    path=tmp_path/'output/checkpoints/finetune_latest.pt'
    # Runtime checkpointing may change on resume without an architecture migration.
    resumed_cfg=replace(cfg,training=replace(cfg.training,gradient_checkpointing=False))
    pipeline.train(resumed_cfg,resume=path)
    resumed=torch.load(path,weights_only=False)
    other=prepared(tmp_path/'continuous',rows=3,dropout=.1)
    pipeline.train(other)
    continuous=torch.load(tmp_path/'continuous/output/checkpoints/finetune_latest.pt',weights_only=False)
    for name,value in resumed['model_state_dict'].items():
        torch.testing.assert_close(value,continuous['model_state_dict'][name],rtol=0,atol=0)


def test_skipped_optimizer_does_not_increment_counter_or_duplicate_tail(tmp_path,monkeypatch):
    from accelerate.optimizer import AcceleratedOptimizer
    cfg=prepared(tmp_path,rows=3)
    original=AcceleratedOptimizer.step
    calls=[]
    def skipped_once(optimizer,*args,**kwargs):
        calls.append(1)
        optimizer._is_overflow = len(calls)==1
        if optimizer._is_overflow:
            return None
        return original(optimizer,*args,**kwargs)
    monkeypatch.setattr(AcceleratedOptimizer,'step',skipped_once)
    state=pipeline.train(cfg,debug=True,debug_samples=100)
    assert len(calls)==2 and state['optimizer_step']==1 and state['global_step']==3
    payload=torch.load(tmp_path/'output/checkpoints/finetune_latest.pt',weights_only=False)
    assert payload['scheduler_state_dict']['last_epoch']==1
    events=[json.loads(x) for x in (tmp_path/'output/debug/debug_events.jsonl').read_text().splitlines()]
    steps=[x['payload'] for x in events if x['section']=='TRAIN_STEP']
    assert [x['optimizer_skipped'] for x in steps]==[True,False]
    assert [x['optimizer_step'] for x in steps]==[0,1]


def test_normal_step_limit_crosses_epoch_instead_of_replaying_sampler(tmp_path):
    cfg=prepared(tmp_path,rows=3)
    cfg=replace(cfg,training=replace(cfg.training,max_steps=4,finetune_epochs=2))
    state=pipeline.train(cfg)
    assert state['global_step']==4 and state['optimizer_step']==3
    assert state['epoch']==1 and state['batch_cursor']==1
    payload=torch.load(tmp_path/'output/checkpoints/finetune_latest.pt',weights_only=False)
    assert payload['scheduler_state_dict']['last_epoch']==1
    cfg=replace(cfg,training=replace(cfg.training,max_steps=6))
    resumed=pipeline.train(cfg,resume=tmp_path/'output/checkpoints/finetune_latest.pt')
    assert resumed['global_step']==6 and resumed['completed_epochs']==2
