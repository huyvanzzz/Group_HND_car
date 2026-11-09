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
    progress = [json.loads(line) for line in (tmp_path/'output/debug/train_progress.jsonl').read_text().splitlines()]
    steps = [row for row in progress if row['event']=='train_step']
    assert steps[0]['step_in_epoch']==1
    assert steps[-1]['step_in_epoch']==(limit or rows)
    assert steps[-1]['estimated_epoch_remaining_seconds']==0
    assert progress[0]['total_steps_epoch']==(limit or rows)
    assert all(row['finite_loss'] for row in steps)



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


def test_progress_native_schema_timing_intervals_and_resume(tmp_path, monkeypatch):
    cfg = prepared(tmp_path, rows=5)
    cfg = replace(cfg, training=replace(cfg.training, progress_log_every_steps=3))
    # Replace the module clock, not Python's shared time module used by Torch.
    from types import SimpleNamespace
    clock = iter(range(0, 1000, 2))
    monkeypatch.setattr(pipeline, 'time', SimpleNamespace(perf_counter=lambda: next(clock)))
    state = pipeline.train(cfg)
    progress_path = tmp_path/'output/debug/train_progress.jsonl'
    rows = [json.loads(line) for line in progress_path.read_text().splitlines()]
    assert all('timestamp' in row and 'event' in row and 'section' not in row for row in rows)
    assert [row['event'] for row in rows] == [
        'epoch_start', 'train_step', 'train_step', 'train_step',
        'validation_start', 'validation_end', 'checkpoint_saved', 'checkpoint_saved', 'epoch_end']
    steps = [row for row in rows if row['event']=='train_step']
    assert [row['step_in_epoch'] for row in steps] == [1, 3, 5]
    assert [row['optimizer_step'] for row in steps] == [0, 1, 3]
    for row, elapsed in zip(steps, (2, 4, 6)):
        assert row['elapsed_seconds'] == elapsed
        assert row['steps_per_second'] == pytest.approx(row['step_in_epoch']/elapsed, abs=1e-6)
        assert row['estimated_epoch_remaining_seconds'] == pytest.approx(
            (5-row['step_in_epoch'])*elapsed/row['step_in_epoch'], abs=.001)
        assert row['cuda_memory_allocated'] is None and row['cuda_memory_peak'] is None
    assert state['optimizer_step']==3
    assert rows[-1]['best_epoch']==1
    # A partial run resumes its cursor, but measures only newly executed work.
    other = prepared(tmp_path/'partial', rows=5)
    other = replace(other, training=replace(other.training, max_steps=2, progress_log_every_steps=3))
    pipeline.train(other)
    path = tmp_path/'partial/output/checkpoints/finetune_latest.pt'
    log = tmp_path/'partial/output/debug/train_progress.jsonl'
    old = log.read_text()
    pipeline.train(replace(other, training=replace(other.training, max_steps=None)), resume=path)
    assert log.read_text().startswith(old)
    resumed = [json.loads(line) for line in log.read_text()[len(old):].splitlines()]
    assert resumed[0]['step_in_epoch']==2
    steps = [row for row in resumed if row['event']=='train_step']
    assert [row['step_in_epoch'] for row in steps]==[3, 5]
    assert steps[0]['elapsed_seconds']==2
    assert steps[0]['steps_per_second']==.5
    assert steps[0]['estimated_epoch_remaining_seconds']==4


def test_debug_events_do_not_duplicate_progress(tmp_path):
    cfg = prepared(tmp_path, rows=1)
    pipeline.event(cfg, 'TRAIN_STEP', {'raw_loss': 1.0})
    assert not (tmp_path/'output/debug/train_progress.jsonl').exists()
    assert (tmp_path/'output/debug/debug_events.jsonl').is_file()


def test_progress_never_reports_failed_checkpoint_as_saved(tmp_path, monkeypatch):
    cfg = prepared(tmp_path, rows=1)
    def fail_save(*args, **kwargs):
        raise OSError('simulated save failure')
    monkeypatch.setattr(checkpoint, 'save', fail_save)
    with pytest.raises(OSError, match='simulated save failure'):
        pipeline.train(cfg)
    rows = [json.loads(line) for line in (tmp_path/'output/debug/train_progress.jsonl').read_text().splitlines()]
    assert rows[-1]['event']=='validation_end'
    assert not any(row['event']=='checkpoint_saved' for row in rows)


def test_recipe_records_rank_seed_and_rejects_old_training_checkpoint(tmp_path):
    cfg = prepared(tmp_path, rows=1)
    assert checkpoint.recipe(cfg)['rng_policy']=='post_prepare_seed_plus_rank_v1'
    pipeline.train(cfg, disable_progress=True)
    path=tmp_path/'output/checkpoints/finetune_latest.pt'
    payload=torch.load(path,weights_only=False)
    assert payload['metadata']['rng_policy']=='post_prepare_seed_plus_rank_v1'
    assert payload['metadata']['seed']==cfg.training.seed
    payload['metadata']['recipe'].pop('rng_policy')
    old=tmp_path/'old.pt'
    torch.save(payload,old)
    model,_=build_model_and_processor(cfg)
    with pytest.raises(ValueError,match='recipe'):
        checkpoint.read(old,model,cfg,training=True,world_size=1)
    assert checkpoint.read(old,model)['architecture_signature']==payload['architecture_signature']


def test_benchmark_timing_excludes_rows_and_keeps_cache(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from efficient_vlm_ad.smolvlm.model import CachedSmolVLM
    cfg=prepared(tmp_path,rows=1)
    pipeline.train(cfg,disable_progress=True)
    cache=tmp_path/'output/cache'
    before={p.name:p.read_bytes() for p in cache.iterdir()}
    clock=[0.0]
    original_inputs=pipeline.inputs
    original_generate=CachedSmolVLM.generate_from_features
    original_loader=pipeline.loader
    def timed_inputs(*args,**kwargs):
        result=original_inputs(*args,**kwargs)
        clock[0]+=1
        return result
    def timed_generate(*args,**kwargs):
        result=original_generate(*args,**kwargs)
        clock[0]+=2
        return result
    class Answers(list):
        def __getitem__(self,index):
            clock[0]+=100
            return super().__getitem__(index)
    def timed_loader(*args,**kwargs):
        result=original_loader(*args,**kwargs)
        collate=result.collate_fn
        def expensive_rows(items):
            batch=collate(items)
            batch['answers']=Answers(batch['answers'])
            return batch
        result.collate_fn=expensive_rows
        return result
    original_setup=pipeline.setup
    def timed_setup(*args,**kwargs):
        model,processor=original_setup(*args,**kwargs)
        decode=processor.tokenizer.batch_decode
        def timed_decode(*a,**kw):
            result=decode(*a,**kw)
            clock[0]+=3
            return result
        processor.tokenizer.batch_decode=timed_decode
        return model,processor
    monkeypatch.setattr(pipeline,'inputs',timed_inputs)
    monkeypatch.setattr(pipeline,'loader',timed_loader)
    monkeypatch.setattr(pipeline,'setup',timed_setup)
    monkeypatch.setattr(CachedSmolVLM,'generate_from_features',timed_generate)
    monkeypatch.setattr(pipeline,'time',SimpleNamespace(perf_counter=lambda:clock[0]))
    # Any attempt to re-extract features must fail. Cached model computation stays real.
    monkeypatch.setattr(CachedSmolVLM,'extract_features',lambda *a,**kw:pytest.fail('vision called from cached inference'))
    result=pipeline.inference(cfg,tmp_path/'output/checkpoints/finetune_latest.pt',benchmark=True,
        max_samples=1,disable_progress=True)
    assert result['e2e_mean_ms']==6000
    assert result['average_generation_ms_per_output_token']>0
    assert before=={p.name:p.read_bytes() for p in cache.iterdir()}


def test_cli_interval_validation_and_routing(tmp_path,monkeypatch):
    from efficient_vlm_ad.smolvlm.cli import main
    config=create_local_run(tmp_path)
    captured=[]
    monkeypatch.setattr(pipeline,'train',lambda cfg,**kw:captured.append((cfg,kw)))
    main(['train','--config',str(config),'--no-progress','--progress-log-every-steps','7'])
    assert captured[0][0].training.progress_log_every_steps==7
    assert captured[0][1]['disable_progress'] is True
    for flag in ('--progress-log-every-steps','--progress-log-every-samples'):
        with pytest.raises(SystemExit) as exc:
            main(['train','--config',str(config),flag,'0'])
        assert exc.value.code==2
