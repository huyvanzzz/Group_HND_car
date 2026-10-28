import pytest
import torch
from efficient_vlm_ad.query_training import window_size, window_boundary, training_recipe, validate_resume_metadata

@pytest.mark.parametrize('batches,accum,expected', [(5,2,[2,2,2,2,1]),(3,4,[3,3,3])])
def test_actual_window(batches, accum, expected):
    assert [window_size(i,batches,accum) for i in range(batches)] == expected

@pytest.mark.parametrize('accelerated', [False, True])
def test_accumulated_gradients_and_epoch_scheduler(accelerated):
    from efficient_vlm_ad.pipeline import _SingleProcessAccelerator
    if accelerated:
        from accelerate import Accelerator
        accelerator = Accelerator(gradient_accumulation_steps=1, step_scheduler_with_optimizer=False)
    else:
        accelerator = _SingleProcessAccelerator(torch.device('cpu'))
    model = torch.nn.Linear(1,1,bias=False)
    model.weight.data.fill_(1.)
    reference = torch.nn.Linear(1,1,bias=False)
    reference.load_state_dict(model.state_dict())
    opt = torch.optim.SGD(model.parameters(),lr=.1)
    refopt = torch.optim.SGD(reference.parameters(),lr=.1)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt,gamma=.9)
    model,opt,scheduler = accelerator.prepare(model,opt,scheduler)
    for epoch in range(2):
        for i in range(5):
            accelerator.backward(model(torch.tensor([[float(i+1)]])).square().mean()/window_size(i,5,2))
            if window_boundary(i,5,2):
                opt.step();opt.zero_grad()
        for start in range(0,5,2):
            x=torch.arange(start+1,min(start+3,6),dtype=torch.float32).reshape(-1,1)
            reference(x).square().mean().backward();refopt.step();refopt.zero_grad()
        scheduler.step()
        for group in refopt.param_groups: group['lr'] *= .9
        assert torch.allclose(accelerator.unwrap_model(model).weight,reference.weight,atol=1e-5)
    assert opt.param_groups[0]['lr'] == pytest.approx(.081)

def test_resume_rejects_partial_and_recipe_mismatch():
    from types import SimpleNamespace as NS
    cfg=NS(training=NS(seed=42,entropy_weight=.5),model=NS(fusion=NS(query=NS(dim=64,init_std=1.,rms_eps=1e-6,position_policy='new_sequence'))))
    metadata={'stage':'finetune','epoch_boundary':True,**training_recipe(cfg)}
    validate_resume_metadata(cfg,metadata)
    with pytest.raises(ValueError,match='epoch'):
        validate_resume_metadata(cfg,{**metadata,'epoch_boundary':False})
    with pytest.raises(ValueError,match='seed'):
        validate_resume_metadata(cfg,{**metadata,'seed':12})

@pytest.mark.parametrize('fallback',[False,True])
def test_dropout_epoch_resume_matches_continuous(tmp_path, monkeypatch, fallback):
    from dataclasses import replace
    from config_helpers import fake_config
    from efficient_vlm_ad import pipeline
    from efficient_vlm_ad.checkpointing import read_checkpoint_metadata
    base=fake_config()
    cfg=replace(base, model=replace(base.model,
        vision=replace(base.model.vision,output_dim=8,seq_len=4),
        fusion=replace(base.model.fusion,name='mvpruner_camera_query',stage1_top_k=12,stage2_top_k=6)),
        training=replace(base.training,max_steps=None,batch_size=1,finetune_epochs=2,entropy_weight=.5),
        runtime=replace(base.runtime,device='cpu',precision='fp32'))
    class Samples(torch.utils.data.Dataset):
        def __len__(self): return 3
        def summary(self): return {'cached_records':3}
        def __getitem__(self,index):
            return {'question':f'question {index}', 'answer':'road', 'visual_features':torch.full((6,4,8),float(index+1)), 'eval_id':str(index)}
    monkeypatch.setattr(pipeline,'_make_dataset_and_collator',lambda cfg,split,tokenizer,**kw:(Samples(),pipeline.CachedBatchCollator(tokenizer)))
    build=pipeline._build_train_model
    def with_dropout(cfg):
        result=build(cfg)
        result[1]([f"question {i}" for i in range(3)] + ["road"],padding=True,return_tensors="pt")
        for module in result[0].modules():
            if isinstance(module,torch.nn.Dropout): module.p=.2
        return result
    monkeypatch.setattr(pipeline,'_build_train_model',with_dropout)
    if fallback:
        monkeypatch.setattr(pipeline,'_build_accelerator',lambda cfg:pipeline._SingleProcessAccelerator(torch.device('cpu')))
    full=replace(cfg,project=replace(cfg.project,output_dir=str(tmp_path/'full')))
    fullpath=pipeline.train_stage(full,'finetune',disable_progress=True)
    first=replace(cfg,project=replace(cfg.project,output_dir=str(tmp_path/'split')),training=replace(cfg.training,finetune_epochs=1))
    firstpath=pipeline.train_stage(first,'finetune',disable_progress=True)
    assert read_checkpoint_metadata(firstpath)['optimizer_step']==2
    from pathlib import Path
    originals = {p:p.read_bytes() for p in [firstpath,*Path(str(firstpath)+'.rng').iterdir()]}
    pipeline.verify_resume_checkpoint(first,str(firstpath),stage='finetune',disable_progress=True)
    assert all(p.read_bytes()==content for p,content in originals.items())
    resumed=replace(first,training=replace(first.training,finetune_epochs=2))
    resumedpath=pipeline.train_stage(resumed,'finetune',resume=str(firstpath),disable_progress=True)
    a=torch.load(fullpath,weights_only=False);b=torch.load(resumedpath,weights_only=False)
    assert b['metadata']['optimizer_step']==4
    for name,value in a['model'].items():
        assert torch.equal(value,b['model'][name]),name
    assert a['scheduler']==b['scheduler']
    partial=replace(first,project=replace(first.project,output_dir=str(tmp_path/'partial')))
    partialpath=pipeline.train_stage(partial,'finetune',max_steps=1,disable_progress=True)
    with pytest.raises(ValueError,match='epoch'):
        pipeline.train_stage(partial,'finetune',resume=str(partialpath),disable_progress=True)


def test_sidecar_validation_requires_every_rank_and_matching_world(tmp_path):
    from efficient_vlm_ad.checkpointing import save_rank_rng_state, write_rank_rng_manifest
    from efficient_vlm_ad.query_training import validate_rank_sidecars
    checkpoint=tmp_path/'two_gpu.pt'
    metadata={'distributed':{'num_processes':2},'completed_epochs':1,'optimizer_step':2}
    for rank in range(2):
        save_rank_rng_state(checkpoint,rank=rank,world_size=2,epoch=1,batch_cursor=3,optimizer_step=2,sampler_state=None)
    write_rank_rng_manifest(checkpoint,world_size=2)
    assert len(validate_rank_sidecars(checkpoint,metadata))==2
    with pytest.raises(ValueError,match='world size'):
        validate_rank_sidecars(checkpoint,metadata,world_size=1)
    (tmp_path/'two_gpu.pt.rng'/'rank_00001.pt').unlink()
    with pytest.raises(FileNotFoundError):
        validate_rank_sidecars(checkpoint,metadata)

def test_camera_query_accelerator_has_one_scaling_owner():
    from dataclasses import replace
    from config_helpers import fake_config
    from efficient_vlm_ad.pipeline import _build_accelerator
    cfg=fake_config()
    cfg=replace(cfg,model=replace(cfg.model,fusion=replace(cfg.model.fusion,name='mvpruner_camera_query')),
                runtime=replace(cfg.runtime,precision='fp32'))
    accelerator=_build_accelerator(cfg)
    assert accelerator.gradient_accumulation_steps==1
    assert accelerator.step_scheduler_with_optimizer is False
    assert accelerator.dataloader_config.use_seedable_sampler is True
    assert accelerator.dataloader_config.data_seed==42
