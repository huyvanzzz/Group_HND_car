from types import SimpleNamespace as NS
import pytest
import torch
from efficient_vlm_ad.query_training import is_camera_query, training_recipe

def config(name='mvpruner_camera_query_qaware', override=None):
    return NS(model=NS(fusion=NS(name=name,query=NS(init_std=1.),beta_start=.2,beta_end=0.,beta_schedule='linear',tau=1.)),training=NS(seed=42,entropy_weight=.5,finetune_epochs=10,gradient_accumulation_steps=2,beta_decay_updates=override))

def test_question_aware_family_and_distinct_recipe():
    from efficient_vlm_ad.query_training import is_question_aware_camera_query
    assert is_camera_query(config())
    assert is_question_aware_camera_query(config())
    assert not is_question_aware_camera_query(config('mvpruner_camera_query'))
    assert training_recipe(config())['training_loop_version'] != training_recipe(config('mvpruner_camera_query'))['training_loop_version']

def test_horizon_uses_prepared_loader_tail_and_preserves_resume():
    from efficient_vlm_ad.query_training import qaware_beta_horizon
    assert qaware_beta_horizon(config(),5)==30
    assert qaware_beta_horizon(config(override=17),5)==17
    assert qaware_beta_horizon(config(),2,stored_horizon=30)==30
    with pytest.raises(ValueError,match='horizon'):
        qaware_beta_horizon(config(override=17),5,stored_horizon=30)

def test_qaware_resume_rejects_counter_disagreement():
    from efficient_vlm_ad.query_training import validate_resume_metadata
    cfg=config()
    metadata={'stage':'finetune','epoch_boundary':True,**training_recipe(cfg),'optimizer_step':2,'beta_step':1,'beta_target_steps':30}
    with pytest.raises(ValueError,match='beta_step'):
        validate_resume_metadata(cfg,metadata)

def test_diagnostic_rng_context_preserves_python_numpy_and_torch():
    import random
    import numpy as np
    import torch
    from efficient_vlm_ad.pipeline import _preserve_rng
    random.seed(2);np.random.seed(2);torch.manual_seed(2)
    expected=(random.random(),np.random.rand(),torch.rand(2))
    random.seed(2);np.random.seed(2);torch.manual_seed(2)
    with _preserve_rng():
        random.random();np.random.rand();torch.rand(19)
    actual=(random.random(),np.random.rand(),torch.rand(2))
    assert expected[:2]==actual[:2]
    assert torch.equal(expected[2],actual[2])

@pytest.mark.parametrize('fallback',[False,True])
def test_qaware_dropout_epoch_resume_matches_continuous(tmp_path, monkeypatch, fallback):
    from dataclasses import replace
    from config_helpers import fake_config
    from efficient_vlm_ad import pipeline
    from efficient_vlm_ad.checkpointing import read_checkpoint_metadata
    base=fake_config()
    cfg=replace(base, model=replace(base.model,
        vision=replace(base.model.vision,output_dim=8,seq_len=4),
        fusion=replace(base.model.fusion,name='mvpruner_camera_query_qaware',stage1_top_k=18,stage2_top_k=8)),
        training=replace(base.training,max_steps=None,batch_size=1,finetune_epochs=2,entropy_weight=.5,beta_decay_updates=4),
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
    assert read_checkpoint_metadata(firstpath)['beta_step']==2
    assert read_checkpoint_metadata(firstpath)['beta_target_steps']==4
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
    assert b['metadata']['beta_step']==4
    assert b['metadata']['beta_target_steps']==4
    def equal_state(x,y):
        if isinstance(x,torch.Tensor): assert torch.equal(x,y)
        elif isinstance(x,dict):
            assert x.keys()==y.keys()
            for key in x: equal_state(x[key],y[key])
        elif isinstance(x,(list,tuple)):
            for left,right in zip(x,y): equal_state(left,right)
        else: assert x==y
    equal_state(a['optimizer'],b['optimizer'])
    import numpy as np
    ra=torch.load(Path(str(fullpath)+'.rng')/'rank_00000.pt',weights_only=False)['rng_state']
    rb=torch.load(Path(str(resumedpath)+'.rng')/'rank_00000.pt',weights_only=False)['rng_state']
    assert ra['python']==rb['python']
    assert ra['numpy'][0]==rb['numpy'][0]
    assert np.array_equal(ra['numpy'][1],rb['numpy'][1])
    assert ra['numpy'][2:]==rb['numpy'][2:]
    assert torch.equal(ra['torch'],rb['torch'])
    if 'cuda' in ra: equal_state(ra['cuda'],rb['cuda'])
    partial=replace(first,project=replace(first.project,output_dir=str(tmp_path/'partial')))
    partialpath=pipeline.train_stage(partial,'finetune',max_steps=1,disable_progress=True)
    assert read_checkpoint_metadata(partialpath)['beta_target_steps']==4
    with pytest.raises(ValueError,match='epoch'):
        pipeline.train_stage(partial,'finetune',resume=str(partialpath),disable_progress=True)




def test_qaware_recipe_includes_beta_policy_but_old_recipe_does_not():
    cfg=config()
    cfg.model.fusion.beta_start=.2
    cfg.model.fusion.beta_end=0.
    cfg.model.fusion.beta_schedule='linear'
    cfg.model.fusion.tau=1.
    assert training_recipe(cfg)['allocation_recipe']=={'beta_start':.2,'beta_end':0.,'beta_schedule':'linear','tau':1.}
    cfg.model.fusion.name='mvpruner_camera_query'
    assert 'allocation_recipe' not in training_recipe(cfg)


def test_qaware_resume_rejects_changed_beta_recipe():
    from efficient_vlm_ad.query_training import validate_resume_metadata
    cfg=config()
    metadata={'stage':'finetune','epoch_boundary':True,**training_recipe(cfg),'optimizer_step':2,'beta_step':2,'beta_target_steps':30}
    cfg.model.fusion.tau=2.
    with pytest.raises(ValueError,match='allocation_recipe'):
        validate_resume_metadata(cfg,metadata)

def test_resume_rejects_beta_buffer_metadata_disagreement():
    from efficient_vlm_ad.pipeline import _configure_qaware_beta
    fusion=NS(beta_step=torch.tensor(2),beta_target_steps=torch.tensor(31))
    with pytest.raises(ValueError,match='buffer'):
        _configure_qaware_beta(fusion,config(),5,{'optimizer_step':2,'beta_step':2,'beta_target_steps':30})
    fusion.beta_target_steps.fill_(30);fusion.beta_step.fill_(1)
    with pytest.raises(ValueError,match='buffer'):
        _configure_qaware_beta(fusion,config(),5,{'optimizer_step':2,'beta_step':2,'beta_target_steps':30})

def test_skipped_optimizer_update_keeps_beta_and_optimizer_counters():
    from efficient_vlm_ad.pipeline import _advance_query_update
    fusion=NS(beta_step=torch.tensor(2))
    fusion.step_beta=lambda: fusion.beta_step.add_(1)
    assert _advance_query_update(config(),fusion,2,skipped=True)==2
    assert fusion.beta_step.item()==2
    assert _advance_query_update(config(),fusion,2,skipped=False)==3
    assert fusion.beta_step.item()==3
    assert _advance_query_update(config('mvpruner_camera_query'),fusion,3,skipped=False)==4
    assert fusion.beta_step.item()==3
