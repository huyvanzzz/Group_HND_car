from unittest.mock import Mock

import pytest
import torch

from efficient_vlm_ad.checkpointing import save_checkpoint, load_checkpoint
from efficient_vlm_ad.modeling.mvpruner import MVPrunerTwoStage


def model(mode):
    net=torch.nn.Module()
    net.visual_fusion=MVPrunerTwoStage(d_model=4,num_views=2,tokens_per_view=3,
                                      stage1_top_k=4,stage2_top_k=2,stage2_score_mode=mode)
    return net


@pytest.mark.parametrize('mode',['attention','l2_variation'])
def test_common_saver_records_mode_without_mutating_metadata(tmp_path,mode):
    net=model(mode); meta={'stage':'finetune','mvpruner':{'beta_step':7}}
    optimizer=torch.optim.AdamW(net.parameters()); path=tmp_path/'new.pt'
    save_checkpoint(path,model=net,optimizer=optimizer,scheduler=None,scaler=None,metadata=meta)
    restored=load_checkpoint(path,model=model(mode))
    assert restored['mvpruner']['stage2_score_mode']==mode
    assert 'stage2_score_mode' not in meta['mvpruner']


@pytest.mark.parametrize('saved,target', [('attention','l2_variation'),('l2_variation','attention'),
                                          (None,'l2_variation'),('unknown','attention')])
@pytest.mark.parametrize('loader',['normal','strict','new_vision'])
def test_mode_mismatch_fails_before_weights(tmp_path,saved,target,loader):
    from efficient_vlm_ad.pipeline import _strict_load_model_from_checkpoint, _load_model_weights_allowing_new_vision
    net=model(target); net.load_state_dict=Mock(side_effect=AssertionError('must not load'))
    meta={'stage1_algorithm_version':2}
    if saved is not None: meta['stage2_score_mode']=saved
    path=tmp_path/'wrong.pt'; torch.save({'metadata':{'mvpruner':meta},'model':{}},path)
    fn={'normal':lambda:load_checkpoint(path,model=net),
        'strict':lambda:_strict_load_model_from_checkpoint(path,net,'cpu'),
        'new_vision':lambda:_load_model_weights_allowing_new_vision(path,net,'cpu')}[loader]
    with pytest.raises(ValueError,match='Stage 2.*score mode'): fn()
    net.load_state_dict.assert_not_called()


def test_legacy_attention_mode_missing_is_accepted(tmp_path):
    net=model('attention'); path=tmp_path/'old.pt'
    save_checkpoint(path,model=net,optimizer=torch.optim.AdamW(net.parameters()),scheduler=None,scaler=None,metadata={})
    data=torch.load(path,weights_only=False); data['metadata']['mvpruner'].pop('stage2_score_mode',None)
    torch.save(data,path)
    load_checkpoint(path,model=model('attention'))


@pytest.mark.parametrize('mode',['wrong','',None])
def test_config_rejects_unknown_score_modes(tmp_path,mode):
    from pathlib import Path
    import yaml
    from efficient_vlm_ad.config import load_config
    data=yaml.safe_load(Path('configs/mvpruner_l2_smoke.yaml').read_text())
    data['model']['fusion']['stage2_score_mode']=mode
    path=tmp_path/'bad_mode.yaml'
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError,match='stage2_score_mode'):
        load_config(path)
