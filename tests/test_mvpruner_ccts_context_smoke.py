import json
import os
from pathlib import Path

import yaml
import pytest
from runtime_helpers import run_cli

from data_helpers import _make_fake_data


@pytest.mark.integration
def test_context_offline_cli_lifecycle(tmp_path):
    data = tmp_path/'data'
    _make_fake_data(data,rows_per_split=3)
    output = tmp_path/'output'
    config = dict(project=dict(output_dir=output.as_posix()),
        data=dict(hf_repo_id='local/fake',local_dir=data.as_posix(),
            view_order=['Front','Front-Left','Front-Right','Back','Back-Left','Back-Right'],
            expected_counts=dict(train=3,val=3,test=3)),
        model=dict(profile='offline_context',
            vision=dict(name='fake_vision',model_id='fake',output_dim=8,seq_len=4,image_size=16),
            text=dict(model_id='fake_t5',d_model=8),projector=dict(type='linear'),
            fusion=dict(name='mvpruner_ccts_context',stage1_top_k=18,stage2_top_k=6,context_tokens=6)),
        cache=dict(dir=(output/'cache').as_posix()),runtime=dict(device='cpu',precision='fp32'),
        training=dict(batch_size=1,gradient_accumulation_steps=2,finetune_epochs=1,num_workers=0),
        generation=dict(max_new_tokens=2,num_beams=2),evaluation=dict(benchmark_max_samples=2))
    path=tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(config),encoding='utf-8')
    checkpoint=str(output/'checkpoints'/'finetune_latest.pt')
    commands=[['prepare-data'],['prepare-features'],['diagnose-train','--stage','finetune'],
        ['train','--stage','finetune'],
        ['verify-resume-checkpoint','--stage','finetune','--checkpoint',checkpoint],
        ['train','--stage','finetune','--resume',checkpoint],
        ['evaluate','--checkpoint',checkpoint,'--max-samples','2'],
        ['benchmark','--checkpoint',checkpoint,'--max-samples','2']]
    env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[1]/'src'),
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for command in commands:
        if command[0]=='train' and '--resume' in command:
            config['training']['finetune_epochs']=2
            path.write_text(yaml.safe_dump(config),encoding='utf-8')
        result=run_cli([*command,'--config',str(path)],env=env)
        assert result.returncode==0,(command,result.stdout[-5000:],result.stderr[-5000:])
    import torch
    payload=torch.load(checkpoint,weights_only=False)
    assert payload['metadata']['optimizer_step']==4
    assert payload['metadata']['mvpruner']['output_visual_tokens']==12
