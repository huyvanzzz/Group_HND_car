"""Shared real-model lifecycle checkpoint audits."""
from dataclasses import asdict

import torch
from config_helpers import fake_config

_HEAD_NAMES = ('fc1.weight', 'fc1.bias', 'fc2.weight')
_HEAD_PREFIX = 'visual_fusion.question_scorer.text_pool.'


def _assert_head_checkpoint_restore(config, checkpoint, payload, expected_steps):
    from efficient_vlm_ad import pipeline
    from efficient_vlm_ad.checkpointing import load_checkpoint
    from efficient_vlm_ad.config import load_config

    cfg = load_config(config)
    model, _, _ = pipeline._build_train_model(cfg)
    pipeline.set_trainable_for_stage(pipeline._module_dict_for_freezing(model), cfg, 'finetune')
    optimizer = torch.optim.AdamW(pipeline._optimizer_param_groups(model, cfg),
                                  lr=cfg.training.learning_rate,
                                  weight_decay=cfg.training.weight_decay)
    load_checkpoint(checkpoint, model=model, optimizer=optimizer, cfg=cfg)
    parameters = dict(model.named_parameters())
    for name in _HEAD_NAMES:
        key = _HEAD_PREFIX + name
        parameter = parameters[key]
        assert parameter.requires_grad
        torch.testing.assert_close(parameter, payload['model'][key], rtol=0, atol=0)
        assert parameter in optimizer.state, key
        state = optimizer.state[parameter]
        assert state['step'].item() == expected_steps
        for slot in ('exp_avg', 'exp_avg_sq'):
            assert state[slot].shape == parameter.shape
            assert torch.isfinite(state[slot]).all()
        if name == 'fc2.weight':
            assert torch.count_nonzero(state['exp_avg']) > 0


def three_stage_config():
    raw = asdict(fake_config())
    raw['runtime'] = {'device': 'cpu', 'precision': 'fp32'}
    raw['model']['vision'].update(output_dim=16, seq_len=4, image_size=16)
    raw['model']['text']['d_model'] = 16
    raw['model']['fusion'].update(
        name='mvpruner_three_stage_qaware', stage1_top_k=18, stage2_top_k=12,
        stage3_top_k=8, query={'dim': 8}, beta_start=.2, beta_end=.1,
        question_scoring={'num_heads': 4, 'hidden_dim': 8})
    raw['training'].update(max_steps=None, batch_size=1, finetune_epochs=1,
                           gradient_accumulation_steps=2, num_workers=0,
                           entropy_weight=0., seed=42, beta_decay_updates=None, align_epochs=0)
    raw['generation'].update(max_new_tokens=3, num_beams=1)
    raw['evaluation']['benchmark_max_samples'] = 2
    return raw


