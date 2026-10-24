from unittest.mock import Mock

import pytest
import torch

from efficient_vlm_ad.checkpointing import load_checkpoint, save_checkpoint
from efficient_vlm_ad.modeling.mvpruner import MVPrunerTwoStage


def model():
    wrapper = torch.nn.Module()
    wrapper.visual_fusion = MVPrunerTwoStage(d_model=4, num_views=2, tokens_per_view=4,
                                            stage1_top_k=4, stage2_top_k=2)
    return wrapper


def test_mvpruner_checkpoint_saves_version_and_preserves_metadata(tmp_path):
    net = model()
    optimizer = torch.optim.AdamW(net.parameters())
    metadata = {'stage': 'finetune', 'mvpruner': {'beta_step': 7}}
    path = tmp_path / 'new.pt'
    save_checkpoint(path, model=net, optimizer=optimizer, scheduler=None, scaler=None, metadata=metadata)
    restored = load_checkpoint(path, model=model())
    assert restored['mvpruner'] == {'beta_step': 7, 'stage1_algorithm_version': 2, 'stage2_score_mode': 'attention'}
    assert metadata == {'stage': 'finetune', 'mvpruner': {'beta_step': 7}}


@pytest.mark.parametrize('loader', ['normal', 'strict', 'new_vision'])
@pytest.mark.parametrize('version', [None, 1])
def test_old_mvpruner_rejected_before_model_load(tmp_path, loader, version):
    from efficient_vlm_ad.pipeline import _strict_load_model_from_checkpoint, _load_model_weights_allowing_new_vision
    net = model()
    net.load_state_dict = Mock(side_effect=AssertionError('weights must not load'))
    path = tmp_path / 'old.pt'
    meta = {'mvpruner': {}} if version is None else {'mvpruner': {'stage1_algorithm_version': version}}
    torch.save({'model': {}, 'metadata': meta}, path)
    fn = {'normal': lambda: load_checkpoint(path, model=net),
          'strict': lambda: _strict_load_model_from_checkpoint(path, net, 'cpu'),
          'new_vision': lambda: _load_model_weights_allowing_new_vision(path, net, 'cpu')}[loader]
    with pytest.raises(ValueError, match='Stage 1.*train'):
        fn()
    net.load_state_dict.assert_not_called()


def test_old_non_mvpruner_checkpoint_still_loads(tmp_path):
    net = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(net.parameters())
    path = tmp_path / 'old_router.pt'
    save_checkpoint(path, model=net, optimizer=optimizer, scheduler=None, scaler=None, metadata={'stage': 'finetune'})
    other = torch.nn.Linear(4, 4)
    assert load_checkpoint(path, model=other) == {'stage': 'finetune'}
    x = torch.randn(2, 4)
    torch.testing.assert_close(net(x), other(x))


def test_legacy_question_router_checkpoint_keeps_output(tmp_path):
    from efficient_vlm_ad.modeling.router import QuestionGuidedTokenRouter
    def build():
        wrapper = torch.nn.Module()
        wrapper.visual_fusion = QuestionGuidedTokenRouter(vision_dim=8, text_dim=4, seq_len=3,
                                                         num_views=2, top_k=3)
        return wrapper.eval()
    torch.manual_seed(7)
    net = build()
    optimizer = torch.optim.AdamW(net.parameters())
    path = tmp_path / 'router.pt'
    save_checkpoint(path, model=net, optimizer=optimizer, scheduler=None, scaler=None, metadata={'stage': 'finetune'})
    payload = torch.load(path, weights_only=False)
    assert 'mvpruner' not in payload['metadata']
    other = build()
    load_checkpoint(path, model=other)
    inputs = (torch.randn(2, 2, 3, 8), torch.randn(2, 4, 4), torch.ones(2, 4))
    with torch.no_grad():
        torch.testing.assert_close(net.visual_fusion(*inputs), other.visual_fusion(*inputs))
