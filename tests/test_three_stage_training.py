"""Shared three-stage beta schedule across training, skipped updates and inference."""
import pytest
import torch
import yaml

from efficient_vlm_ad.config import load_config
from efficient_vlm_ad import pipeline
from efficient_vlm_ad.query_training import qaware_beta_horizon
from lifecycle_helpers import three_stage_config
from test_three_stage_model import fusion, model, batch


def config(tmp_path, **training):
    raw = three_stage_config()
    raw['training'].update(training)
    raw['project']['output_dir'] = str(tmp_path / 'output')
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(raw), encoding='utf-8')
    return load_config(path)


@pytest.mark.parametrize('batches,accum,epochs,expected', [(4,2,3,6), (5,2,3,9), (1,4,1,1), (10669,2,10,53350)])
def test_prepared_horizon_includes_tail(tmp_path, batches, accum, epochs, expected):
    cfg = config(tmp_path, gradient_accumulation_steps=accum, finetune_epochs=epochs, max_steps=1)
    assert qaware_beta_horizon(cfg, batches) == expected


@pytest.mark.parametrize('horizon', [1,2,5])
def test_forward_beta_and_post_update_beta_have_distinct_endpoints(tmp_path, horizon):
    cfg = config(tmp_path)
    f = fusion(beta_start=.2, beta_end=.1)
    f.configure_beta(horizon, reset=True)
    before = []
    counter = 0
    for step in range(horizon):
        before.append(f.current_beta())
        counter = pipeline._advance_query_update(cfg, f, counter, skipped=False)
    assert before == pytest.approx([.2 - .1 * min(t / max(1,horizon-1), 1) for t in range(horizon)])
    assert f.current_beta() == pytest.approx(.1)
    assert counter == int(f.beta_step) == horizon
    old = (f.current_beta(), int(f.beta_step), counter)
    counter = pipeline._advance_query_update(cfg, f, counter, skipped=True)
    assert (f.current_beta(), int(f.beta_step), counter) == old


def test_eval_forward_and_generation_preserve_shared_counter_and_rng():
    m = model().eval()
    m.visual_fusion.configure_beta(7, reset=True)
    m.visual_fusion.step_beta()
    b = batch()
    rng = torch.get_rng_state().clone()
    with torch.no_grad():
        m(**b)
        m.generate_from_features(**{k:v for k,v in b.items() if k != 'labels'}, max_new_tokens=2, num_beams=3)
    assert int(m.visual_fusion.beta_step) == 1
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("prepared_batches,expected_horizon", [(3,6), (1,3)])
def test_debug_step_limit_preserves_full_training_horizon(tmp_path, monkeypatch, prepared_batches, expected_horizon):
    cfg = config(tmp_path, finetune_epochs=3)
    class Samples(torch.utils.data.Dataset):
        def __len__(self): return 3
        def summary(self): return {'cached_records':3}
        def __getitem__(self, index):
            return {'question':'question', 'answer':'road', 'visual_features':torch.full((6,4,16), float(index+1)), 'eval_id':str(index)}
    monkeypatch.setattr(pipeline, '_make_dataset_and_collator', lambda cfg, split, tokenizer, **kw: (Samples(), pipeline.CachedBatchCollator(tokenizer)))
    class PreparedLoader:
        def __init__(self, original):
            self.original = original
        def __getattr__(self, name):
            return getattr(self.original, name)
        def __len__(self):
            return prepared_batches
        def __iter__(self):
            from itertools import islice
            return islice(iter(self.original), prepared_batches)
    class PreparedAccelerator(pipeline._SingleProcessAccelerator):
        def prepare(self, *objects):
            prepared = list(objects)
            for i, obj in enumerate(prepared):
                if isinstance(obj, torch.utils.data.DataLoader):
                    prepared[i] = PreparedLoader(obj)
            return tuple(prepared)
    monkeypatch.setattr(pipeline, '_build_accelerator', lambda cfg: PreparedAccelerator(torch.device('cpu'), gradient_accumulation_steps=2))
    path = pipeline.train_stage(cfg, 'finetune', max_steps=1, disable_progress=True)
    stored = torch.load(path, weights_only=False)
    assert stored['metadata']['beta_target_steps'] == expected_horizon
    assert stored['metadata']['beta_step'] == stored['metadata']['optimizer_step'] == 1
    assert stored['metadata']['epoch_boundary'] is (prepared_batches == 1)
