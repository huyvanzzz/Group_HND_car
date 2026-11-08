"""Real native causal CE and synchronized full/tail windows on two CPU ranks."""
from pathlib import Path
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _native_accumulation_worker(rank, store, output):
    from smolvlm_helpers import tiny_components
    from efficient_vlm_ad.smolvlm.model import CachedSmolVLM
    from efficient_vlm_ad.smolvlm.text import CachedBatchCollator
    from efficient_vlm_ad.pipeline import _SingleProcessAccelerator
    from efficient_vlm_ad.training_accumulation import window_size, window_boundary, loss_for_backward
    import copy

    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=Path(store).as_uri(), rank=rank, world_size=2)
    try:
        native, processor = tiny_components(dropout=.1)
        wrapper = CachedSmolVLM(native, processor, gradient_checkpointing=True)
        model = torch.nn.parallel.DistributedDataParallel(wrapper)
        reference = copy.deepcopy(wrapper)
        collator = CachedBatchCollator(processor, image_seq_len=4, max_length=512)
        optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=.01)
        reference_optimizer = torch.optim.SGD((p for p in reference.parameters() if p.requires_grad), lr=.01)
        backend = _SingleProcessAccelerator(torch.device('cpu'), gradient_accumulation_steps=2)
        frozen = {name: value.clone() for name, value in wrapper.native.model.vision_model.state_dict().items()}
        snapshots = []
        updates = 0

        def batch(other_rank, index):
            generator = torch.Generator().manual_seed(170 + other_rank * 10 + index)
            pixels = torch.rand(1, 6, 3, 16, 16, generator=generator)
            features = wrapper.extract_features(pixels)[0]
            collated = collator([dict(question='What is visible?' if index % 2 else 'What changed?',
                answer='A road.' if other_rank == 0 else 'A person.', visual_features=features,
                sample_id=f'{other_rank}-{index}', eval_id=index)])
            return {k: v for k,v in collated.items() if k in {'input_ids','attention_mask','visual_features','labels'}}

        batches = {(other_rank, index): batch(other_rank, index)
                   for other_rank in range(2) for index in range(3)}

        def seed(epoch, other_rank, index):
            torch.manual_seed(710 + epoch * 100 + other_rank * 10 + index)

        for epoch in range(2):
            for index in range(3):
                size = window_size(index, 3, 2)
                seed(epoch, rank, index)
                result = model(**batches[rank, index])
                expected_ce = torch.nn.functional.cross_entropy(
                    result.logits[:, :-1].reshape(-1, result.logits.shape[-1]),
                    batches[rank,index]['labels'][:,1:].reshape(-1), ignore_index=-100)
                torch.testing.assert_close(result.loss, expected_ce)
                backend.backward(loss_for_backward(result.loss, 2, size))
                for name, parameter in model.named_parameters():
                    if not parameter.requires_grad:
                        assert parameter.grad is None, name
                        continue
                    assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
                    gathered = [torch.empty_like(parameter.grad) for _ in range(2)]
                    dist.all_gather(gathered, parameter.grad)
                    torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0, msg=name)
                if window_boundary(index, 3, 2):
                    for other_rank in range(2):
                        for micro in range(index // 2 * 2, index + 1):
                            seed(epoch, other_rank, micro)
                            (reference(**batches[other_rank,micro]).loss / (2 * size)).backward()
                    for (name, actual), (_, expected) in zip(wrapper.named_parameters(), reference.named_parameters()):
                        if actual.requires_grad:
                            torch.testing.assert_close(actual.grad, expected.grad, rtol=5e-4, atol=4e-6, msg=name)
                    backend.clip_grad_norm_(model.parameters(), 1.)
                    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.)
                    optimizer.step()
                    reference_optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    reference_optimizer.zero_grad(set_to_none=True)
                    updates += 1
            assert updates == (epoch + 1) * 2
            for name, value in wrapper.state_dict().items():
                torch.testing.assert_close(value, reference.state_dict()[name], rtol=5e-4, atol=4e-6, msg=name)
            for name, value in wrapper.native.model.vision_model.state_dict().items():
                torch.testing.assert_close(value, frozen[name], rtol=0, atol=0)
            snapshots.append({'updates': updates, 'weights': {name: value.clone() for name,value in wrapper.state_dict().items()}})
        torch.save(snapshots, Path(output) / f'native_rank_{rank}.pt')
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason='CPU Gloo unavailable')
def test_native_two_cpu_ranks_full_tail_gradients_and_updates(tmp_path):
    context = mp.spawn(_native_accumulation_worker, args=(str(tmp_path / 'store'), str(tmp_path)), nprocs=2, join=False)
    started = time.monotonic()
    try:
        while not context.join(timeout=1):
            if time.monotonic() - started > 120:
                pytest.fail('two-rank native SmolVLM accumulation exceeded 120 seconds')
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    first = torch.load(tmp_path / 'native_rank_0.pt', weights_only=True)
    second = torch.load(tmp_path / 'native_rank_1.pt', weights_only=True)
    for epoch in range(2):
        assert first[epoch]['updates'] == second[epoch]['updates'] == 2 * (epoch + 1)
        for name, value in first[epoch]['weights'].items():
            torch.testing.assert_close(value, second[epoch]['weights'][name], rtol=0, atol=0)


def _production_train_worker(rank, store, config, output, resume=None):
    import os
    from efficient_vlm_ad.smolvlm.config import load_config
    from efficient_vlm_ad.smolvlm.pipeline import train
    torch.set_num_threads(1)
    os.environ.update(RANK=str(rank), WORLD_SIZE='2', LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE='2',
                      MASTER_ADDR='127.0.0.1', MASTER_PORT='29597')
    dist.init_process_group('gloo', init_method=Path(store).as_uri(), rank=rank, world_size=2)
    try:
        state = train(load_config(config), resume=resume, debug=True)
        torch.save(state, Path(output) / f'production_rank_{rank}.pt')
    finally:
        dist.destroy_process_group()


def _run_production_train(tmp_path, config, name, resume=None):
    context = mp.spawn(_production_train_worker,
        args=(str(tmp_path / (name + '_store')), str(config), str(tmp_path),
              None if resume is None else str(resume)), nprocs=2, join=False)
    started = time.monotonic()
    try:
        while not context.join(timeout=1):
            if time.monotonic() - started > 120:
                pytest.fail('production two-rank native training exceeded 120 seconds')
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)


def _assert_same_state(actual, expected):
    import numpy as np
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_same_state(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_same_state(left, right)
    else:
        assert actual == expected


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason='CPU Gloo unavailable')
def test_production_native_train_two_ranks_full_tail_counters_and_rng(tmp_path):
    import json
    import yaml
    from smolvlm_helpers import create_local_run
    from efficient_vlm_ad.hf_data import prepare_data
    from efficient_vlm_ad.smolvlm.config import load_config
    from efficient_vlm_ad.smolvlm.model import build_model_and_processor
    from efficient_vlm_ad.smolvlm.cache import prepare_features
    config = create_local_run(tmp_path, rows=5, dropout=.1)
    raw = yaml.safe_load(config.read_text())
    raw['training']['finetune_epochs'] = 2
    config.write_text(yaml.safe_dump(raw))
    cfg = load_config(config)
    prepare_data(cfg, disable_progress=True)
    model, processor = build_model_and_processor(cfg)
    frozen = {name: value.clone() for name,value in model.native.model.vision_model.state_dict().items()}
    prepare_features(cfg, model, processor, disable_progress=True)
    _run_production_train(tmp_path, config, 'continuous')
    first = torch.load(tmp_path / 'production_rank_0.pt', weights_only=True)
    second = torch.load(tmp_path / 'production_rank_1.pt', weights_only=True)
    assert first == second
    assert first['world_size'] == 2
    assert first['optimizer_step'] == 4 and first['global_step'] == 6
    assert first['completed_epochs'] == 2
    checkpoint = Path(cfg.project.output_dir) / 'checkpoints' / 'finetune_latest.pt'
    payload = torch.load(checkpoint, weights_only=False)
    state = payload['model_state_dict']
    for name,value in frozen.items():
        torch.testing.assert_close(state['native.model.vision_model.' + name], value, rtol=0, atol=0)
    rng = json.loads((checkpoint.parent / (checkpoint.name + '.rng') / 'manifest.json').read_text())
    assert rng['world_size'] == 2
    events = [json.loads(line) for line in (Path(cfg.project.output_dir) / 'debug' / 'debug_events.jsonl').read_text().splitlines()]
    steps = [row['payload'] for row in events if row['section'] == 'TRAIN_STEP']
    assert [row['window_size'] for row in steps] == [2, 1, 2, 1]
    assert [row['optimizer_step'] for row in steps] == [1, 2, 3, 4]

    # An interrupted epoch-boundary run must restore both ranks' dropout RNG.
    split_config = create_local_run(tmp_path / 'split', rows=5, dropout=.1)
    split_cfg = load_config(split_config)
    prepare_data(split_cfg, disable_progress=True)
    split_model, split_processor = build_model_and_processor(split_cfg)
    prepare_features(split_cfg, split_model, split_processor, disable_progress=True)
    _run_production_train(tmp_path, split_config, 'first_epoch')
    split_checkpoint = Path(split_cfg.project.output_dir) / 'checkpoints' / 'finetune_latest.pt'
    partial = torch.load(split_checkpoint, weights_only=False)
    assert partial['metadata']['optimizer_step'] == 2
    assert partial['metadata']['completed_epochs'] == 1
    split_raw = yaml.safe_load(split_config.read_text())
    split_raw['training']['finetune_epochs'] = 2
    split_config.write_text(yaml.safe_dump(split_raw))
    _run_production_train(tmp_path, split_config, 'resumed_epoch', resume=split_checkpoint)
    resumed = torch.load(split_checkpoint, weights_only=False)
    for key in ('model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 'metadata'):
        _assert_same_state(resumed[key], payload[key])
    for rank in range(2):
        suffix = f'.rng/rank_{rank:05d}.pt'
        continuous_rng = torch.load(str(checkpoint) + suffix, weights_only=False)
        resumed_rng = torch.load(str(split_checkpoint) + suffix, weights_only=False)
        _assert_same_state(resumed_rng, continuous_rng)

    # Verification uses the checkpoint's stored world size without creating ranks.
    from efficient_vlm_ad.smolvlm.checkpoint import read
    checked = read(checkpoint, model, cfg, training=True, world_size=None)
    assert checked['metadata']['world_size'] == 2
    # Inference consumes native weights without requiring training RNG sidecars.
    rng_directory = Path(str(checkpoint) + '.rng')
    rng_directory.rename(Path(str(checkpoint) + '.rng-hidden'))
    from efficient_vlm_ad.smolvlm.pipeline import inference
    report = inference(cfg, checkpoint, max_samples=1, skip_meteor=True)
    assert report is not None
    predictions = [json.loads(line) for line in (Path(cfg.project.output_dir) / 'predictions.jsonl').read_text().splitlines()]
    assert len(predictions) == 1
