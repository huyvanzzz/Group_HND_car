"""Native T5 attention participates in real CPU distributed answer-CE training."""
from datetime import timedelta
from pathlib import Path
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, store, output):
    from test_mvpruner_stage2_only_lifecycle import _config, _batch
    from efficient_vlm_ad.modeling.factory import build_vlm_model
    from efficient_vlm_ad.pipeline import _SingleProcessAccelerator
    from efficient_vlm_ad.training_accumulation import loss_for_backward, window_size, window_boundary
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=Path(store).as_uri(), rank=rank,
                            world_size=2, timeout=timedelta(seconds=60))
    try:
        torch.manual_seed(42)
        model, tokenizer = build_vlm_model(_config(output))
        ddp = torch.nn.parallel.DistributedDataParallel(model)
        optimizer = torch.optim.AdamW(ddp.parameters(), lr=1e-3)
        accelerator = _SingleProcessAccelerator(torch.device('cpu'), gradient_accumulation_steps=2)
        questions = ['what is ahead', 'where is traffic'] if rank == 0 else ['is the lane clear', 'which car turns']
        for index in range(3):
            batch = _batch(tokenizer, seed=100 * rank + index)
            batch.update(tokenizer(questions, padding=True, return_tensors='pt'))
            raw = ddp(**batch).loss
            accelerator.backward(loss_for_backward(raw, 2, window_size(index, 3, 2)))
            attention = [(name, parameter) for name, parameter in model.text_model.named_parameters()
                         if name.startswith('encoder.') and 'SelfAttention' in name and ('.q.' in name or '.k.' in name)]
            assert attention
            for name, parameter in attention:
                assert parameter.grad is not None, name
                assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0, name
                gathered = [torch.empty_like(parameter.grad) for _ in range(2)]
                dist.all_gather(gathered, parameter.grad)
                torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)
            if window_boundary(index, 3, 2):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                model.visual_fusion.step_beta()
                state = model.state_dict()
                for name, value in state.items():
                    gathered = [torch.empty_like(value) for _ in range(2)]
                    dist.all_gather(gathered, value)
                    torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)
        torch.save(model.state_dict(), Path(output) / f'stage2_only_rank_{rank}.pt')
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason='CPU Gloo unavailable')
def test_two_cpu_ranks_real_t5_attention_gradients_and_tail_updates_synchronize(tmp_path):
    context = mp.spawn(_worker, args=(str(tmp_path / 'store'), str(tmp_path)), nprocs=2, join=False)
    started = time.monotonic()
    try:
        while not context.join(timeout=1):
            if time.monotonic() - started > 90:
                pytest.fail('Stage-2-only CPU distributed test exceeded 90 seconds')
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
    left = torch.load(tmp_path / 'stage2_only_rank_0.pt', weights_only=True)
    right = torch.load(tmp_path / 'stage2_only_rank_1.pt', weights_only=True)
    assert left.keys() == right.keys()
    for name, value in left.items():
        torch.testing.assert_close(value, right[name], rtol=0, atol=0)
    assert all(value.item() == 2 for name, value in left.items() if name.endswith('beta_step'))
