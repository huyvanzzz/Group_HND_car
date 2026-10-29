"""Two CPU ranks synchronize every microbatch, including a short final window."""
from pathlib import Path
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, store, output):
    from efficient_vlm_ad.pipeline import _SingleProcessAccelerator
    from efficient_vlm_ad.training_accumulation import window_size, window_boundary, loss_for_backward
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=Path(store).as_uri(), rank=rank, world_size=2)
    try:
        torch.manual_seed(42)
        model = torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(2, 1))
        reference = torch.nn.Linear(2, 1)
        reference.load_state_dict(model.module.state_dict())
        optimizer = torch.optim.SGD(model.parameters(), lr=.02)
        ref_optimizer = torch.optim.SGD(reference.parameters(), lr=.02)
        accelerator = _SingleProcessAccelerator(torch.device('cpu'), gradient_accumulation_steps=2)
        snapshots = []
        for epoch in range(2):
            for index in range(3):
                x = torch.tensor([[float(rank + index + 1), (-1.) ** index]])
                target = torch.tensor([[.2 * (rank + index)]])
                raw = (model(x) - target).square().mean()
                accelerator.backward(loss_for_backward(raw, 2, window_size(index, 3, 2)))
                # This assertion runs after every backward, not only optimizer updates.
                for parameter in model.parameters():
                    assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                    gathered = [torch.empty_like(parameter.grad) for _ in range(2)]
                    dist.all_gather(gathered, parameter.grad)
                    torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)
                if window_boundary(index, 3, 2):
                    start = index // 2 * 2
                    losses = []
                    for other_rank in range(2):
                        for j in range(start, index + 1):
                            values = torch.tensor([[float(other_rank + j + 1), (-1.) ** j]])
                            expected = torch.tensor([[.2 * (other_rank + j)]])
                            losses.append((reference(values) - expected).square().mean())
                    torch.stack(losses).mean().backward()
                    for actual, expected in zip(model.parameters(), reference.parameters()):
                        torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-5, atol=1e-6)
                    accelerator.clip_grad_norm_(model.parameters(), .8)
                    torch.nn.utils.clip_grad_norm_(reference.parameters(), .8)
                    optimizer.step(); ref_optimizer.step()
                    optimizer.zero_grad(); ref_optimizer.zero_grad()
            for actual, expected in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-6)
            snapshots.append({name: value.clone() for name, value in model.module.state_dict().items()})
        torch.save(snapshots, Path(output) / f'accum_rank_{rank}.pt')
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason='CPU Gloo unavailable')
def test_two_cpu_ranks_tail_window_stays_synchronized_across_epochs(tmp_path):
    context = mp.spawn(_worker, args=(str(tmp_path / 'store'), str(tmp_path)), nprocs=2, join=False)
    started = time.monotonic()
    try:
        while not context.join(timeout=1):
            if time.monotonic() - started > 90:
                pytest.fail('two CPU rank accumulation exceeded 90 seconds')
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    first = torch.load(tmp_path / 'accum_rank_0.pt', weights_only=True)
    second = torch.load(tmp_path / 'accum_rank_1.pt', weights_only=True)
    assert len(first) == len(second) == 2
    for epoch in range(2):
        for name, tensor in first[epoch].items():
            torch.testing.assert_close(tensor, second[epoch][name], rtol=0, atol=0)
