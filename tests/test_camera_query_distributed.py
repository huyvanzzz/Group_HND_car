"""CPU/Gloo coverage for different active query prefixes on different ranks."""
from pathlib import Path
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _camera_query_worker(rank, store_path, output_dir):
    from efficient_vlm_ad.modeling.camera_query import CameraQueryCompressor

    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=Path(store_path).as_uri(), rank=rank, world_size=2)
    try:
        torch.manual_seed(42)
        model = torch.nn.parallel.DistributedDataParallel(
            CameraQueryCompressor(d_model=8, num_views=2, output_tokens=3, query_dim=4),
            find_unused_parameters=True,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        torch.manual_seed(100 + rank)
        source = torch.randn(1, 6, 8)
        camera_ids = torch.tensor([[0, 0, 0, 1, 1, 1]])
        budgets = torch.tensor([[2, 1] if rank == 0 else [1, 2]])
        for _ in range(2):
            output = model(source, camera_ids, budgets)
            loss = output.visual_tokens.square().mean() + 0.5 * output.entropy_loss
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        torch.save(model.module.state_dict(), Path(output_dir) / f'rank_{rank}.pt')
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason='CPU Gloo unavailable')
def test_different_camera_budgets_on_two_cpu_ranks(tmp_path):
    context = mp.spawn(_camera_query_worker, args=(str(tmp_path / 'store'), str(tmp_path)), nprocs=2, join=False)
    started = time.monotonic()
    try:
        while not context.join(timeout=1):
            if time.monotonic() - started > 90:
                pytest.fail('two-rank query compression did not complete within 90 seconds')
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    first = torch.load(tmp_path / 'rank_0.pt', weights_only=True)
    second = torch.load(tmp_path / 'rank_1.pt', weights_only=True)
    for name, tensor in first.items():
        torch.testing.assert_close(tensor, second[name], rtol=0, atol=0)
