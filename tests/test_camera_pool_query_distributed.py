"""Real two-rank Gloo training with independently ranked camera inputs."""
from datetime import timedelta
from pathlib import Path
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, store, output):
    from test_camera_pool_query_lifecycle import _config, _batch
    from efficient_vlm_ad.modeling.factory import build_vlm_model
    from efficient_vlm_ad.pipeline import _SingleProcessAccelerator
    from efficient_vlm_ad.training_accumulation import loss_for_backward, window_size, window_boundary
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=Path(store).as_uri(), rank=rank,
                            world_size=2, timeout=timedelta(seconds=60))
    try:
        torch.manual_seed(42)
        model, tokenizer = build_vlm_model(_config(Path(output)))
        ddp = torch.nn.parallel.DistributedDataParallel(model)
        optimizer = torch.optim.AdamW(ddp.parameters(), lr=1e-3)
        accelerator = _SingleProcessAccelerator(torch.device('cpu'), gradient_accumulation_steps=2)
        updates = 0
        for index in range(3):
            batch = _batch(tokenizer, seed=100 * rank + index)
            loss = ddp(**batch).loss
            if index == 0:
                camera_rank = model.visual_fusion.last_debug['camera_rank']
                gathered = [torch.empty_like(camera_rank) for _ in range(2)]
                dist.all_gather(gathered, camera_rank)
                assert not torch.equal(gathered[0], gathered[1]), 'Expected differing camera ranks across inputs'
            accelerator.backward(loss_for_backward(loss, 2, window_size(index, 3, 2)))
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                assert parameter.grad is not None, name
                assert torch.isfinite(parameter.grad).all(), name
                gathered = [torch.empty_like(parameter.grad) for _ in range(2)]
                dist.all_gather(gathered, parameter.grad)
                torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)
            if window_boundary(index, 3, 2):
                optimizer.step()
                updates += 1
                optimizer.zero_grad(set_to_none=True)
                for name, value in model.state_dict().items():
                    gathered = [torch.empty_like(value) for _ in range(2)]
                    dist.all_gather(gathered, value)
                    torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)
        assert updates == 2
        assert model.visual_fusion.beta_step.item() == 0
        torch.save(model.state_dict(), Path(output) / f'pool_rank_{rank}.pt')
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason='CPU Gloo unavailable')
def test_two_cpu_ranks_all_trainable_gradients_and_tail_updates_synchronize(tmp_path):
    context = mp.spawn(_worker, args=(str(tmp_path / 'store'), str(tmp_path)), nprocs=2, join=False)
    started = time.monotonic()
    try:
        while not context.join(timeout=1):
            if time.monotonic() - started > 90:
                pytest.fail('Pooled-query CPU distributed test exceeded 90 seconds')
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
    left = torch.load(tmp_path / 'pool_rank_0.pt', weights_only=True)
    right = torch.load(tmp_path / 'pool_rank_1.pt', weights_only=True)
    assert left.keys() == right.keys()
    for name, value in left.items():
        torch.testing.assert_close(value, right[name], rtol=0, atol=0)
    assert all(value.item() == 0 for name, value in left.items() if name.endswith('beta_step'))
