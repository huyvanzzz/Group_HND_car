from datetime import timedelta
from pathlib import Path
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, store, output):
    from test_mvpruner_ccts_context_encoder import make_model, inputs
    from efficient_vlm_ad.pipeline import _SingleProcessAccelerator
    from efficient_vlm_ad.training_accumulation import loss_for_backward, window_size, window_boundary
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=Path(store).as_uri(), rank=rank,
        world_size=2, timeout=timedelta(seconds=60))
    try:
        model = make_model()
        model.visual_fusion.configure_beta(2)
        ddp = torch.nn.parallel.DistributedDataParallel(model)
        optimizer = torch.optim.AdamW(ddp.parameters(), lr=1e-3)
        accelerator = _SingleProcessAccelerator(torch.device('cpu'), gradient_accumulation_steps=2)
        for index in range(3):
            torch.manual_seed(rank * 100 + index)
            loss = ddp(**inputs()).loss
            accelerator.backward(loss_for_backward(loss, 2, window_size(index,3,2)))
            for name, parameter in model.named_parameters():
                assert parameter.grad is not None, name
                assert torch.isfinite(parameter.grad).all(), name
                gathered = [torch.empty_like(parameter.grad) for _ in range(2)]
                dist.all_gather(gathered, parameter.grad)
                torch.testing.assert_close(gathered[0],gathered[1],rtol=0,atol=0)
            if window_boundary(index,3,2):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                model.visual_fusion.step_beta()
        torch.save(model.state_dict(),Path(output)/f'context_rank_{rank}.pt')
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(),reason='CPU Gloo unavailable')
def test_two_rank_context_training_full_and_tail_windows(tmp_path):
    context = mp.spawn(_worker,args=(str(tmp_path/'store'),str(tmp_path)),nprocs=2,join=False)
    started = time.monotonic()
    try:
        while not context.join(timeout=1):
            if time.monotonic()-started>100:
                pytest.fail('context CPU distributed test exceeded 100 seconds')
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
    left = torch.load(tmp_path/'context_rank_0.pt',weights_only=True)
    right = torch.load(tmp_path/'context_rank_1.pt',weights_only=True)
    for key,value in left.items():
        torch.testing.assert_close(value,right[key],rtol=0,atol=0)
    assert left['visual_fusion.beta_step'].item()==2
