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


class _QuestionAwareCompression(torch.nn.Module):
    """Exercise the real fusion, including camera embeddings and learned scorer."""
    def __init__(self):
        super().__init__()
        from efficient_vlm_ad.modeling.question_camera_query import QuestionAwareCameraQueryFusion
        self.fusion = QuestionAwareCameraQueryFusion(
            d_model=16, num_views=2, tokens_per_view=4, stage1_top_k=8,
            stage2_top_k=4, query_dim=8, scoring_hidden_dim=8,
        )

    def forward(self, values, questions, question_mask, camera_ids):
        values = values + self.fusion.camera_embeddings(camera_ids)
        result = self.fusion.compress_from_question(
            values, questions, question_mask, camera_ids,
            training=True, compute_entropy=True,
        )
        return result.visual_tokens, result.entropy_loss, result.budgets


def _qaware_worker(rank, store_path, output_dir):
    import efficient_vlm_ad.modeling.question_camera_query as module

    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=Path(store_path).as_uri(), rank=rank, world_size=2)
    try:
        torch.manual_seed(42)
        model = torch.nn.parallel.DistributedDataParallel(_QuestionAwareCompression())
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        original = module.build_prefix_utilities

        def fixed_noise(log_shares, capacities, bank_size, **kwargs):
            noise = log_shares.new_zeros(log_shares.shape + (bank_size,))
            # Rank zero chooses [3,1], rank one chooses [1,3]. Both the hard
            # and differentiable DP receive exactly this same perturbation.
            noise[:, rank, 2] = 10.
            noise[:, 1 - rank, 0] = 10.
            return original(log_shares, capacities, bank_size, noise=noise, **kwargs)

        module.build_prefix_utilities = fixed_noise
        torch.manual_seed(100 + rank)
        values = torch.randn(1, 8, 16)
        questions = torch.randn(1, 3, 16)
        mask = torch.tensor([[1, 1, 0]])
        ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
        expected = [[3, 1] if rank == 0 else [1, 3]]
        recorded = []
        for _ in range(2):
            output, entropy, budgets = model(values, questions, mask, ids)
            assert budgets.tolist() == expected
            recorded.append(budgets.tolist())
            loss = output.square().mean() + 0.5 * entropy
            loss.backward()
            for name, parameter in model.named_parameters():
                assert parameter.grad is not None, name
                assert torch.isfinite(parameter.grad).all(), name
                gradients = [torch.empty_like(parameter.grad) for _ in range(2)]
                dist.all_gather(gradients, parameter.grad)
                torch.testing.assert_close(gradients[0], gradients[1], rtol=0, atol=0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        torch.save({'state': model.module.state_dict(), 'budgets': recorded},
                   Path(output_dir) / f'qaware_rank_{rank}.pt')
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason='CPU Gloo unavailable')
def test_question_aware_fusion_distinct_budgets_two_cpu_ranks(tmp_path):
    context = mp.spawn(_qaware_worker, args=(str(tmp_path / 'qaware_store'), str(tmp_path)),
                       nprocs=2, join=False)
    started = time.monotonic()
    try:
        while not context.join(timeout=1):
            if time.monotonic() - started > 90:
                pytest.fail('two-rank question-aware compression exceeded 90 seconds')
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    first = torch.load(tmp_path / 'qaware_rank_0.pt', weights_only=True)
    second = torch.load(tmp_path / 'qaware_rank_1.pt', weights_only=True)
    assert first['budgets'] == [[[3, 1]], [[3, 1]]]
    assert second['budgets'] == [[[1, 3]], [[1, 3]]]
    for name, value in first['state'].items():
        torch.testing.assert_close(value, second['state'][name], rtol=0, atol=0)
