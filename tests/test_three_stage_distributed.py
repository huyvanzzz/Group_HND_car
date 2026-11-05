"""Actual three-stage VLM training on two CPU/Gloo ranks."""
import copy
from pathlib import Path
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _tiny_vlm():
    from transformers import T5Config, T5ForConditionalGeneration
    from efficient_vlm_ad.modeling.three_stage_qaware import MVPrunerThreeStageQaware
    from efficient_vlm_ad.modeling.vlm import EfficientVLMForAD
    text = T5ForConditionalGeneration(T5Config(
        vocab_size=32, d_model=16, d_kv=4, d_ff=32, num_layers=4,
        num_decoder_layers=2, num_heads=4, dropout_rate=.1,
        decoder_start_token_id=0, pad_token_id=0, eos_token_id=1))
    fusion = MVPrunerThreeStageQaware(
        d_model=16, num_views=2, tokens_per_view=4, stage1_top_k=8,
        stage2_top_k=6, stage3_top_k=4, query_dim=8,
        scoring_hidden_dim=8, beta_start=.2, beta_end=.1, entropy_weight=0.)
    fusion.configure_beta(2)
    return EfficientVLMForAD(text_model=text, visual_fusion=fusion,
                            vision_dim=16, d_model=16, seq_len=4, projector_type='linear')


def _worker(rank, store_path, output_dir):
    import efficient_vlm_ad.modeling.question_camera_query as allocation
    from efficient_vlm_ad.checkpointing import save_rank_rng_state, write_rank_rng_manifest, load_rank_rng_state
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=Path(store_path).as_uri(), rank=rank, world_size=2)
    try:
        torch.manual_seed(42)
        model = torch.nn.parallel.DistributedDataParallel(_tiny_vlm())
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        original = allocation.build_prefix_utilities
        recorded = []

        def rank_noise(log_shares, capacities, bank_size, **kwargs):
            # Force each rank to use a different query prefix, independent of
            # the learned scorer, while retaining its soft-DP gradient graph.
            noise = log_shares.new_zeros(log_shares.shape + (bank_size,))
            noise[:, rank, 2] = 1000.
            noise[:, 1-rank, 0] = 1000.
            return original(log_shares, capacities, bank_size, noise=noise, **kwargs)

        allocation.build_prefix_utilities = rank_noise
        # Both cameras must have capacity three after the attention prune.
        model.module.visual_fusion.min_tokens_per_camera = 3
        # Stage3 itself still has minimum one/query; temporarily restoring the
        # fusion minimum around compression keeps stage2 capacity deterministic.
        fusion = model.module.visual_fusion
        compress = fusion.compress_from_question

        def compression(*args, **kwargs):
            fusion.min_tokens_per_camera = 1
            try:
                result = compress(*args, **kwargs)
                recorded.append(result.budgets.tolist())
                return result
            finally:
                fusion.min_tokens_per_camera = 3
        fusion.compress_from_question = compression
        torch.manual_seed(100 + rank)
        batch = dict(input_ids=torch.tensor([[2, 3, 0]]), attention_mask=torch.tensor([[1, 1, 0]]),
                     visual_features=torch.randn(1, 2, 4, 16), labels=torch.tensor([[4, 5, 1]]))

        def update():
            before = int(fusion.beta_step)
            output = model(**batch)
            assert torch.isfinite(output.loss)
            assert int(fusion.beta_step) == before
            output.loss.backward()
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                assert parameter.grad is not None, name
                assert torch.isfinite(parameter.grad).all(), name
                gradients = [torch.empty_like(parameter.grad) for _ in range(2)]
                dist.all_gather(gradients, parameter.grad)
                torch.testing.assert_close(gradients[0], gradients[1], rtol=0, atol=0)
            optimizer.step()
            fusion.step_beta()
            assert int(fusion.beta_step) == before + 1
            optimizer.zero_grad(set_to_none=True)
            return output.loss.detach()

        update()
        assert abs(fusion.current_beta() - .1) < 1e-6
        checkpoint = Path(output_dir) / 'resume.pt'
        saved = copy.deepcopy(model.module.state_dict())
        optimizer_saved = copy.deepcopy(optimizer.state_dict())
        save_rank_rng_state(checkpoint, rank=rank, world_size=2, epoch=1, batch_cursor=0, optimizer_step=1)
        dist.barrier()
        if rank == 0:
            write_rank_rng_manifest(checkpoint, world_size=2)
        dist.barrier()
        expected_loss = update()
        expected = copy.deepcopy(model.module.state_dict())
        model.module.load_state_dict(saved)
        optimizer.load_state_dict(optimizer_saved)
        load_rank_rng_state(checkpoint, rank=rank, world_size=2)
        actual_loss = update()
        torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
        for key, tensor in model.module.state_dict().items():
            torch.testing.assert_close(tensor, expected[key], rtol=0, atol=0)
        assert recorded == [[[3, 1] if rank == 0 else [1, 3]]] * 3
        torch.save({'state': expected, 'budgets': recorded}, Path(output_dir) / f'three_rank_{rank}.pt')
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason='CPU Gloo unavailable')
def test_three_stage_distinct_quotas_gradients_and_rank_rng_replay(tmp_path):
    context = mp.spawn(_worker, args=(str(tmp_path / 'store'), str(tmp_path)), nprocs=2, join=False)
    started = time.monotonic()
    try:
        while not context.join(timeout=1):
            if time.monotonic() - started > 120:
                pytest.fail('two-rank three-stage training exceeded 120 seconds')
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    first = torch.load(tmp_path / 'three_rank_0.pt', weights_only=True)
    second = torch.load(tmp_path / 'three_rank_1.pt', weights_only=True)
    assert first['budgets'] != second['budgets']
    assert first['state']['visual_fusion.beta_step'].item() == 2
    for name, value in first['state'].items():
        torch.testing.assert_close(value, second['state'][name], rtol=0, atol=0)


def _accumulation_worker(rank, store_path, output_dir):
    from types import SimpleNamespace
    from efficient_vlm_ad.pipeline import _SingleProcessAccelerator, _advance_query_update
    from efficient_vlm_ad.training_accumulation import window_size, window_boundary, loss_for_backward
    from efficient_vlm_ad.query_training import qaware_beta_horizon
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=Path(store_path).as_uri(), rank=rank, world_size=2)
    try:
        torch.manual_seed(42)
        model = torch.nn.parallel.DistributedDataParallel(_tiny_vlm())
        reference = _tiny_vlm()
        reference.load_state_dict(model.module.state_dict(), strict=True)
        cfg = SimpleNamespace(
            model=SimpleNamespace(fusion=SimpleNamespace(name='mvpruner_three_stage_qaware')),
            training=SimpleNamespace(finetune_epochs=2, gradient_accumulation_steps=2,
                                     beta_decay_updates=None))
        horizon = qaware_beta_horizon(cfg, 3)
        assert horizon == 4
        for instance in (model.module, reference):
            instance.visual_fusion.configure_beta(horizon, reset=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.01)
        accelerator = _SingleProcessAccelerator(torch.device('cpu'), gradient_accumulation_steps=2)
        snapshots = []
        optimizer_step = 0

        def batch(other_rank, index):
            generator = torch.Generator().manual_seed(100 + 10 * other_rank + index)
            return dict(input_ids=torch.tensor([[2, 3 + index, 0]]),
                        attention_mask=torch.tensor([[1, 1, 0]]),
                        visual_features=torch.randn(1, 2, 4, 16, generator=generator),
                        labels=torch.tensor([[4 + other_rank, 5 + index, 1]]))

        def seed(epoch, other_rank, index):
            torch.manual_seed(500 + epoch * 100 + other_rank * 10 + index)

        for epoch in range(2):
            for index in range(3):
                size = window_size(index, 3, 2)
                seed(epoch, rank, index)
                raw = model(**batch(rank, index)).loss
                assert torch.isfinite(raw)
                accelerator.backward(loss_for_backward(raw, 2, size))
                assert int(model.module.visual_fusion.beta_step) == optimizer_step
                for name, parameter in model.named_parameters():
                    assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
                    gradients = [torch.empty_like(parameter.grad) for _ in range(2)]
                    dist.all_gather(gradients, parameter.grad)
                    torch.testing.assert_close(gradients[0], gradients[1], rtol=0, atol=0)
                if window_boundary(index, 3, 2):
                    # Explicit mean over both ranks and every microbatch in this
                    # window is the reference for both full and tail windows.
                    for other_rank in range(2):
                        for micro in range(index // 2 * 2, index + 1):
                            seed(epoch, other_rank, micro)
                            loss = reference(**batch(other_rank, micro)).loss
                            (loss / (2 * size)).backward()
                    for (name, actual), (_, expected) in zip(
                            model.module.named_parameters(), reference.named_parameters()):
                        torch.testing.assert_close(actual.grad, expected.grad, rtol=3e-4, atol=3e-6, msg=name)
                    optimizer.step()
                    reference_optimizer.step()
                    optimizer_step = _advance_query_update(cfg, model.module.visual_fusion,
                                                           optimizer_step, skipped=False)
                    reference.visual_fusion.step_beta()
                    assert int(model.module.visual_fusion.beta_step) == optimizer_step
                    optimizer.zero_grad(set_to_none=True)
                    reference_optimizer.zero_grad(set_to_none=True)
            assert optimizer_step == 2 * (epoch + 1)
            for name, actual in model.module.state_dict().items():
                torch.testing.assert_close(actual, reference.state_dict()[name], rtol=3e-4, atol=3e-6, msg=name)
            snapshots.append({name: value.detach().clone() for name, value in model.module.state_dict().items()})
        torch.save(snapshots, Path(output_dir) / f'three_accum_rank_{rank}.pt')
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason='CPU Gloo unavailable')
def test_three_stage_two_ranks_full_and_tail_accumulation_across_epochs(tmp_path):
    context = mp.spawn(_accumulation_worker, args=(str(tmp_path / 'accum_store'), str(tmp_path)),
                       nprocs=2, join=False)
    started = time.monotonic()
    try:
        while not context.join(timeout=1):
            if time.monotonic() - started > 120:
                pytest.fail('two-rank three-stage accumulation exceeded 120 seconds')
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    first = torch.load(tmp_path / 'three_accum_rank_0.pt', weights_only=True)
    second = torch.load(tmp_path / 'three_accum_rank_1.pt', weights_only=True)
    assert len(first) == len(second) == 2
    for epoch in range(2):
        assert first[epoch]['visual_fusion.beta_step'].item() == 2 * (epoch + 1)
        assert first[epoch]['visual_fusion.beta_target_steps'].item() == 4
        for name, value in first[epoch].items():
            torch.testing.assert_close(value, second[epoch][name], rtol=0, atol=0)
