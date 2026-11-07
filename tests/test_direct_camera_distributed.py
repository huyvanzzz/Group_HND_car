"""Real native-CE direct-camera training with full and tail DDP windows."""
from pathlib import Path
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _tiny_vlm():
    from transformers import T5Config, T5ForConditionalGeneration
    from efficient_vlm_ad.modeling.mvpruner_direct_camera import MVPrunerDirectCamera
    from efficient_vlm_ad.modeling.vlm import EfficientVLMForAD
    text = T5ForConditionalGeneration(T5Config(
        vocab_size=32, d_model=16, d_kv=4, d_ff=32, num_layers=4,
        num_decoder_layers=2, num_heads=4, dropout_rate=.1,
        decoder_start_token_id=0, pad_token_id=0, eos_token_id=1))
    fusion = MVPrunerDirectCamera(
        d_model=16, num_views=2, tokens_per_view=4, stage1_top_k=6,
        stage2_top_k=4, beta_start=.2, beta_end=.1)
    fusion.configure_beta(2)
    return EfficientVLMForAD(text_model=text, visual_fusion=fusion,
                            vision_dim=16, d_model=16, seq_len=4, projector_type='linear')


def _accumulation_worker(rank, store_path, output_dir):
    from efficient_vlm_ad.pipeline import _SingleProcessAccelerator
    from efficient_vlm_ad.training_accumulation import window_size, window_boundary, loss_for_backward
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=Path(store_path).as_uri(), rank=rank, world_size=2)
    try:
        torch.manual_seed(42)
        model = torch.nn.parallel.DistributedDataParallel(_tiny_vlm())
        reference = _tiny_vlm()
        reference.load_state_dict(model.module.state_dict(), strict=True)
        horizon = 4
        assert horizon == 4
        for instance in (model.module, reference):
            instance.visual_fusion.configure_beta(horizon, reset=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.01)
        accelerator = _SingleProcessAccelerator(torch.device('cpu'), gradient_accumulation_steps=2)
        snapshots = []
        camera_counts = []
        optimizer_step = 0

        def batch(other_rank, index):
            generator = torch.Generator().manual_seed(100 + 10 * other_rank + index)
            visual = torch.randn(1, 2, 4, 16, generator=generator) * 100
            repeated_camera = (other_rank + index) % 2
            visual[:, repeated_camera] = visual[:, repeated_camera, :1]
            return dict(input_ids=torch.tensor([[2, 3 + index, 0]]),
                        attention_mask=torch.tensor([[1, 1, 0]]),
                        visual_features=visual,
                        labels=torch.tensor([[4 + other_rank, 5 + index, 1]]))

        def seed(epoch, other_rank, index):
            torch.manual_seed(500 + epoch * 100 + other_rank * 10 + index)

        for epoch in range(2):
            for index in range(3):
                size = window_size(index, 3, 2)
                seed(epoch, rank, index)
                output = model(**batch(rank, index))
                raw = output.loss
                camera_counts.append(model.module.visual_fusion.last_debug["camera_counts"].tolist())
                expected_ce = torch.nn.functional.cross_entropy(output.logits.flatten(0, 1),
                    batch(rank, index)["labels"].flatten())
                torch.testing.assert_close(raw, expected_ce)
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
                    optimizer_step += 1
                    model.module.visual_fusion.step_beta()
                    reference.visual_fusion.step_beta()
                    assert int(model.module.visual_fusion.beta_step) == optimizer_step
                    optimizer.zero_grad(set_to_none=True)
                    reference_optimizer.zero_grad(set_to_none=True)
            assert optimizer_step == 2 * (epoch + 1)
            for name, actual in model.module.state_dict().items():
                torch.testing.assert_close(actual, reference.state_dict()[name], rtol=3e-4, atol=3e-6, msg=name)
            snapshots.append({name: value.detach().clone() for name, value in model.module.state_dict().items()})
        assert len({tuple(row) for counts in camera_counts for row in counts}) > 1
        torch.save(snapshots, Path(output_dir) / f'direct_accum_rank_{rank}.pt')
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason='CPU Gloo unavailable')
def test_direct_camera_two_ranks_full_and_tail_accumulation_across_epochs(tmp_path):
    context = mp.spawn(_accumulation_worker, args=(str(tmp_path / 'accum_store'), str(tmp_path)),
                       nprocs=2, join=False)
    started = time.monotonic()
    try:
        while not context.join(timeout=1):
            if time.monotonic() - started > 120:
                pytest.fail('two-rank direct-camera accumulation exceeded 120 seconds')
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    first = torch.load(tmp_path / 'direct_accum_rank_0.pt', weights_only=True)
    second = torch.load(tmp_path / 'direct_accum_rank_1.pt', weights_only=True)
    assert len(first) == len(second) == 2
    for epoch in range(2):
        assert first[epoch]['visual_fusion.beta_step'].item() == 2 * (epoch + 1)
        assert first[epoch]['visual_fusion.beta_target_steps'].item() == 4
        for name, value in first[epoch].items():
            torch.testing.assert_close(value, second[epoch][name], rtol=0, atol=0)
