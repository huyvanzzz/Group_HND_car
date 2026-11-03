# Question-aware spatial pooled-query compression

Profile mới **`mvpruner_camera_pool_qaware`** finetune-only từ pretrained T5, bỏ Stage 1, giữ 384 visual tokens qua hai block đầu rồi nén thành 46 token. Trọng số entropy mặc định **0.0**, backward chỉ dùng CE. Chạy notebook `notebooks/kaggle_train_dinov2_camera_pool_qaware_finetune.ipynb`. Quota `[16, 9, 9, 4, 4, 4]` gán theo hạng camera từ scorer câu hỏi; pooling **2D** tạo query. Không dùng learned query bank, Gumbel/DP/STE hoặc beta schedule. Các profile cũ giữ nguyên.

The `mvpruner_camera_pool_qaware` profile is a finetune-only experiment with frozen DINOv2 features. Six cameras supply an 8 by 8 grid each: 384 visual tokens. Stage 1 is disabled. Embeddings and the projector follow the existing preprocessing order; visual tokens and the complete masked prompt pass through T5 blocks 0 and 1 exactly once.

The existing question-conditioned scorer computes RMS-normalized question cross-attention and token scores `s`. Camera scores are `logsumexp(s_c) - log(64)` and their softmax gives camera shares. Stable descending hard ranking assigns quotas `[16, 9, 9, 4, 4, 4]`, with ties resolved by original camera order. Only rank matters; outputs are packed in original camera order and always contain 46 tokens.

For each camera, let `N_c = RMS(X_c)` with epsilon `1e-6`. Reshape the 64 slots into their 8 by 8 grid and use PyTorch adaptive average pooling to a square grid of side `sqrt(m_c)`. Flatten in row-major order to obtain `Z_c`. The 8 to 3 pooling windows overlap. Queries may attend to all 64 slots in their camera:

```text
Q_c = Z_c W_q
K_c = N_c W_k
P_c = softmax(((Q_c * omega) K_c^T) / sqrt(64) + gamma s_c^T)
Y_c = P_c X_c
```

`gamma = 0.1`; the query/key dimension is 64. Values use the original, unnormalized hidden states. Bias-free projections are shared across cameras and quotas. Initialization copies one Gaussian matrix with standard deviation `1/sqrt(64)` into two independently trainable projection parameters; omega starts at ones. There is no value/output projection, residual, or new embedding after compression. Grouped pooling handles B, 2B, and 3B cameras for quotas 16, 9, and 4 without per-sample loops or CPU quota transfers.

The remaining T5 blocks consume 46 visual tokens plus all prompt states and the original prompt padding mask. The `new_sequence` policy builds relative position bias for the shortened sequence from the pretrained block 0 table; no source-index gather is used. Subsequent T5 attention is unrestricted. Camera-local compression limits direct source slots, but those hidden states already contain cross-camera and prompt information. It does not recover purely local image features.

The loss is `CE + entropy_weight * H(P)`, where entropy averages over the 46 output rows and batch. The supplied config sets `entropy_weight: 0.0`, so backward uses CE directly. Entropy is observed during training/diagnose and omitted during generation/benchmark. No scorer entropy, collapse loss, or auxiliary loss is added. Integer hard ranking has no direct gradient; positive gamma supplies a differentiable CE path through the score bias. Sharpness and output-token similarity are observations of possible collapse, not evidence that collapse is prevented.

Use `configs/dinov2_t5_efficient_mini_kaggle_2gpu_camera_pool_qaware.yaml` and `notebooks/kaggle_train_dinov2_camera_pool_qaware_finetune.ipynb`. The protocol retains FP32, feature cache, batch 16 per process, accumulation 2, two GPUs (nominal effective batch 64), 10 epochs, learning rate 1e-4, seed 42, and gradient checkpointing false. Generation settings and pinned model revisions match the existing question-aware recipe. Both allocator environment variables are set before Torch/CUDA. Runtime YAML is separate from the tracked config and entropy weight has one source in that config.

The notebook validates prepared counts, cache identity and complete feature coverage, then runs diagnose, accumulation verification, finetune/resume, evaluation, benchmark, and archive. Prepared data, cache, checkpoints, and output share the new profile directory. `FINETUNE_CHECKPOINT` selection retains local-latest priority and optional Hugging Face checkpoint downloads with both rank RNG sidecars. Resume validation requires a same-profile completed epoch; debug checkpoints may not be resumable. Archives retain checkpoints, rank RNG sidecars, runtime config, metrics, predictions and diagnostics. Existing profiles and artifacts are unchanged.

Diagnose should show Stage 1 disabled, 384 to 46 tokens, pooling/position policy, camera scores/ranks/quotas in camera order, score range, gamma, compression entropy and sharpness, output similarity, weighted entropy, CE/total loss and finite gradients to scorer, W_q, W_k, omega, projector, embeddings and T5. Under the supplied config, weighted entropy is zero and total loss equals answer loss. Accumulation diagnostics report configured/backend accumulation and successful optimizer updates. No learned query bank, Gumbel, DP, STE or active beta schedule participates.

Benchmark starts from cached features and includes scorer, ranking, pooling and compression; DINO extraction is excluded. CPU tests cannot establish GPU latency, VRAM, training quality, or two-GPU replay. This combines ideas for the current task rather than claiming an exact reproduction of a paper. Reproduction and comparison require separate experiments.

Research sources: [MetaCompress](https://arxiv.org/html/2603.21701v1), [TokenFLEX](https://arxiv.org/html/2504.03154v1), [Oryx implementation](https://github.com/Oryx-mllm/Oryx/blob/main/oryx/model/multimodal_resampler/perceiver.py). MetaCompress describes average pooling in the main method but fractional max pooling in its appendix; this profile explicitly uses adaptive average pooling 2D, without the paper's distillation/collapse objective.

## Kết quả kiểm chứng triển khai

- **59 tests riêng pass**: core, schema/signature, artifacts, tiny-T5 integration, hai rank CPU/Gloo và smoke CLI offline. Smoke chạy prepare-data → prepare-features → diagnose → accumulation verify → finetune → verify-resume → resume → evaluate → benchmark.
- Khi entropy weight bằng 0, CE vẫn truyền gradient hữu hạn tới scorer, W_q/W_k/omega. Grouped implementation khớp reference vòng lặp về output và gradient. Các đường Stage 1, scorer T5 gốc và beta được chặn trong regression test.
- Resume ở epoch boundary với dropout khớp chạy liên tục về trọng số, optimizer, scheduler và RNG trên cả Accelerate và CPU fallback. Nhóm accumulation dư được kiểm tra; hai rank đồng bộ gradient/model state dù thứ hạng camera khác nhau.
- So riêng cả **bảy profile cũ** với HEAD trước sửa: signature, state dict, eval/train logits, loss và gradients khớp tuyệt đối trên dữ liệu giả cùng seed.
- Test artifact inventory được mở rộng để nhận config/notebook mới; expected signature cũ vẫn loại trường pooled-query bất hoạt. Không sửa config/notebook cũ để làm suite xanh.
- Notebook Python/Bash syntax và compile Python được kiểm tra. Không tải model/dataset lớn, không chạy full training hoặc commit/push. Chưa xác minh CUDA, latency, VRAM và chất lượng mô hình.
- Đối chiếu bản HEAD trước sửa: **18 lỗi nền** vẫn tái hiện khi chạy với đúng PYTHONPATH. Gồm 12 tests còn đòi cơ chế RESUME_TRAINING cũ; 2 tests qaware còn đòi entropy 0.5 trong khi config là 0.0; 4 tests về config/signature/protocol còn lệch với MVPruner tracked đang dùng 64 thay vì 49 token. Không sửa các hành vi/config đó trong đợt này.
- Lượt toàn suite: **758 passed, 3 skipped, 18 failed**. Tập 18 test thất bại khớp chính xác baseline trước sửa; không có failure mới. Lượt kiểm tra riêng cuối cùng của profile mới có **59 passed**, bao gồm regression metadata benchmark bổ sung sau khi toàn suite đã bắt đầu collect.
