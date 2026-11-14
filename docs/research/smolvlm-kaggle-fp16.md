# SmolVLM Kaggle FP16 camera-order experiment

The native backend uses the existing pinned SmolVLM-256M pretrained revision and the existing output directory. Training and validation use AMP FP16 with FP32 parameters and optimizer state, non-reentrant gradient checkpointing, batch 6 per rank, accumulation 8, and two GPUs (nominal effective batch 96). Two epochs are required; each notebook session runs at most one. Training uses two workers per rank, pinned memory, and persistent workers only when workers are nonzero. Both allocator environment variables are set before importing Torch.

The prompt order is Back-Right, Back-Left, Back, Front-Right, Front-Left, Front. The collator pairs each camera name with its corresponding features once. Canonical data order, prepared data, stored cache dtype, cache identity, processor/cache policy and pretrained revision are unchanged. Existing cache validation and publishing/Input flows remain authoritative. Old canonical checkpoints cannot resume this reversed-camera experiment; an incompatible checkpoint is an error, never an implicit fresh start.

Run the notebook in order. After cache publication/readiness, single-process FP16 diagnose and accumulation verification run first. Two fresh two-rank processes then run `diagnose-train --stress --stress-phase train` and `diagnose-train --stress --stress-phase evaluate` with the same runtime YAML. The train probe includes validation with live optimizer states; the evaluate probe forces FP32 and uses the configured beam count of 3. The notebook records successful diagnose and accumulation checks separately; rerunning either clears downstream approval, and manually running stress cannot bypass them. These disposable probes write no real training checkpoint. They consume GPU time and must succeed before training; a nonzero exit aborts the cell without retrying OOM or changing batch settings. CPU tests do not establish T4 memory fit or throughput.

Resume precedence remains local latest, configured local filename, then Hugging Face. Restore latest/best checkpoints together with all matching `.rng` directories. The native validator checks compatibility and complete state, including FP16 scaler state, before launch. No `RESUME_TRAINING` switch is needed. Training launches through Accelerate with `--mixed_precision fp16`; validation uses FP16 batch 16 per GPU.

After both epochs, evaluation launches a new two-rank process with `--mixed_precision no`, FP32 model/features and batch 16 per GPU. It selects best, falling back to latest. Benchmark launches a standalone fresh native process, forces FP32, and uses batch 1 with 100 samples from cached features. Timing excludes vision extraction and retains the native token/timing definitions. LR 1e-4, weight decay 0.05, epoch scheduler gamma 0.9, clipping 1.0, seed 42, native CE and generation settings remain unchanged.

The artifact archive keeps the existing allowlist, including checkpoints, RNG sidecars, runtime YAML, logs and results. Gzip compression level 1 reduces packaging time; feature cache and raw images are excluded. Offline artifact tests validate notebook syntax, settings, launch modes, preflight gating, resume precedence and archive contents. Actual Kaggle GPU stress success remains a runtime requirement.

## Hướng dẫn kiểm tra

- Dùng notebook SmolVLM hiện tại, chạy theo thứ tự các cell. Cache Kaggle đã có vẫn dùng được; không extract lại hoặc sửa manifest.
- Kiểm tra debug có `precision: fp16`, `feature_input_dtype: float16`, `parameter_dtype: float32`, đúng `prompt_camera_order`, checkpointing active, accumulation config/backend bằng 8 và effective batch 96 khi train hai rank.
- Diagnose ghi `autocast_active`, `loss_dtype`, `gradient_norms_unscaled` và `grad_scaler_scale`. Mọi `optimizer_skipped: true` đều được log, kể cả ngoài interval định kỳ; successful optimizer counter không tăng cho bước đó.
- `debug/stress_train.json` và `debug/stress_evaluate.json` ghi các source indices khác nhau, chiều dài thực và allocated/reserved/free VRAM từng rank. Stress train có hai cửa sổ đủ và một cửa sổ dư, validation giữ optimizer states trong bộ nhớ. Không có checkpoint thật được ghi từ stress.
- Khi chuyển phiên, giữ cả `.pt` và toàn bộ thư mục `.pt.rng`. Validator kiểm tra optimizer/scheduler/scaler trước khi phục hồi, không tạo bản sao model để kiểm tra state.
- Train vẫn dùng trung bình native CE của microbatch; validation dùng tổng CE theo target tokens trên hai rank. Không đổi objective training thành loss của batch ghép.
- Evaluate/benchmark dùng process FP32 mới. Câu sinh chỉ được `.strip()` khoảng trắng ngoài. Định nghĩa `average_generation_ms_per_output_token`, prefill/FLOPs và phạm vi benchmark từ features giữ nguyên.
- Các script Vast cũ là snapshot cho commit nguồn được ghi trong README riêng, không phải launcher của notebook mới.

## Kiểm chứng offline

Các test dùng model/tokenizer nhỏ thật, bao phủ camera/processor parity, cache bất biến, scaler/overflow, accumulation đủ/dư, no-sync, CPU/Gloo với shard rỗng, checkpoint/RNG resume và lifecycle CLI. Môi trường kiểm tra: Torch `2.14.1+cpu`, Transformers `4.46.3`, Accelerate `1.15.0`, dùng các dependency đã có, không tải model/dataset lớn.

CPU không xác nhận T4 đủ VRAM, tốc độ hay chất lượng DriveLM. Hai phép stress trên Kaggle vẫn phải pass trước khi train. Chưa commit/push hoặc chạy full training.

Kết quả ngày 2026-11-14: full suite **1.152 passed, 4 skipped**, 428,89 giây; notebook syntax và `git diff --check` pass. Các cảnh báo trong lượt chạy là pin memory trên CPU và fixture beam 1 có `early_stopping=True`. Kiểm tra bổ sung xác nhận log checkpointing active của stress đúng với mode training và mọi skipped update ngoài interval cũng được ghi.

## Verify accumulation FP16 khi GPU có dao động số học

Log Kaggle thực tế cho thấy ngay cả reference backward chạy lại với cùng RNG cũng vượt `allclose` cũ: relative L2 khoảng 0,0275% ở cửa sổ đủ và 0,0510% ở nhóm dư; backend lệch khoảng 0,0216% và 0,0566%. Vì vậy `verify-accumulation` của SmolVLM FP16 dùng marker `comparison_implementation: fp16_replay_noise_v1`:

- Mỗi cửa sổ có reference, replay reference và backend; cùng microbatches, RNG và GradScaler scale. Không có optimizer step hoặc scaler update; không tạo thêm model hoặc giữ thêm một bản gradient đầy đủ.
- Giữ `mismatched_tensor_count` và các chênh lệch elementwise cũ làm diagnostics. Điều kiện pass dùng relative L2, cosine và norm ratio ở cả mức toàn mô hình và từng tensor, để tensor lớn không che lỗi của tensor nhỏ.
- Giới hạn relative L2 là `min(0.002, max(0.0001, 3 * replay_relative_L2))`; norm ratio lệch tối đa 0,002, cosine tối thiểu 0,999998. Reference replay cũng phải ổn định. Tensor có gradient gần zero vẫn được xét bằng absolute tolerance cũ. Đây là guardrail của phép chẩn đoán, không phải ngưỡng được paper bảo đảm về chất lượng model.
- Gradient thiếu/NaN/Inf, reference không ổn định, sai scaling hoặc sai tensor vượt giới hạn đều fail. JSON có thêm `reference_repeat`, `relative_L2`, `cosine_similarity`, `relative_L2_limit`, `failed_parameters` và `failed_parameter_diagnostics`.
- Nhánh FP32 giữ nguyên phép so cũ. Không sửa training, loss, config, notebook, checkpoint hay cache. Verify FP16 thêm một lượt reference backward, nên tốn thời gian hơn phép kiểm tra trước.

Sau khi cập nhật source trên Kaggle, chạy lại cell verify; chỉ khi pass mới tiếp tục các cell stress và train. Không bỏ gate hoặc tự tăng giới hạn nếu còn fail; xem diagnostics từng tensor. Chưa có phép chạy CUDA local để xác nhận gate mới trên T4.

Kiểm chứng bản sửa verify: **125 tests passed** (50,09 giây), bao gồm replay noise có kiểm soát, chia thừa 8 lần, sai tensor nhỏ, reference không ổn định, NaN/Inf, native model có dropout/checkpointing, FP32 accumulation và lifecycle resume; `git diff --check` pass. Không commit/push.

Follow-up từ JSON Kaggle: toàn mô hình pass về sai lệch, nhưng một `layers.22.self_attn.k_proj.weight` bị loại do replay cục bộ tình cờ rất nhỏ (relative L2 0,00002574), tạo limit 0,0001 trong khi backend lệch 0,00011114. Rule tensor được sửa thành `min(0.002, max(0.0001, 3*global_replay_L2, 3*local_replay_L2))`: dùng dao động hệ thống làm sàn, không tăng trần 0,2% hoặc bỏ kiểm tra từng tensor. JSON thêm `tensor_relative_L2_floor` và policy `tensor_noise_floor: global_reference_replay`; diagnostics tensor fail ghi limit riêng. Không thêm lượt backward so với bản replay trước.
