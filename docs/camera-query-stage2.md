# Camera-query Stage 2

Profile `mvpruner_camera_query` giữ Stage 1 DRA/CCTS 196 tokens, question relevance `shifted_max_cosine` và IRA như MVPruner. Sáu camera × 64 tokens DINOv2 (384 chiều) đi qua projector, Stage 1, T5 blocks 0–1, rồi được nén thành 49 tokens trước blocks 2–3 và decoder. Camera ID chỉ là metadata để mask; không cộng camera embedding mới.

## Công thức và khởi tạo

IRA cấp budget `m_bc`, tổng 49, mỗi camera ít nhất 1 và không vượt số nguồn. Query bank dùng chung giữa các camera: `Q_bank ∈ R^(44×64)`, IID `Normal(0,1)` với seed 42; `44 = 49 − (6−1)×1`. Mỗi camera lấy prefix `Q_bank[:m_bc]`. `W_k:384→64`, `bias=False`, khởi tạo Normal std `1/sqrt(384)`; vector omega 64 chiều khởi tạo toàn 1. Chỉ khởi tạo bộ nén mới, không khởi tạo lại pretrained T5; resume giữ queries đã học.

Với `X:[B,196,384]`, tính FP32:

```text
X_hat = X / sqrt(mean_d(X²) + 1e-6)
K = X_hat W_k
S = ((Q ⊙ omega) Kᵀ) / sqrt(64)
M[b,i,j] = 0 nếu query_camera[b,i] == source_camera[b,j], ngược lại −inf
P = softmax_source(S + M)       # [B,49,196], mỗi hàng tổng 1
Z = P X                       # [B,49,384], values X gốc
```

RMS không affine chỉ áp dụng cho keys; values không normalize. Key projection chạy một lần trên toàn bộ nguồn. Tính sáu camera theo batch bằng mask cộng, không lặp từng query/mẫu. Query bank chia sẻ tham số nhưng không trộn nguồn chéo camera trong compressor. Hai block T5 cuối vẫn có thể tương tác chéo camera. Không Gumbel, STE, ΔQ, value/output projection, MLP, distillation hay regularizer anti-collapse; budget vẫn rời rạc, không yêu cầu gradient qua IRA.

Output theo thứ tự camera chuẩn rồi query slot. Token tổng hợp không có một source index duy nhất: không giả lập `selected_indices`. Khi budget bằng số nguồn vẫn dùng compression, không bypass identity.

## Vị trí T5 và loss

Ghép `[Z;prompt]` thành sequence mới dài `49+Lq`, giữ nguyên prompt và prompt mask. Dùng bảng relative bias của encoder block 0, gọi `compute_bias(49+Lq,49+Lq)`, tạo padding mask mới `[49 positions hợp lệ | prompt mask]`, cộng đúng một lần. Block 2 nhận bias mới; block 3 dùng bias trả về, không cộng mask lại. `cache_position` cũng theo sequence mới. Vị trí mới biểu diễn latent slots, không phải tọa độ ảnh trung bình.

```text
entropy = −sum_bij P[b,i,j] log(max(P[b,i,j],1e-8)) / (B×49)
train_loss = answer_CE + 0.5 × entropy
validation_loss = answer_CE
```

Dùng natural log, không chia thêm cho camera hay `log N_c`. Entropy giữ graph riêng; debug cache chỉ giữ tensors detach. `.loss` khi train có labels là tổng loss, khi eval có labels là CE, không labels là None. Generation chạy encoder/compressor một lần. Validation/best checkpoint dựa trên CE.

Finetune toàn bộ T5, embeddings, projector và compressor; DINOv2 frozen với `vision_training: feature_cache`. Theo dõi answer/entropy/weighted entropy/total loss, budget, query slot usage, logit scale, max probability, output norms, output trùng nhau và gradients queries/keys/omega/values/projector/T5. Các hàng cuối bank có thể ít active (cold rows do prefix); không đòi mọi hàng có gradient mỗi batch. Camera một nguồn có scoring gradient bằng 0 là hợp lệ. Entropy và output trùng nhau giúp phát hiện collapse; không tự thêm regularizer.

## Chạy Kaggle

Mở [notebook riêng](../notebooks/kaggle_train_dinov2_camera_query_finetune.ipynb), chọn hai GPU, branch `huy-dinov2`, đặt secret `HF_TOKEN` và các HF repo tùy chọn. [Config riêng](../configs/dinov2_t5_efficient_mini_kaggle_2gpu_camera_query.yaml) giữ đúng revisions DINO/T5, preprocessing và geometry MVPruner; output/cache mang hậu tố `_camera_query`.

Cell code đầu đặt cả `PYTORCH_ALLOC_CONF` và `PYTORCH_CUDA_ALLOC_CONF` thành `expandable_segments:True` trước Torch/CUDA. Chỉ notebook mới pin Transformers 4.46.3, Tokenizers 0.20.4 và Accelerate 1.15.0. Pip/CLI dùng Python của kernel. Gradient checkpointing là false; true chưa hỗ trợ, không có toggle sau OOM.

Pipeline: environment/dependencies → runtime YAML riêng (không ghi đè config nguồn) → chuẩn bị/tải data/cache và validate → diagnose forward/backward → verify và resume cùng profile hoặc finetune mới → evaluate/benchmark → archive. Không chạy align hoặc warm-start từ checkpoint align/MVPruner/unpruned. Không checkpoint cùng profile thì tải pretrained T5 mới.

Prepared repo và cache repo phải tách biệt, đặt files ở root. Prepared gồm `manifest.json`, `train.jsonl`, `val.jsonl`, `test.jsonl`; phải là full subset và counts train 341381, val 19785, test 16817. Cache gồm `manifest.json`, `index.json`, `features.float16.memmap`. Kiểm tra manifest v2, feature shape `[N,6,64,384]`, revisions, preprocessing, đúng hashes prepared/index, fingerprint, coverage và byte length. Chỉ reuse khi identity khớp chính xác; không sửa manifest để ép hợp lệ. Thiếu cache và không có cache repo thì phải materialize raw images trước extraction; prepared JSON máy khác không thay thế raw images.

## Training và resume

Mặc định FP32, batch 16/process, application accumulation 2, hai GPU (batch hiệu dụng danh nghĩa 64), 10 epochs, AdamW lr 1e-4, weight decay .05, max grad norm 1.0, gamma .9 mỗi epoch.

Chỉ profile mới dùng `Accelerator(gradient_accumulation_steps=1)` và application accumulation 2. Chia total loss cho số microbatches thực tế của mỗi cửa sổ, kể cả cửa sổ thiếu cuối epoch; reset cửa sổ mỗi epoch, clip/step/zero ở cuối cửa sổ. Counter optimizer_step chỉ tăng khi optimizer thực sự step. `step_scheduler_with_optimizer=False`; scheduler gọi đúng một lần cuối epoch, không nhân theo world size. Metadata ghi training_loop_version. MVPruner/unpruned cũ giữ scaling cũ; không quy toàn bộ khác biệt chất lượng cho selector khi so checkpoint cũ.

Resume chỉ tại ranh giới epoch, gồm weights, optimizer, scheduler, counters, completed epochs và RNG sidecars từng rank. Checkpoint debug dừng giữa epoch bị đánh dấu partial và từ chối resume train. Phải cùng architecture signature và training metadata (init recipe, seed, entropy coefficient, training-loop version); đổi output path hoặc tổng epochs không thay kiến trúc. Seedable sampler data seed 42, đặt epoch trước iterator. Restore RNG sau preparation model/data/Accelerator, trước iterator tiếp theo; không seed lại hoặc reset queries sau restore.

Checkpoint local `finetune_latest.pt` ưu tiên hơn HF. HF model repo phải chứa checkpoint cùng `<filename>.rng/manifest.json`, `rank_00000.pt`, `rank_00001.pt`; world size phải là 2. Notebook chạy `verify-resume-checkpoint` trước launch. Probe này không sửa nguồn và không chứng minh replay DDP bằng một process. Kiểm tra thực nghiệm hai GPU vẫn cần chạy riêng.

Alias `best_model.pt` chỉ dành cho inference; dùng `finetune_best.pt` hoặc `finetune_latest.pt` cùng thư mục RNG tương ứng để resume.

Archive chứa runtime YAML, checkpoints/model weights và RNG sidecars, logs/debug, metrics, benchmark và predictions. Giữ archive đủ sidecars để resume.

## Benchmark và giới hạn

CLI sau khi cài package trong môi trường đúng dependency:

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CONFIG=configs/dinov2_t5_efficient_mini_kaggle_2gpu_camera_query.yaml
python -m efficient_vlm_ad diagnose-train --config "$CONFIG" --stage finetune --debug --debug-numerics
python -m accelerate.commands.launch --multi_gpu --num_processes 2 --mixed_precision no -m efficient_vlm_ad train --config "$CONFIG" --stage finetune
```

So cùng batch, precision và generation protocol với MVPruner/unpruned, trong phạm vi từ features: gồm packing, projector, allocation/compression, bias adaptation, encoder và decoder. Không cộng DINO extraction offline vào riêng một phía. Chưa có kết quả CUDA để khẳng định VRAM, latency, DDP hoặc chất lượng tốt hơn. Query initialization, shared bank và position reassignment là giả thuyết cần đo.

## Kết quả kiểm chứng implementation — 2026-10-29

- Toàn bộ suite offline: `python -m pytest -q` — **399 passed, 3 skipped** (624.83 giây). Môi trường CPU, Transformers 4.46.3, Accelerate 1.15.0.
- Sau các sửa cuối về diagnostics và API tải checkpoint notebook: `python -m pytest tests/test_camera_query.py tests/test_mvpruner_diagnostics.py tests/test_camera_query_notebook.py -q` — **31 passed**. Lượt này bao phủ thêm test notebook được thêm sau khi suite tổng đã bắt đầu thu thập tests.
- Đã kiểm tra lifecycle CLI offline từ prepare data/features đến finetune, verify-resume, resume, evaluate và benchmark; kiểm tra replay liên tục so với resume tại ranh giới epoch trên CPU, gồm dropout và shuffle.
- DDP hai process Gloo trên CPU kiểm tra compressor với budget khác nhau giữa các rank. Đây không phải kiểm chứng toàn bộ training/resume hai GPU.
- Notebook JSON/Python syntax hợp lệ, outputs được xóa; `git diff --check` không có lỗi whitespace. Các config/notebook MVPruner và unpruned hiện có được giữ nguyên.

Chưa xác minh CUDA, mức VRAM, latency, chất lượng sau full training hoặc replay toàn bộ training trên hai GPU. Các kết quả CPU không chứng minh những số đo này.

[Kế hoạch đã duyệt](superpowers/plans/2026-10-29-camera-query-stage2.md) giữ nguyên nội dung attachment UTF-8.
