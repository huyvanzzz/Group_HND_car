# Thay selector Stage 2 bằng camera-query compression — kế hoạch đã rà soát

## 1. Mục tiêu và các điểm đã sửa trong plan

Giữ nguyên Stage 1, question relevance và IRA; chỉ thay Top-K/Gumbel/STE bằng learned-query compression. Chạy **finetune-only từ pretrained T5**, entropy weight **0.5**, có config và notebook Kaggle riêng.

```text
6 camera × 64 tokens = 384
→ Stage 1 DRA/CCTS: 196
→ T5 encoder block 0–1 cùng prompt
→ question relevance + IRA: budget từng camera
→ camera-query compression: tổng 49 tokens
→ 49 tokens mới + prompt
→ T5 encoder block 2–3 → decoder
```

Rà soát code cho thấy plan trước thiếu những điểm cần khóa rõ:

- Phải thay contract `selected_indices` và diagnostics; token tổng hợp không có source index duy nhất.
- T5 bias trả về từ block đầu đã chứa padding mask; không được cộng lại mask qua mỗi block.
- Train loop hiện tại chia loss trước `Accelerator.backward()`, trong khi Accelerate cũng chia theo accumulation. Probe CPU đã xác nhận gradient bị scale thêm.
- Accelerated scheduler mặc định có thể gọi scheduler nhiều lần theo world size, không đúng ý nghĩa “gamma mỗi epoch”.
- Code hiện chủ yếu checkpoint cuối epoch; không được hứa resume chính xác giữa epoch chỉ vì đã lưu batch cursor.
- Profiler hiện chỉ nhận tên Stage 2 cũ, cần nhận diện compression và bias mới.

**Theo lựa chọn đã xác nhận:** sửa accumulation/scheduler cho profile query mới; giữ nguyên hành vi MVPruner và unpruned hiện tại. Những khác biệt training này phải được ghi khi so chất lượng.

Các lựa chọn nghiên cứu vẫn là giả thuyết cần đo: learned queries thay pooled queries, shared query bank, khởi tạo và position reassignment. Không coi kết quả MetaCompress là bảo đảm cho DriveLM.

## 2. Bộ nén: công thức, khởi tạo và implementation

### 2.1. Đầu vào và budget

Sau block encoder thứ hai:

\[
X\in\mathbb R^{B\times196\times384},\qquad
T\in\mathbb R^{B\times L_q\times384}
\]

Giữ camera IDs \(C\) của các token được Stage 1 chọn.

Tách phần tính budget hiện tại thành helper dùng chung, giữ đúng công thức và tie-break:

\[
m=\operatorname{IRA}_{old}(r,C),\qquad
\sum_c m_{bc}=49,\qquad
1\le m_{bc}\le N_{bc}
\]

IRA vẫn rời rạc, không có gradient. Không thêm Gumbel hoặc STE để chọn số query.

### 2.2. Tham số và initialization

| Thành phần | Shape/profile thật | Khởi tạo |
|---|---|---|
| Query bank dùng chung | \(44\times64\) | IID \(\mathcal N(0,1)\) |
| Key projection, không bias | \(384\rightarrow64\) | Std \(1/\sqrt{384}\) |
| \(\omega\), học được | \(64\) | Toàn bộ bằng 1 |
| RMS normalization | Chiều 384 | Không affine, epsilon \(10^{-6}\) |

Bank size tổng quát:

\[
Q_{\max}=K_{\text{out}}-(V-1)m_{\min}
\]

Với profile này:

\[
Q_{\max}=49-5=44
\]

Seed mặc định **42**, đặt trước khi tạo model. Chỉ khởi tạo compressor mới; không gọi initialization toàn model làm ghi đè pretrained T5. Resume nạp nguyên weights đã học, không khởi tạo lại queries.

Cơ sở lựa chọn:

- [BLIP-2](https://github.com/salesforce/LAVIS/blob/main/lavis/models/blip2_models/blip2.py) dùng Gaussian std theo BERT nhưng còn có LayerNorm/Q-Former.
- [OpenFlamingo](https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py) dùng latents Normal(0,1) cùng normalization trước attention.

Không có bằng chứng std 1 tối ưu cho bộ nén này; đây là mặc định cụ thể có kiểm tra numerics. RMS cho keys đã được người dùng xác nhận.

### 2.3. Forward tính chung cả sáu camera

Camera \(c\) nhận:

\[
Q_{bc}=Q_{\text{bank}}[:m_{bc}]
\]

Dùng cumulative budgets/searchsorted để tạo:

- \(Q:[B,49,64]\).
- \(G:[B,49]\): camera ID của mỗi query.
- `query_slot:[B,49]`: hàng tương ứng trong query bank.

Không lặp theo từng mẫu hoặc từng query.

\[
\widehat X=
\frac{X}{\sqrt{\operatorname{Mean}_{d}(X^2)+10^{-6}}}
,\qquad K=\widehat XW_k
\]

\[
S=\frac{(Q\odot\omega)K^\top}{\sqrt{64}}
\]

\[
M_{bij}=
\begin{cases}
0&G_{bi}=C_{bj}\\
-\infty&G_{bi}\ne C_{bj}
\end{cases}
\]

\[
P=\operatorname{softmax}_{j}(S+M)
\in\mathbb R^{B\times49\times196}
,\qquad
Z=PX\in\mathbb R^{B\times49\times384}
\]

Quy định implementation:

- RMS, scoring, softmax và entropy tính FP32.
- Chỉ normalized features đi vào key projection; **values vẫn là X gốc**.
- Tính key projection một lần trên toàn bộ nguồn.
- Không nhân mask 0/1 vào logits thay cho additive mask.
- Camera ID là metadata; không cộng camera embedding lần hai.
- Không thêm ΔQ, value/output projection, FFN hoặc dropout của compressor.
- Queries khác nhau ngay từ initialization; train/eval cùng công thức compression.
- Không bypass thành identity khi budget bằng số nguồn: profile query luôn có semantics compression.

Các query cuối bank có thể ít active. Log usage theo slot; không yêu cầu mọi hàng đều có gradient khác 0 trong mỗi batch.

### 2.4. Tổ chức module

Tạo `src/efficient_vlm_ad/modeling/camera_query.py` chứa:

- `CameraQueryCompressor`: nhận values, camera IDs và budgets.
- `CameraQueryFusion`: tái sử dụng Stage 1 của `MVPrunerTwoStage`, gọi budget helper chung và compressor.

Fusion mới giữ tương thích các type guards hiện có bằng kế thừa `MVPrunerTwoStage`, nhưng có đường `compress_from_scores` riêng. Không thay kiểu trả về của `stage2_from_scores` cũ.

Kết quả compression gồm:

```text
visual_tokens       [B,49,384]
visual_camera_ids   [B,49]
budgets             [B,6]
entropy_loss        scalar hoặc None
debug               dữ liệu diagnostics
```

Không trả `selected_indices` giả. Split encoder phân nhánh selector theo fusion mới; đường MVPruner cũ giữ nguyên.

## 3. T5, entropy và contracts dùng chung

### 3.1. Position bias và padding

Sau compression:

\[
H=[Z;T],\qquad S_{\text{new}}=49+L_q
\]

Thứ tự output là camera chuẩn, sau đó query slot. Dùng lại bảng relative bias của **encoder block đầu** để tính:

\[
B_{\text{rel,new}}
=\operatorname{compute\_bias}(S_{\text{new}},S_{\text{new}})
\]

Mask mới:

```text
[49 visual positions hợp lệ | prompt mask cũ]
```

Tạo extended padding mask bằng helper T5 rồi cộng:

\[
B_{\text{combined}}=B_{\text{rel,new}}+M_{\text{pad,new}}
\]

Truyền `B_combined` cho block 2, tái sử dụng bias trả về cho block 3:

- Không dùng bias cũ đã chứa mask.
- Không cộng mask lại ở từng block.
- Không đặt bias thành `None` và giả định block 2 tự có bảng relative bias.
- `cache_position` theo sequence mới.
- Giữ final norm, dropout và decoder hiện có.

Đây là vị trí của latent slots, không phải tọa độ ảnh trung bình. Các token được phép tương tác chéo camera trong hai block T5 cuối; giới hạn cùng camera chỉ áp dụng tại compressor.

### 3.2. Entropy và output model

\[
L_{\text{ent}}
=-\frac1{49B}\sum_{b,i,j}
P_{bij}\log\big(\max(P_{bij},10^{-8})\big)
\]

\[
L_{\text{train}}=L_{\text{answer}}+0.5L_{\text{ent}}
\]

Dùng natural log; không chuẩn hóa thêm theo số camera hoặc \(\log N_c\).

- Mở rộng kết quả split encoder bằng auxiliary losses mặc định rỗng.
- Giữ entropy tensor có graph riêng; không lấy nó từ `last_debug` đã detach.
- Giữ các interface `forward`, `forward_from_features`, `forward_debug`, `generate_from_features`.
- Output profile mới bổ sung `answer_loss`, `entropy_loss`; `.loss` là total khi training có labels, answer loss khi eval có labels, `None` khi không có labels.
- `no_grad()` không quyết định objective; dùng `model.training` để train/debug có semantics nhất quán.
- Generation không tính entropy hoặc chạy lại compressor ở từng decoder token.
- Validation và best checkpoint dựa trên answer loss.

Không thêm anti-collapse hoặc distillation. Theo dõi entropy, max probability và mức trùng output để phát hiện collapse; không tự bật regularizer khác.

### 3.3. Diagnostics và profiler

Giữ diagnostics cũ cho MVPruner. Profile mới báo:

- Budget và camera/slot của query.
- Answer loss, entropy, weighted entropy, total loss.
- Query usage, logit scale, xác suất lớn nhất và output norms.
- Gradient của queries, keys, omega, values, projector và T5.

Không đòi gradient qua relevance/IRA hoặc gradient khác 0 ở inactive query rows. Khi mọi camera chỉ có một nguồn, zero scoring gradients là trường hợp suy biến hợp lệ.

Không giữ graph trong debug cache giữa steps; chỉ lưu đầy đủ \(P\) khi diagnostics yêu cầu.

Thêm profiler ranges riêng cho allocation, compression và bias adaptation; cập nhật bộ tổng hợp để không bỏ sót event mới.

## 4. Training, checkpoint và notebook

### 4.1. Profile và cấu hình

Fusion mới: `mvpruner_camera_query`. Config/output/cache/notebook dùng hậu tố `_camera_query`.

Config mới khai báo query dimension 64, init std 1, RMS epsilon \(10^{-6}\), position policy `new_sequence`, entropy weight 0.5 và seed 42.

Mặc định giữ:

- DINOv2/T5 revisions và preprocessing của profile MVPruner.
- Feature-cache, FP32.
- Batch 16/process, accumulation 2, hai GPU: batch hiệu dụng danh nghĩa 64.
- Finetune 10 epochs; AdamW lr \(10^{-4}\), weight decay 0.05.
- ExponentialLR gamma 0.9 mỗi epoch; clipping 1.0.
- Generation và evaluation settings hiện tại.
- Gradient checkpointing `false`; bật cho profile mới báo chưa hỗ trợ trong phiên bản này.

Không nạp checkpoint align, MVPruner hoặc unpruned để warm-start. Resume chỉ nhận checkpoint finetune cùng kiến trúc mới.

### 4.2. Accumulation và scheduler: sửa riêng profile mới

Dùng manual accumulation với một nơi duy nhất thực hiện scaling:

- Accelerator của profile mới có `gradient_accumulation_steps=1`.
- Cấu hình ứng dụng vẫn là accumulation 2.
- Chia total loss cho số microbatches thực tế trong cửa sổ hiện tại rồi gọi `accelerator.backward`.
- Step/clip/zero-grad ở cuối cửa sổ.
- Cửa sổ cuối epoch thiếu microbatch được chuẩn hóa theo kích thước thực tế.
- Reset cửa sổ tại đầu epoch; không dùng modulo của global microbatch counter xuyên epoch.
- Counter `optimizer_step` chỉ tăng khi optimizer thực sự step.

Đặt `step_scheduler_with_optimizer=False` cho profile mới, gọi scheduler một lần cuối epoch để tránh nhân số scheduler steps theo world size.

MVPruner/unpruned cũ tiếp tục dùng hành vi hiện tại. Ghi `training_loop_version` trong metadata mới. Không quy toàn bộ chênh lệch chất lượng với checkpoint cũ cho selector khi training semantics khác nhau.

### 4.3. Checkpoint và resume

Architecture signature mới ghi fusion, Stage 1 version, query dimensions/bank policy, RMS và position policy. Không đưa fields mới vào signature lịch sử của hai profile cũ.

Initialization recipe, entropy coefficient, seed và training-loop version thuộc training metadata. Resume yêu cầu chúng khớp; changing output paths hoặc tổng epochs không làm đổi kiến trúc.

Hỗ trợ **resume tại ranh giới epoch**, gồm:

- Model, optimizer, scheduler và optimizer-step counter.
- Completed epochs và global microbatch step.
- RNG sidecars từng rank và world-size validation.
- Seedable sampler, data seed 42; đặt epoch trước khi tạo iterator.

Khôi phục RNG từng rank sau khi hoàn tất model/data-loader/Accelerator preparation, trước iterator train tiếp theo. Không reset queries hoặc seed lại sau restore.

Không thêm checkpoint giữa epoch. Checkpoint do debug step-limit dừng giữa epoch được đánh dấu partial và từ chối training resume, tránh âm thầm bỏ qua dữ liệu.

`verify-resume` là probe trên model tạm; không sửa checkpoint hoặc RNG sidecars nguồn. Với checkpoint hai GPU, kiểm tra đủ manifest/sidecars; probe một process không được báo đã xác minh replay DDP.

Giữ nguyên ngoại lệ loader end-to-end chỉ cho thiếu vision weights; không nới loader dùng chung cho thiếu compressor weights.

### 4.4. Notebook riêng

Notebook mới:

```text
Allocator/environment
→ dependency và hai GPU
→ runtime YAML riêng
→ prepared/cache validation
→ diagnose forward/backward
→ fresh finetune hoặc verify + resume
→ evaluate/benchmark
→ archive
```

- Đặt cả allocator variables thành `expandable_segments:True` trước Torch/CUDA.
- Pin Transformers 4.46.3, Tokenizers 0.20.4 và Accelerate 1.15.0 trong notebook mới.
- Seedable sampler dùng API đã kiểm tra của Accelerate này; không đổi dependency/runtime profile cũ.
- Không có lệnh align hoặc warm-start checkpoint cũ.
- Cache reuse phải khớp prepared identity và manifest hiện tại.
- Không sửa YAML nguồn khi tạo runtime config.
- Archive gồm checkpoint, sidecars, config runtime, logs và metrics của profile mới.

## 5. Thứ tự thực hiện, tests và tiêu chí hoàn thành

Thực hiện theo bốn chặng, mỗi chặng có tests trước khi sang chặng kế tiếp:

1. Config, budget helper và compressor độc lập.
2. Split T5, bias/mask, model output và generation.
3. Objective, diagnostics, training-loop mới và checkpoint/resume.
4. Notebook, benchmark integration và tài liệu.

Các tests bắt buộc:

| Nhóm | Kiểm chứng |
|---|---|
| Bảo toàn | Stage 1 và IRA budgets khớp cũ; selector/bias/output/checkpoint MVPruner cũ không đổi |
| Khởi tạo | Seed tái lập; đúng shapes/std/omega; queries không giống nhau; không ghi đè pretrained weights |
| Compression | Vector hóa khớp bản tham chiếu từng camera; tổng 49; đúng camera mask; hàng \(P\) tổng 1 |
| Numerics | Zero input, input tăng scale, camera một nguồn, budgets khác nhau; không NaN/Inf |
| Entropy | Uniform cho \(\log N_c\), one-hot cho 0; tổng loss đúng hệ số 0.5; không tính masked sources |
| T5 | Đủ `49+Lq`; giữ prompt; bias khớp `compute_bias` sequence mới; padding mask cộng đúng một lần |
| Gradient | Finite gradients qua compressor và model; không đòi gradient qua budget hoặc inactive rows |
| Accumulation | So với gradient trung bình trực tiếp; cửa sổ đầy/thiếu; nhiều epoch; Accelerator và CPU fallback |
| Scheduler | Mỗi epoch đúng một lần gamma; không nhân theo world size |
| Resume | Round-trip weights/state; counter đúng; partial/wrong-profile bị từ chối; resume epoch tiếp theo khớp run liên tục |
| Generation | Encoder/compressor chạy một lần; decoder cache/generation settings giữ đúng |
| DDP | Budgets khác nhau giữa rank không lỗi; state/counter nhất quán; kiểm tra RNG và sampler sau resume |

Smoke offline dùng tiny T5 thật, không tải model:

```text
prepare → features → diagnose → finetune đủ epoch
→ verify-resume → resume epoch tiếp
→ evaluate → benchmark
```

Dùng dropout bật trong test resume để phát hiện sai RNG. Trong test parity deterministic, tắt dropout và dùng cùng weights.

Chạy toàn bộ test suite, kiểm tra notebook JSON/Python syntax, artifact allowlists và `git diff --check`. Cập nhật tài liệu bằng công thức cuối cùng, khởi tạo, loss, phạm vi resume và hướng dẫn benchmark.

Benchmark cùng batch/precision/generation protocol, đo cả packing, projection, compression và bias adaptation; giữ phạm vi từ features, không cộng DINO extraction offline vào một phía.

Hoàn thành phần implementation khi lifecycle và regression pass. Chỉ kết luận chất lượng/tốc độ tốt hơn sau thí nghiệm thực tế; môi trường hiện tại chưa có CUDA nên VRAM, latency và DDP hai GPU vẫn cần xác minh riêng.
