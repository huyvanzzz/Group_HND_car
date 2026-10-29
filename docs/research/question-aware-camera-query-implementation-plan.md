# Plan đã rà soát: Stage 2 theo câu hỏi với Gumbel + Prefix-DP/STE

## 1. Kết luận rà soát và thiết kế cố định

Plan trước **đúng về công thức hard/soft DP và bridge STE**, nhưng chưa đủ chặt ở các điểm: hook beta trong training, ý nghĩa gradient của entropy, sai số FP32, candidate pool, và checkpoint/resume. Bản này thay thế toàn bộ plan trước.

Giữ luồng:

```text
6 camera → DINOv2 frozen
→ Stage 1 nguyên trạng: 384 → 196
→ T5 blocks 0–1
→ Cross-attention câu hỏi–ảnh + scorer từng token
→ Trung bình trọng số theo camera
→ Gumbel + Prefix-DP/STE phân bổ đúng 49 query
→ Camera-query compression có bias +0.1 × s
→ T5 blocks 2–3 → decoder
```

Các quyết định đã chốt:

- **Một bank chung `[44,64]`**, camera nhận \(m_c\) query luôn dùng prefix `0 … m_c−1`.
- Không thêm bank riêng, camera-specific query offset hoặc chọn hàng query tùy ý.
- Chấp nhận compression ứng viên phụ khi train; chỉ đưa **49 visual tokens** vào hai block T5 cuối.
- **γ = 0.1 cố định**, không học hoặc schedule.
- Giữ entropy trên compression \(P\), hệ số **0.5**; không thêm entropy loss trên \(\alpha\).
- Finetune-only từ pretrained T5; không align hoặc warm-start từ profile khác.
- MVPruner, camera-query hiện tại và unpruned phải giữ nguyên hành vi, config và notebook.

Cơ sở phương pháp là [structured stochastic relaxation](https://arxiv.org/pdf/2006.08063) và [differentiable dynamic programming](https://proceedings.mlr.press/v80/mensch18a.html). **Prefix-DP và utility dưới đây là thiết kế riêng cho repository**, chưa có bằng chứng tốt hơn trên DriveLM.

Kiểm chứng trong lần rà soát này:

- Bridge tiết kiệm memory có forward cứng và gradient khớp reference trên bài toán nhỏ.
- Soft selection ở kích thước sáu camera/49 query hữu hạn trong probe FP32; tổng hàng có sai số đến khoảng \(6\times10^{-5}\).
- Đây là kiểm chứng toán và số học, chưa phải kiểm chứng implementation đầy đủ hoặc hiệu năng GPU.

## 2. Scorer theo câu hỏi và camera shares

### Đầu vào và cross-attention

Lấy hidden states sau T5 block 1, tức sau hai block đầu:

\[
X\in\mathbb R^{B\times196\times384},
\qquad
T\in\mathbb R^{B\times L_q\times384}
\]

\(T\) là prompt/question tokens đang dùng trong pipeline. Tách bằng ranh giới sequence hiện có; camera IDs đi theo tokens qua Stage 1. Không sử dụng labels hoặc decoder states.

Chuẩn hóa RMS không affine:

\[
\operatorname{RMS}(x)
=x\left(\operatorname{mean}(x^2)+10^{-6}\right)^{-1/2}
\]

Dùng cross-attention chuẩn:

- `embed_dim=384`, `num_heads=4`, `bias=False`, `dropout=0`.
- Query từ \(\operatorname{RMS}(T)\); key/value từ \(\operatorname{RMS}(X)\).
- Production không yêu cầu trả attention weights.
- Không thêm positional/camera embeddings mới trong module này.

\[
M=\operatorname{MHA}
\left(\operatorname{RMS}(T),
      \operatorname{RMS}(X),
      \operatorname{RMS}(X)\right)
\]

\[
m_b=
\frac{\sum_t \mathrm{mask}_{bt}M_{bt}}
     {\sum_t\mathrm{mask}_{bt}}
\]

Question padding không tham gia mean; mẫu không có token hợp lệ bị từ chối.

### Scorer dùng chung

\[
h_{bj}=\operatorname{ReLU}
\left(
W_1[\operatorname{RMS}(X_{bj});
    \operatorname{RMS}(m_b)]+b_1
\right)
\]

\[
s_{bj}=w_2^\top h_{bj},
\qquad
\alpha_{bj}=\operatorname{softmax}_j(s_{bj})
\]

MLP: **768 → 64 → 1**; linear cuối không bias.

Khởi tạo:

- Attention projections và linear đầu: Xavier uniform, gain 1.
- Bias linear đầu: 0.
- Linear cuối: Normal, std 0.02.
- Giữ query bank Normal std 1, key projection std \(1/\sqrt{384}\), omega bằng 1.
- Không gọi initialization lên pretrained T5.

### Trung bình theo camera

\[
r_{bc}=\frac{1}{N_{bc}}
\sum_{j:C_{bj}=c}\alpha_{bj},
\qquad
p_{bc}=\frac{r_{bc}}{\sum_v r_{bv}}
\]

Tính đường allocation trong log-space, tránh lấy log của xác suất đã underflow:

\[
a_{bc}
=\operatorname{logsumexp}_{j:C_{bj}=c}s_{bj}
-\log N_{bc}
\]

\[
\ell_{bc}=\operatorname{logsoftmax}_c(a_{bc})
=\log p_{bc}
\]

Hai cách tính phải được đối chiếu bằng test. Không dùng min–max normalization hoặc hiệu chỉnh IRA trong profile mới.

Scorer và DP chạy FP32 trong production. Backend toán DP vẫn nhận FP64 để kiểm chứng numerical gradient trực tiếp.

## 3. Prefix-DP, STE và compression

### Không gian allocation và utility

Với mỗi mẫu:

\[
1\le m_c\le u_c=\min(N_c,44),
\qquad
\sum_{c=0}^{5}m_c=49
\]

Định nghĩa:

\[
U_c(k)=k\ell_c-\log(k!),
\qquad k=1,\ldots,u_c
\]

Prior factorial tạo lợi ích biên giảm dần. Khi không có noise và \(\tau=1\), phân phối mềm tương ứng với multinomial được điều kiện hóa bởi minimum/capacity; **không có nghĩa hard budget hoặc expected budget luôn bằng chính xác \(49p_c\)**.

Train dùng:

\[
\widetilde U_c(k)=U_c(k)+\beta G_{c,k},
\qquad G_{c,k}\sim\mathrm{Gumbel}(0,1)
\]

- Sinh một noise tensor `[B,6,44]` mỗi train forward khi \(\beta>0\).
- Hard và soft paths dùng cùng tensor noise.
- Chỉ state \(1\ldots u_c\) hợp lệ.
- Eval/generation không sinh noise.
- Đây là perturb-and-MAP có cấu trúc; không khẳng định phân phối hard sampler chính là phân phối Gibbs của soft-DP.

Trước DP, trừ max utility hợp lệ của từng camera dưới `stopgrad`. Đây là phép đổi mốc số học, không đổi nghiệm hoặc phân phối.

### Hard-DP và soft-DP

Hard allocation:

\[
m^*=\arg\max_{m\ \mathrm{hợp\ lệ}}
\sum_c\widetilde U_c(m_c)
\]

Dùng max-DP theo số camera và tổng query, với backpointers. Tie-break: \(k\) tăng dần, chọn max đầu tiên. Từ chối capacity không khả thi.

Soft relaxation, **\(\tau=1.0\) cố định**:

\[
q_\tau(m)
\propto
\exp\left(\sum_c\widetilde U_c(m_c)/\tau\right)
\]

Soft-DP phải chạy trên cùng tập allocation hợp lệ. Đặt \(F_c(t)\) là log-weight cấp tổng \(t\) query cho các camera trước \(c\), và \(B_c(t)\) cho phần từ camera \(c\) trở đi:

\[
F_{c+1}(t)=
\operatorname{logsumexp}_{k}
\left(F_c(t-k)+\widetilde U_c(k)/\tau\right)
\]

\[
B_c(t)=
\operatorname{logsumexp}_{k}
\left(\widetilde U_c(k)/\tau+B_{c+1}(t-k)\right)
\]

Khởi tạo \(F_0(0)=B_6(0)=0\); trạng thái khác không hợp lệ. \(\log Z=F_6(49)\).

Không lặp Python theo mẫu, output position hoặc budget; chỉ lặp theo sáu camera, kể cả backtracking.

### Ma trận lựa chọn prefix

Candidate được định danh bằng cặp `(camera c, query row r)`, flatten thành `c*44+r`. Tensor có 264 slot padding; candidate hợp lệ thỏa \(r<u_c\).

Với allocation \(m\), \(S(m)\) đóng gói query theo camera rồi query row:

\[
S(m)\in\{0,1\}^{49\times264},
\qquad
S_{\mathrm{soft}}=\mathbb E_{q_\tau}[S(m)]
\]

Với output position \(i=0,\ldots,48\):

\[
S^{\mathrm{soft}}_{i,c,r}
=
\sum_{k=r+1}^{u_c}
\exp\left(
F_c(i-r)+
\widetilde U_c(k)/\tau+
B_{c+1}(49-i+r-k)
-\log Z
\right)
\]

Index/state không hợp lệ đóng góp 0. Dùng posterior bảng `(previous total, camera count)` và suffix cumulative sums để tính tensor này; không tạo tensor liệt kê allocation.

Yêu cầu số học:

- Masked log-sum-exp phải xử lý riêng hàng không có state hợp lệ để tránh NaN backward.
- Kiểm tra finite, trọng số không âm và tổng hàng trước normalization.
- Nếu sai số tổng hàng vượt **\(10^{-3}\)** hoặc tổng hàng không dương, báo lỗi số học.
- Sau khi kiểm tra, chuẩn hóa hàng để loại sai số FP32 nhỏ.
- Lưu `soft_row_mass_error` trước normalization; không dùng normalization để che sai thuật toán.
- FP64 phải khớp oracle exhaustive ở tolerance \(10^{-10}\).

Soft-DP là surrogate conditional theo noise đã lấy. STE dùng gradient surrogate có bias, không phải gradient chính xác của quyết định hard.

### Compression và bridge tiết kiệm memory

Giữ keys và values như compressor hiện tại:

\[
K_j=\operatorname{RMS}(X_j)W_k
\]

\[
P_{c,r,j}
=\operatorname{softmax}_{j:C_j=c}
\left(
\frac{(Q_r\odot\omega)^\top K_j}{\sqrt{64}}
+0.1s_j
\right)
\]

\[
Z_{c,r}=\sum_{j:C_j=c}P_{c,r,j}X_j
\]

Values là \(X\) gốc; không thêm projection hoặc camera embedding sau nén. Noise allocation không đi vào compression logits.

Train:

1. Scorer và soft-DP giữ graph.
2. Tính candidate outputs dưới `no_grad`, dùng detached values/keys/scores/query parameters; candidate không hợp lệ được đặt 0.
3. Tính đúng 49 query hard-selected với autograd.
4. Tạo:

\[
Y_s=S_{\mathrm{soft}}Z_{\mathrm{candidate}}^{\mathrm{detach}}
\]

\[
Y=Y_h+(Y_s-\operatorname{stopgrad}(Y_s))
\]

Đặt phép trừ trong ngoặc trước khi cộng để giữ hard forward khi giá trị hữu hạn.

Candidate path không sinh RNG, cập nhật counter hoặc ghi đè diagnostic tensors của active path. Có thể group nguồn theo camera để tránh nhân toàn bộ 264 queries với tất cả 196 keys.

**Giới hạn cần ghi đúng:** chỉ 49 query compression giữ activation graph, nhưng scorer/DP vẫn có graph riêng. Pool phụ vẫn tốn thời gian và memory. Query bank rows không active không nhận direct compression gradient; Gumbel không bảo đảm mọi row được sử dụng.

Eval/generation bỏ candidate pool và soft-DP, chỉ tạo 49 query hard-selected.

### Loss và T5

\[
L_{\mathrm{train}}
=L_{\mathrm{answer}}(Y)
+0.5H(P_{\mathrm{active}})
\]

\[
H(P_{\mathrm{active}})
=
-\frac1{B\cdot49}
\sum_{b,i,j}
P_{bij}\log\max(P_{bij},10^{-8})
\]

- Entropy chỉ dùng 49 query active và hard membership.
- Không đưa entropy qua soft allocation hoặc lấy trung bình trên pool.
- Đường STE allocation trực tiếp nhận gradient từ answer CE. Entropy vẫn cập nhật scorer qua bias trong active compression.
- Không thêm loss trên \(\alpha\), rate penalty, warmup allocation đặc biệt hoặc distillation.

Ghép `[49 visual; question]`, giữ question states/mask và tính lại T5 relative bias cho sequence mới; padding mask cộng đúng một lần. Validation/best checkpoint dùng answer CE. Generation encode một lần, không tính entropy.

## 4. Interfaces, training và artifacts

### Phân nhánh an toàn

Fusion mới: **`mvpruner_camera_query_qaware`**.

Thêm fusion class riêng kế thừa camera-query và entrypoint nội bộ:

```text
compress_from_question(
    values, question_states, question_mask, camera_ids,
    training, compute_entropy
) → CameraQueryResult
```

Giữ VLM forward/generation interfaces và output loss hiện có.

- T5 split encoder dispatch profile mới trước nhánh camera-query cũ.
- Profile mới không gọi `_visual_relevance`, IRA hoặc raw T5 attention reconstruction của Stage 2 cũ.
- Giữ các debug fields `stage2_budgets`, `query_camera_ids`, `query_slots`, `query_usage`; không giả lập source `selected_indices`.
- Mở rộng predicate camera-query family cho accumulation/resume, nhưng có predicate riêng cho learned allocation.
- Không dựa chỉ vào `hasattr(compressor)` để quyết định schedule hoặc benchmark identity.

### Config và train policy

Thêm `fusion.question_scoring` với defaults:

```yaml
num_heads: 4
hidden_dim: 64
gamma: 0.1
rms_eps: 1.0e-6
```

Dùng các trường fusion hiện có cho `beta_start: 0.2`, `beta_end: 0.0`, `beta_schedule: linear`, `tau: 1.0`. Thêm `training.beta_decay_updates: null`, chỉ hỗ trợ override này cho profile mới.

Profile chạy chính giữ:

- Sáu camera, 64 tokens/camera; Stage 1 = 196, Stage 2 = 49, minimum = 1.
- DINO/T5 revisions và preprocessing hiện tại; runtime FP32, feature-cache.
- Finetune 10 epochs; batch 16/process, accumulation 2, hai GPU.
- AdamW lr \(10^{-4}\), weight decay 0.05; scheduler gamma 0.9/epoch; clipping 1.0.
- T5, projector, embeddings, scorer và compressor được finetune; DINO frozen/offline.
- Gradient checkpointing `false`; bật cho profile mới báo chưa hỗ trợ.

Backend scorer/DP hỗ trợ kích thước nhỏ cho offline tests; số hàng bank được suy ra từ tổng budget/số view/minimum, không hard-code 44 trong backend.

### Schedule và resume

**Phải thêm hook riêng cho profile mới**, vì pipeline hiện tại bỏ qua beta update cho camera-query.

Fresh-run horizon:

\[
H=
\begin{cases}
\text{beta\_decay\_updates},&\text{nếu override}\\
\text{finetune\_epochs}\,
\left\lceil
\frac{\operatorname{len}(\text{prepared loader})}
{\text{accumulation}}
\right\rceil,&\text{nếu auto}
\end{cases}
\]

Tính sau `accelerator.prepare`. Debug step limit không thay horizon khai báo này.

Với \(t\) optimizer updates đã hoàn thành:

\[
\beta(t)=0.2\left(1-\min\left(\frac{t}{\max(1,H-1)},1\right)\right)
\]

- Advance đúng một lần sau optimizer update thành công, không theo microbatch hoặc world size.
- Diagnose/verify/eval/generation không advance.
- Reuse beta buffers/methods hiện có trong fusion; không tạo counter thứ hai độc lập.
- Checkpoint phải có `beta_step == optimizer_step` cho profile mới.
- Resume giữ \(H\) đã lưu. Override explicit khác \(H\) bị từ chối; `null` dùng stored horizon.
- Giữ tail accumulation và epoch scheduler của camera-query hiện tại.

Model init seed 42; data sampler seed 42. Fresh train đặt process RNG theo `42 + rank` sau preparation, trước iterator. Resume khôi phục rank RNG ở cùng vị trí và không reseed.

### Checkpoint, notebook và benchmark

- Signature mới ghi scorer, gamma, geometry và `qaware_prefix_dp_v1`; beta/tau/horizon là training recipe/state.
- Không thêm các trường mới không hoạt động vào signature/recipe của profile cũ.
- Recipe mới nhận diện initialization, RNG policy và loop version riêng; optimizer/scheduler/RNG sidecars giữ đủ.
- Chỉ resume cùng profile tại epoch boundary và cùng world size. Model-only `best_model.pt` dành cho inference.
- Giữ ngoại lệ thiếu vision weights của loader dùng chung.
- Debug forward phụ phải bảo toàn RNG để không âm thầm đổi noise/dropout của forward được diagnose.

Tạo riêng:

- `configs/dinov2_t5_efficient_mini_kaggle_2gpu_camera_query_qaware.yaml`
- `notebooks/kaggle_train_dinov2_camera_query_qaware_finetune.ipynb`
- `docs/research/question-aware-camera-query.md`

Notebook giữ allocator env vars trước Torch/CUDA, dependency pins hiện tại, runtime YAML riêng, finetune-only, cache validation, verify-resume và archive đầy đủ sidecars.

Benchmark bắt đầu từ features và bao gồm scorer, hard-DP, compression cùng T5/decoder. Eval benchmark không chạy surrogate pool. FLOPs được ghi là major-operator estimate, nêu rõ DP/indexing/log-sum-exp không được counter bao phủ đầy đủ.

## 5. Kiểm thử và tiêu chí bàn giao

Thực hiện theo thứ tự: **scorer/DP → bridge/compression → VLM → lifecycle → notebook/docs → regression**.

### Toán và gradient

- Scorer với batch/padding khác nhau; thay labels không thay scores/budget khi inputs và RNG giữ nguyên.
- Hoán vị visual tokens cùng camera IDs tại đầu vào scorer phải hoán vị token scores tương ứng.
- Log-space shares khớp mean \(\alpha\).
- Hard-DP và soft selection đối chiếu exhaustive oracle trên bài nhỏ, gồm ties, skewed shares, minimum/capacity và trường hợp chỉ còn một allocation.
- Mọi hard output đúng tổng, prefix, capacity và canonical order.
- Soft rows, expected budgets và prefix inclusion marginals đúng; unique-optimum soft solution tiến về hard solution khi giảm tau.
- FP64 `gradcheck` cho soft-DP; không numerical-gradcheck STE.
- FP32 stress test ở sáu camera/49 query, shares lệch mạnh và states không khả thi.
- Tắt gamma/entropy: answer CE vẫn tạo allocation gradient trên dữ liệu synthetic phân biệt được các camera.
- Detach soft selection: allocation gradient biến mất.
- Bridge tối ưu khớp gradient reference cho scorer, source states, query bank, keys và omega.
- Không yêu cầu gradient nonzero khi allocation bị forced hoặc mọi candidate output giống nhau.

### Integration và lifecycle

- Stage 1 không đổi; profile mới không gọi selector/relevance Stage 2 cũ.
- Hard forward không trộn nguồn camera; block 2–3 luôn nhận `49 + Lq`.
- Entropy chỉ lấy active queries; kiểm tra riêng đường CE-allocation và entropy-compression.
- Candidate path detached và không ghi đè diagnostics; eval không chạy pool/soft-DP hoặc tiêu thụ Gumbel RNG.
- Generation encode một lần; position bias/padding/loss đúng.
- Schedule đúng với accumulation tail, skipped update và resume; profile cũ không bắt đầu update beta ngoài ý muốn.
- Continuous/resume replay dùng **cùng horizon explicit**, dropout, shuffle và noise; đối chiếu weights, optimizer, scheduler, counters và RNG.
- Verify-resume không sửa checkpoint/sidecars nguồn; checkpoint sai profile, partial epoch hoặc thiếu rank sidecar bị từ chối.
- Smoke offline toàn lifecycle; DDP CPU hai rank có budget khác nhau; notebook syntax/artifact tests.
- Chạy toàn suite, regression ba profile cũ và `git diff --check`.

Diagnostics ghi compact camera shares, hard/expected budgets, beta/tau/gamma, soft-row error, scorer scale, alpha entropy metric, active \(P\) entropy, query overlap, usage và gradients. Không lưu graph hoặc dump ma trận lớn trong log thường.

**Hoàn thành code** khi các hợp đồng/offline tests pass và profile cũ không đổi hành vi. Nếu chưa có CUDA, ghi rõ chưa xác minh VRAM, latency, full training và replay hai GPU.

**Chất lượng tốt hơn** chỉ được kết luận từ ablation cùng protocol: camera-query hiện tại, profile mới γ=0 và profile đầy đủ γ=0.1. Báo cáo riêng overhead train của candidate pool; không dùng kết quả toán/CPU để khẳng định tăng chất lượng hoặc tốc độ GPU.
