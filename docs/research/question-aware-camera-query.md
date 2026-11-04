# Question-aware camera-query Stage 2

This experimental profile adds question-conditioned camera allocation after the first two T5 blocks. The frozen DINO feature pipeline and Stage 1 remain unchanged: six cameras provide 384 tokens, Stage 1 retains 196, and Stage 2 emits exactly 49 visual tokens before the final two T5 blocks. The [approved implementation plan](question-aware-camera-query-implementation-plan.md) is retained verbatim as the design record.

Use `configs/dinov2_t5_efficient_mini_kaggle_2gpu_camera_query_qaware.yaml` with `notebooks/kaggle_train_dinov2_camera_query_qaware_finetune.ipynb`. Fusion identity is `mvpruner_camera_query_qaware`; its architecture version is `qaware_prefix_dp_v1`. Existing camera-query, MVPruner and unpruned artifacts retain their own identities.

## Decoder CE ổn định số học

Riêng fusion `mvpruner_camera_query_qaware`, answer loss dùng target-excluded logsumexp và softplus:

\[
L_t=\operatorname{softplus}\left(
\operatorname{LogSumExp}_{j\ne y_t}(z_j-z_{y_t})\right).
\]

Đây vẫn là hard-target CE, đổi cách tính số học theo [A Fix for Softmax and Cross-Entropy](https://arxiv.org/abs/2610.31642). Loss lấy trung bình trên các label hợp lệ, bỏ `-100`; teacher forcing giữ nguyên T5 `_shift_right`. Forward và forward-debug dùng chung implementation, không tính CE native dư thừa. Tổng loss vẫn ghép entropy theo trọng số config hiện có; không sửa scorer, softmax compression, allocation, beta hoặc hai stage.

Không cần thêm/sửa config. Các profile khác, kể cả `mvpruner_camera_pool_qaware`, tiếp tục dùng CE hiện tại. Generation không gọi fixed CE. Debug numerics ghi `answer_loss_implementation: target_excluded_softplus_v1` để nhận biết implementation.

Signature, state-dict keys và điều kiện resume không đổi; checkpoint cùng profile vẫn nạp theo validator hiện có. Khi tiếp tục train bằng code mới, phép tính CE mới được áp dụng, không cam kết tái hiện bitwise trajectory của lượt dùng CE cũ. Chưa có bằng chứng cải thiện metric DriveLM hoặc số đo GPU/VRAM.

## Question scorer

Let X be the retained visual states and T the prompt/question states after T5 block 1. No labels or decoder states enter scoring. Define non-affine RMS(x) = x / sqrt(mean(x**2) + 1e-6). Four-head, bias-free cross-attention uses RMS(T) as queries and RMS(X) as keys/values, without dropout. Its question-mask-weighted mean m excludes question padding; an empty valid question is rejected.

A shared 768 -> 64 -> 1 ReLU MLP scores concatenated [RMS(X_j); RMS(m)]. The last linear layer has no bias. Let s_j be scores and alpha = softmax(s). Camera shares average alpha over that camera, then normalize across cameras. Allocation computes the equivalent stable log shares:

    a_c = logsumexp(s_j for camera j = c) - log(N_c)
    ell_c = logsoftmax(a)_c

Scorer and production DP use FP32; mathematical backend tests may use FP64. No new positional/camera embedding, min-max normalization, IRA correction, or raw T5 attention reconstruction is used in this Stage 2.

## Prefix allocation and compression

For camera c, 1 <= m_c <= min(N_c, 44), and sum(m_c) = 49. A shared [44,64] query bank supplies the prefix rows 0 through m_c-1. Utility is U_c(k) = k ell_c - log(k!). During training one Gumbel tensor [B,6,44], scaled by beta, perturbs valid utilities. Hard max-DP and soft log-sum-exp DP share this noise. Tau is fixed at 1.0. Hard ties select the first maximum with ascending k. Infeasible capacity is rejected. Perturb-and-MAP hard samples are not asserted to follow the soft-DP Gibbs distribution.

The factorial prior gives diminishing marginal utility; hard or expected budgets need not equal 49 times the shares. The soft selection S_soft is the expected canonical camera-then-row packing matrix over feasible allocations. Invalid states have zero mass. Pre-normalization row mass error must be at most 1e-3, with finite nonnegative weights and positive row sums. Rows are normalized only after these checks; diagnostics retain the original error.

Compression stays within each camera:

    K_j = RMS(X_j) W_k
    P_c,r,j = softmax_j(((Q_r * omega) dot K_j)/sqrt(64) + 0.1 s_j)
    Z_c,r = sum_j P_c,r,j X_j

Values are original X. Gamma = 0.1 is fixed; allocation noise never enters compression logits. Training computes detached candidate outputs for the padded 264-slot pool and separately computes 49 active outputs Y_h with autograd. The bridge is:

    Y_s = S_soft @ stopgrad(Z_candidate)
    Y = Y_h + (Y_s - stopgrad(Y_s))

The parenthesized subtraction preserves finite hard forward values. The allocation gradient is a biased STE surrogate. Only active compression retains its activation graph; scorer and DP retain separate graphs, and the auxiliary candidate pool still consumes memory and time. Inactive query rows receive no direct compression gradient; Gumbel does not guarantee every row is used. Evaluation and generation use only hard-DP and 49 active queries, without soft-DP, candidate pool, or Gumbel draws.

Training loss is answer CE + lambda H(P_active), with lambda from `training.entropy_weight` (currently 0.0 in the tracked profile), where H is mean entropy across batch and 49 active queries, using log(max(P,1e-8)). Entropy uses hard membership and does not traverse soft allocation. Answer CE directly trains allocation through the bridge; entropy still trains the scorer through the active compression bias. There is no alpha entropy loss. Validation/best-checkpoint selection uses answer CE. The final sequence is [49 visual; question], with new relative position bias and padding mask applied once. Generation encodes once and omits entropy.

## Training and resume

### Numerical softmax and answer-CE implementation

For this profile only, answer CE uses the target-excluded `logsumexp` +
`softplus` expression (Eq. 6 of *A Fix for Softmax and Cross-Entropy*,
arXiv:2610.31642v1). Standalone softmax/log-softmax now use Eq. 5:

    a = argmax(z); d_a = constant(0); d_i = z_i - z_a for i != a
    u = logsumexp_{i != a}(d_i)
    log_softmax_fixed(z) = d - softplus(u)
    softmax_fixed(z) = exp(log_softmax_fixed(z))

The optimized implementation covers Stage-1 camera weights, token alpha, camera
log-shares, active compression and detached candidate compression, every native
T5 encoder/decoder self-attention and decoder cross-attention, and generation
beam log-softmax (or sampling softmax). The scorer's MultiheadAttention is the
explicit exception: it keeps its original fused/native SDPA and its internal
softmax. T5 4.46.3 attention is already eager; its existing mask/cache/dropout
implementation is reused with only its normalization dependency specialized.

There is no TorchFunctionMode or operator interception. Repository call sites
call the fixed functions directly. T5/generation functions are specialized once
per model instance with a private `nn.functional` dependency; neither PyTorch nor
Transformers globals are patched. No parameters or state-dict keys are added,
and other profiles keep their existing implementations.

The forward calculation uses the equivalent, max-shifted expressions:

    S = sum_{i != a} exp(d_i)
    log_softmax_fixed(z) = d - log1p(S)
    softmax_fixed(z) = exp(d) / (1 + S)

This avoids redundant logsumexp max reductions and exponentials. The softmax
probability at a can still round to 1, so autodiff through that ratio is NOT used.
The custom analytic backward computes all non-maximum gradients first and sets
the maximum gradient to their negative sum. For upstream gradient g:

    softmax: h_i = g_i - g_a; mu = sum_i p_i h_i
             dz_i = p_i (h_i - mu), i != a
    log-softmax: dz_i = g_i - p_i sum_j g_j, i != a
    both: dz_a = -sum_{i != a} dz_i

Only output and argmax indices are saved for backward. Tests check FP64 first
and second derivatives, arbitrary upstream gradients, tied logits, padding,
one-valid-key rows, and gradients beyond both FP32 and FP64 precision cliffs.
The answer-CE fix remains unchanged. Masks, normalization axes, dropout rates,
quotas, Gumbel/STE, beta, entropy definition/weight and generation settings are
unchanged. DP log-partition/posterior arithmetic remains unchanged. DINO
extraction is offline and outside this VLM execution scope. No config,
notebook, checkpoint migration or architecture-signature change is required.

Diagnostics now report:

    softmax_implementation: max_excluded_softplus_direct_v2
    scorer_attention_backend: native_sdpa
    answer_loss_implementation: target_excluded_softplus_v1

The numeric marker identifies an implementation change, not a new model
architecture or proof of improved task metrics. Benchmark metadata distinguishes
native scorer SDPA from direct fixed T5 normalization. GPU measurements are
still required before drawing speed or VRAM conclusions.

CPU smoke comparison used identical initial weights, inputs, Gumbel seeds,
FP32, dropout zero, batch two, six cameras, 384 -> 196 -> 49 tokens, T5 width
384, four encoder/four decoder blocks and vocabulary 32128. Alternating timing
order, warmup and median measurements gave:

| Operation | Native softmax/CE | Previous full fix | Optimized fix + native scorer SDPA |
|---|---:|---:|---:|
| Forward + backward | 427.61 ms | 556.65 ms | 502.48 ms |
| Generate 8 tokens, beam 1 | 200.75 ms | 319.11 ms | 239.19 ms |
| Generate 8 tokens, beam 3 | 284.45 ms | 377.29 ms | 286.05 ms |

These are synthetic random-weight CPU measurements, with substantial sample
timing variation, not GPU latency estimates. Compared with the previous full
fix, initial logits differed by at most 4.06e-6 and gradients by 1.38e-7
(relative L2 8.62e-7). Camera budgets and generated token IDs matched in this
probe; three paired optimizer updates completed with finite gradients and
nearly identical losses. These observations do not establish model quality.
Raw benchmark output is in the session's temporary `huy-softmax-opt-*` folder.

Verification after optimization: the complete offline suite ran 804 tests,
with **782 passed, 3 skipped, 19 existing failures** in 963.15 seconds. Failed
node IDs exactly matched the 18 previously recorded baseline failures plus
`test_camera_pool_query_artifacts.py::test_config_geometry_and_training_protocol`.
The latter was separately reproduced with the pre-optimization source snapshot:
its assertion expects entropy 0.0 while that separate profile currently has 1.0.
Neither its config nor its assertion was changed. The 18 older failures concern
removed notebook resume flags and stale entropy/token-budget/signature expectations.
No additional failed node IDs appeared. The full suite includes the numeric
fix/gradgrad/mask checks, native-SDPA routing, T5 and beam routing, state identity,
CLI lifecycle, accumulation, checkpoint/resume and two-process CPU/Gloo tests.
Syntax compilation and `git diff --check` passed. There was no GPU run, pretrained
model/data download, config/notebook edit, commit or push for this optimization.

The dedicated profile uses FP32 feature-cache training, frozen/offline DINO, trainable T5/projector/embeddings/scorer/compressor, ten epochs, batch 16 per process, accumulation 2, AdamW lr 1e-4 and weight decay 0.05, epoch scheduler gamma 0.9, clipping 1.0, seed 42, and gradient checkpointing false. Gradient checkpointing is unsupported for this profile. The notebook starts from pretrained T5; it has no alignment or cross-profile warm start.

The automatic beta horizon H is epochs times ceil(len(prepared loader)/accumulation), computed after accelerator preparation. `training.beta_decay_updates` can override it; null chooses automatic for a fresh run and stored H on resume. For t completed optimizer updates:

    beta(t) = 0.2 * (1 - min(t / max(1, H-1), 1))

Advance once per successful optimizer update, including accumulation tails; skipped updates and diagnostics/evaluation do not advance. Debug limits do not shorten H. Checkpoints require beta_step = optimizer_step. Resume retains H and rejects a conflicting explicit override. Initialization and sampler seed are 42; fresh process RNG is seeded 42 + rank after preparation and before iteration. Resume restores rank RNG there without reseeding.

Resume supports this same profile at an epoch boundary with the same world size. Keep the complete checkpoint and `<checkpoint>.rng/manifest.json` plus every rank sidecar. Model-only inference weights are insufficient for training resume. The notebook validates the two-rank manifest and sidecars and runs `verify-resume-checkpoint` before launch. A one-process verification probe establishes compatibility, not two-GPU replay equivalence. Architecture signatures identify scorer/gamma/geometry; recipe and state retain schedule, horizon, optimizer, scheduler and RNG policy.

Both allocator environment variables are set before Torch/CUDA imports. Dependency pins remain transformers 4.46.3, tokenizers 0.20.4 and accelerate 1.15.0. The notebook writes a distinct runtime YAML, uses separate output/cache/archive paths, validates prepared counts and feature manifest v2/model revisions/prepared-file identity/index coverage, and archives runtime YAML, the checkpoint directory with sidecars, metrics, benchmark, predictions and debug records.

## Learned text pooling (2026-11-04)

Chỉ profile `mvpruner_camera_query_qaware` thay masked mean bằng learned pooling.
Scorer vẫn chạy đúng một cross-attention SDPA sẵn có, tạo
`M = CrossAttention(RMS(T), RMS(X), RMS(X))` có shape `[B,Lq,D]`.
Head nhận trực tiếp `M`, chưa thêm normalization:

```text
u = tanh(Linear(D, H)(M))
e = Linear(H, 1, bias=False)(u).squeeze(-1)
a = StableSoftmax(e.masked_fill(~question_mask, -inf), dim=text)
m = bmm(a.unsqueeze(1), M).squeeze(1)
s_j = scorer_MLP(concat(RMS(X_j), RMS(m)))
```

`H` dùng `question_scoring.hidden_dim` hiện có: production D=384, H=64,
thêm 24.704 tham số trong ba tensor. Đây là head độc lập với MLP chấm visual
token. Mọi token prompt có mask 1 đều tham gia; padding có trọng số 0, câu hỏi
không có token hợp lệ vẫn bị từ chối. Không dùng decoder `<pad>`, không thêm
attention hoặc entropy loss cho trọng số text. `a_t` là trọng số text, khác
`alpha_j` của visual tokens. Allocation, candidate pool và compression dùng lại
scores hiện có; generation chỉ encode/pool một lần, không chạy head mỗi token sinh.

Khởi tạo lớp đầu Xavier uniform gain 1, bias 0; lớp cuối zero. Vì vậy `a` ban đầu
uniform trên token hợp lệ, output gần masked mean trong sai số số học, không
cam kết bitwise. Toàn bộ khởi tạo head ở trong `fork_rng(devices=[])`, sau các
module cũ, giữ nguyên RNG CPU và khởi tạo các weight chung. Không đổi seed.
Backward đầu có gradient tensor zero hợp lệ ở lớp đầu; lớp cuối có thể học ngay.
AdamW vẫn có thể decay weight lớp đầu dù gradient zero. Sau cập nhật lớp cuối,
lớp đầu có thể nhận tín hiệu; mẫu chỉ một token hợp lệ hoặc mọi `M_t` giống nhau
không bảo đảm gradient khác zero. Ý tưởng tanh attention tham khảo
[HAN, mục 2.2](https://aclanthology.org/N16-1174/); khởi tạo uniform là lựa chọn
thực nghiệm này, không phải khởi tạo nguyên bản của HAN.

Chỉ `QuestionAwareCameraQueryFusion` bật tùy chọn constructor nội bộ.
Scorer mặc định, profile spatial pooled-query và qaware-pruning vẫn dùng mean,
không thêm tham số hoặc state-dict keys. Giữ 384 → 196 → 49 token, các chuẩn hóa
và CE đã có, native SDPA, optimizer/LR/weight decay, beta/Gumbel/STE và detach.
Không đổi config, notebook, signature, recipe hoặc checkpoint loader.

Lượt mới finetune từ pretrained T5 bằng notebook hiện có. Checkpoint mới tự lưu
head trong `visual_fusion.question_scorer.text_pool.{fc1.weight,fc1.bias,fc2.weight}`
và dùng lifecycle resume hiện có. Checkpoint cũ thiếu head sẽ bị strict loading
từ chối; không migration hoặc bỏ qua missing keys.

Marker trong `forward_debug`:

```text
question_pooling_implementation: masked_attentive_mlp_v1
scorer_attention_backend: native_sdpa
mvpruner.text_pool_entropy
mvpruner.text_pool_max_weight_mean
```

Hai thống kê text chỉ tính trong diagnostic capture, không cộng vào loss.
Báo cáo gradient scorer có `text_pool.*` với `present`, `finite`, `grad_norm`,
`nonzero_ratio`, `ok`. Zero hữu hạn được chấp nhận; thiếu gradient hoặc NaN/Inf
bị đánh dấu không đạt. Không dump toàn bộ `a` vào JSONL hoặc lưu graph trong head.

Kiểm chứng offline mới: 45 test liên quan pass, gồm công thức/reference độc lập,
FP64 input/parameter gradcheck, masking/permutation, khởi tạo/RNG, CE qua T5 nhỏ,
optimizer đăng ký mỗi tensor đúng một lần, native SDPA và đúng một lần pooling
mỗi encode. Diagnostic test chấp nhận gradient zero, phát hiện thiếu/NaN.
Hai test lifecycle riêng pass: CLI prepare/features → diagnose → accumulation
verify → finetune → verify/resume → evaluate/benchmark; kiểm tra khôi phục chính
xác ba tensor head và Adam slots ở update 2/4. Hai rank CPU/Gloo có gradient và
head state đồng bộ sau cập nhật, kể cả hard quota khác nhau giữa các rank.
Fixture SGD riêng dùng LR head 1 để quan sát thay đổi rất nhỏ của lớp đầu ở
FP32; đây không phải thay đổi LR production.

CPU smoke dùng cùng common weights, features, mask, seed, FP32, dropout 0,
batch 2, D384, sáu camera 64 token, 196 → 49, prompt tensor 16, labels 12,
vocabulary 32128, T5 bốn encoder/bốn decoder blocks. Hai lượt warmup, tám mẫu
timing luân phiên thứ tự, median (ms):

| Thời điểm | Phép đo | Mean | Learned |
|---|---|---:|---:|
| Khởi tạo | Forward + backward | 472.38 | 466.22 |
| Khởi tạo | Generate tối đa 8 token, beam 1 | 212.22 | 215.99 |
| Khởi tạo | Generate tối đa 8 token, beam 3 | 347.62 | 327.15 |
| Sau 3 update | Forward + backward | 479.15 | 462.42 |
| Sau 3 update | Generate tối đa 8 token, beam 1 | 192.00 | 189.03 |
| Sau 3 update | Generate tối đa 8 token, beam 3 | 285.22 | 284.13 |

Đo riêng pooling `[16,38,384]`, H64, 80 mẫu: mean 0.151 ms, learned 0.677 ms
lúc khởi tạo; sau ba update là 0.151/0.672 ms. Head thêm công việc; các median
toàn mô hình nhỏ hơn ở một số lượt không chứng minh head làm mô hình nhanh hơn.
Độ dao động timing CPU và chi phí toàn pipeline lớn hơn phần pooling nhỏ này.
Ban đầu max absolute logit difference 4.29e-6, loss 10.573593/10.573594, quota
giống nhau. Ba cặp update có loss hữu hạn gần nhau và head lớp cuối học được
(norm 0.00124). Đây là model/features ngẫu nhiên, không đo chất lượng DriveLM,
VRAM, latency GPU hoặc train hai GPU. Raw smoke script/results nằm trong thư
mục tạm của phiên `huy-text-pool-*`; không thêm công cụ benchmark vào repo.

Full offline suite sau thay đổi: **793 passed, 3 skipped, 19 failed** trên 815
test, 992.18 giây. Cả 11 test mới trong `test_text_pooling.py` và
`test_text_pool_lifecycle.py` pass trong lượt tổng. Chạy lại riêng 19 test lỗi
trên snapshot source trước thay đổi cũng tái hiện đúng 19 test ID đó: assertion
cũ về notebook resume flag, entropy, token budget và signature. Không phát sinh
test ID lỗi mới; không sửa config/assertion ngoài phạm vi để làm suite xanh.
Syntax compilation và `git diff --check` pass. Không tải model/dataset lớn,
chạy full training, sửa notebook/config/checkpoint loader hoặc commit/push.

## Evidence and measurement limits

The design review established small-case hard/soft mathematics and bridge-gradient checks, plus a six-camera/49-query FP32 numerical probe; these are not evidence of model quality, GPU speed, or VRAM capacity. Offline implementation tests should cover exhaustive DP oracles, FP64 gradient checks, FP32 stress, hard-forward/bridge gradients, question padding/permutation, split T5 behavior, active entropy, lifecycle schedules and same-horizon replay. Record actual executed checks separately from these design requirements.

Executed offline artifact checks: 34 tests passed across the camera-query, qaware, MVPruner and unpruned notebook artifact suites, including notebook Python and embedded heredoc syntax checks. The separate real CLI lifecycle and CPU/Gloo suite passed all four tests: existing and qaware profiles completed prepare-data, prepare-features, diagnose, train, verify-resume, epoch-boundary resume, evaluate and benchmark. Qaware used an explicit four-update beta horizon and checked persisted counters, architecture identity and scorer state. Two CPU ranks ran the complete question scorer/DP/compressor with camera embeddings for two optimizer updates, forced distinct [3,1] versus [1,3] hard budgets, checked finite identical reduced gradients for every trainable parameter, and finished with identical model states. This establishes CPU distributed synchronization for the tested geometry, not full two-GPU training or replay.

Final repository regression on 2026-10-29: **448 passed, 3 skipped** in 645.57 seconds, using `python -m pytest -q` with `PYTHONPATH=src`, `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`. This includes exhaustive hard/soft DP oracles, FP64 gradcheck, FP32 stress, dense-reference bridge gradients, active entropy, native T5 integration, and exact model/optimizer/scheduler/rank-RNG replay with dropout, shuffle and Gumbel. Production geometry was checked on CPU with batch two, hidden dimension 384, 196 source tokens, 49 outputs and a [44,64] bank; Stage 1 indices, budgets and outputs matched the existing camera-query fusion exactly. All notebook JSON/Python/embedded-heredoc syntax checks and `git diff --check` passed. Previously tracked config and notebook files remain unchanged.

Feature-starting evaluation benchmarks include scorer, hard-DP, active compression and T5/decoder; they exclude image decode/DINO and the training-only surrogate pool. FLOPs are a major-operator estimate: DP, indexing and log-sum-exp are not fully counted. Report training candidate-pool overhead separately. No GPU VRAM, latency, full training, or two-GPU replay claim follows from CPU tests. Quality improvement requires matched-protocol ablations against existing camera-query, this profile with gamma zero, and the full gamma 0.1 profile. Compact diagnostics include shares, hard/expected budgets, beta/tau/gamma, soft-row error, scorer scale, alpha entropy, active compression entropy, overlap, usage and gradients without retaining graphs or dumping large matrices.
