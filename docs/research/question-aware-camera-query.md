# Question-aware camera-query Stage 2

This experimental profile adds question-conditioned camera allocation after the first two T5 blocks. The frozen DINO feature pipeline and Stage 1 remain unchanged: six cameras provide 384 tokens, Stage 1 retains 196, and Stage 2 emits exactly 49 visual tokens before the final two T5 blocks. The [approved implementation plan](question-aware-camera-query-implementation-plan.md) is retained verbatim as the design record.

Use `configs/dinov2_t5_efficient_mini_kaggle_2gpu_camera_query_qaware.yaml` with `notebooks/kaggle_train_dinov2_camera_query_qaware_finetune.ipynb`. Fusion identity is `mvpruner_camera_query_qaware`; its architecture version is `qaware_prefix_dp_v1`. Existing camera-query, MVPruner and unpruned artifacts retain their own identities.

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

Training loss is answer CE + 0.5 H(P_active), where H is mean entropy across batch and 49 active queries, using log(max(P,1e-8)). Entropy uses hard membership and does not traverse soft allocation. Answer CE directly trains allocation through the bridge; entropy still trains the scorer through the active compression bias. There is no alpha entropy loss. Validation/best-checkpoint selection uses answer CE. The final sequence is [49 visual; question], with new relative position bias and padding mask applied once. Generation encodes once and omits entropy.

## Training and resume

The dedicated profile uses FP32 feature-cache training, frozen/offline DINO, trainable T5/projector/embeddings/scorer/compressor, ten epochs, batch 16 per process, accumulation 2, AdamW lr 1e-4 and weight decay 0.05, epoch scheduler gamma 0.9, clipping 1.0, seed 42, and gradient checkpointing false. Gradient checkpointing is unsupported for this profile. The notebook starts from pretrained T5; it has no alignment or cross-profile warm start.

The automatic beta horizon H is epochs times ceil(len(prepared loader)/accumulation), computed after accelerator preparation. `training.beta_decay_updates` can override it; null chooses automatic for a fresh run and stored H on resume. For t completed optimizer updates:

    beta(t) = 0.2 * (1 - min(t / max(1, H-1), 1))

Advance once per successful optimizer update, including accumulation tails; skipped updates and diagnostics/evaluation do not advance. Debug limits do not shorten H. Checkpoints require beta_step = optimizer_step. Resume retains H and rejects a conflicting explicit override. Initialization and sampler seed are 42; fresh process RNG is seeded 42 + rank after preparation and before iteration. Resume restores rank RNG there without reseeding.

Resume supports this same profile at an epoch boundary with the same world size. Keep the complete checkpoint and `<checkpoint>.rng/manifest.json` plus every rank sidecar. Model-only inference weights are insufficient for training resume. The notebook validates the two-rank manifest and sidecars and runs `verify-resume-checkpoint` before launch. A one-process verification probe establishes compatibility, not two-GPU replay equivalence. Architecture signatures identify scorer/gamma/geometry; recipe and state retain schedule, horizon, optimizer, scheduler and RNG policy.

Both allocator environment variables are set before Torch/CUDA imports. Dependency pins remain transformers 4.46.3, tokenizers 0.20.4 and accelerate 1.15.0. The notebook writes a distinct runtime YAML, uses separate output/cache/archive paths, validates prepared counts and feature manifest v2/model revisions/prepared-file identity/index coverage, and archives runtime YAML, the checkpoint directory with sidecars, metrics, benchmark, predictions and debug records.

## Evidence and measurement limits

The design review established small-case hard/soft mathematics and bridge-gradient checks, plus a six-camera/49-query FP32 numerical probe; these are not evidence of model quality, GPU speed, or VRAM capacity. Offline implementation tests should cover exhaustive DP oracles, FP64 gradient checks, FP32 stress, hard-forward/bridge gradients, question padding/permutation, split T5 behavior, active entropy, lifecycle schedules and same-horizon replay. Record actual executed checks separately from these design requirements.

Executed offline artifact checks: 34 tests passed across the camera-query, qaware, MVPruner and unpruned notebook artifact suites, including notebook Python and embedded heredoc syntax checks. The separate real CLI lifecycle and CPU/Gloo suite passed all four tests: existing and qaware profiles completed prepare-data, prepare-features, diagnose, train, verify-resume, epoch-boundary resume, evaluate and benchmark. Qaware used an explicit four-update beta horizon and checked persisted counters, architecture identity and scorer state. Two CPU ranks ran the complete question scorer/DP/compressor with camera embeddings for two optimizer updates, forced distinct [3,1] versus [1,3] hard budgets, checked finite identical reduced gradients for every trainable parameter, and finished with identical model states. This establishes CPU distributed synchronization for the tested geometry, not full two-GPU training or replay.

Final repository regression on 2026-10-29: **448 passed, 3 skipped** in 645.57 seconds, using `python -m pytest -q` with `PYTHONPATH=src`, `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`. This includes exhaustive hard/soft DP oracles, FP64 gradcheck, FP32 stress, dense-reference bridge gradients, active entropy, native T5 integration, and exact model/optimizer/scheduler/rank-RNG replay with dropout, shuffle and Gumbel. Production geometry was checked on CPU with batch two, hidden dimension 384, 196 source tokens, 49 outputs and a [44,64] bank; Stage 1 indices, budgets and outputs matched the existing camera-query fusion exactly. All notebook JSON/Python/embedded-heredoc syntax checks and `git diff --check` passed. Previously tracked config and notebook files remain unchanged.

Feature-starting evaluation benchmarks include scorer, hard-DP, active compression and T5/decoder; they exclude image decode/DINO and the training-only surrogate pool. FLOPs are a major-operator estimate: DP, indexing and log-sum-exp are not fully counted. Report training candidate-pool overhead separately. No GPU VRAM, latency, full training, or two-GPU replay claim follows from CPU tests. Quality improvement requires matched-protocol ablations against existing camera-query, this profile with gamma zero, and the full gamma 0.1 profile. Compact diagnostics include shares, hard/expected budgets, beta/tau/gamma, soft-row error, scorer scale, alpha entropy, active compression entropy, overlap, usage and gradients without retaining graphs or dumping large matrices.
