# MVPruner + L2 variation + IRA + Gumbel STE

This experiment adapts V2Drop's residual-update score to the four-block T5 encoder.
It is not a reproduction of V2Drop: Stage 1, camera IRA and training-time STE belong
to this repository. No L2 auxiliary loss or learned scoring parameters are added.

## Source mapping and invariants

- V2Drop research source: commit `2a9431150f8ace3a1d9abcdc96a7d46acca597df`,
  `llava/model/language_model/V2Drop.py` and Qwen's `modeling_qwen2_vl_self.py`.
  Both compute a raw L2 norm of consecutive residual-stream states, without
  normalization, squared distances or multi-layer averaging. Their hard-coded
  image spans, batch assumptions and pruning schedules are not copied.
- MVPruner IRA reference: commit `53ecc443fd7664a92e71ab6213f5611ef1ec5c7d`,
  `drivelmm-o1/modeling/modeling_internlm2_mvpruner.py`. The subtracted term is
  the fraction of tokens surviving Stage 1, not freshly recomputed diversity.
- This branch starts at `cc48d91`. Stage 1/CCTS/allocator retain their algorithms.
  RepViT, preprocessing, the float16 feature cache schema, projector, embeddings,
  tokenizer `<`, post-helper, FP32 compute and generation settings are unchanged.

## Forward

`[B,6,49,512] -> spatial/camera -> Linear(512,384) -> DRA/CCTS(196)`
`-> modal embeddings + question -> encoder dropout -> block 0 -> block 1`
`-> IRA/selection(49) -> blocks 2,3 -> final norm/dropout -> decoder/CE`.

`model.fusion.stage2_score_mode` is `attention` by default or `l2_variation`.
L2 uses `norm(block1_output.float() - block1_input.float(), dim=-1)` on visual
tokens only. No detach, additional block execution, final layer norm or scorer
attention matrix is involved. Native block dropout remains enabled in training.
The L2 score is therefore question-conditioned in the bidirectional T5 encoder;
that is not generally true for visual-prefix tokens in a causal decoder.

Per camera, mean token scores are min-max normalized using the existing epsilon
clamp and softmax. With normalized score share `r` and Stage 1 fraction `u`, weights
are `k1 * clamp(1 + r - u, 0, 2)`. The bounded allocator retains exactly 49 tokens
and at least one per camera in the real profile. Allocation is discrete.

Training selection uses `log(clamp_min(score, eps)) + beta * Gumbel`, relaxed
slots and `P_STE @ hidden_visual`. Only CE trains the model. Score gradients are
surrogates, not exact Top-K derivatives. Zero norms/clamped scores may have zero
gradient; dropped value-path gradients need not be nonzero. Large variation is
an empirical importance proxy, not a mathematical guarantee of safe deletion.

L2 inference keeps the same clamped logits, hard-slot count, camera budgets and
ordering as beta-zero training, but gathers directly without soft assignment.
Attention mode keeps its original implementation. This optimization also saves
selection overhead: do not attribute every latency change solely to L2 scoring.

## Checkpoints and diagnostics

Common checkpoint saving writes `metadata.mvpruner.stage2_score_mode` from the
model. Missing mode means legacy attention only. Cross-mode/unknown modes are
rejected before weights or optimizer loading, along with the existing Stage 1
version guard. Start the new L2 experiment from pretrained T5, not an attention
checkpoint. Later L2 resumes retain optimizer, scheduler, beta and rank RNG states.

`diagnose-train` captures scores, block states and value-path gradients from the
actual CE backward. `stage2_gradients` distinguishes selected/dropped score
gradients and the direct value path. It performs no optimizer step. Capture is
disabled and graph references are released even on exceptions. The separate
no-grad `forward_debug` only reports numerics, scores, camera shares and budgets.

## Running and measurement

Use the retained attention config or the `_mvpruner_l2.yaml` profile and their
debug/full-train notebooks. Both real profiles train FP32 finetune-only in the
notebooks. L2 artifacts have a separate output/cache location; compatible cached
features and prepared data may be restored from the existing separate HF repos.
The full notebook exports `PYTORCH_ALLOC_CONF=expandable_segments:True` before
launch. Local execution and remote cloning use branch `huy-v2drop`.

Offline profiles are `mvpruner_attention_smoke.yaml` and `mvpruner_l2_smoke.yaml`.
They use fake vision and a locally initialized four-block T5, never downloads.

For a fair quality experiment use matching pretrained T5, multimodal initialization,
seed, data and training budget. Do not bypass the checkpoint guard for user runs.
Use matching samples, beams, precision and hardware for speed comparisons.
Profiler scopes identify `mvpruner.stage2_l2_scoring`, `stage2_attention_scoring`,
`stage2_allocation`, `stage2_selection` and existing Stage 1/T5 scopes. CUDA profiling
must synchronize appropriately. Existing benchmark prefill/E2E/memory definitions
are preserved; report metadata records the scorer. Major-op FLOPs exclude much of
L2/elementwise/sorting work, so GFLOPs alone cannot establish a speedup.

Artifact deletion and preserved/replaced test coverage are listed separately in
`v2drop-artifacts.md`. CUDA/T4/DDP behavior needs verification on the target hardware;
CPU offline tests do not establish GPU latency, memory or distributed reproducibility.

## Verification record (2026-10-24)

- Final full suite: `243 passed, 3 skipped` (447.18 s), Python 3.12 environment
  `D:/PythonEnvs/mvpruner-stage1`, Transformers 4.46.3, CPU PyTorch.
- Skips: two CUDA-only checks and one Windows symlink capability check.
- Both-mode CLI smoke covers fake data/features, finetune, mode-aware checkpoint
  verification/resume, diagnose CE backward, evaluation and prefill/FLOP benchmark.
- Against commit `cc48d91`, Stage 1/CCTS/allocator AST is unchanged, and fixed-input
  Stage 1 plus attention Stage 2 outputs/indices/camera IDs match bitwise (batch 2,
  six views, budgets 294 -> 196 -> 49). L2 tests additionally verify real CE gradients,
  dropout/checkpointing parity, pruned batch/bias invariance, repeated generation,
  zero/tied scores and exception cleanup.
- CLI help, Python compilation and `git diff --check` pass. Independent review found
  no blocking issues; its two coverage suggestions were added and passed.
- CUDA/Tesla T4/multi-GPU execution and actual pretrained-model quality/speed have
  not been verified here. Other branch worktrees remain unchanged; no commit/push.
