# MVPruner QAware Stage II experiment

This additive profile keeps MVPruner Stage I and the existing encoder pruning wrapper. Only Stage II importance scoring changes. The original MVPruner configuration, notebook, training, cache provenance and inference entrypoints remain the comparison protocol.

For encoder visual states $V$ and question states $Q$, apply RMS normalization: $X=\mathrm{RMS}(V)$ and $U=\mathrm{RMS}(Q)$. The reused four-head attention computes $A=\mathrm{MHA}(U,X,X)$, with zero dropout. Form a masked mean summary of attended question positions, $q=\mathrm{RMS}(\sum_t m_t A_t/\sum_t m_t)$. Score each visual token with the MLP $s_i=W_2\mathrm{ReLU}(W_1[X_i;q]+b_1)$, then apply a single global softmax over the 196 valid Stage I visual positions: $\alpha_i=\mathrm{softmax}(s)_i$. Defaults are four attention heads, MLP hidden dimension 64 and RMS epsilon $10^{-6}$. Scorer settings live in `fusion.stage2_scoring`; no query tokens, `fusion.query`/`fusion.question_scoring` sections or gamma control are introduced.

Feed $\alpha$ to the existing IRA procedure: mean importance per camera, min/max normalization, softmax camera weights and bounded integer allocation respecting camera capacity and the minimum one-token floor. The reused scorer also computes extra `camera_shares`; this experiment discards that output. Its global alpha is the sole scorer output consumed by the allocator and selector. Camera allocation is hard and has no direct budget gradient.

Within each camera use the inherited clamped $\log\alpha$, add a single Gumbel sample during training and apply the inherited straight-through top-k estimator. Evaluation uses deterministic top-k. Select exactly 49 ORIGINAL visual tokens, without replacing their values with normalized or scored vectors. Preserve question/padding positions and gather the existing relative-position bias rows and columns with the selected old indices; do not recompute bias for a new sequence. Keep Stage I's 384-to-196 selection unchanged.

Training minimizes answer cross-entropy only (`entropy_weight: 0.0`). It finetunes for 10 epochs from pretrained T5 and new multimodal modules, or resumes a same-profile finetune checkpoint with complete rank-specific RNG sidecars. There is no align stage or cross-profile warmstart. The hard camera allocator does not receive a direct gradient through integer budgets; scorer gradients flow through the inherited token-selection surrogate. Keep current original-MV beta horizon calculation from the pre-prepare dataloader deliberately unchanged, including its relationship to distributed preparation. This profile does not fix that behavior and has no `beta_decay_updates` override. Accumulation and resume behavior are inherited unchanged.

The isolated config is `configs/dinov2_t5_efficient_mini_kaggle_2gpu_mvpruner_qaware.yaml`; output and cache use the matching `_mvpruner_qaware` folder. The companion notebook copies that config to a Kaggle runtime YAML and never writes the tracked source. Both allocator environment variable names are set to `expandable_segments:True` before Torch import or CUDA discovery. It retains the original prepared-data downloads, manifest-v2 cache checks, dependency pins, diagnose command, existing accumulation-debug command, checkpoint autodiscovery, distributed RNG validation and verify-resume command.

Vision/text revisions, preprocessing, 384/196/49 budgets, FP32, batch 16, accumulation 2, clipping, optimizer and generation are unchanged. Gradient checkpointing is disabled and seed is 42. Evaluate and benchmark use the same repository entrypoints and protocol; timing includes the new scorer. No separate inference implementation or unrelated benchmark is added. No GPU quality or speed result is claimed: validation here covers implementation and artifacts, not a completed GPU experiment.

## Running and validation

Use `notebooks/kaggle_train_dinov2_mvpruner_qaware_finetune.ipynb` for the two-GPU run. For the CLI, after preparing and validating data/features, start a fresh run with:

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
accelerate launch --multi_gpu --num_processes 2 --mixed_precision no -m efficient_vlm_ad train \
  --config configs/dinov2_t5_efficient_mini_kaggle_2gpu_mvpruner_qaware.yaml \
  --stage finetune
```

To resume, supply `--resume` with a same-profile finetune checkpoint and retain its rank-specific RNG sidecars. Use the existing `evaluate` and `benchmark` commands with the same config and `--checkpoint`; generation settings remain in YAML.

CPU validation: 114 targeted tests passed, including original architecture-signature and notebook checks and two-process Gloo gradient synchronization. An independent nine-command CLI smoke run passed prepare-data, prepare-features, diagnose, accumulation verification, fresh finetune, checkpoint verification, resume, evaluate and benchmark. Notebook JSON, Python and Bash syntax checks passed. These runs used tiny offline fixtures, not pretrained model quality measurements.

The full-suite run reported 580 passed, 3 skipped and 16 failed. Two failures were expectations affected by adding the profile (generic signature and artifact inventory); those expectations were updated and retested. The other 14 are existing notebook tests expecting the removed `RESUME_TRAINING` switch or hard-coded entropy 0.5. They were left unchanged; the full suite is not claimed to be green.
