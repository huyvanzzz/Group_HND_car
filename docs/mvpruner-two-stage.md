# DINOv2 MVPruner two-stage profile

Active profile: `configs/dinov2_t5_efficient_mini_kaggle_2gpu_mvpruner.yaml`. The runtime supports MVPruner only; historical GPA/router/adapter execution paths have been removed. See the README compatibility section for loading existing MVPruner checkpoints.

## Model flow

1. Frozen `facebook/dinov2-small` extracts the last hidden state from six 224x224 camera views. Remove the single CLS token and reshape the 256 patch tokens into a 16x16 grid.
2. Average pool with kernel/stride 2 to obtain 64 tokens per camera. Features have shape `[N, 6, 64, 384]`; storage uses float16 and model input uses float32.
3. Add spatial and camera embeddings, flatten 384 visual tokens and apply an explicitly learned linear `384 -> 384` projector.
4. Stage 1 uses DRA plus CCTS to select exactly 196 tokens. Diversity allocates camera budgets; CCTS combines uniqueness with shifted max-cosine relevance over valid question tokens.
5. Add modal embeddings and run T5 encoder blocks 0 and 1. Attention probabilities before dropout from block 1 provide visual relevance.
6. IRA allocates exactly 49 retained tokens with camera capacity constraints. During training, hard Gumbel Top-K uses slot-wise Relaxed Top-K gradients: `P_STE = P_hard - stopgrad(P_soft) + P_soft`, `V_out = P_STE @ V`. Evaluation uses deterministic Top-K.
7. Gather both axes of position bias and the attention mask, then run the remaining T5 blocks, final normalization and decoder.

The objective remains T5 cross-entropy. Main-profile batch size is 16 per process with accumulation 2: effective batch size 64 on two GPUs.

## Numeric and cache compatibility

Install a binary `tokenizers==0.20.4` wheel before `pip install -e ".[mvpruner]"`. The tested Transformers version is exactly `4.46.3`. Both Kaggle notebooks require two GPUs, runtime FP32 and Accelerate `--mixed_precision no`.

DINO manifest version 2 signs the pinned vision revision, processor policy, pooling/token policy, camera order, geometry, prepared-data hashes and cache-index hash. It records float16 storage and float32 loading. Notebooks validate the signature, byte length and sample coverage before accepting a downloaded cache.

T5 attention scale is 1.0. Attention masks are incorporated once in position bias. Negative infinity is accepted only for masked keys; invalid values in active attention logits, relevance, allocations or Gumbel scores fail explicitly. Gradient checkpointing uses the T5 4.46.3 interface.

## Training and resume

The active notebooks run finetune directly from pretrained T5 and newly initialized multimodal modules, or resume an existing finetune checkpoint. The debug notebook writes a temporary runtime config with accumulation 1 and a 20-step limit, isolating smoke artifacts from full training.

Beta decreases linearly from 0.2 to 0 by optimizer step; tau is 1.0. Checkpoints record the architecture signature and MVPruner budget/schedule state. Each rank saves `<checkpoint>.rng/rank_XXXXX.pt` with Python, NumPy, Torch and CUDA RNG, sampler state and cursor; the RNG manifest records world size.

The full notebook prefers local `finetune_latest.pt`. Otherwise it downloads the requested model-repo checkpoint and all `<filename>.rng/**` sidecars. It checks both rank files and world size 2, then invokes `verify-resume-checkpoint --stage finetune` before training. Its final archive includes checkpoints with RNG directories, metrics, predictions, benchmark and debug logs.

Prepared-data and DINO-cache dataset repos are separate. Required files live at repository root and are copied directly into `PROFILE/prepared_data` and `PROFILE/cache`, without nested `cache/cache` paths. When cache files are missing and no cache repo is configured, full data preparation always runs before feature extraction to materialize raw images and refresh the manifest root. Prepared JSON downloaded from another machine is reused only when cache files are already present. Configured cache repositories missing required files fail explicitly. Full prepared-data counts must match the configured dataset.

## Validation and benchmarks

Run `python -m pytest tests -q` for offline tests. Coverage includes DINO geometry and processor policy, manifest v2 provenance, architecture signatures, MVPruner allocation/STE/numerics, T5 split encoder, offline CLI workflows and notebook download/resume contracts. Source-level RepViT tests remain.

End-to-end benchmarks use the CLI `benchmark` command. Component timing is available with:

```bash
python scripts/benchmark_mvpruner_components.py \
  --config configs/dinov2_t5_efficient_mini_kaggle_2gpu_mvpruner.yaml \
  --checkpoint outputs/dinov2_t5_efficient_mini_kaggle_2gpu_mvpruner/checkpoints/finetune_best.pt \
  --mode eval
```

Use `--mode train` to measure Relaxed Top-K and backward overhead.
