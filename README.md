# Efficient VLM for Autonomous Driving: MVPruner

This branch compares attention and L2 variation scoring at MVPruner stage 2 using
RepViT-M1.5 and T5-Efficient-Mini. Both use the same stage 1 selector, camera
minimums and feature-cache format.

| Profile | Stage 2 | Purpose |
| --- | --- | --- |
| `configs/repvit_t5_efficient_mini_kaggle_2gpu_mvpruner.yaml` | attention | Real Kaggle baseline |
| `configs/repvit_t5_efficient_mini_kaggle_2gpu_mvpruner_l2.yaml` | l2_variation | Real Kaggle comparison |
| `configs/mvpruner_attention_smoke.yaml` | attention | Offline CPU fake-model smoke |
| `configs/mvpruner_l2_smoke.yaml` | l2_variation | Offline CPU fake-model smoke |

Real profiles train FP32, finetune only. L2 variation selects using the L2 norm
of each visual token's hidden-state change across the configured encoder block.
Its profile uses a separate output/cache directory. Cached original RepViT
features remain float16; selecting the L2 score does not change the cache schema.
Attention checkpoints and L2 checkpoints cannot resume across scoring modes.

## Kaggle

Use `notebooks/kaggle_debug_mvpruner_finetune.ipynb` or
`notebooks/kaggle_train_mvpruner_finetune.ipynb` for attention.
Use the matching `kaggle_debug_mvpruner_l2_finetune.ipynb` or
`kaggle_train_mvpruner_l2_finetune.ipynb` for L2 variation.
All default to branch `huy-v2drop`. Full notebooks configure
`PYTORCH_ALLOC_CONF=expandable_segments:True` before shell Python processes.

The notebooks preserve separate Hugging Face repositories for prepared data,
feature caches and optional finetune checkpoints, including distributed RNG
sidecars. Checkpoint repositories default to empty. Set `HF_TOKEN` through Kaggle
Secrets or the environment.

## CLI

Choose one real profile, then run:

```bash
python -m efficient_vlm_ad prepare-data --config CONFIG --subset full
python -m efficient_vlm_ad prepare-features --config CONFIG --subset full
python -m efficient_vlm_ad diagnose-train --config CONFIG --stage finetune --debug-numerics
python -m efficient_vlm_ad train --config CONFIG --stage finetune
python -m efficient_vlm_ad verify-resume-checkpoint --config CONFIG --stage finetune --checkpoint CHECKPOINT
python -m efficient_vlm_ad evaluate --config CONFIG --checkpoint CHECKPOINT
python -m efficient_vlm_ad benchmark --config CONFIG --checkpoint CHECKPOINT
```

For offline smoke profiles, supply a local synthetic dataset via `data.local_dir`.
The tests create temporary images and annotations and require no network access.

## Verification

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest tests -q
```

Artifact removals and retained coverage are recorded in [docs/v2drop-artifacts.md](docs/v2drop-artifacts.md).
Reference source, shared data, caches and existing outputs are preserved.
