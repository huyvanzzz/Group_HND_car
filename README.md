# Efficient VLM For Autonomous Driving

Refactor workspace for EM-VLM4AD experiments on Kaggle.

The original upstream code is kept unchanged in `em_vlm4ad/` as a reference. New code lives in
`src/efficient_vlm_ad/` and is designed around config-driven profiles:

- `configs/legacy_vit_t5_base.yaml`: faithful ViT-B/32 patch-projection + T5-Base baseline.
- `configs/repvit_t5_efficient_mini.yaml`: main RepViT-M1.5 + T5-Efficient-Mini profile.
- `configs/repvit_t5_efficient_tiny_smoke.yaml`: quick smoke/debug profile.

## Current Entrypoints

```bash
python -m efficient_vlm_ad inspect-data --config configs/repvit_t5_efficient_mini.yaml
python -m efficient_vlm_ad prepare-data --config configs/repvit_t5_efficient_mini.yaml
python -m efficient_vlm_ad prepare-features --config configs/repvit_t5_efficient_mini.yaml
python -m efficient_vlm_ad train --config configs/repvit_t5_efficient_mini.yaml --stage align
python -m efficient_vlm_ad train --config configs/repvit_t5_efficient_mini.yaml --stage finetune --resume CHECKPOINT
python -m efficient_vlm_ad evaluate --config configs/repvit_t5_efficient_mini.yaml --checkpoint CHECKPOINT
python -m efficient_vlm_ad benchmark --config configs/repvit_t5_efficient_mini.yaml --checkpoint CHECKPOINT
```

`HF_TOKEN` must come from the environment. Do not put it in YAML, CLI args, notebooks, or Git.

## Kaggle

On Kaggle, store the Hugging Face token as a secret named `HF_TOKEN`, set `GITHUB_REPO_URL`, and run:

```bash
python scripts/kaggle_bootstrap.py
```

The bootstrap clones branch `huy`, installs the package editable, reads the Kaggle secret, and runs
the smoke sequence. For a full run, use the Mini config and remove `--subset smoke`,
`--max-steps`, and `--max-samples`.

## Verification

```bash
python -m pytest tests -q
```
