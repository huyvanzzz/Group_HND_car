# Efficient VLM For Autonomous Driving

Config-driven EM-VLM4AD experiments on Kaggle. The active profile is DINOv2-S + T5-Efficient-Mini with two-stage MVPruner:

- `configs/dinov2_t5_efficient_mini_kaggle_2gpu_mvpruner.yaml`: frozen DINOv2-S, 64 tokens per camera, learned linear projector, MVPruner 196 → 49 tokens, FP32 finetuning.
- `configs/legacy_vit_t5_base.yaml`: retained ViT-B/32 + T5-Base baseline.

The upstream reference remains in `em_vlm4ad/`; implementation lives in `src/efficient_vlm_ad/`. RepViT extractors remain available in source for compatibility, while old RepViT configs and Kaggle notebooks have been removed.

## Kaggle

Enable two GPUs and store the Hugging Face token as a Kaggle secret named `HF_TOKEN`. Use branch `huy-dinov2` and one of these entrypoints:

- [Debug finetuning](notebooks/kaggle_debug_dinov2_mvpruner_finetune.ipynb): real DINO/T5 smoke data and features, manifest v2 validation, gradient diagnosis, exactly 20 optimizer steps with runtime accumulation 1, then small evaluation and benchmark. Debug outputs use a separate directory.
- [Full finetuning](notebooks/kaggle_train_dinov2_mvpruner_finetune.ipynb): full data/cache preparation or validated downloads, optional finetune resume with per-rank RNG files, two-GPU FP32 training, evaluation, benchmark and artifact archive.

Both notebooks install the binary `tokenizers==0.20.4` wheel first, then `.[mvpruner]`, and assert `transformers==4.46.3`. A missing compatible tokenizer wheel stops installation with a clear error. They preserve `HF_HUB_DISABLE_XET=1` and a 120-second download timeout.

For reuse, set separate dataset repositories for prepared data and the DINO cache. Each repository stores its files at root:

| Dataset repository | Required root files | Local destination |
| --- | --- | --- |
| Prepared data | `manifest.json`, `train.jsonl`, `val.jsonl`, `test.jsonl` | `PROFILE/prepared_data/` |
| DINO cache | `manifest.json`, `index.json`, `features.float16.memmap` | `PROFILE/cache/` |

The notebook validates full split counts and DINO manifest v2 provenance, shape, index coverage and memmap length before reuse. An incompatible cache fails validation. If the cache is absent and no cache repo is configured, the notebook first runs full data preparation to materialize raw images and refresh prepared JSON, then extracts features. Prepared JSON from another machine cannot supply the images required for extraction; its dataset repo is reused when cache files are already available.

The optional model repository must contain a finetune checkpoint and `<filename>.rng/**`, including the RNG manifest and both rank files. Local `finetune_latest.pt` takes priority; resume is verified before training. With no checkpoint, finetuning starts from pretrained T5 and new multimodal modules. The notebooks run only the finetune stage.

You can also set `GITHUB_REPO_URL` and run `python scripts/kaggle_bootstrap.py` for the DINO finetune smoke workflow.

## CLI

```bash
python -m pip install --only-binary=:all: tokenizers==0.20.4
python -m pip install --only-binary=tokenizers -e ".[mvpruner]"
CONFIG=configs/dinov2_t5_efficient_mini_kaggle_2gpu_mvpruner.yaml
python -m efficient_vlm_ad inspect-data --config "$CONFIG"
python -m efficient_vlm_ad prepare-data --config "$CONFIG" --subset full
python -m efficient_vlm_ad prepare-features --config "$CONFIG" --subset full
python -m efficient_vlm_ad diagnose-train --config "$CONFIG" --stage finetune --debug --debug-numerics
accelerate launch --multi_gpu --num_processes 2 --mixed_precision no -m efficient_vlm_ad train --config "$CONFIG" --stage finetune
python -m efficient_vlm_ad evaluate --config "$CONFIG" --checkpoint CHECKPOINT
python -m efficient_vlm_ad benchmark --config "$CONFIG" --checkpoint CHECKPOINT
```

Read [MVPruner details](docs/mvpruner-two-stage.md) for architecture, cache and checkpoint contracts. Supply `HF_TOKEN` through the environment or Kaggle secrets.

## Verification

```bash
python -m pytest tests -q
```
