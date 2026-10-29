# Efficient VLM For Autonomous Driving

Config-driven EM-VLM4AD experiments on Kaggle. The active profile is DINOv2-S + T5-Efficient-Mini with two-stage MVPruner:

- `configs/dinov2_t5_efficient_mini_kaggle_2gpu_mvpruner.yaml`: frozen DINOv2-S, 64 tokens per camera, learned linear projector, MVPruner 196 → 49 tokens, FP32 finetuning.

The runtime in `src/efficient_vlm_ad/` supports MVPruner and a native unpruned baseline. GPA, question-guided router and instruction-adapter execution paths and their baseline config have been removed. The upstream reference in `em_vlm4ad/` and research documents remain historical references; standalone vision utilities remain available.

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

## Unpruned baseline

[Full-token finetuning notebook](notebooks/kaggle_train_dinov2_unpruned_finetune.ipynb) uses [the separate unpruned config](configs/dinov2_t5_efficient_mini_kaggle_2gpu_unpruned.yaml): all 384 visual tokens enter native T5 without scoring or pruning. It runs only finetuning from pretrained T5 and new multimodal modules, with separate output/cache/checkpoint paths and unpruned-only checkpoint resume. DINO, projector, training and generation settings match the MVPruner profile. See [baseline commands and reuse rules](docs/unpruned-baseline.md).

Gradient checkpointing defaults to `False`. The notebook sets both allocator environment variables before torch starts, writes a separate runtime YAML and archives it with results. After a CUDA OOM, manually set `GRADIENT_CHECKPOINTING = True`, restart the kernel and resume the last compatible checkpoint. It does not automatically retry or change batch size. Benchmark timing excludes DINO feature extraction; GPU memory and performance have not been validated in the CPU test environment.

## Historical MVPruner compatibility

Existing MVPruner Stage-1-v2 checkpoints still load for training/resume, verification, evaluation and benchmarking. The loader validates the active architecture, then drops only the four unused historical GPA tensors and ignores an explicit list of inactive signature fields. Unknown weights, missing active weights, active historical adapters and incompatible MVPruner signatures still fail. Optimizer, scheduler and per-rank RNG state are preserved; source checkpoint files are not rewritten.

Old MVPruner YAML may retain `adapter: {name: none}`, `training.gpa_hidden_size` and inactive router fields. They are ignored. The historical `fusion.top_k` fallback remains when `stage2_top_k` is absent; an explicit `stage2_top_k` wins. GPA/router configs and active adapters are rejected. DINO feature-cache files and manifest identity are unchanged.

New model instances no longer allocate the unused GPA module, so total parameter counts decrease and fresh random initialization may differ for the same seed. Given the same active weights and RNG at execution time, the MVPruner computation is preserved.

## Verification

```bash
python -m pytest tests -q
```

## Camera-query Stage 2

[Hướng dẫn camera-query](docs/camera-query-stage2.md), [config riêng](configs/dinov2_t5_efficient_mini_kaggle_2gpu_camera_query.yaml) và [notebook Kaggle hai GPU](notebooks/kaggle_train_dinov2_camera_query_finetune.ipynb) chạy finetune-only từ pretrained T5: giữ Stage 1/IRA, nén 196 thành 49 tokens, entropy weight 0.5, seed 42. Profile mới sửa accumulation/scheduler riêng và chỉ resume checkpoint camera-query tại ranh giới epoch.

Question-aware camera-query has a separate [research guide](docs/research/question-aware-camera-query.md), [config](configs/dinov2_t5_efficient_mini_kaggle_2gpu_camera_query_qaware.yaml), and [two-GPU finetune notebook](notebooks/kaggle_train_dinov2_camera_query_qaware_finetune.ipynb). It uses question-conditioned scoring and prefix-DP allocation; GPU performance and quality require measurement.
