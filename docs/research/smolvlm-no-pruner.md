# Native SmolVLM-256M without pruning

The isolated [config](../../configs/smolvlm_256m_kaggle_2gpu_no_pruner.yaml) and [Kaggle notebook](../../notebooks/kaggle_train_smolvlm_256m_no_pruner_finetune.ipynb) use `HuggingFaceTB/SmolVLM-256M-Instruct` pinned to revision `7e3e67edbbed1bf9888184d9df282b700a323964`. The native vision encoder is frozen. The pretrained connector, causal language model and language head train together; no random projector, alignment stage, pruning, camera embedding or auxiliary entropy loss is added.

The processor uses one global image per camera, with splitting disabled, in order Front, Front-Left, Front-Right, Back, Back-Left, Back-Right. Frozen native vision outputs contain 1024 tokens of dimension 768 per camera. The native connector maps each camera to 64 tokens of dimension 576, giving 384 visual tokens in the causal prompt. Native image placeholder expansion and multimodal merging preserve the pretrained model's input convention. Offset-aligned labels cover the assistant continuation, including its native separating space, answer and EOS; prompt, image slots, padding and trailing newline after EOS use -100. The model shifts labels exactly once internally. Generation decodes causal continuation tokens only.

## Training protocol

Dataset: `minhdang0901/drivelm-nuscenes-v1-0`, revision `main`, with full split counts 341381 / 19785 / 16817. Images are shared across questions and deduplicated for feature extraction. The seed is 42. FP32 training uses two GPUs, batch 4 per rank and 8 gradient accumulation steps: effective batch 64. Ten finetune epochs use learning rate 1e-4, weight decay 0.05, scheduler gamma 0.9 and gradient clipping at 1.0. Gradient checkpointing defaults to strict boolean `true`. Both PyTorch allocator environment variables are set to `expandable_segments:True` before Torch/CUDA imports in the notebook. There is no automatic batch-size or precision adjustment.

Generation uses 512 maximum new tokens, 3 beams, early stopping and length penalty 1.0. Evaluation batch size is 4; benchmark sample cap is 200. The notebook pins Transformers 4.46.3, Tokenizers 0.20.4 and Accelerate 1.15.0. No FlashAttention installation is required.

Attention is selected separately: native eager for the vision sub-config and native SDPA for the language sub-config. Decoder gradient checkpointing uses `use_reentrant=False`; the cached/frozen vision encoder does not use checkpointing. Teacher forcing passes `use_cache=False`, while generation retains native KV caching. The model's global generation configuration is preserved.

Training averages the native CE of each microbatch within an accumulation window. Accelerate divides by A=8; a tail window of k microbatches multiplies each raw loss by A/k before backward. This is distinct from averaging all supervised tokens across a concatenated batch. Validation uses target-token-weighted CE across the full validation split, and this value selects the best checkpoint. Scheduler gamma is applied once per completed epoch.

## Commands

Every backend command uses its own module; existing DINO/T5 commands and artifacts retain their original paths.

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CONFIG=configs/smolvlm_256m_kaggle_2gpu_no_pruner.yaml
python -m efficient_vlm_ad.smolvlm prepare-data --config "$CONFIG" --subset full
python -m efficient_vlm_ad.smolvlm prepare-features --config "$CONFIG" --subset full
python -m efficient_vlm_ad.smolvlm validate-cache --config "$CONFIG"
python -m efficient_vlm_ad.smolvlm diagnose-train --config "$CONFIG" --debug --debug-numerics --debug-samples 1
python -m efficient_vlm_ad.smolvlm verify-accumulation --config "$CONFIG"
python -m accelerate.commands.launch --multi_gpu --num_processes 2 --num_machines 1 --mixed_precision no --dynamo_backend no -m efficient_vlm_ad.smolvlm train --config "$CONFIG" --stage finetune
python -m efficient_vlm_ad.smolvlm verify-resume-checkpoint --config "$CONFIG" --stage finetune --checkpoint outputs/smolvlm_256m_kaggle_2gpu_no_pruner/checkpoints/finetune_latest.pt
python -m efficient_vlm_ad.smolvlm evaluate --config "$CONFIG" --checkpoint outputs/smolvlm_256m_kaggle_2gpu_no_pruner/checkpoints/finetune_best.pt --batch-size 4
python -m efficient_vlm_ad.smolvlm benchmark --config "$CONFIG" --checkpoint outputs/smolvlm_256m_kaggle_2gpu_no_pruner/checkpoints/finetune_best.pt --max-samples 200
```

Use `--subset smoke` for a small preparation run, `train --max-steps N` to cap total microbatches, and `evaluate --max-samples N` for a short evaluation. Ordinary limited training respects epoch boundaries and persists a partial-epoch cursor; explicit `--debug --max-steps N` may replay a short loader to reach the requested debug count. `diagnose-train` reports dtypes, backend, frozen/trainable modules, gradient checkpointing, loss and gradient norms. `verify-accumulation` runs in one process and compares reference gradients for complete and partial accumulation windows. No model-specific attention debug scope is exposed.

## Cache and disk requirements

All outputs use `outputs/smolvlm_256m_kaggle_2gpu_no_pruner`, with `prepared_data`, `cache`, `checkpoints` and `debug` underneath. The runtime YAML is a separate `/kaggle/working/smolvlm_no_pruner_runtime.yaml`; the tracked config is never overwritten.

Raw cache files are `features.float16.memmap`, `index.json`, `manifest.json`. The memmap shape is `[N,6,1024,768]` and uses float16 on disk, converted to FP32 for training. Its byte length is `N * 6 * 1024 * 768 * 2` (9 MiB per distinct camera group). For N=4072 this is 35.8 GiB; actual N comes from the deduplicated prepared-data index. Raw photos, model files, temporary storage and checkpoints need additional space. The notebook checks the remote sizes of all selected files before HF downloads and requires a further 6 GiB free reserve; unknown remote sizes fail closed. The reserve is a minimum, not a guarantee that raw-image extraction or a complete training run fits the available disk.

Prepared data and feature cache use separate optional HF dataset repositories with exact files at repository root. `hf_hub_download(local_dir=destination)` writes directly into the profile directory and avoids a second copied cache. After files are available, the notebook invokes the backend's `validate-cache` command. It does not reuse DINO geometry validation. Native provenance includes backend `smolvlm_no_pruner`, model ID/revision, processor policy, prepared identity, index identity and geometry (`num_views=6`, `vision_tokens=1024`, `vision_dim=768`, `visual_tokens_per_view=64`, `text_dim=576`). The backend checks exact bytes, mapping coverage and prepared-data identity.

A complete valid cache and its corresponding prepared JSON files work without raw photos. When the cache is missing and no HF cache repo is configured, run full data preparation to materialize images before native feature extraction. A copied prepared JSON alone can contain another machine's image paths and cannot substitute for that step. Full split counts remain mandatory.

## Checkpoint and resume

The notebook selects local `finetune_latest.pt` first, then the configured local checkpoint filename, then the optional HF model repository. HF download includes the selected file and `<filename>.rng/**`. A two-rank checkpoint requires its RNG manifest plus `rank_00000.pt` and `rank_00001.pt`. Standalone `verify-resume-checkpoint` runs before Accelerate launch; it validates the stored checkpoint world size and sidecars even though the validator itself has one process. Training checks distributed compatibility again.

Resume adds `--resume PATH` to `train --stage finetune`. The native backend rejects old T5/DINO profile checkpoints; no cross-architecture migration is attempted. The notebook evaluates `finetune_best.pt` when present, otherwise `finetune_latest.pt`. Its archive includes checkpoints and RNG sidecars, runtime configuration, prepared JSON, metrics, predictions and debug artifacts; it excludes the large cache and raw photos.

Architecture identity is mandatory and checked before loading checkpoint weights or optimizer state. Native weight tying is retained without calling the T5 retie helper. Resume checks the training recipe, world size, counters, sampler seed/epoch/cursor and complete per-rank RNG payloads. CUDA RNG state sizes are recorded and validated; CUDA-capable verification also validates those states with a separate CUDA generator. Inference loads the same weights strictly but does not require training RNG sidecars. Gradient checkpointing is runtime metadata, so changing it does not change checkpoint architecture identity.

## Debug artifacts

`debug/debug_events.jsonl` contains `NUMERICS` entries for `diagnose_forward` and `diagnose_backward`, backend/revision, actual visual-token count, cache/compute dtype, active attention backends and checkpointing state. Backward diagnostics list finite gradient norms and missing trainable gradients. Training adds `TRAINING`, `TRAIN_STEP` and `EPOCH` records with prepared-loader lengths, accumulation config/backend, effective batch, actual window size, raw CE, optimizer counters, skipped updates, gradient norm before clipping and CUDA peak memory when available. Step and epoch records also appear in `debug/train_progress.jsonl`.

`debug/gradient_accumulation_verify.json` compares a full and tail window against an independent mean-of-microbatch-loss reference, preserving model weights/buffers. Its scope is one process; it is not a distributed replay test. The separate offline two-rank Gloo tests cover DDP gradient synchronization and rank-specific resume.

## Measurement limits and validation

The end-to-end benchmark includes host-to-device feature transfer, generation and decoding. The generation region includes connector, native multimodal merge and causal LM generation, excluding vision extraction, DataLoader work and text tokenization. Reported T5 prefill/FLOPs fields are unsupported for this backend and must be `null` with explanatory metadata rather than misleading zeroes.

Offline verification on Windows CPU: **1024 passed, 3 skipped** in **379.87 seconds** for the full repository suite, including **62 new native-backend tests**. Versions used were Torch 2.14.0+cpu, Transformers 4.46.3, Tokenizers 0.20.4 and Accelerate 1.15.0. Existing skips were two CUDA checks and one Windows symlink-permission check; both new Gloo tests executed successfully. The artifact allowlist was extended for the two new files; existing model/config/notebook files were unchanged.

Executed checks include the full offline CLI lifecycle, native raw-image/cache parity, labels and causal CE, frozen vision and trainable connector/LM gradients, real decoder checkpointing parity, beam 1/3 with differently padded multi-sample inputs, full/tail/debug accumulation, skipped optimizer updates, token-weighted validation, strict checkpoint identity, and one-process inference from a two-rank checkpoint without RNG sidecars. Two-rank CPU/Gloo tests compare synchronized gradients against an independent microbatch/rank reference and confirm exact epoch-boundary resume parity for model weights, optimizer, scheduler, counters and both ranks' RNG/sampler state with dropout enabled. A separate mid-epoch test verifies cursor replay at an update boundary while changing runtime checkpointing. Python/notebook syntax and tracked/new-file whitespace checks also passed.

Reproduce with the pinned dependency versions and `python -m pytest -q --durations=15`; run native-only checks with `python -m pytest -q tests/test_smolvlm_*.py`. All model weights and datasets used by these tests are tiny local fixtures. Real pretrained native-model GPU execution, two-GPU Kaggle memory fit, throughput, benchmark values and training quality remain unverified. No large pretrained weights or datasets were downloaded, and no full training was run. The fixed FP32 protocol may exceed available memory despite checkpointing; measured hardware results are required before claiming feasibility.

Implementation references: [pinned model](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/tree/7e3e67edbbed1bf9888184d9df282b700a323964), [Idefics3 model source, Transformers 4.46.3](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/idefics3/modeling_idefics3.py), [native image processor](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/idefics3/image_processing_idefics3.py), [native prompt expansion](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/idefics3/processing_idefics3.py).
