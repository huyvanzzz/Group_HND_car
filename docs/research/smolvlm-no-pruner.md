# Native SmolVLM-256M without pruning

The isolated [config](../../configs/smolvlm_256m_kaggle_2gpu_no_pruner.yaml) and [Kaggle notebook](../../notebooks/kaggle_train_smolvlm_256m_no_pruner_finetune.ipynb) use `HuggingFaceTB/SmolVLM-256M-Instruct` pinned to revision `7e3e67edbbed1bf9888184d9df282b700a323964`. The native vision encoder is frozen. The pretrained connector, causal language model and language head train together; no random projector, alignment stage, pruning, camera embedding or auxiliary entropy loss is added.

The processor uses one global image per camera, with splitting disabled, in order Front, Front-Left, Front-Right, Back, Back-Left, Back-Right. Frozen native vision outputs contain 1024 tokens of dimension 768 per camera. The native connector maps each camera to 64 tokens of dimension 576, giving 384 visual tokens in the causal prompt. Native image placeholder expansion and multimodal merging preserve the pretrained model's input convention. Offset-aligned labels cover the assistant continuation, including its native separating space, answer and EOS; prompt, image slots, padding and trailing newline after EOS use -100. The model shifts labels exactly once internally. Generation decodes causal continuation tokens only.

## Training protocol

Dataset: `minhdang0901/drivelm-nuscenes-v1-0`, revision `main`, with full split counts 341381 / 19785 / 16817. Images are shared across questions and deduplicated for feature extraction. The seed is 42. FP32 training uses two GPUs, batch 4 per rank and 8 gradient accumulation steps: effective batch 64. Ten finetune epochs use learning rate 1e-4, weight decay 0.05, scheduler gamma 0.9 and gradient clipping at 1.0. Gradient checkpointing defaults to strict boolean `true`. Both PyTorch allocator environment variables are set to `expandable_segments:True` before Torch/CUDA imports in the notebook. There is no automatic batch-size or precision adjustment.

Generation uses 512 maximum new tokens, 3 beams, early stopping and length penalty 1.0. Evaluation batch size is 4; benchmark sample cap is 200. The notebook pins Transformers 4.46.3, Tokenizers 0.20.4 and Accelerate 1.15.0. No FlashAttention installation is required.

Attention is selected separately: native eager for the vision sub-config and native SDPA for the language sub-config. Decoder gradient checkpointing uses `use_reentrant=False`; the cached/frozen vision encoder does not use checkpointing. Teacher forcing passes `use_cache=False`, while generation retains native KV caching. The model's global generation configuration is preserved.

Kaggle setup prints the global `pip check` report, then strictly validates the workflow's full active dependency tree (project, pinned model libraries, Kaggle clients and metrics), including transitive requirements, environment markers and requested extras. Missing or incompatible workflow dependencies still stop setup. Conflicts belonging only to unrelated preinstalled apps are reported without blocking training; no packages are uninstalled and pins are unchanged. The success marker is `[ENV] SmolVLM dependency tree OK`. After updating this notebook, restart the Kaggle session/kernel and run from the first cell to avoid using previously imported library versions. This metadata check is followed by imports of the actual Idefics3 classes; it does not replace GPU diagnose.

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

The tracked CLI config keeps its relative paths under `outputs/smolvlm_256m_kaggle_2gpu_no_pruner`. The Kaggle notebook puts newly extracted or HF-downloaded features in `/kaggle/tmp/cache_smolvlm`; subsequent sessions can read a public Kaggle Dataset directly from Input. Prepared data, checkpoints and debug stay in the existing output profile under working. The separate runtime YAML `/kaggle/working/smolvlm_no_pruner_runtime.yaml` sets `cache.dir` to the actual tmp/Input path; the tracked config is never overwritten.

Raw cache files are `features.float16.memmap`, `index.json`, `manifest.json`. Shape `[N,6,1024,768]` uses FP16 on disk: `N * 6 * 1024 * 768 * 2` bytes (9 MiB per distinct camera group). For N=4072 this is 35.8 GiB; actual N comes from prepared-data deduplication. Samples are converted to FP32 in RAM before batching/device transfer; the full cache is never converted or written back to disk as FP32.

Raw photos, model files and checkpoints need additional space. Checks use the actual cache filesystem and keep a 6 GiB checkpoint reserve on the output filesystem, combining requirements once when they are shared. Unknown remote sizes fail closed. Existing partial downloads already occupy space; checks conservatively require room for a fresh remaining file rather than promising SDK byte-level resume. Moving to tmp does not guarantee extraction fits the session disk.

Prepared data and feature cache keep separate optional HF dataset repos, with their files at root. Set `PREPARED_DATA_HF_REPO_ID` to the existing prepared repo when local prepared files are missing, and `CACHE_HF_REPO_ID` to the cache repo. Complete valid prepared files are reused without rewriting; conflicting partial files are rejected. Cache downloads use `hf_hub_download(local_dir=destination)` directly into tmp, pinned to one resolved Hub commit per attempt. Manifest/index identity is checked before the large feature download. Partial caches remember their source/commit; a different source is rejected instead of overwriting them.

Native validation retains backend/model revision, processor policy, prepared hashes, index identity, geometry, byte length, coverage and finite values. Production geometry is `num_views=6`, `vision_tokens=1024`, `vision_dim=768`, `visual_tokens_per_view=64`, `text_dim=576`. Metadata fingerprint is not a checksum of the full memmap.

A complete valid cache and matching prepared JSON work without raw photos. Without a cache source, extraction requires photos accessible at the existing prepared manifest root. The notebook never reruns `prepare-data` or rewrites that root to repair a mismatch. Missing prepared data or raw photos causes a clear error. Full split counts remain mandatory.

### Public Kaggle cache, prepared data kept separate

The notebook pins `kaggle==2.2.5` and `kagglehub==1.0.3` (Python >=3.11), preserves model pins and checks `pip check`. For first extraction/HF download leave `CACHE_KAGGLE_DATASET = ''` and add Secret `KAGGLE_API_TOKEN`, or legacy `KAGGLE_USERNAME` + `KAGGLE_KEY`. Credentials load before SDK import and are never written to YAML/logs/dataset.

After validation the cache is published as public **`<authenticated_owner>/cache-smolvlm`**, title **`cache_smolvlm`**, before diagnose/training. Payload is exactly:

```text
features.float16.memmap
index.json
manifest.json
README.md
```

Prepared manifests and train/val/test records are never uploaded. `prepared_hashes` in the cache manifest identify the separate prepared data. Stable sibling staging uses a verified hardlink to the memmap rather than a feature copy/archive. No images, checkpoint, credentials or download metadata are staged. Source-data attribution and conditions remain applicable.

An identical existing public/ready dataset is reused. Mismatch, private state, HTTP 401 or failed processing stops the notebook. HTTP 404 means absent. If status lookup returns HTTP 403, confirm the configured owner and enumerate every page of the authenticated personal dataset listing (`mine=True`), including private datasets. Only a completed listing without the target allows the new-dataset path; failed/malformed/repeated pagination or a listed-but-inaccessible target still stops setup. This fallback does not verify create permissions; upload/create errors continue to stop the workflow. No version creation, overwrite, deletion or visibility change occurs. Processing is polled every 15 seconds for up to 30 minutes per cell run. On upload/error/timeout keep tmp cache/staging and rerun **only the publish cell**. Training stays blocked until publication succeeds. The cell prints the dataset URL/handle. SDK upload resume may help but is not guaranteed at an exact byte offset.

For later sessions set `CACHE_KAGGLE_DATASET = 'your_kaggle_username/cache-smolvlm'` (optional `/versions/N` is accepted). KaggleHub attaches public Input without publisher credentials. The backend reads its returned path directly, validates against separate prepared files, and does not copy or republish it. Attachment mode is restricted to Kaggle notebooks to prevent an accidental large local download. Priority: explicit Kaggle handle, otherwise valid tmp, otherwise HF, otherwise extraction. Invalid explicit sources never silently fall back.

Tmp is session scratch; a ready Kaggle Dataset is the durable copy. Storage checks print source/revision/path, required/free bytes and fingerprint. The notebook records a `STORAGE` event with publish/reuse status in `debug/debug_events.jsonl`. The archive excludes cache and staging. Storage does not change checkpoint selection, architecture identity, training recipe or model computation.

## Checkpoint and resume

The notebook selects local `finetune_latest.pt` first, then the configured local checkpoint filename, then the optional HF model repository. HF download includes the selected file and `<filename>.rng/**`. A two-rank checkpoint requires its RNG manifest plus `rank_00000.pt` and `rank_00001.pt`. Standalone `verify-resume-checkpoint` runs before Accelerate launch; it validates the stored checkpoint world size and sidecars even though the validator itself has one process. Training checks distributed compatibility again.

Resume adds `--resume PATH` to `train --stage finetune`. The native backend rejects old T5/DINO profile checkpoints; no cross-architecture migration is attempted. The notebook evaluates `finetune_best.pt` when present, otherwise `finetune_latest.pt`. Its archive includes checkpoints and RNG sidecars, runtime configuration, prepared JSON, metrics, predictions and debug artifacts; it excludes the large cache and raw photos.

Architecture identity is mandatory and checked before loading checkpoint weights or optimizer state. Native weight tying is retained without calling the T5 retie helper. Resume checks the training recipe, world size, counters, sampler seed/epoch/cursor and complete per-rank RNG payloads. CUDA RNG state sizes are recorded and validated; CUDA-capable verification also validates those states with a separate CUDA generator. Inference loads the same weights strictly but does not require training RNG sidecars. Gradient checkpointing is runtime metadata, so changing it does not change checkpoint architecture identity.

## Debug artifacts

`debug/debug_events.jsonl` contains `NUMERICS` entries for `diagnose_forward` and `diagnose_backward`, backend/revision, actual visual-token count, cache/compute dtype, active attention backends and checkpointing state. Backward diagnostics list finite gradient norms and missing trainable gradients. Training adds `TRAINING`, `TRAIN_STEP` and `EPOCH` records with prepared-loader lengths, accumulation config/backend, effective batch, actual window size, raw CE, optimizer counters, skipped updates, gradient norm before clipping and CUDA peak memory when available. Step and epoch records also appear in `debug/train_progress.jsonl`.

`debug/gradient_accumulation_verify.json` compares a full and tail window against an independent mean-of-microbatch-loss reference, preserving model weights/buffers. Its scope is one process; it is not a distributed replay test. The separate offline two-rank Gloo tests cover DDP gradient synchronization and rank-specific resume.

## Measurement limits and validation

The end-to-end benchmark includes host-to-device feature transfer, generation and decoding. The generation region includes connector, native multimodal merge and causal LM generation, excluding vision extraction, DataLoader work and text tokenization. Reported T5 prefill/FLOPs fields are unsupported for this backend and must be `null` with explanatory metadata rather than misleading zeroes.

Initial native-backend verification on Windows CPU: **1024 passed, 3 skipped** in **379.87 seconds** for the full repository suite, including **62 new native-backend tests**. Versions used were Torch 2.14.0+cpu, Transformers 4.46.3, Tokenizers 0.20.4 and Accelerate 1.15.0. Existing skips were two CUDA checks and one Windows symlink-permission check; both new Gloo tests executed successfully. The artifact allowlist was extended for the two new files; existing model/config/notebook files were unchanged.

Executed checks include the full offline CLI lifecycle, native raw-image/cache parity, labels and causal CE, frozen vision and trainable connector/LM gradients, real decoder checkpointing parity, beam 1/3 with differently padded multi-sample inputs, full/tail/debug accumulation, skipped optimizer updates, token-weighted validation, strict checkpoint identity, and one-process inference from a two-rank checkpoint without RNG sidecars. Two-rank CPU/Gloo tests compare synchronized gradients against an independent microbatch/rank reference and confirm exact epoch-boundary resume parity for model weights, optimizer, scheduler, counters and both ranks' RNG/sampler state with dropout enabled. A separate mid-epoch test verifies cursor replay at an update boundary while changing runtime checkpointing. Python/notebook syntax and tracked/new-file whitespace checks also passed.

Reproduce with the pinned dependency versions and `python -m pytest -q --durations=15`; run native-only checks with `python -m pytest -q tests/test_smolvlm_*.py`. All model weights and datasets used by these tests are tiny local fixtures. Real pretrained native-model GPU execution, two-GPU Kaggle memory fit, throughput, benchmark values and training quality remain unverified. No large pretrained weights or datasets were downloaded, and no full training was run. The fixed FP32 protocol may exceed available memory despite checkpointing; measured hardware results are required before claiming feasibility.

Implementation references: [pinned model](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/tree/7e3e67edbbed1bf9888184d9df282b700a323964), [Idefics3 model source, Transformers 4.46.3](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/idefics3/modeling_idefics3.py), [native image processor](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/idefics3/image_processing_idefics3.py), [native prompt expansion](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/idefics3/processing_idefics3.py).


## Cache storage verification (2026-11-09)

The full repository suite after the storage/notebook changes completed with **1065 passed, 3 skipped, 16 warnings in 396.41 seconds** on Windows CPU. This includes both native two-rank Gloo tests, all existing profiles, the real cache-only Input lifecycle (diagnose, accumulation verify, train, resume, evaluate and benchmark), and **31 publishing + 9 storage** tests. Public network operations are replaced by transport fixtures; model computations use genuine tiny local Idefics3 weights. The notebook readiness tests also execute its publish/logging cell and verify that cache resolution invalidates stale publication readiness and binds it to the current cache fingerprint/runtime path.

Regressions cover auto-compressed Kaggle metadata, existing dataset reuse, processing states, conflict/error/timeout, hardlink allowlist, source identity, preserved prepared bytes, changed upstream HF HEAD during interrupted download, and separate/shared filesystem accounting. Python/notebook syntax, new-file whitespace, `git diff --check`, and environment `pip check` passed. Existing skips remain two CUDA-only checks and one Windows symlink-permission check; warnings concern CPU pin_memory and the tiny beam-1 fixture's early_stopping flag.

Reproduce with the project src directory on `PYTHONPATH`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, the pinned model dependencies, and `python -m pytest -q --durations=15`. Kaggle SDK APIs were checked against the pinned SDK source; no credentials, real publication, large model/dataset download or full training was used. Actual Kaggle publication, disk quota behavior, GPU fit and throughput still require the configured notebook run. No model, training, checkpoint, config or older-profile notebook behavior was changed.

When no complete feature cache is available, extraction reuses existing local source images or automatically downloads only the referenced photos from `data.hf_repo_id` at a resolved commit. Photos are deduplicated and stored beside the tmp feature cache in `cache_smolvlm.raw-images`; private source repositories use HF_TOKEN. Free-space checks include photos plus feature bytes plus the output checkpoint reserve. Existing prepared JSON/manifest bytes are preserved, even when their image root belongs to an earlier session; extraction uses a runtime image-root override. A valid feature cache never triggers photo downloads. An explicitly configured local dataset with missing photos still fails instead of silently switching sources.
