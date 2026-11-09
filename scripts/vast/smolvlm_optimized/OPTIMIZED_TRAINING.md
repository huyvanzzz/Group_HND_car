# Vast optimized training

Server-only entrypoint; tracked source and original runtime.yaml untouched.

Config: optimized.yaml + optimization.json. Native model, loss, LR and eight epochs unchanged. Batch6/GPU, accumulation8, FP16 autocast, FP16 feature transport, FP32 weights/AdamW/native CE, GradScaler(initial_scale256), two persistent workers/GPU, no_sync except window boundary, checkpointing off, expandable_segments True.

Checkpoint payload adds grad_scaler_state_dict; resume checks optimization recipe and restores scaler plus original rank RNG/sampler/optimizer/scheduler. Old FP32 training checkpoints are not accepted for optimized training. Inference uses original FP32 native path/protocol.

Verification: real two-GPU train full/tail windows -> validation -> latest/best with rank RNG -> strict verify -> resume passed. Fresh FP16 no_sync comparison: relative gradient L2 difference 0.00015848 in full window; single-microbatch tail identical. Not bitwise parity and not a quality guarantee.

Smoke files live only in optimized_smoke and are never used for full training.

Start manually when approved: supervisorctl start smolvlm_train
Watch: tail -f /workspace/smolvlm-vast/logs/training.log
Full job is NOT started by setup.

Validation and evaluate: batch32/GPU on two GPUs. Validation shards without padding and reduces token-weighted loss sums/counts. Evaluate merges predictions by source indices, computes metrics once on rank0; retains native FP32 beam3 generation. Benchmark: one GPU, batch1, at most100 samples, original benchmark protocol. Two-GPU uneven-shard validation/evaluation smoke passed.
