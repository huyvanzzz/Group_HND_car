# Stage 1 CCTS v2

Reference: Zizzzzzzz/MVPruner commit `53ecc443fd7664a92e71ab6213f5611ef1ec5c7d`,
`transformers/src/transformers/models/llama/modeling_llama_mvpruner.py`.

Stage 1 runs before T5 block 0, on projected visual features enriched with
spatial/camera embeddings and valid question/prompt embeddings. Modal embeddings
are added after selection. DINO cache contents and schema are unchanged.

- DRA: mean nearest-neighbor cosine distance per camera; exclude the diagonal.
  Normalize `(s - max(s)) / (max(s) - min(s) + 1e-5)`, then softmax per sample.
- Unlike upstream independent floor/clamp, bounded allocation keeps exactly 196
  tokens across cameras, with minimum one and capacity 64 per camera.
- Relevance intentionally remains shifted max cosine over unmasked text tokens:
  `clamp((1 + max_cosine) / 2, 0, 1)`. Upstream code uses mean dot product.
- CCTS seeds with maximum relevance. Subsequent candidates maximize minimum
  distance to the selected set times relevance. Running minimum updates are
  mathematically equivalent to recomputing distances to all selected tokens.
- Ties choose the lowest local index. Output is sorted in original token order.
- Hard decisions do not have selection gradients; gathered values still train
  the projector. Stage 2 IRA/Gumbel STE is unchanged.

The previous implementation initialized running uniqueness with the minimum
distance to *all* camera tokens, making later updates ineffective. Its selected
set was equivalent to static top-k scores. Tests now compare against an
independent sequential Algorithm-1 oracle, not just shapes and budgets.

Checkpoint architecture signature records `mvpruner.stage1_algorithm_version=2`.
Old MVPruner checkpoints must not resume/evaluate silently under the new
selection behavior. Start finetune from pretrained T5; reuse existing DINO cache.
If the notebook finds a local `finetune_latest.pt` from v1, move the old
checkpoint directory elsewhere before starting v2, keeping `prepared_data` and
`cache` in place. Leave the optional checkpoint HF repository empty for a fresh run.

Benchmark without downloads:

```powershell
$env:PYTHONPATH='src'
python scripts/benchmark_stage1.py --device cpu --warmup 20 --iterations 100
# On Kaggle, replace cpu with cuda. JSON reports wall/CUDA times and memory.
```

CPU measurements do not establish Kaggle GPU latency or end-to-end speedup.

## Verification (2026-10-21)

Offline full-suite run: 308 passed, 1 skipped. Additional updated algorithm,
diagnostic and architecture tests: 139 passed, 1 CUDA test skipped. Both
align-to-finetune and fresh finetune-only smoke paths passed train/evaluate/benchmark.

CPU benchmark, FP32, one thread, 20 warmups and 100 measured iterations:

| Batch | Previous Stage 1 mean ms | Stage 1 v2 mean ms | Previous allocation ms | New allocation ms |
|---|---:|---:|---:|---:|
| 1 | 46.062 | 10.048 | 25.493 | 0.819 |
| 2 | 46.670 | 11.096 | 36.832 | 0.939 |
| 8 | 60.929 | 23.250 | 37.853 | 2.045 |

Hardware: Intel Family 6 Model 140; PyTorch 2.14.1+cpu, Transformers 4.46.3.
Full report: `outputs/stage1_v2_cpu_benchmark.json` (generated local artifact).
Timing cases are measured separately and are subject to host scheduling noise;
their means need not add up. Previous Stage 1 and v2 have different semantics;
allocator and reference-vs-optimized CCTS equivalence is checked independently.
Kaggle CUDA performance and end-to-end real-model speed remain unmeasured.
