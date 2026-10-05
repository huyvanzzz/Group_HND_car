# Efficient VLM AD: T5 Internal Pruning Ablations

This branch is scoped to the T5-internal MVPruner-style pruning experiments:

- `configs/repvit_t5_efficient_mini_internal_pruning_a1_global_topk.yaml`
- `configs/repvit_t5_efficient_mini_internal_pruning_a2_per_view_uniform.yaml`
- `configs/repvit_t5_efficient_mini_internal_pruning_a3_ira.yaml`

The GPA baseline is intentionally not kept on this branch. Use branch `nguyet` for the Question-GPA baseline.

For Kaggle training, use:

```text
notebooks/kaggle_train_t5_internal_mvpruner_ablations.ipynb
```

The notebook can switch between A1/A2/A3 with the `ABLATION` variable and keeps at most two downloadable checkpoint zip files under `/kaggle/working/checkpoint_zips`.

## Entrypoints

```bash
python -m efficient_vlm_ad inspect-data --config configs/repvit_t5_efficient_mini_internal_pruning_a1_global_topk.yaml
python -m efficient_vlm_ad prepare-data --config configs/repvit_t5_efficient_mini_internal_pruning_a1_global_topk.yaml
python -m efficient_vlm_ad prepare-features --config configs/repvit_t5_efficient_mini_internal_pruning_a1_global_topk.yaml

python -m efficient_vlm_ad train --config configs/repvit_t5_efficient_mini_internal_pruning_a1_global_topk.yaml --stage align
python -m efficient_vlm_ad train --config configs/repvit_t5_efficient_mini_internal_pruning_a2_per_view_uniform.yaml --stage align
python -m efficient_vlm_ad train --config configs/repvit_t5_efficient_mini_internal_pruning_a3_ira.yaml --stage align
```

Use each experiment's own align checkpoint for finetune:

```bash
python -m efficient_vlm_ad train --config configs/repvit_t5_efficient_mini_internal_pruning_a1_global_topk.yaml --stage finetune --resume outputs/repvit_t5_efficient_mini_internal_pruning_a1_global_topk/checkpoints/align_best.pt
```

## Notes

All three configs share:

```yaml
architecture: t5_internal_pruning
pruning:
  layer_policy: middle
  keep_ratio: 0.5
  min_keep_per_view: 2
```

Only `selection_policy` changes across A1/A2/A3. See `docs/research/t5-internal-mvpruner-ablation-design.md` for the experiment design.

`HF_TOKEN` must come from the environment. Do not put it in YAML, CLI args, notebooks, or Git.

## Verification

```bash
python -m pytest tests -q
```
