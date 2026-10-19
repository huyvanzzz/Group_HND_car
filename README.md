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

## Kaggle: giới hạn thời gian theo epoch

Notebook `notebooks/kaggle_train_t5_internal_mvpruner_ablations.ipynb` đặt một
deadline chung khi chạy cell cấu hình, trước bước chuẩn bị dữ liệu. Chỉnh
`SESSION_TIME_LIMIT_HOURS` theo thời gian còn lại của session. Chạy lại cell không
gia hạn deadline; session mới cần kernel mới hoặc xóa biến môi trường
`TRAINING_DEADLINE_MONOTONIC` trước khi đặt lại.

Đầu mỗi epoch, training chỉ chạy tiếp khi thời gian còn lại lớn hơn
`reserve + estimated_epoch_duration`. Epoch đầu dùng `INITIAL_EPOCH_MINUTES`;
các epoch tiếp theo dùng thời lượng lớn nhất đã đo cho stage, nhân
`EPOCH_SAFETY_FACTOR`. Thời lượng gồm train, validation, lưu checkpoint và ZIP.
Lịch sử trong `debug/train_progress.jsonl` được dùng lại khi resume cùng profile.
Chỉnh ước lượng epoch đầu cho phù hợp dữ liệu và GPU, nhất là khi đổi cấu hình.

Cuối mỗi epoch, process chính hoàn tất checkpoint và ZIP trước khi mọi GPU cùng
bắt đầu epoch mới. `CHECKPOINT_ZIP_DIR` bật tạo ZIP; `MAX_CHECKPOINT_ZIPS` mặc định
giữ 2 ZIP mới nhất cho profile. Không chạy watcher cũ đồng thời với cơ chế này.
Nếu không đặt deadline, training vẫn chạy đủ số epoch như trước.

Notebook gọi `scripts/run_training_stage.py` cho cả align và finetune. Launcher
chỉ gửi SIGTERM tại `deadline - TRAINING_EMERGENCY_RESERVE_SECONDS` (mặc định 2
phút), sau đó SIGKILL nếu process chưa thoát. Mốc này sát deadline hơn reserve
của training (mặc định 15 phút). Dừng khẩn cấp có thể cắt ngang epoch nếu thời
lượng thực tế vượt ước lượng; checkpoint và ZIP của epoch đã hoàn tất vẫn được giữ.

Các biến môi trường tương ứng cho lệnh train trực tiếp:
`TRAINING_DEADLINE_MONOTONIC` (deadline tuyệt đối từ `time.monotonic()` trên cùng
máy/session), `TRAINING_RESERVE_SECONDS`, `TRAINING_INITIAL_EPOCH_SECONDS`,
`TRAINING_EPOCH_SAFETY_FACTOR`. Launcher dùng thêm
`TRAINING_EMERGENCY_RESERVE_SECONDS`. Ngân sách notebook mặc định là 11 giờ tính
từ cell cấu hình, không tự suy ra giờ session Kaggle bắt đầu.
