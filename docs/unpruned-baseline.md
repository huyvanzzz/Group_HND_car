# Baseline giữ nguyên toàn bộ token

[Notebook Kaggle](../notebooks/kaggle_train_dinov2_unpruned_finetune.ipynb) dùng [config riêng](../configs/dinov2_t5_efficient_mini_kaggle_2gpu_unpruned.yaml). Toàn bộ **384 visual tokens = 6 camera × 64 tokens/camera** đi qua linear projector vào native T5 encoder. Không tính relevance/diversity score, chọn top-k hoặc prune ở bất kỳ layer nào. Đặt budget MVPruner thành 384/384 vẫn tính scoring nên không thể thay thế baseline này.

DINOv2-S được đóng băng. Revision, preprocessing, projector, pretrained T5-Efficient-Mini, FP32, batch size 16 mỗi rank, accumulation 2, learning rate 0.0001, 10 epoch finetune, generation và evaluation giống MVPruner. Effective batch trên hai GPU là 64. Chỉ chạy finetune từ pretrained T5 và các module multimodal mới; không chạy alignment.

## Chạy trên Kaggle

Bật hai GPU, cung cấp secret `HF_TOKEN`, chọn branch `huy-dinov2` rồi chạy notebook từ đầu. Cell đầu tiên đặt `GRADIENT_CHECKPOINTING = False` và cả hai biến allocator trước khi import torch hoặc kiểm tra CUDA. Cờ chỉ nhận boolean `True` hoặc `False`.

Notebook sao chép YAML vào `/kaggle/working/dinov2_unpruned_runtime.yaml`, ghi cờ vào `training.gradient_checkpointing`, rồi cập nhật `CONFIG` và `cfg` trước prepare-data, diagnose và train. YAML trong repo không bị ghi đè. Settings in ra ghi rõ 384 tokens và cờ thực tế; archive chứa cả runtime YAML.

Nếu gặp CUDA OOM, tự đổi `GRADIENT_CHECKPOINTING = True`, **restart kernel**, chạy lại các cell và resume từ `finetune_latest.pt` tương thích gần nhất. Notebook không tự retry hoặc đổi batch size. Gradient checkpointing giảm activation lưu trong lúc train bằng cách tính lại một phần forward khi backward; không thay đổi số visual tokens. Khi evaluate/generate ở chế độ eval không thực hiện phép tính lại của gradient checkpointing. Môi trường kiểm tra hiện tại chỉ có CPU, vì vậy chưa xác nhận mức VRAM hoặc bảo đảm chạy vừa GPU Kaggle.

Các lệnh CLI chạy từ repo root sau khi cài dependencies:

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m pip install --only-binary=:all: tokenizers==0.20.4
python -m pip install --only-binary=tokenizers -e ".[mvpruner]"
CONFIG=configs/dinov2_t5_efficient_mini_kaggle_2gpu_unpruned.yaml
PROFILE=outputs/dinov2_t5_efficient_mini_kaggle_2gpu_unpruned
python -m efficient_vlm_ad prepare-data --config "$CONFIG" --subset full
python -m efficient_vlm_ad prepare-features --config "$CONFIG" --subset full
python -m efficient_vlm_ad diagnose-train --config "$CONFIG" --stage finetune --debug --debug-numerics
accelerate launch --multi_gpu --num_processes 2 --num_machines 1 --mixed_precision no --dynamo_backend no -m efficient_vlm_ad train --config "$CONFIG" --stage finetune
CHECKPOINT="$PROFILE/checkpoints/finetune_best.pt"
python -m efficient_vlm_ad evaluate --config "$CONFIG" --checkpoint "$CHECKPOINT" --batch-size 16
python -m efficient_vlm_ad benchmark --config "$CONFIG" --checkpoint "$CHECKPOINT"
```

CLI dùng cờ mặc định false của YAML; để bật, tạo YAML runtime riêng, đặt `training.gradient_checkpointing: true` và trỏ `CONFIG` đến bản đó. Export allocator trước khi khởi động bất kỳ tiến trình torch nào. Nếu chưa có best checkpoint, dùng `finetune_latest.pt`. Extra dependencies vẫn có tên lịch sử `mvpruner`; fusion được chọn là `unpruned`.

## Dữ liệu, cache và resume

Output riêng ở `outputs/dinov2_t5_efficient_mini_kaggle_2gpu_unpruned/`, gồm `prepared_data/`, `cache/`, `checkpoints/`. Config và notebook MVPruner không bị thay đổi.

Có thể tái sử dụng DINO cache vì backbone và preprocessing giống nhau, nhưng phải dùng đúng prepared files tương ứng. Không ghép cache với JSON từ lần prepare khác: camera paths phải khớp keys trong cache index. Notebook kiểm tra full split counts (train 341381, val 19785, test 16817), manifest v2 provenance, shape `[N,6,64,384]`, index coverage và kích thước memmap. Prepared JSON từ máy khác không thay thế raw images khi cần extract features.

Đặt prepared data và features trong hai Hugging Face dataset repositories riêng, với files ở root:

| Repository | Files tại root |
| --- | --- |
| Prepared data | `manifest.json`, `train.jsonl`, `val.jsonl`, `test.jsonl` |
| DINO cache | `manifest.json`, `index.json`, `features.float16.memmap` |

Nếu cache thiếu và không cấu hình cache repo, notebook tải raw images, refresh prepared data rồi extract features. Các bước validation, download và kiểm tra RNG giữ nguyên hợp đồng của notebook full.

Resume chỉ nhận checkpoint **unpruned finetune** tương thích. Runtime kiểm tra architecture/signature và từ chối checkpoint MVPruner trước khi restore model, optimizer, scheduler và RNG. Local `finetune_latest.pt` được ưu tiên. Model repo tùy chọn phải chứa checkpoint và toàn bộ `<filename>.rng/**`, gồm manifest cùng RNG của cả hai rank.

```bash
CHECKPOINT="$PROFILE/checkpoints/finetune_latest.pt"
python -m efficient_vlm_ad verify-resume-checkpoint --config "$CONFIG" --stage finetune --checkpoint "$CHECKPOINT"
accelerate launch --multi_gpu --num_processes 2 --num_machines 1 --mixed_precision no --dynamo_backend no -m efficient_vlm_ad train --config "$CONFIG" --stage finetune --resume "$CHECKPOINT"
```

Notebook lưu `dinov2_unpruned_finetune_artifacts.tar.gz` riêng, gồm runtime YAML, checkpoints, metrics, benchmark, predictions và debug logs. Benchmark đo ở mức feature: không tính chi phí DINO extraction. Khi evaluate/generate ở chế độ eval không có chi phí tính lại của gradient checkpointing. Cần so sánh cùng dữ liệu, generation settings và thiết bị với MVPruner; chưa có kết quả GPU từ môi trường CPU này.

## Kiểm chứng implementation

Ngày 2026-10-27, toàn bộ test suite offline đạt **352 passed, 3 skipped** trong 457,78 giây, với Transformers 4.46.3 và tokenizers 0.20.4. Các skip liên quan CUDA và symbolic link trên Windows.

Tests kiểm tra đủ 384 visual tokens ở mọi encoder block, masks và gradients, native T5 tương đương MVPruner identity khi dùng cùng active weights, checkpointing bật/tắt có logits/gradients tương đương, generation vẫn dùng KV cache, cùng vòng đời finetune → diagnose → verify-resume → resume từ false sang true → evaluate → benchmark. Notebook có tests áp dụng cờ vào YAML runtime mà không thay đổi config gốc.

Parity MVPruner trước/sau vẫn khớp 9/9 trường hợp, gồm geometry 384 → 196 → 49. Chưa chạy pretrained-model finetune, đo giảm VRAM hoặc smoke DDP hai GPU vì môi trường kiểm thử chỉ có CPU. Không tải dataset/model và không tự commit/push; trước khi chạy notebook Kaggle, branch được clone phải chứa implementation baseline này.
