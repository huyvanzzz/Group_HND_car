# SmolVLM tối ưu trên Vast.ai

Bản sao code/config đang dùng trên server, dựa trên source SmolVLM tại commit
`7638958d6bbe3e60a91f0ebd6c2e97e035d4f16c` của repository này. Không chứa
dataset, feature cache, checkpoint, RNG sidecars, log hoặc credentials.

## Cấu hình đã chọn

- Train: hai GPU, batch 6/GPU, accumulation 8, effective batch 96, tổng 8 epoch.
- AMP FP16, feature CPU/H2D FP16; weights, AdamW và native CE giữ FP32.
- GradScaler với initial scale 256, lưu/khôi phục cùng checkpoint.
- Hai persistent DataLoader worker/GPU; `no_sync` ngoài ranh giới update.
- Gradient checkpointing tắt; cả hai biến allocator dùng `expandable_segments:True`.
- Validation: FP16, batch 32/GPU, hai GPU, tổng hợp loss theo target-token count.
- Evaluate: FP32, batch 32/GPU, hai GPU, beam 3, tối đa 512 token sinh.
- Benchmark: một GPU, batch 1, tối đa 100 mẫu; giữ protocol benchmark native.

`optimized.yaml` giữ `runtime.precision: fp32` theo schema native. Chế độ
**compute AMP FP16** được đặt trong `optimization.json` và entrypoint, không
phải training thuần FP32. Kiến trúc, loss, LR và cache không thay đổi.

## Các file và cách chạy trên server hiện tại

Đặt các file trong thư mục này tại `/workspace/smolvlm-vast/`. Đây là snapshot
cho môi trường hiện tại, **chưa phải launcher portable**: code dùng đường dẫn
repo `/workspace/huy-smolvlm-256m-no-pruner` và Python
`/workspace/smolvlm-venv/bin/python`. Chạy `run_training.py` từ thư mục repo.

```bash
supervisorctl start smolvlm_train
tail -f /workspace/smolvlm-vast/logs/training.log
```

Service `smolvlm_train` trên server gọi `run_training.py`. Script chạy train,
evaluate best (hoặc latest nếu thiếu best), benchmark rồi archive artifacts.
Nó không tự dừng instance Vast. Credentials nếu cần phải được cấp bên ngoài;
không ghi vào các file này.

`optimized_entry.py` tạo module lifecycle riêng từ source native bằng các
thay thế có kiểm tra, không ghi đè file production. `distributed_evaluate.py`
chia mẫu không padding/lặp, ghép lại theo source index và tính metrics một lần.
Checkpoint FP16 thêm scaler state và identity recipe; không dùng checkpoint
FP32 cũ để resume luồng tối ưu.

`OPTIMIZED_READY.json` lưu checksum và kết quả kiểm chứng trên server hiện tại.
Nó không chứng minh môi trường mới đủ VRAM. Launcher từ chối khi checksum đổi;
nếu thay code/config hoặc chuyển máy, cần kiểm chứng lại trước khi cập nhật
manifest, không đơn thuần đặt `ready: true`.

## Kiểm chứng đã thực hiện

- Smoke train hai GPU với cửa sổ accumulation đủ/dư, validation, latest/best,
  RNG từng rank, GradScaler và resume.
- Chia validation/evaluate số mẫu lẻ, không mất hoặc lặp mẫu. Validation khớp
  reference trong tolerance FP16; metrics được tính sau khi ghép predictions.
- Stress train bằng sáu mẫu dài nhất khác nhau: không OOM, VRAM trống thấp nhất
  quan sát được khoảng 4,33 GiB/GPU.
- Một lượt timing train ngắn: 55,20–56,41 samples/s tổng hai GPU. Đây không phải
  cam kết tốc độ toàn epoch hoặc chất lượng model.
- `no_sync` từ cùng weights: relative gradient L2 difference khoảng 0,016%
  trong cửa sổ đủ; tail một microbatch khớp hoàn toàn. Không phải bitwise parity.

Các script `verify_optimized.py`, `check_validation_shards.py` và
`check_eval_changes.py` là probe riêng. Chúng ghi vào thư mục smoke, không chạy
full training. Smoke evaluate chỉ dùng năm mẫu và rút ngắn generation để kiểm
tra lifecycle; không dùng metrics đó làm kết quả thực nghiệm.

Môi trường đã đo: hai RTX 4090, Python 3.12, Torch 2.14.1+cu130,
Transformers 4.46.3, Tokenizers 0.20.4 và Accelerate 1.15.0.
