# Kaggle MVPruner finetune notebooks

Tạo hai notebook Kaggle self-contained cho profile
`configs/repvit_t5_efficient_mini_kaggle_2gpu_mvpruner.yaml`.

- Notebook debug dùng một runtime config chỉ đổi output/cache directory để không
  làm bẩn artifact train thật. Nó giữ nguyên `runtime.precision: fp32`, chuẩn bị
  smoke data/cache, diagnose, train finetune 20 steps và evaluate/benchmark 32 mẫu.
- Notebook train thật dùng trực tiếp config đã track, chuẩn bị full data/cache,
  train `finetune` từ pretrained khi chưa có checkpoint, hoặc resume
  `finetune_latest.pt` cùng stage khi checkpoint tồn tại. Không chạy align và
  không yêu cầu align checkpoint.
- Cả hai cài extra `.[mvpruner]`, kiểm tra Transformers 4.46.3, dùng hai GPU với
  `--mixed_precision no`, lấy `HF_TOKEN` từ Kaggle Secrets và in các artifact cần
  thiết để debug/tải về.

Không chỉnh sửa config gốc và không thay đổi model/training code.
