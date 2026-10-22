# Stage 1 MVPruner v2 trên huy-router

Chuyển riêng DRA, CCTS và allocator từ nhánh `huy-dinov2`. Backbone vẫn là
RepViT, feature cache `[B,6,49,512]`, projector sang 384 chiều và ngân sách
`294 → 196 → 49`. Config, notebook và dependency không đổi.

Nguồn đối chiếu: MVPruner chính thức commit
`53ecc443fd7664a92e71ab6213f5611ef1ec5c7d`, file
`transformers/src/transformers/models/llama/modeling_llama_mvpruner.py`.
Baseline nhánh này: `9779658`.

- DRA giữ cosine distance và mean nearest-neighbor distance. Chuẩn hóa camera
  bằng `(s-max(s))/(max(s)-min(s)+1e-5)` rồi softmax theo từng sample.
- Budget vẫn được phân bổ chính xác 196 token, minimum 1, capacity 49/camera.
  Đây là điều chỉnh so với floor/clamp độc lập của repo chính thức.
- Relevance giữ shifted max cosine của code hiện tại, không chuyển sang mean
  dot-product của repo chính thức.
- CCTS chọn seed bằng relevance; các lượt sau dùng minimum distance tới tập
  đã chọn nhân relevance. Cập nhật minimum tích lũy tương đương greedy tham chiếu.
- Xử lý song song batch/camera, không chuyển sang NumPy; chỉ lấy một scalar
  maximum budget để giới hạn vòng lặp. Gather vẫn truyền gradient về projector.
- Allocator dùng stable marginal ranking thay vòng sửa cố định. Kết quả phải
  khớp allocator cũ, bao gồm budget Stage 2 và quy tắc hòa điểm.
- Debug có diversity, weights, budget và local seed index; không nhập hệ thống
  gradient-diagnostics riêng của DINOv2.

Checkpoint MVPruner mới lưu `metadata.mvpruner.stage1_algorithm_version=2`.
Checkpoint MVPruner thiếu version hoặc khác version bị chặn trước khi nạp weights.
Checkpoint router/GPA cũ vẫn dùng được. Không nhập architecture-signature DINOv2.

Để train mới, giữ prepared data/cache, để trống checkpoint HF và chuyển thư mục
checkpoint MVPruner cũ ra chỗ khác nếu notebook đang tự tìm `finetune_latest.pt`.

Kiểm chứng offline:

```powershell
$env:PYTHONPATH='src'
$env:OMP_NUM_THREADS='1'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
python -m pytest tests -q
python scripts/benchmark_stage1.py --device cpu --warmup 20 --iterations 100 --output outputs/stage1_v2_cpu_benchmark.json
```

Benchmark dùng projected features `[B,6,49,384]`, batch 1/2/8, không chạy
vision encoder hoặc tải model. Trên Kaggle đổi `--device cuda`. CPU timings
không chứng minh GPU hoặc toàn VLM nhanh hơn. Script kiểm tra allocator và CCTS
optimized khớp reference trước khi đo; Stage 1 cũ/mới khác thuật toán nên kết
quả timing không phải so sánh chất lượng.

## Kết quả kiểm chứng CPU

Môi trường: PyTorch 2.14.1+cpu, Transformers 4.46.3, tokenizers 0.20.3,
FP32, 1 CPU thread, 20 warm-up và 100 lượt đo. Không đổi dependency của repo.

| Batch | Stage 1 cũ (mean ms) | Stage 1 v2 (mean ms) |
|---|---:|---:|
| 1 | 43.396 | 9.389 |
| 2 | 43.519 | 10.434 |
| 8 | 50.563 | 22.447 |

Report đầy đủ ở `outputs/stage1_v2_cpu_benchmark.json` (artifact local).
Chưa đo Kaggle GPU, multi-GPU DDP hoặc latency toàn VLM trên pretrained models.

Các test cấu hình/notebook cũ vẫn tham chiếu 6 file không tồn tại trong baseline:
3 config `safe` của mini/DI/end-to-end và các notebook `kaggle_smoke_debug`,
`kaggle_e2e_debug_repvit_t5_efficient_mini_2gpu`, `kaggle_verify_resume_checkpoints`.
Chúng không được khôi phục/xóa/sửa trong bản chuyển Stage 1 này.

Full suite: **211 passed, 6 failed, 2 skipped**; cả 6 failure là FileNotFoundError
cho các file baseline đã liệt kê. Các test Stage 1/checkpoint và smoke
finetune-only đều pass. Test checkpoint bổ sung xác nhận router cũ vẫn load được
và giữ nguyên output eval. Kiểm tra AST xác nhận Stage 1/allocator khớp bản
DINOv2 v2, còn Stage 2, Gumbel relaxation, beta schedule và camera embeddings
không đổi so với baseline `9779658`.
