# Dọn runtime về MVPruner duy nhất

Ngày thực hiện: 2026-10-27. Source tham chiếu trước thay đổi: `974f922ec7d6bd7e72a5b40d3e8e26852363ae30`.

## Phạm vi thay đổi

- Xóa module thực thi GPA, question-guided router và DynamicInstructionAdapter, cùng factory/forward/generation/debug/freeze branches tương ứng.
- Bỏ config baseline ViT/T5/GPA và tests chỉ dành cho các nhánh bị loại. Tests dùng chung được chuyển sang MVPruner với T5 nhỏ bốn encoder blocks.
- Bỏ `AdapterConfig`, `ModelConfig.adapter`, `TrainingConfig.gpa_hidden_size` và các trường router khỏi runtime dataclasses. Constructor VLM không còn nhận `visual_adapter` hoặc `gpa_hidden_size`.
- Các helpers trước đây mang tên router nhưng điều khiển beta của MVPruner được đổi tên; cách tính target steps và thời điểm cập nhật không đổi.
- Giữ upstream `em_vlm4ad/` và nghiên cứu cũ làm tham chiếu lịch sử. Giữ standalone vision utilities, builder end-to-end dùng chương trình và ngoại lệ nạp thiếu vision weights; không mở end-to-end qua config MVPruner.

## Phần MVPruner được giữ nguyên

Không sửa các file thuật toán `modeling/mvpruner.py`, `modeling/t5_split_encoder.py`, `modeling/vision.py`, `tokenization.py`, `cache.py`, `data.py`, `hf_data.py`. Notebook và scripts chạy/benchmark hiện hữu không đổi.

Giữ DINO preprocessing/pooling/cache, camera order, spatial/camera/modal embeddings, projector, DRA/CCTS, IRA/Gumbel/STE, masks/position bias, T5 loss/generation, beta schedule và profiler event names. Training giữ align/finetune, optimizer grouping/order, accumulation, clipping, scheduler và checkpoint/RNG behavior hiện có. Evaluation/benchmark/prefill không sửa semantics.

Tổng số tham số giảm do bỏ GPA vốn không được dùng trong MVPruner. Khởi tạo model mới có thể khác dù cùng seed vì không còn tiêu thụ RNG để khởi tạo GPA. Người dùng đã chấp nhận khác biệt này. Parity được so bằng cùng active weights và cùng RNG khi thực thi.

## Tương thích cấu hình và checkpoint

### YAML cũ

Config phải chọn `fusion.name: mvpruner_two_stage`. GPA/router và adapter đang bật bị từ chối. YAML MVPruner cũ có `adapter.name: none`, `gpa_hidden_size` và inactive router fields vẫn đọc được; những trường này không xuất hiện trong dataclasses runtime mới.

Giữ nguyên fallback: nếu thiếu `stage2_top_k`, lấy `fusion.top_k` nếu có, sau đó mới tới fallback cũ theo vision sequence length. `stage2_top_k` khai báo trực tiếp luôn có ưu tiên. Giữ validation `feature_cache`, budgets, relevance modes, beta, tau và split layer.

### Checkpoint cũ

Mọi đường nạp model đi qua `checkpoint_model_state` trong [checkpointing.py](../../src/efficient_vlm_ad/checkpointing.py): train same-stage resume, loader thường, strict helper, helper cho phép vision mới, diagnose, evaluate, benchmark và verification.

Sau khi kiểm tra signature, loader chỉ loại bốn keys không hoạt động của checkpoint MVPruner v2 cũ có `adapter.name: none`:

```text
gpa.z.0.weight
gpa.g.0.weight
gpa.w.weight
gpa.w.bias
```

Signature normalization chỉ bỏ danh sách trường GPA/adapter/router cũ được liệt kê rõ. Các trường MVPruner còn lại vẫn được kiểm tra; stage 1 algorithm version cũ, unknown signature fields, unknown weights, sai shape và thiếu active weights vẫn bị từ chối. Không dùng `strict=False` để bỏ qua mismatch. Helper cho phép vision mới vẫn chỉ chấp nhận thiếu `vision_encoder.*` như trước.

Giữ PyTorch state-dict version metadata, optimizer/scheduler state, tokenizer metadata và rank RNG sidecars. Không sửa checkpoint nguồn trên đĩa. Checkpoint mới không còn GPA weights hoặc inactive signature fields.

## Kiểm chứng

Môi trường test tạm riêng: Python 3.12, PyTorch `2.14.0+cpu`, Transformers `4.46.3`, tokenizers `0.20.4`. Không tải pretrained weights hoặc dataset; fake vision và T5 khởi tạo cục bộ được dùng cho kiểm chứng offline.

### Parity trước/sau

Source cũ được sao chép ngoài repository trước khi sửa. Harness tạo checkpoint bằng code cũ, nạp vào code mới và khôi phục cùng RNG. Kết quả **9/9 trường hợp khớp**:

| Nhóm | Trường hợp |
|---|---|
| T5 nhỏ, hai mẫu có padding | Align/finetune × shifted-max-cosine/diversity-only × pruning/identity: 8 |
| Geometry profile chính | 6 camera × 64 tokens × 384 chiều, budgets 196 → 49, finetune: 1 |

So sánh gồm active trainable names/order, logits, loss, stage allocations/indices/relevance, position bias, gradients, active model state sau update, AdamW/scheduler state, eval output, diagnostics và generation token IDs/decoded text. Float tolerance: `rtol=1e-5`, `atol=1e-6`.

Geometry profile chính dùng features tổng hợp và T5 khởi tạo cục bộ; không phải kiểm chứng chất lượng pretrained DINO/T5 trên DriveLM.

### Regression tests được bổ sung

[test_mvpruner_cleanup.py](../../tests/test_mvpruner_cleanup.py) kiểm tra:

- Runtime không còn GPA hoặc visual adapter.
- YAML cũ giữ giá trị hiệu lực, kể cả `top_k` fallback và precedence.
- Config GPA/router/thiếu fusion bị từ chối.
- Ba weight loaders đọc được checkpoint cũ và giữ toàn bộ active weights.
- Unknown/missing/wrong-shape state, unknown signature, active adapter hoặc version sai bị từ chối.
- State migration giữ metadata của PyTorch và không sửa payload nguồn.
- Helper vision mới chỉ chấp nhận thiếu vision weights.
- Checkpoint lịch sử đi qua verify, evaluate, benchmark, same-stage resume với AdamW/scheduler đã có state, và align → finetune. Checkpoint nguồn không bị ghi lại.

Lần chạy toàn bộ cuối: **335 passed, 3 skipped**, không có failure, trong **469,62 giây**. Các skip liên quan đến CUDA và khả năng tạo symbolic link trong môi trường Windows. Riêng nhóm kiểm thử cleanup/lifecycle có 19 test cases.

Lệnh kiểm chứng, chạy bằng Python của môi trường test tạm:

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:OMP_NUM_THREADS = '1'
$env:PYTHONPATH = 'D:/AD/huy-dinov2/src'
python -m pytest tests -q --disable-warnings
```

Kiểm tra bổ sung: 12 file lõi liên quan đến thuật toán, dữ liệu, cache, evaluation, benchmark và CLI giữ nguyên từng byte so với snapshot trước thay đổi; cấu hình hiệu lực profile chính không đổi sau khi loại inactive fields; `git diff --check`, Python syntax, notebook JSON và các liên kết trong báo cáo đều hợp lệ. Rà soát code độc lập không phát hiện lỗi critical/important.

## Giới hạn

Máy kiểm thử không có CUDA. Chưa chạy smoke DDP hai GPU, real-model finetune hoặc benchmark Kaggle. Không suy diễn parity CPU thành bảo đảm hiệu năng hoặc tái lập distributed trên GPU.

Không kết hợp sửa các vấn đề accumulation/scaling, beta target sau sharding, mid-epoch resume hoặc metric fallback đã ghi nhận trong báo cáo khảo sát ban đầu.
