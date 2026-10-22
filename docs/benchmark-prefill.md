# Prefill latency và FLOPs

Lệnh benchmark hiện tại vẫn ghi `benchmark.json`; không cần train lại,
đổi checkpoint, config hoặc notebook để dùng các chỉ số mới.

## Output

Thêm sáu trường:

```text
prefill_mean_ms
prefill_p50_ms
prefill_p95_ms
prefill_flops_mean_gflops
prefill_flops_p50_gflops
prefill_flops_p95_gflops
```

Bỏ `generation_mean_ms`, `generation_p50_ms`, `generation_p95_ms`.
E2E, average generation ms/output token, tokens/second và thông tin thiết bị giữ
nguyên định nghĩa. Duration generation vẫn được dùng nội bộ để tính tốc độ token.
Percentile dùng cùng nearest-index convention đang có của benchmark.

## Phạm vi

Từ visual features và text tensors đã ở trên thiết bị, qua embeddings,
adapter/fusion/projector/pruning, T5 encoder và decoder forward đầu tiên,
bao gồm LM head tạo logits. Không tính đọc dữ liệu, CPU→GPU transfer, vision
extraction, beam/token selection, decoding tiếp theo hay post-processing.

Đây là cách quy đổi prefill cho T5 encoder–decoder, không phải tuyên bố tái lập
chính xác protocol của paper MVPruner. Feature-cache không cần chạy vision
encoder. Nếu benchmark end-to-end, vision encoder vẫn nằm trong E2E cũ nhưng
được chạy ngoài timer/counter prefill mới.

Generation settings được giữ nguyên, kể cả `num_beams`. Các lượt prefill dùng
`max_new_tokens=1` và hook tại output của first T5 forward, trước khi generation
finalize token/beam. Không sửa settings của lượt generation thường.

## FLOPs là ước tính

Dùng `torch.utils.flop_counter.FlopCounterMode`, chỉ lấy tổng Global một lần,
không cộng trùng các module con; 1 MAC = 2 FLOPs, 1 GFLOP = 10^9 FLOPs.
Đếm các phép toán chính được hỗ trợ: matmul/linear/batched matmul và attention,
gồm các phép nhân ma trận của projector, pruning và extra attention scorer.

Không đếm normalization, softmax, activation/phép toán phần tử, sorting,
indexing/gather và CPU control flow. Các giới hạn được ghi trong
`prefill_measurement`, cùng scope, beams và warm-up count.

Chỉ lượt đếm FLOPs tắt native MHA fastpath và dùng math SDPA để tránh fused
kernels không được bộ đếm hỗ trợ. Có probe kiểm tra SDPA counter; flag/backend
được khôi phục khi kết thúc hoặc lỗi. Các lượt latency dùng backend gốc.
Không lấy FLOPs từ latency/parameter count và không nhân đôi số đếm đã có.
Runtime thiếu counter/SDPA support hoặc kết quả FLOPs không hợp lệ báo lỗi rõ.

## Cách chạy

Tối đa ba sample đầu được warm-up cho generation và prefill rồi loại khỏi
thống kê. Batch size benchmark là 1. Ba lượt riêng chạy trên cùng samples:

1. `benchmark`: generation thường, E2E, throughput, snapshot peak CUDA memory.
2. `benchmark-prefill`: synchronized wall-clock latency đến logits đầu.
3. `benchmark-flops`: FLOPs; thời gian instrumentation không vào latency.

Peak CUDA memory được reset sau warm-up và chụp trước instrumentation.
Các phase có progress riêng, cả trên console và benchmark progress JSONL.
Nếu dataset rỗng, benchmark báo lỗi thay vì ghi report toàn 0.

Model giả `TinySeq2Seq` không có decoder forward thật: sáu trường mới là
`null`, với status `unsupported_synthetic`. Kiểm thử số liệu dùng T5 nhỏ tạo
từ config, không tải pretrained model. Điều này không ảnh hưởng các model T5
thật của MVPruner/router/GPA.

Kiểm thử offline:

```text
python -m pytest tests/test_prefill_benchmarking.py tests/test_evaluation_benchmark.py tests/test_generation.py -q
```

CUDA/Kaggle phải được kiểm chứng riêng; kết quả CPU hoặc CUDA giả lập trong
unit tests không chứng minh latency, memory hay FLOPs coverage trên Tesla T4.
