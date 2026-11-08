# Rà soát và tinh gọn test suite

## Phạm vi và cách chạy

Chỉ thay tests, cấu hình pytest và tài liệu kiểm thử. Không thay model, training,
checkpoint loader, config thí nghiệm hoặc notebook. Không đổi chính sách resume,
accumulation, beta, softmax, fixed CE hay learned pooling.

```bash
# Mặc định vẫn chạy tất cả integration và distributed
python -m pytest -q

# Chủ động chọn kiểm tra nhanh khi sửa cục bộ
python -m pytest -q -m "not integration and not distributed"

# Lưu bằng chứng thời gian và kết quả
python -m pytest -q -ra --junitxml=test-results.xml --durations=30
```

`pyproject.toml` chỉ đăng ký hai marker; không có `addopts` loại nhóm.
`tests/conftest.py` gắn marker theo file lifecycle/smoke/distributed và một số
entrypoint integration nằm trong file hỗn hợp. Hook không deselect test.

## Baseline trước chỉnh sửa

Collection hiện tại trước dọn: **915 cases**, **475 hàm test**, **57 vị trí gọi
subprocess**, **11 vị trí spawn DDP**. Số vị trí gọi được đếm bằng AST; không
phải số process thực tế, vì một vị trí có thể nằm trong vòng lặp/parametrization.

Không chạy thêm một full suite chỉ để đo baseline. Dùng báo cáo full suite trước
đó `huy-direct-camera-suite.xml`: **906 cases, 882 pass, 3 skip, 21 fail**,
**1.411,16 giây** wall time, tổng testcase khoảng **1.399,99 giây**. Collection
915 là snapshot mới hơn sau các test bổ sung cuối lượt triển khai trước; vì thế
so sánh runtime là toàn suite trên cùng môi trường, không phải hai collection
hoàn toàn giống nhau.

Đã chạy lại đúng 21 ID fail trước khi chỉnh: **19 fail, 2 pass, 123,39 giây**.
Hai test direct-camera đã pass ở code hiện tại; không xóa hoặc sửa chúng để né lỗi.
19 failure còn lại là assertion về fresh-run/`RESUME_TRAINING`, giá trị entropy/
budget thí nghiệm và digest phụ thuộc YAML. Không suy từ việc fail rằng hợp đồng
production phải thay đổi.

## Bảng quyết định

Thời gian dưới đây là tổng testcase của báo cáo baseline, làm tròn giây.

| Test/nhóm | Hành vi bảo vệ | Trước (s) | Quyết định và nơi giữ coverage |
|---|---|---:|---|
| `test_smoke_pipeline.py` | MVPruner align→finetune; finetune-only; epoch aliases; chuẩn bị cache; lỗi NaN | 269 | Giữ toàn bộ; CLI trong process, riêng workflow align→finetune chạy một child process |
| `test_camera_query_smoke.py` | Classic query và qaware lifecycle, recipe/beta/checkpoint/log | 198 | Giữ classic; chuyển toàn bộ assertion qaware sang `test_text_pool_lifecycle.py` |
| `test_verify_align_checkpoint.py` | Align/finetune verification, strict loading, optimizer và RNG sidecars, lỗi checkpoint | 146 | Giữ mọi case; module fixture train một lần, mỗi test nhận bản sao checkpoint/sidecars riêng |
| `test_text_pool_lifecycle.py` | Learned head, optimizer restore, diagnose, accumulation verify, resume, evaluate/benchmark | 122 | Một workflow qaware trong một child process; gộp DDP audit vào worker qaware sẵn có |
| `test_camera_pool_query_smoke.py` | Spatial pooled query end-to-end | 108 | Giữ mọi bước/assertion; bỏ khởi động Python mỗi lệnh |
| `test_mvpruner_ccts_context_smoke.py` | 49 main + 15 context lifecycle | 101 | Giữ; gọi CLI thật trong process |
| `test_three_stage_lifecycle.py` | Prune→merge Stage 3, optimizer/beta boundaries, resume | 89 | Giữ; dùng helper config/checkpoint audit chung, CLI trong process |
| `test_debugging.py` | Debug sample, diagnose boundaries/numerics, JSONL, token secrecy | 69 | Giữ; prepared data/features fixture chỉ đọc, sao chép trước khi ghi debug |
| `test_direct_camera_lifecycle.py` | Direct quota lifecycle, beta cũ, skipped updates | 49 | Giữ; bỏ subprocess lặp, không đổi beta hoặc thuật toán |
| `test_progress.py` | CLI progress flags, event logs, prepare no-progress | 27 | Hai train-help test thành một với cả hai assertion; help còn lại gọi CLI thật trong process |
| Các `*_distributed.py` và learned-pool DDP cũ | Quota khác nhau giữa rank, STE, gradient sync, full/tail accumulation, rank-RNG replay | 135 | Giữ các failure mode riêng; chỉ bỏ spawn learned-pool trùng worker qaware; số này có phần đã tính trong nhóm text-pool phía trên |
| `test_accumulation_notebooks.py` | Runtime YAML, một output folder, cache identity và checkpoint precedence | — | Bỏ guard fresh-run cũ; thực thi runtime 10 profile và 5 tình huống Bash/profile |
| Artifact/config tests | Geometry/budget hợp lệ, entropy hợp lệ, runtime propagation, protocol, syntax, allocator/download | — | Cập nhật assertion lỗi thời; không dùng example YAML làm signature oracle |
| `test_mvpruner_qaware_config.py` | Signature đầy đủ và mismatch rejection | — | 10 fixture cấu hình/signature cố định; MVPruner lịch sử giữ 49 token, độc lập config thí nghiệm 64 |
| `test_mvpruner_cleanup.py` và legacy validation | Reject GPA/router/adapter, lịch sử checkpoint còn hỗ trợ | 24 | Giữ; không xóa vì tên cleanup/legacy |
| Công thức, mask/bias/padding, native softmax/CE, fixed CE, pooling, gradient, accumulation/beta | Hợp đồng thuật toán và gradient | — | Giữ regression; không tăng tolerance hoặc thêm skip/xfail |

### Những hàm/case bị thay hoặc gộp

Không xóa cả file test. Các hàm bị bỏ có mapping sau:

| Hàm/case cũ | Lý do | Thay thế |
|---|---|---|
| `test_accumulation_runtime_is_fresh_and_keeps_cache_and_profile` | Tên/assertion fresh-run không còn đúng | `test_accumulation_runtime_preserves_configured_architecture_and_cache`, mở rộng 4→10 profile |
| `test_fresh_run_ignores_hf_resume_and_rejects_existing_checkpoint` | Chính sách `RESUME_TRAINING` đã bỏ | `test_checkpoint_selection_prefers_latest_then_configured_then_download`, 50 case thực thi Bash |
| `test_resume_reuses_configured_local_checkpoint_before_download` | Trùng một nhánh của kiểm tra precedence | Gộp vào cùng 50 case; giữ kiểm tra không download/không sửa checkpoint local |
| `test_existing_signatures_match_before_change` | Digest đọc YAML thí nghiệm nên fail khi đổi budget hợp lệ | `test_historical_profile_signature_contract`, so toàn bộ expected signature cố định, reject revision/profile/version khác |
| `test_cli_help_exposes_no_progress` và `test_cli_help_exposes_progress_log_every_steps` | Cùng lệnh train-help, hai subprocess | `test_cli_help_exposes_train_progress_flags`, giữ cả hai assertion |
| `test_text_pool_optimized_and_synchronized_on_two_cpu_ranks` | Spawn lại chính worker qaware | `test_question_aware_fusion_distinct_budgets_two_cpu_ranks`: giữ gradient/head update lần 1/2, optimizer registration, quota khác nhau và state sync |
| Qaware case của `test_camera_query_offline_cli_lifecycle` | Lifecycle trùng text-pooling | `test_text_pool_offline_cli_checkpoint_resume`, thêm toàn bộ recipe/beta/state/log assertion cũ |

Các giá trị cố định trong fixture unit là đầu vào kiểm thử có chủ ý, không đọc từ
profile YAML. Các ràng buộc kiến trúc bắt buộc (ví dụ spatial pooling 46 token,
Stage-1-disabled input bằng geometry) vẫn được kiểm tra chính xác.

Đếm theo AST ở cấp hàm: **7 hàm cũ được thay/gộp**, **7 hàm mới**, **39 hàm
giữ tên có body/decorator cập nhật**; tổng vẫn **475 hàm**. Trong 7 hàm mới có
hai test helper restoration/parser và một smoke module entrypoint. Số case sau
parametrization tăng vì notebook coverage mở rộng từ bốn lên mười profile.

Call sites giảm **subprocess 57→7**, **DDP spawn 11→10**. Bảy subprocess call
sites còn lại gồm một helper cho hai child workflows, một entrypoint Python,
một bảng Bash checkpoint selection và bốn file kiểm tra Bash syntax. Đây là
đếm tĩnh, không tuyên bố số process chạy thực tế bằng bảy.

## Tổ chức thực thi và cách ly trạng thái

- `runtime_helpers.py`: gọi `cli.main(argv)` thật, bắt stdout/stderr/exit code;
  khôi phục env, cwd, argv, Python/NumPy/Torch RNG cả khi có exception. Không nhận
  timeout giả; chỉ subprocess workflow có timeout thật. CUDA RNG chỉ đọc nếu đã
  khởi tạo, không khởi tạo CUDA để kiểm thử helper.
- Chỉ hai lifecycle child workflows: MVPruner align→finetune và qaware learned
  pooling/recipe/RNG resume. Mỗi child gọi lại chính test được chỉ định qua pytest,
  thực hiện nhiều CLI trong process đó; mọi assertion vẫn chạy trong child.
- Một subprocess riêng kiểm tra thật `python -m efficient_vlm_ad --help`.
  Subprocess Bash còn lại kiểm tra syntax/selection; không import Torch.
- `data_helpers.py`, `checkpoint_helpers.py`, `lifecycle_helpers.py`: tách helper
  trước khi gộp test dùng chung. Không chia sẻ live model/optimizer giữa các test.
  Fixture checkpoint được sao chép cùng RNG sidecars trước mọi mutation.
- `test_runtime_helpers.py` kiểm tra restoration khi exception và exit code/parser
  capture qua CLI thật. Không mock forward/backward, optimizer, load hoặc generate
  của các lifecycle.
- Worker DDP qaware kiểm tra learned head ngay trong hai update sẵn có; nhóm LR
  head giúp quan sát update FP32 như audit cũ. Các DDP profile và replay riêng giữ nguyên.

## Kết quả kiểm chứng

Lượt full hoàn chỉnh ngày **08/11/2026**: **962 passed, 3 skipped, 0 failed,
0 errors**, tổng **965 cases**, pytest exit code **0**. Thời gian pytest báo
**407,62 giây (6 phút 47 giây)**; JUnit ghi **407,540 giây**. Không có deselection
ở lượt full; cả **10 cases distributed** đều chạy.

| Chỉ số | Trước | Sau |
|---|---:|---:|
| Collection hiện tại trước/sau cleanup | 915 | 965 |
| Hàm test (AST) | 475 | 475 |
| Subprocess call sites (AST) | 57 | 7 |
| DDP spawn call sites / cases | 11 | 10 |
| Full wall time pytest | 1.411,16 s | 407,62 s |
| Kết quả full trong báo cáo timing | 882 pass / 3 skip / 21 fail (906 cases ở snapshot trước) | 962 pass / 3 skip / 0 fail (965 cases) |

Giảm **71,1%** thời gian, tương đương khoảng **3,46 lần** nhanh hơn báo cáo
baseline, vượt mục tiêu 50%. Collection tăng 50 cases vì mở rộng notebook và
signature contracts, dù các workflow/process trùng đã được gộp. Không dùng số
test giảm làm tiêu chí. Hai snapshot khác số cases và baseline có failure; đây
là so sánh suite thực tế trên cùng máy, không phải benchmark hai collection
đồng nhất hoặc cam kết thời gian trên máy khác.

| Nhóm | Baseline testcase time (s) | Sau (s) |
|---|---:|---:|
| MVPruner smoke | 269,47 | 49,50 |
| Camera-query smoke | 197,65 | 13,55 |
| Checkpoint verification | 145,99 | 1,80 |
| Text-pooling lifecycle + DDP cũ | 121,58 | 24,50 |
| Spatial pooled-query smoke | 107,61 | 15,31 |
| CCTS context smoke | 100,50 | 11,63 |
| Three-stage lifecycle | 89,03 | 14,10 |
| Debugging | 69,11 | 0,63 |
| Direct-camera lifecycle | 49,14 | 18,67 |
| Progress | 27,29 | 0,15 |

Các hàng không phải so sánh workflow một-một sau khi gộp: assertion qaware từ
camera-query chuyển sang text-pooling; learned-pool DDP chuyển sang worker
qaware trong `test_camera_query_distributed.py`. Số đo toàn suite mới là căn cứ
cho mức giảm 71,1%, không lấy riêng các hàng đã chuyển coverage để kết luận.

Ba skip giữ nguyên: quyền symlink Windows không có; CUDA prefill; CUDA Stage 1.
Không thêm skip/xfail, không nới tolerance, không sửa production để che failure.
Python syntax của **96 module test/helper** pass; `git diff --check` sạch.
Đối chiếu SHA256 **61 file source/config/notebook** với snapshot trước cleanup:
không có thay đổi.

Bằng chứng cục bộ nằm dưới
`C:/Users/Admin/AppData/Local/Temp/huy-test-cleanup/`: `before-inventory.json`,
`baseline-failures.xml`, `fast.xml`, `full-complete.xml`, `full-complete.log`,
`comparison.json`, `test-decisions.json` và `protected.json`. Collection gốc là
`../huy-test-cleanup-before-collection.log`; timing baseline là
`../huy-direct-camera-suite.xml`. Các file log/binary/checkpoint test đều ở thư
mục tạm, không thêm vào repository.

Lượt nhanh trung gian: **850 pass, 3 skip có sẵn, 112 deselected, 68,85 giây**.
Smoke MVPruner: **7 pass, 62,99 giây**, so baseline nhóm này **269,47 giây**.

Một lượt full đang chạy đã được dừng giữa chừng sau khi review phát hiện phép
so sánh training dict vẫn buộc entropy của hai profile bằng nhau. Sau khi loại
riêng ràng buộc đó và kiểm tra lại file liên quan (**3 pass**), chạy lại một lượt
full hoàn chỉnh trên test code ổn định. Một lượt tiếp theo cũng mất process khi
phiên làm việc kết thúc, chưa xuất được JUnit. Các lượt bị ngắt đều không được
tính là pass hoặc dùng làm số đo; kết quả bàn giao chỉ dùng lượt hoàn chỉnh.

Môi trường: Windows, Python virtualenv có Torch CPU và Transformers 4.46.3,
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, Hugging Face/Transformers offline. DDP
dùng hai process CPU/Gloo. Không suy từ các số đo này ra latency/VRAM GPU hoặc
chất lượng mô hình. Không cài dependency, tải model/dataset, commit hoặc push.
