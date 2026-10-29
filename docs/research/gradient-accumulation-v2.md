# Gradient accumulation v2

Bốn notebook Kaggle finetune dùng chung quy tắc accumulation: `unpruned`, `mvpruner`, `camera_query` và `camera_query_qaware`. YAML gốc giữ `gradient_accumulation_steps: 2`; không ép nội bộ về 1. Kiến trúc, fusion/profile, allocator, phiên bản thư viện và mặc định gradient checkpointing giữ nguyên.

## Loss và cửa sổ cập nhật

Gọi A là số microbatch cấu hình trong một cửa sổ. Truyền raw loss vào backend; backend chia A đúng một lần. Không chia loss thêm A trước backward. Với cửa sổ đủ A microbatch, gradient là trung bình các gradient microbatch.

Cuối epoch, nếu chỉ còn k microbatch (k < A), truyền `raw_loss * A / k` vào backend. Sau phép chia A của backend, mỗi microbatch đóng góp `raw_loss / k`. Cửa sổ cuối không bị giảm gradient vì thiếu microbatch. Đây là trung bình theo microbatch; không khẳng định tương đương batch lớn nếu số mẫu/token mỗi microbatch khác nhau.

Các profile dùng chung ranh giới cửa sổ cho backward, đồng bộ distributed, clipping, optimizer step và zero_grad. Chỉ cập nhật optimizer khi kết thúc cửa sổ; không xóa gradient giữa các microbatch cùng cửa sổ. Scheduler chạy đúng một lần mỗi epoch hoàn tất, không chạy theo microbatch hay optimizer step. Các bộ đếm và lịch phụ thuộc optimizer step phải theo bước optimizer thực tế.

DDP vẫn đồng bộ gradient sau từng microbatch như trước; thay đổi này không thêm `no_sync` hoặc dùng `accelerator.accumulate`. Ranh giới cửa sổ quyết định clipping và cập nhật trọng số, không trì hoãn đồng bộ DDP.

## Chạy mới và resume

Mỗi notebook tạo runtime YAML riêng, nối `_accumulation_v2` vào output_dir của profile gốc và đặt `PROFILE` theo output_dir đó. Ví dụ camera-query ghi vào `outputs/dinov2_t5_efficient_mini_kaggle_2gpu_camera_query_accumulation_v2`. Bốn profile giữ danh tính fusion/model riêng và nguyên đường dẫn cache đã cấu hình.

`RESUME_TRAINING = False` là mặc định. Notebook bắt đầu từ pretrained T5 và khởi tạo mới các module multimodal; không tự chọn checkpoint local hoặc tải checkpoint Hugging Face để tiếp tục. Nếu thư mục checkpoints mới đã có bất kỳ nội dung nào, notebook dừng và yêu cầu chọn output mới hoặc bật resume rõ ràng. Không xóa hay ghi đè checkpoint cũ để chạy lại.

Khi đặt `RESUME_TRAINING = True`, luồng resume chọn local latest, tệp local được chỉ định hoặc checkpoint Hugging Face, kiểm tra metadata bằng `verify-resume-checkpoint` rồi mới khởi chạy hai GPU. Tệp local được ưu tiên trước download để tránh ghi đè. Checkpoint phải khớp hợp đồng accumulation v2, profile, cấu hình training và trạng thái resume/RNG cần thiết. Metadata cũ hoặc thiếu bị chặn khi tiếp tục training; checkpoint cũ vẫn dùng được cho inference/evaluate/benchmark nếu qua kiểm tra kiến trúc hiện có. Không sửa metadata để hợp thức hóa checkpoint cũ.

## Tái sử dụng dữ liệu

Runtime mới có thể sao chép `prepared_data` từ output gốc khi thư mục prepared mới chưa tồn tại. Sau đó vẫn kiểm tra full subset, số dòng từng split và expected_counts như trước. Cache DINO không đổi tên, không đổi fingerprint và không nới điều kiện tương thích: manifest v2, processor identity, shape `[N,6,64,384]`, index, dung lượng memmap và camera keys vẫn được kiểm tra chính xác. Khi thiếu cache, luồng chuẩn bị raw images rồi trích xuất feature giữ nguyên.

Các kiểm thử CPU kiểm tra hợp đồng runtime và shell guard, cú pháp notebook, phép chuẩn hóa accumulation và metadata resume. Chúng không thay thế phép đo chất lượng, bộ nhớ hoặc tốc độ trên Kaggle hai GPU.

## Kết quả xác minh ngày 2026-10-30

Toàn bộ repository: **542 passed, 3 skipped** trong 603.76 giây khi chạy `python -m pytest -q` với `PYTHONPATH=src`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`. Ba test bị skip do môi trường không hỗ trợ symlink hoặc không có CUDA.

Kiểm thử bao phủ A=1/2/4, cửa sổ đủ/dư/ngắn hơn A, nhiều epoch, debug giới hạn lẻ và lặp loader; so gradient trước clipping và trọng số sau cập nhật với reference. Cả bốn profile đã chạy qua Accelerate thật và fallback. DDP hai tiến trình CPU kiểm tra đồng bộ từng microbatch, nhóm dư, reference trung bình trên hai rank và trọng số cuối giống nhau. Lifecycle CLI, resume phiên bản mới, từ chối recipe cũ/sai, counter khi skipped update và bảo toàn checkpoint cũ đều được kiểm tra.

Cú pháp JSON/Python và Python nhúng trong notebook hợp lệ; `git diff --check` pass. Các config gốc vẫn có accumulation bằng 2, không đổi model/fusion, batch size, learning rate hoặc precision. Không có thay đổi trong code mô hình. Chưa xác minh VRAM, latency hoặc full training hai GPU.
