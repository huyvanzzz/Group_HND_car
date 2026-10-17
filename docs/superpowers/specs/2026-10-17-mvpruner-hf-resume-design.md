# Thiết kế resume MVPruner từ Hugging Face trên Kaggle

## Mục tiêu

Bổ sung vào `notebooks/kaggle_train_mvpruner_finetune.ipynb` khả năng khôi phục checkpoint finetune từ Hugging Face qua nhiều Kaggle session. Notebook vẫn chỉ chạy stage `finetune` và vẫn có thể bắt đầu từ pretrained weights khi không cấu hình checkpoint.

## Cấu hình

Cell cấu hình khai báo:

- `FINETUNE_CHECKPOINT_HF_REPO_ID`, mặc định rỗng.
- `FINETUNE_CHECKPOINT_HF_REPO_TYPE`, mặc định `model`.
- `FINETUNE_CHECKPOINT_FILENAME`, mặc định `finetune_latest.pt`.

Các giá trị được đưa vào `os.environ` để mọi cell `%%bash` kế thừa.

## Luồng khôi phục

1. Nếu `$PROFILE/checkpoints/finetune_latest.pt` đã tồn tại, dùng bản local và không tải lại.
2. Nếu checkpoint local chưa có và repo ID được cấu hình, tải vào `$PROFILE/checkpoints`:
   - checkpoint được chỉ định bởi `FINETUNE_CHECKPOINT_FILENAME`;
   - toàn bộ sidecar `${FINETUNE_CHECKPOINT_FILENAME}.rng/**`.
3. Nếu filename tải về khác `finetune_latest.pt`, copy nó cùng sidecar sang tên canonical mà pipeline hiện dùng.
4. Khi có checkpoint, chạy `verify-resume-checkpoint` với stage `finetune`. Verification thất bại phải dừng notebook trước khi launch DDP.
5. Khi không có local checkpoint và repo ID rỗng, giữ hành vi hiện tại: bắt đầu finetune từ pretrained T5 và các multimodal module mới khởi tạo.

## Chính sách lỗi

- Có cấu hình Hugging Face checkpoint nhưng tải checkpoint hoặc RNG sidecar thất bại: dừng rõ lỗi, không âm thầm train lại từ đầu.
- Checkpoint có nhưng verification không hợp lệ: dừng rõ lỗi.
- Resume DDP yêu cầu sidecar RNG đủ cho world size hiện tại; logic verification hiện có chịu trách nhiệm kiểm tra manifest/rank state.

## Kiểm thử

Test notebook xác nhận:

- ba biến cấu hình checkpoint tồn tại và được export;
- lệnh `hf download` tải checkpoint và pattern RNG sidecar;
- local checkpoint được ưu tiên;
- `verify-resume-checkpoint` chạy trước lệnh train;
- `--resume` chỉ trỏ tới checkpoint finetune canonical;
- notebook không thêm stage align và vẫn giữ FP32/2-GPU.
