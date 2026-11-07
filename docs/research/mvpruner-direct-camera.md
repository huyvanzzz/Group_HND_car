# MVPruner direct camera với Stage1 gốc

Profile `mvpruner_direct_camera` dùng một thư mục riêng `outputs/dinov2_t5_efficient_mini_kaggle_2gpu_mvpruner_direct_camera`, không thêm V2. Notebook: `notebooks/kaggle_train_dinov2_mvpruner_direct_camera_finetune.ipynb`.

Đây là ablation mới của implementation DINOv2/T5 hiện tại, không tuyên bố tái hiện nguyên bản [upstream MVPruner](https://github.com/Zizzzzzzz/MVPruner). Đối chứng đúng là MVPruner gốc **cùng budget 49**, cùng weights/input/settings khi đo; config MVPruner gốc hiện đang là 64 và không bị sửa trong đợt này. Không gán chênh lệch giữa 49 và 64 token riêng cho công thức quota.

Stage1 giữ nguyên 384 → 196 token, DRA/CCTS và shifted max-cosine của implementation hiện tại. Stage2 dùng attention native của T5 tại layer 2, giữ 49 token bằng Gumbel STE cũ. Chỉ đổi allocation thành `w = R`: min-max relevance rồi softmax trực tiếp, không bù số token Nc. Quota vẫn bị chặn bởi capacity, tổng bằng 49. Không thêm module học hoặc context token. Loss là native CE, không dùng fixed CE.

Với \(s_{bj}\) là attention trung bình từ các prompt query hợp lệ qua mọi head, camera ID \(C_{bj}\), số token camera \(N_{bc}\):

\[
r_{bc}=\frac{\sum_{j:C_{bj}=c}s_{bj}}{N_{bc}},\quad
z_{bc}=\frac{r_{bc}-\min r_b}{\max(\max r_b-\min r_b,\varepsilon)},\quad
R_{bc}=\operatorname{softmax}_c(z_{bc}),\quad w_{bc}=R_{bc}.
\]

Bounded allocation lấy quota lý tưởng \(49R_{bc}\), làm tròn và điều chỉnh để tổng đúng 49, minimum 1/camera, capacity \(N_{bc}\). Khi camera hết capacity, helper phân bổ phần dư theo quy tắc hiện có. Điểm bằng nhau cho shares 1/6; quota gần đều nếu capacity cho phép. Không softmax thêm trên 196 visual scores, không đổi sang tổng attention camera. Camera có số token khác nhau được tính bằng IDs thực, không reshape thành các nhóm bằng nhau.

Blocks 0,1 nhận `[B,196+Lq,384]`; blocks 2,3 và decoder memory nhận `[B,49+Lq,384]`. 49 token được giữ nguyên từ hidden states sau block 1, theo thứ tự nguồn; mọi prompt/padding được giữ. Position bias và mask gather từ chuỗi 196-token nguồn, không reset bias hoặc tạo token merge. Khởi tạo và toàn bộ parameter/state keys của fusion bằng MVPruner gốc (2.304 tham số camera embedding, không thêm module).

Giữ beta_start=0.2, beta_end=0.0 và horizon pre-prepare cũ có chủ ý. Trên hai GPU, beta cuối 10 epoch xấp xỉ 0.1 vì số bước thực tế nhỏ hơn horizon trước Accelerate.prepare; không sửa lịch này.

\[
H=\max(1,E\lceil L_{\mathrm{before\ prepare}}/A\rceil),\qquad
\beta(t)=0.2\left(1-\min(t/\max(1,H-1),1)\right).
\]

Với 21.337 microbatch trước prepare, A=2 và 10 epoch: H=106.690. Hai GPU có 10.669 microbatch/rank, khoảng 53.350 updates/rank. Counter tăng đúng một lần/update thành công; skipped update không tăng. Debug giới hạn microbatch và resume giữ semantics MVPruner cũ, gồm tính lại horizon theo config chạy tiếp; không áp dụng chính sách horizon cố định của qaware.

FP32, batch 16 mỗi GPU, accumulation 2, hai GPU, 10 epoch finetune; align_epochs=0, entropy_weight=0, gradient_checkpointing=false. Các tham số LR, weight decay, scheduler, generation và evaluation giữ nguyên. Không thêm seed hoặc beta_decay_updates.

Notebook đặt cả PYTORCH_ALLOC_CONF và PYTORCH_CUDA_ALLOC_CONF trước import torch, ghi YAML runtime riêng. Giữ prepare-data, prepare-features, kiểm định manifest/cache, diagnose-train, accumulation, verify-resume-checkpoint, train finetune, evaluate, benchmark và archive. FINETUNE_CHECKPOINT giữ hành vi cũ, ưu tiên local latest; không thêm RESUME_TRAINING. Resume cần RNG manifest và sidecar của cả hai rank. Archive giữ cả thư mục checkpoints.

Chẩn đoán dùng log mặc định `debug/debug_events.jsonl`, không xóa log cũ. Notebook lấy bản ghi NUMERICS mới nhất cho từng stage diagnose_forward và diagnose_backward, rồi kiểm tra riêng mvpruner và mvpruner_gradients: stage1_enabled=true, stage2_method=t5_attention_direct_camera_stage1_v1, camera_allocation=direct_minmax_softmax_v1, camera_counts, camera_relevance, camera_shares, quota stage2_budgets, stage2_indices, stage2_source_indices và beta. Cả hai báo cáo phải hữu hạn; gradients và mvpruner_gradients phải có ok=true.

Attention native ở đây là attention của T5 qua helper tái dựng pre-dropout attention trong split encoder; không phải attention được cache sẵn. Chi phí tái dựng phải được tính trong benchmark Stage2 và forward. Feature DINO có cache là một cơ chế riêng.

Các kiểm tra RNG sidecar nói trên mô tả đúng protocol của notebook này. Chúng không phải tuyên bố rằng mọi checkpoint hoặc mọi lệnh resume đều có cùng mức kiểm tra sidecar.

Checkpoint của profile khác bị từ chối dù có cùng state-dict keys/shapes. Chỉ finetune từ pretrained T5 hoặc resume checkpoint finetune cùng profile; không warm-start align/MVPruner gốc, không migration hoặc nới strict loading. Loader dùng chung và ngoại lệ thiếu vision weights của nó không bị sửa.

Sau khi chuẩn bị dữ liệu/cache, có thể chạy bằng shell (hoặc dùng notebook từ đầu đến cuối):

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CONFIG=configs/dinov2_t5_efficient_mini_kaggle_2gpu_mvpruner_direct_camera.yaml
python -m efficient_vlm_ad diagnose-train --config "$CONFIG" --stage finetune --debug --debug-numerics
python -m efficient_vlm_ad.accumulation_debug --config "$CONFIG"
python -m accelerate.commands.launch --multi_gpu --num_processes 2 --mixed_precision no -m efficient_vlm_ad train --config "$CONFIG" --stage finetune
```

Để resume, giữ cơ chế `FINETUNE_CHECKPOINT` trong notebook hoặc thêm `--resume <checkpoint>` vào lệnh train; verify checkpoint bằng CLI hiện có trước khi chạy. Notebook chỉ ghi runtime YAML riêng, không ghi đè config tracked; mọi tái sử dụng prepared/cache phải qua validator identity/coverage, không chỉnh manifest để ép nhận.

Kiểm thử trọng tâm ngày 2026-11-07: **131 passed** trong 135,24 giây, dùng Python offline `huy-context-validation-kpx98f2v`, `PYTHONPATH=src`, một thread CPU. Gồm 59 test mới: model 16, config 34, artifact 4, lifecycle 4, DDP 1; thêm 72 regression signature/notebook. Quota được so reference độc lập cho camera counts không đều, capacity nhỏ, scores đều/gần đều/chênh lệch lớn. Stage1, state keys và khởi tạo khớp MVPruner cũ; selector/STE gradient khớp khi cấp cùng quota và nhiễu.

Đã chạy geometry visual production `[B,6,64,384]` trên T5 nhỏ offline, xác nhận block lengths, source indices, mask/bias gather và CE/gradient hữu hạn; generation beam 1/3 chỉ encode một lần và không tăng counter/RNG ở eval. CLI thật chạy prepare/features → diagnose → accumulation verify → finetune → verify/resume → evaluate/benchmark, kiểm tra JSONL forward/backward. Hai rank CPU/Gloo có counts camera thay đổi, gradient đồng bộ và model states bằng nhau; accumulation đủ/dư qua hai epoch khớp reference trung bình độc lập. Skipped optimizer update không tăng beta, horizon vẫn là kế hoạch cũ trước prepare.

Full suite đã chạy: **882 passed, 3 skipped, 21 failed** trong 1.411,16 giây. Đối chiếu JUnit với baseline xác nhận đủ **19 failure cũ trùng hoàn toàn**. Hai failure còn lại là assertion của bản test đã collect trước khi chỉnh: test artifact còn đòi tên JSONL riêng sau khi notebook đã dùng log mặc định; test lifecycle còn đọc beta metadata ở top-level thay vì nhóm `mvpruner` của MVPruner gốc. Cả hai đã được sửa và **pass trong lượt 131 tests trên**; không có regression mới chưa xử lý. Một số test bổ sung sau khi full suite đã collect cũng nằm trong lượt 131 tests cuối.

19 failure nền gồm: 12 assertion notebook còn đòi `RESUME_TRAINING` đã bỏ theo yêu cầu trước; ba assertion entropy kỳ vọng giá trị cũ khác config hiện tại; bốn assertion config/protocol/signature còn kỳ vọng MVPruner 49 trong khi config gốc hiện là 64. Không sửa các config/notebook cũ hoặc thuật toán để làm xanh chúng. Signature của cả chín config hiện có đã được so với implementation HEAD và giữ nguyên. Source Python, notebook JSON/Python/heredoc/Bash và `git diff --check` đều pass.

Phép đo CPU ngày 2026-11-07: FP32, một thread, batch 2, sáu camera × 64 token, prompt dài 12 (một mẫu có ba padding), labels dài 5. T5 nhỏ `D=32`, FFN 64, bốn encoder block/hai decoder block, dropout 0; hai model cùng state dict, cùng input và generation settings. Warmup 3, 15 lượt đảo thứ tự, median; không chạy suite song song khi đo. Đối chứng chỉ thay budget trong model đo thành 49, không sửa config gốc.

| Phép đo | MVPruner gốc 49 | Direct camera 49 |
| --- | ---: | ---: |
| Eval forward có CE | 48,112 ms | 48,588 ms |
| Train forward + backward | 75,632 ms | 72,177 ms |
| Generation beam 1, tối đa 5 token mới | 54,567 ms | 55,317 ms |
| Generation beam 3, tối đa 5 token mới | 73,670 ms | 69,801 ms |
| Prefill major-operator FLOPs, beam 1 | 56.594.432 | 56.594.432 |
| Prefill major-operator FLOPs, beam 3 | 58.853.376 | 58.853.376 |

Cả hai luồng đều sinh output shape `[2,6]` ở beam 1 và 3. Component đo riêng: Stage1 khoảng 8,625/8,601 ms; helper attention 14,310/14,146 ms; Stage2 selector/allocation 4,225/4,434 ms (gốc/direct). Tổng chi phí CPU gần nhau trong lần đo này; các tỷ lệ nhỏ lên/xuống không đủ để kết luận luồng mới nhanh hơn. FLOPs bằng nhau chỉ nói về major operators của counter hiện có, không bao phủ hết indexing, sorting, softmax, normalization và control flow. Phạm vi từ features, gồm Stage1/scorer/quota/selector/T5/decoder, không gồm DINO extraction offline.

Chưa chạy huấn luyện Kaggle, chất lượng QA hoặc benchmark GPU; không suy ra mức cải thiện chất lượng/latency/VRAM GPU từ CPU smoke. Không tải model/dataset lớn, không sửa checkpoint cũ, chưa commit/push.
