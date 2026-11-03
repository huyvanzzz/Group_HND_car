# MVPruner: 49 token chính + 15 CCTS context

Profile `mvpruner_ccts_context` chạy finetune-only từ pretrained T5. Giữ Stage 1 hiện tại (384 → 196), Stage 2 gốc chọn 49 token bằng Gumbel/STE. Tập bù 147 token tạo 15 token tổng hợp; không thêm learned compressor hoặc loss phụ.

```text
6 camera → DINO frozen [B,6,64,384] → embeddings/projector
→ Stage 1 giữ 196 → T5 blocks 0,1
→ 49 main STE + 147 residual → 15 CCTS context
→ [49 main; 15 context; prompt] → T5 blocks 2,3 → decoder → CE
```

## Quota và chọn tâm

$X,Q$ là hidden states ảnh/prompt sau block 1. $s_j$ là attention score Stage 2 hiện có, trước dropout, lấy từ đầu vào block 1. Giữ nguyên 49 output STE; không thay bằng hard gather. $D$ là phần bù 49 hard indices trong tập 196; $D_c$ chỉ chứa camera $c$.

$$n_c=|D_c|,\qquad u_c=\sum_{j\in D_c}s_j,\qquad q_c=15u_c/\sum_vu_v.$$

Camera rỗng nhận 0; camera có nguồn nhận tối thiểu 1 và không vượt $n_c$. Floor rồi clamp quota; thêm vào camera có $q_c-m_c$ lớn nhất, bớt ở camera có $m_c-q_c$ lớn nhất, đến đúng 15. Hòa chọn camera index nhỏ hơn. Tổng attention bằng 0 thì dùng $n_c$. Không min-max, softmax camera hay bù Stage 1.

Relevance theo InternLM2: dot product rồi softmax trên residual **trong từng camera**, mean qua prompt hợp lệ:

$$p_{jt}=\operatorname{softmax}_{j\in D_c}(x_j^\top q_t),\qquad r_j=\frac{\sum_t\mathrm{mask}_t p_{jt}}{\sum_t\mathrm{mask}_t}.$$

Không temperature, cosine normalization hoặc scaling cho dot product. Prompt padding bị loại; camera rỗng không chạy softmax toàn mask. CCTS chọn tâm đầu $t_1=\arg\max_jr_j$, tiếp theo:

$$t_{next}=\arg\max_{j\notin T_c}r_j\min_{i\in T_c}[1-\cos(x_j,x_i)].$$

Cosine FP32, normalize có epsilon, clamp [-1,1]. Không chọn lại tâm/padding; hòa chọn index nguồn nhỏ hơn. Stage 1 local giữ nguyên relevance cũ, chỉ bước chọn tâm mới dùng công thức trên.

## Gán nhóm, merge và gradient

Mỗi residual không phải tâm được gán vào tâm cosine hidden-state giống nhất trong cùng camera; hòa chọn tâm có index nhỏ hơn. Tâm không nằm trong tập token cần gán:

$$a(j)=\arg\max_{i\in T_c}\cos(x_j,x_i),\qquad z_i=x_i+\frac{\sum_{j:a(j)=i}x_j}{\max(1,|\{j:a(j)=i\}|)}.$$

Đúng kiểu code VisionZip: **tâm + trung bình thành viên**, không phải mean cả nhóm, không attention-weighted merge. Nhóm rỗng giữ nguyên tâm. Không cộng lại embedding; xếp context theo camera rồi index tâm.

Quota/relevance/CCTS/assignment là quyết định hard trên tensor detached. **Gradient** qua values vẫn giữ: hệ số trực tiếp 1 cho tâm, $1/|C_i|$ cho thành viên. Không detach $X$ dùng merge và không thêm Gumbel cho context. Tái sử dụng hidden states sau block 1 trong cùng forward; không cache xuyên batch/update. Không dùng relevance/cosine Stage 1 vì biểu diễn đã đổi. Tính một ma trận cosine mỗi camera để dùng chung cho CCTS/assignment; batched matmul, cập nhật minimum-distance tăng dần và merge batched tránh vòng lặp từng mẫu/nhóm.

## Mask và position bias

Chuỗi mới: main 0–48; context 49–63; prompt từ 64, giữ cả slot padding. Tính lại relative position bias toàn chuỗi từ bảng pretrained encoder block 0, không gather source bias hoặc tạo bảng mới. Bias giữa main tokens cũng thay đổi so với MVPruner gốc.

| Query / Key | Main | Context | Prompt hợp lệ |
|---|---|---|---|
| Main | Cho phép | Chặn | Cho phép |
| Context | Chặn | Cho phép | Cho phép |
| Prompt | Cho phép | Cho phép | Cho phép |

Group mask và key-padding mask cộng đúng một lần vào bias cho blocks 2,3. Decoder dùng padding mask 2D, nhìn đủ memory. Không giảm/tăng attention hoặc token-size correction. Hai nhóm đã tương tác ở blocks 0,1 và vẫn truyền thông tin gián tiếp qua prompt. Dense mask không tự giảm FLOPs thành sparse attention.

Indices main/residual/center tham chiếu tập 196, map qua Stage 1 để truy nguồn 384. Một context đại diện nhiều sources: không tạo retained index giả để gather bias. Vị trí chuỗi mới khác source index.

## Chạy, cache và resume

Config và notebook riêng mang hậu tố `mvpruner_ccts_context`. FP32, batch 16/process, accumulation 2, hai GPU: effective batch danh nghĩa 64. Finetune 10 epoch; CE-only, entropy 0, gradient checkpointing false. Giữ revisions, optimizer/LR/scheduler/clipping/generation/evaluation của MVPruner. Beta giữ semantics **pre-prepare** hiện tại có chủ ý; không gọi horizon này là số update thực sau sharding.

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m efficient_vlm_ad diagnose-train --config configs/dinov2_t5_efficient_mini_kaggle_2gpu_mvpruner_ccts_context.yaml --stage finetune --debug --debug-samples 1 --debug-numerics
python -m efficient_vlm_ad.accumulation_debug --config configs/dinov2_t5_efficient_mini_kaggle_2gpu_mvpruner_ccts_context.yaml
accelerate launch --multi_gpu --num_processes 2 --mixed_precision no -m efficient_vlm_ad train --config configs/dinov2_t5_efficient_mini_kaggle_2gpu_mvpruner_ccts_context.yaml --stage finetune
```

Notebook đặt hai allocator vars trước Torch/CUDA, viết runtime YAML riêng trong `/kaggle/working`, không sửa tracked config. Prepare/cache validation → diagnose → accumulation verify → finetune/resume → evaluate → benchmark → archive. Không align, không thêm RESUME_TRAINING. Local latest được ưu tiên, sau đó filename/HF repo từ FINETUNE_CHECKPOINT. Không có checkpoint thì train từ pretrained.

Một folder `outputs/dinov2_t5_efficient_mini_kaggle_2gpu_mvpruner_ccts_context` chứa prepared data/cache/debug/checkpoints. Tái sử dụng cache chỉ qua validator identity/coverage/manifest hiện có; không chỉnh manifest. Chỉ resume finetune cùng profile, đủ RNG sidecars/world size. Không warm-start checkpoint MVPruner/align hoặc nới loader. Signature ghi main/context budgets và relevance/CCTS/assignment/merge/mask/position; khác profile bị từ chối dù shapes giống nhau.

## Debug và giới hạn

Log cần có counts 384→196→49+15, finite CE/gradients, camera residual counts/attention mass/quota, tâm/cluster sizes/nhóm rỗng, relevance range/entropy, mask/position policy, beta/horizon và accumulation config/backend. Không dump đầy đủ cosine/attention matrices.

Benchmark từ features, gồm scorer/quota/selector/CCTS/assignment/merge/T5/generation; không gồm DINO extraction offline. Profiler regions không thêm CUDA synchronization vào inference thông thường. Chi phí vòng chọn tâm và dense mask phải đo trên GPU; không suy diễn latency từ FLOPs.

Tests đối chiếu reference độc lập về quota/relevance/CCTS/assignment/merge/gradient; kiểm tra T5 mask/bias, checkpoint/lifecycle/accumulation/DDP và bảo toàn profile cũ. CPU/Gloo không chứng minh VRAM, latency, độ ổn định hai GPU hoặc chất lượng DriveLM. Không cam kết tốt hơn MVPruner49/64 trước thực nghiệm.

Nguồn: [MVPruner InternLM2](https://github.com/Zizzzzzzz/MVPruner/blob/main/drivelmm-o1/modeling/modeling_internlm2_mvpruner.py), [VisionZip merge](https://github.com/JIA-Lab-research/VisionZip/blob/main/visionzip/clip_encoder.py), [VisionZip vị trí](https://github.com/JIA-Lab-research/VisionZip/blob/main/visionzip/llava_arch.py). Biến thể này thay uniform centers bằng CCTS, dùng cosine hidden states thay key, thích nghi mask/vị trí sang T5; không tuyên bố tái hiện nguyên bản VisionZip.

## Kiểm chứng triển khai

Ngày 2026-11-03, **63 tests riêng của profile pass** trên CPU với Transformers 4.46.3 và PyTorch 2.14.0+cpu. Gồm reference độc lập cho allocation/CCTS/merge và gradient; parity selector gốc với cùng RNG; mask/bias hai block cuối; beam generation; đo prefill/FLOPs; checkpoint/resume; accumulation đủ/dư và hai rank CPU/Gloo.

Smoke CLI đã chạy thật bằng dataset ảnh nhỏ và model T5 khởi tạo offline: prepare-data → prepare-features → diagnose → finetune → verify-resume → resume epoch tiếp theo → evaluate → benchmark. Diagnose kiểm tra riêng gradient output context và gradient tới các residual hidden states dùng làm nguồn merge.

Toàn suite: **700 passed, 3 skipped, 18 failed**. Cả 18 failure đã được tái hiện độc lập trên snapshot `HEAD` trước thay đổi; danh sách failure khớp hoàn toàn, không có failure mới. Các test cũ còn kỳ vọng `RESUME_TRAINING`/entropy hoặc config, artifact và signature MVPruner 49-token trong khi config hiện tại đã là 64. Giữ nguyên các test/lựa chọn cũ ngoài phạm vi; chỉ cập nhật allowlist cho artifacts mới và expectation loại field context không hoạt động khỏi signature cũ. `git diff --check` pass.

Lệnh chạy nhóm test:

```bash
python -m pytest tests -k ccts_context -q
git diff --check
```

Không tải pretrained weights/dataset và không chạy full training. Chưa kiểm chứng CUDA, mức giảm/tăng VRAM, latency hai GPU hoặc chất lượng thực nghiệm.
