# MVPruner Stage2-only

Ablation độc lập `mvpruner_stage2_only`: DINO đóng băng, cache `[N,6,64,384]`, spatial/camera embeddings, projector tuyến tính, modal embeddings và T5-efficient-mini dùng revision ghim như MVPruner gốc. Stage 1 tắt hoàn toàn: không gọi DRA, CCTS, pairwise distance, cosine relevance hoặc counterfactual probe. `stage1_top_k: 384` chỉ để tương thích constructor; profile quyết định bypass, không có công tắc Stage 1 riêng.

Giữ attention T5 của block 1 trước dropout, tính từ hidden states đi vào block 1; chọn token từ hidden states sau block 1. Attention softmax trên tất cả visual + prompt keys hợp lệ; điểm visual không qua softmax bổ sung. Gọi $m_t$ là mask prompt, $H$ là số head và $A_{h,t,j}$ là attention từ prompt query tới visual key:

$$s_j=\frac{\sum_h\sum_t m_t A_{h,384+t,j}}{H\sum_t m_t},\qquad r_c=\frac1{64}\sum_{j:C_j=c}s_j.$$

$$\hat r_c=\frac{r_c-\min r}{\max(\max r-\min r,\varepsilon)},\qquad R_c=\operatorname{softmax}_c(\hat r),\qquad w_c=R_c.$$

Đây là `min-max` rồi `softmax` giữa camera, không dùng $N_c(1+R_c-S_c)$. Helper bounded allocation chia tổng 49 theo trọng số, capacity 64 và minimum 1/camera; quota nguyên không có gradient trực tiếp. Dùng nguyên Gumbel/STE của Stage 2: train chọn bằng log-score có nhiễu beta, eval không nhiễu. Forward giữ 49 token gốc theo thứ tự nguồn; backward qua relaxed slots. Giữ toàn bộ prompt và gather mask/position bias theo indices chuỗi 384 + prompt, không tái tạo bias. Log kỳ vọng: `stage1_enabled=false`, `stage2_method=t5_attention_direct_camera_v1`, `loss_type=ce`, token `384 -> 49`, score range/clamp ratio, camera relevance/shares/quota, beta/counter/horizon.

Giữ FP32, batch 16 mỗi GPU, accumulation 2 (effective batch 64 trên hai GPU), 10 epoch finetune, optimizer/LR/scheduler/clip và generation/evaluate/benchmark như bản gốc. Entropy weight 0, gradient checkpointing false; không thêm seed policy hoặc align epoch. Beta giữ semantics pre-prepare gốc có chủ ý, vẫn tác động nhiễu chọn token Stage 2. Với 21.337 batch trước prepare, 10.669 batch/rank, A=2 và 10 epoch, horizon cũ là 106.690 và beta cuối khoảng 0.1; không gọi đây là horizon đúng số optimizer updates sau prepare.

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m efficient_vlm_ad diagnose-train --config configs/dinov2_t5_efficient_mini_kaggle_2gpu_mvpruner_stage2_only.yaml --stage finetune --debug --debug-samples 1 --debug-numerics
python -m efficient_vlm_ad.accumulation_debug --config configs/dinov2_t5_efficient_mini_kaggle_2gpu_mvpruner_stage2_only.yaml
accelerate launch --multi_gpu --num_processes 2 --mixed_precision no -m efficient_vlm_ad train --config configs/dinov2_t5_efficient_mini_kaggle_2gpu_mvpruner_stage2_only.yaml --stage finetune
```

Notebook `notebooks/kaggle_train_dinov2_mvpruner_stage2_only_finetune.ipynb` tạo YAML runtime riêng trong `/kaggle/working`, không ghi YAML tracked. Hai biến allocator đặt trước Torch/CUDA. Các bước chuẩn bị dữ liệu/cache, diagnose, accumulation_debug, evaluate và benchmark giữ protocol gốc.

Output ở `outputs/dinov2_t5_efficient_mini_kaggle_2gpu_mvpruner_stage2_only`; prepared data, cache, checkpoints và debug nằm trong các thư mục con tương ứng. Tái sử dụng feature DINO cần manifest v2, provenance, prepared-file identity, shape, index, byte length và sample coverage đều hợp lệ. Không sửa manifest để ép tái sử dụng.

Không resume checkpoint MVPruner hai-stage hoặc QAware: signature phải khớp profile mới, không có stage1 algorithm version. Notebook ưu tiên `finetune_latest.pt` cục bộ, rồi filename cấu hình, rồi model repo tùy chọn. Resume cần đủ RNG sidecar hai rank, world size 2. Không có checkpoint thì bắt đầu từ pretrained T5 và multimodal module khởi tạo mới; không chạy align.

Hai block đầu nhận `[B,384+Lq,384]`, hai block cuối nhận `[B,49+Lq,384]`. Benchmark dùng protocol hiện tại, từ features đến generation/prefill, bao gồm attention scoring/quota/selector; DINO extraction offline bị loại khỏi thời gian đo. Bỏ chi phí Stage 1 nhưng tăng chiều dài chuỗi hai block đầu so với MVPruner hai-stage; không mặc định nhanh hơn hoặc tốt hơn.

## Kiểm chứng

192 test liên quan pass trên CPU: config/signature, quota so reference scalar độc lập, hard gather/STE và Gumbel parity, native T5 attention/identity, padding và position bias, block lengths 384 → 49, CE gradients, checkpoint strict/resume, notebook JSON/Python/Bash, accumulation và hai rank CPU/Gloo. Test bắt lỗi nếu Stage 1/CCTS/pairwise distance được gọi. Câu hỏi không có token hợp lệ bị từ chối dù Stage 1 đã bypass.

Chuỗi CLI thật trên dataset/model giả offline pass: prepare-data → prepare-features → diagnose → accumulation verify → finetune → verify-resume → resume → evaluate → benchmark. Sau hai epoch nhỏ có 6 microbatch, 4 optimizer updates và beta_step=4; verify-resume không ghi lại checkpoint. Accumulation verify kiểm tra 69 gradient tensors, cả nhóm đủ/dư có norm ratio=1, max error=0, không optimizer update và không đổi weights/buffers.

So với code HEAD trước thay đổi: signature cả năm profile cũ giữ nguyên. Với cùng active weights, bốn profile dùng split encoder giữ nguyên state-dict keys, logits và CE ở eval; source model unpruned không sửa. Các config/notebook cũ và helper MVPruner/T5 split/scorer/loader không thay đổi. Suite tổng được báo riêng vì repository có các test notebook resume/entropy đã lệch hành vi hiện tại.

Suite tổng: **635 passed, 3 skipped, 14 failed**. Cả 14 failure đã được xác nhận riêng trước khi chạy suite tổng: 12 test trong `test_accumulation_notebooks.py` còn yêu cầu công tắc `RESUME_TRAINING` đã bỏ và hai test trong `test_camera_query_qaware_notebook.py` còn hard-code entropy 0.5 trong khi config hiện tại là 0. Không sửa các test ngoài phạm vi này và không tuyên bố toàn suite xanh. Bốn trường hợp quota, identity native và block-length bổ sung được xác nhận trong nhóm 192 test cuối.

GPU Kaggle, VRAM, tốc độ và chất lượng full training chưa kiểm chứng. Không tải model/dataset lớn hoặc chạy full training trong kiểm chứng offline. Đây là ablation đồng thời bỏ Stage 1 và thay công thức quota, không tách riêng ảnh hưởng của từng thay đổi.
