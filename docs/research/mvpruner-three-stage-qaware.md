# MVPruner ba stage với question-aware merge

Profile riêng `mvpruner_three_stage_qaware` finetune từ pretrained T5, không chạy align (`align_epochs: 0`). Luồng: DINOv2 frozen `[B,6,64,384]` → spatial/camera embeddings → projector → DRA/CCTS `384 → 196` → encoder blocks 0,1 → attention/IRA/Gumbel-STE prune `196 → 98` → blocks 2,3 → final norm/dropout → qaware camera-query merge `98 → 49` → decoder.

Notebook: [kaggle_train_dinov2_mvpruner_three_stage_qaware_finetune.ipynb](../../notebooks/kaggle_train_dinov2_mvpruner_three_stage_qaware_finetune.ipynb). Config: [profile ba stage](../../configs/dinov2_t5_efficient_mini_kaggle_2gpu_mvpruner_three_stage_qaware.yaml).

| Ranh giới | Shape |
| --- | --- |
| Blocks 0,1 | `[B,196+Lq,384]` |
| Blocks 2,3 | `[B,98+Lq,384]` |
| Đầu vào merge cuối encoder | `[B,98,384]`, `[B,Lq,384]` |
| Decoder memory | `[B,49+Lq,384]` |

Stage 2 giữ thuật toán hiện có: attention trước dropout của block index 1, IRA theo camera, bounded allocation tối thiểu một token/camera, hard Top-K và STE. Điểm visual là trung bình attention từ các prompt query hợp lệ; không thêm softmax riêng trên visual scores. Token được chọn từ hidden states sau block, giữ thứ tự nguồn; mask và position bias được gather theo từng mẫu, không cộng mask hai lần.

Stage 3 dùng **cả visual lẫn prompt hidden states cuối encoder**. Giữ learned attentive text pooling, native softmax, camera log-shares, prefix hard/soft DP, detached candidate pool, RMS keys và raw values. Budget prune 98 độc lập budget query 49; query bank chung `[44,64]`. Khởi tạo query/key/omega/scorer/pooling và LM head giữ nguyên protocol qaware hiện có. Camera ID chỉ nguồn token, không bảo đảm hidden states chưa trao đổi thông tin giữa camera.

Với scorer logits \(s_j\), query \(q_i\), key \(k_j\), compressor dùng:

\[
P_{ij}=\operatorname{softmax}_j\left((q_i\odot\omega)^\top k_j/\sqrt{64}+0.1s_j+\mathrm{CameraMask}_{ij}\right),\qquad Y=PX_f.
\]

Decoder nhận \([Y;T_f]\), mask \([\mathbf1_{49};m_q]\). Không thêm embedding, normalization, encoder block hay position bias sau merge; bias đã gather thuộc đường encoder 98 token. Loss giữ fixed cross-entropy và `entropy_weight: 0.0`.

Một counter update thành công dùng chung cho hai stage:

\[
H=\max(1,E\lceil L_{\mathrm{prepared}}/A\rceil),\qquad
\beta(t)=0.2-0.1\min(t/\max(1,H-1),1).
\]

Tính horizon sau `accelerator.prepare`. Với 10,669 microbatch/rank, accumulation 2, 10 epoch: **53,350** planned updates/rank. Debug max_steps không rút ngắn horizon; skipped update không tăng counter. Resume giữ horizon/counter và kiểm tra `beta_step == optimizer_step`. Hai stage lấy Gumbel độc lập; hard/soft DP của Stage 3 dùng chung draw riêng của nó. Eval/generation không thêm noise.

Config/notebook giữ revisions, preprocessing, camera order, linear projector, FP32 feature cache, batch 16/process, accumulation 2, hai GPU, 10 epoch, optimizer/LR/scheduler/clipping, seed 42 và generation/evaluation của camera-query-qaware. Gradient checkpointing false. Notebook đặt cả hai allocator variables trước Torch/CUDA, ghi runtime YAML riêng, giữ `FINETUNE_CHECKPOINT`, latest/best và optimizer/scheduler cùng RNG sidecars từng rank. Output/prepared/cache/checkpoint dùng một thư mục profile, không hậu tố phiên bản. Chỉ tái dùng cache sau validator, không sửa manifest.

Chạy notebook theo thứ tự prepare/cache validation → diagnose → accumulation verify → finetune/resume → evaluate → benchmark. Ví dụ CLI sau khi đã chuẩn bị cache:

```bash
CONFIG=configs/dinov2_t5_efficient_mini_kaggle_2gpu_mvpruner_three_stage_qaware.yaml
python -m efficient_vlm_ad diagnose-train --config "$CONFIG" --stage finetune --debug --debug-numerics
python -m efficient_vlm_ad.accumulation_debug --config "$CONFIG"
python -m accelerate.commands.launch --multi_gpu --num_processes 2 --mixed_precision no -m efficient_vlm_ad train --config "$CONFIG" --stage finetune
```

JSONL forward dùng `mvpruner.stage2_prune`/`mvpruner.stage3_merge`; backward dùng `mvpruner_gradients.stage2_prune`/`mvpruner_gradients.stage3_merge`. Mỗi nhóm có `input_count`/`output_count`; quota prune là `stage2_budgets`, quota merge là `stage3_budgets`. Kiểm tra `text_pool_entropy` và `text_pool_max_weight_mean` ở cả hai pha. Không dump attention matrix hay graph. Resume chỉ nhận cùng profile/recipe; không warm-start checkpoint align/qaware cũ.

Benchmark bắt đầu từ features, không gồm DINO extraction. Prefill gồm encode/reduction và decoder pass đầu với LM head. Metadata phân biệt Stage 1:196, hai encoder block cuối:98, decoder visual memory:49 cộng prompt; FLOPs đo đường chạy thật. Đối chứng chính là qaware hai stage ra 49 token; so với MVPruner 64 token phải ghi rõ decoder memory khác nhau. Dùng cùng input/settings, warmup, đảo thứ tự và median cho component timing, forward/backward và generation.

Fusion production có **693.568 tham số**, bằng qaware hai stage; mọi weight chung khởi tạo khớp chính xác khi dùng cùng seed. Bank vẫn `[44,64]`. Đã kiểm tra parity Stage 2 và Stage 3 ở train/eval, gradient, padding, source indices/bias, learned pooling học từ CE, generation beam 1/3 và FLOPs đường chạy thật. Hai rank CPU/Gloo có quota khác nhau nhưng gradient/model state đồng bộ; kiểm tra accumulation đủ/dư qua hai epoch so với reference trung bình độc lập và replay RNG sidecars khớp chính xác.

Kiểm chứng cuối ngày 2026-11-05: **125 tests trọng tâm pass** trong 116,51 giây, gồm toàn bộ **53 test mới** và regression signature/notebook. Có CLI thật prepare/features → diagnose → accumulation verify → train → verify/resume → evaluate/benchmark; kiểm tra trực tiếp cả JSONL forward/backward và optimizer/head state sau resume. Mọi source Python và notebook mới qua kiểm tra syntax; Python heredoc/Bash cells hợp lệ; `git diff --check` pass.

Full suite đã chạy: **817 passed, 3 skipped, 22 failed** trong 1.136,22 giây. Đối chiếu JUnit với baseline xác nhận **19 failure cũ trùng hoàn toàn**. Ba failure còn lại là test kỳ vọng field `stage3_top_k=None` xuất hiện trong signature cũ, test liệt kê artifact chưa nhận file mới, và test so protocol chưa loại override `align_epochs: 0`; cả ba đã được cập nhật và pass trong lượt 125 tests trên. Không còn failure mới chưa xử lý. Một số test mới được bổ sung sau khi full suite đã collect; lượt 125 tests cuối bao phủ đầy đủ bản mới.

Các failure nền giữ nguyên ngoài phạm vi: 12 test accumulation-notebook còn đòi biến `RESUME_TRAINING` đã bỏ theo yêu cầu trước; ba test entropy kỳ vọng giá trị cũ khác config hiện tại; bốn test config/protocol/signature MVPruner còn kỳ vọng budget 49 trong khi config hiện tại là 64. Không sửa config/notebook cũ hoặc thay đổi thuật toán để làm xanh các assertion này. CPU smoke không xác lập chất lượng, latency hay VRAM GPU; chưa chạy CUDA, full training hoặc tải model/dataset lớn trong kiểm chứng cục bộ. Chưa commit/push.

Phép đo CPU nhỏ ngày 2026-11-05, FP32, một thread, batch 2, sáu camera × 64 token, prompt dài 12 (mẫu thứ hai có ba padding), label dài 5; T5 nhỏ `D=32`, FFN 64, bốn encoder block/hai decoder block, query dim 8, scorer hidden 8, dropout 0. Hai model dùng trọng số chung giống hệt; warmup 3, 15 lần đo đảo thứ tự, báo median. Không chạy test suite song song khi đo.

| Phép đo | Qaware hai stage 196→49 | Ba stage 196→98→49 |
| --- | ---: | ---: |
| Eval forward có teacher forcing/CE | 26,307 ms | 48,469 ms |
| Train forward + backward | 76,030 ms | 118,404 ms |
| Generation, beam 1, tối đa 5 token mới | 38,927 ms | 65,463 ms |
| Generation, beam 3, tối đa 5 token mới | 46,721 ms | 71,991 ms |
| Prefill major-operator FLOPs, beam 1 | 52.588.672 | 67.613.952 |
| Prefill major-operator FLOPs, beam 3 | 54.847.616 | 69.872.896 |

Đo component riêng để hooks không làm nhiễu tổng thời gian: Stage 1 khoảng 7,074/7,199 ms; blocks 0,1 khoảng 7,739/8,112 ms; blocks 2,3 khoảng 1,656/2,375 ms; qaware merge khoảng 4,599/3,858 ms (hai stage/ba stage). Nhánh mới thêm helper dựng lại attention khoảng 16,607 ms và selector khoảng 6,682 ms. Giữ nguyên helper/selector theo plan, không đưa fast path khác thuật toán. Số đo này chỉ mô tả CPU/model nhỏ; đặc biệt overhead Python, indexing và kiểm tra numerics không phản ánh tỷ lệ latency của T5 production trên GPU. FLOPs vẫn chỉ bao phủ major operators của counter hiện có, không đếm đủ DP/indexing/normalization.

Nguồn thiết kế: [MVPruner §3.6](https://arxiv.org/html/2606.27660v1#S3.SS6), [T5 Transformers 4.46.3](https://raw.githubusercontent.com/huggingface/transformers/v4.46.3/src/transformers/models/t5/modeling_t5.py), [Accelerate prepare_data_loader](https://huggingface.co/docs/accelerate/v1.15.0/en/package_reference/accelerator#accelerate.Accelerator.prepare_data_loader).
