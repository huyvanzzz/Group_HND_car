# Nghiên cứu thay bước chọn token Stage 2 bằng query compression theo camera

Ngày đối chiếu: 2026-10-28. Trạng thái: đề xuất nghiên cứu, chưa triển khai hoặc benchmark. Không phải thiết kế đã được người dùng phê duyệt.

## 1. Phạm vi đã chốt trong trao đổi

Giữ nguyên DINOv2, preprocessing, embeddings, projector, Stage 1 DRA/CCTS, hai block T5 đầu, relevance theo câu hỏi và IRA phân bổ camera. Chỉ thay bước chọn token trong từng camera của Stage 2 bằng phép nén qua learned queries.

```text
6 camera × 64 tokens = 384
→ Stage 1 giữ nguyên: 196 tokens
→ T5 encoder block 0–1 cùng prompt
→ relevance theo câu hỏi + IRA hiện tại → ngân sách m_c cho từng camera
→ learned-query compression riêng từng camera: N_c → m_c
→ tổng 49 visual tokens + toàn bộ prompt
→ T5 encoder block 2–3 → decoder như hiện tại
```

Không quay lại nhánh 49 token gốc + 15 summaries, không bỏ Stage 1, không chuyển bộ nén lên trước T5, không thêm mạng học phân bổ camera trong nghiên cứu này.

Ở đây “tỷ lệ query” là biến budget IRA hiện tại thành số query đầu ra. Nó không phải học lại tỷ lệ bằng một mạng mới.

## 2. Nguồn đã kiểm tra và giới hạn

- PDF người dùng cung cấp: [2603.21701v1.pdf](D:/AD/Paper/2603.21701v1.pdf), đủ 16 trang, gồm phụ lục. Đọc lại phương pháp §5, setup §6.1, Tables 1–3 và 5–8, phụ lục §8–12; kiểm tra hình Table 3 trên trang PDF 8.
- [Paper trên arXiv](https://arxiv.org/html/2603.21701v1).
- [Repository chính thức](https://github.com/MArSha1147/MetaCompress): kiểm tra qua GitHub API, thư mục gốc chỉ có README.md; README thông báo code chưa công bố. Chưa thể xác minh implementation, pooling, normalization và hyperparameters bằng code tác giả. README SHA khi đọc: `9990f69d82de8713efdac4c7b9dd9f97f30da1ae`.
- [BLIP-2, paper gốc, §3.1](https://proceedings.mlr.press/v202/li23q/li23q.pdf): bằng chứng tiền lệ cho learned queries truy xuất visual features. Không phải bằng chứng cấu hình nén nhẹ đang đề xuất sẽ cải thiện DriveLM.
- Code và tests cục bộ, tại HEAD `164894c` khi khảo sát. Không chạy training, tải model/dataset hay sửa source code.

Không diễn giải lời văn của tác giả thành bảo đảm lý thuyết hoặc kết quả cho mô hình DINOv2/T5 của mình.

## 3. Những gì paper thực sự làm

Với image features X, paper sinh P từ pooled image queries và projected keys:

\[
Q=\operatorname{Pool}(X+E_{pos})W_q,\quad K=(X+E_{pos})W_k
\]
\[
P=\operatorname{softmax}_{row}(Q\operatorname{diag}(\omega)K^T/\sqrt{d_c}),\quad Z=PX.
\]

- Values là X, không bắt buộc có value/output projection hoặc Transformer mới.
- d_c nhỏ hơn chiều hidden để giảm chi phí scoring.
- Queries phụ thuộc ảnh. Không phải learned query bank cố định.
- Nén trước decoder-only LLM, backbone đóng băng; tối ưu bộ nén bằng discrepancy/KL từ teacher không nén, cộng entropy và anti-collapse.
- Trong khi đó, thiết kế của mình dùng learned queries ở giữa encoder T5, có budget theo câu hỏi và theo camera. Đây là phương pháp lai lấy cảm hứng từ paper, không phải tái tạo MetaCompress.
- Ma trận P của mình vẫn phụ thuộc mẫu qua keys; vì hidden states đã tương tác với prompt, P cũng có thể phụ thuộc câu hỏi.

### Bằng chứng về chất lượng cần ưu tiên

Table 3: LLaVA-NeXT-7B, MT-GQA, bỏ 90% visual tokens:

| Loss/thành phần | Accuracy |
| --- | ---: |
| L_pred | 61.98 |
| L_pred + entropy | 62.42 |
| L_pred + anti-collapse, không clipping | 56.34 |
| L_pred + anti-collapse + clipping | 62.13 |
| L_pred + entropy + anti-collapse + clipping | 62.70 |

Table 6 lặp lại xu hướng trên LLaVA-1.5-7B và XComposer-2.5-7B. Vì clipping chỉ xuất hiện trong một số hàng, không thể quy toàn bộ cải thiện của cấu hình đầy đủ cho anti-collapse riêng lẻ.

Ý nghĩa thực dụng:

1. Không nên chỉ copy công thức QK rồi bỏ qua objective. Entropy có bằng chứng cải thiện trong setup paper.
2. Anti-collapse đáng thử cùng clipping và theo dõi numerics; thêm riêng có thể gây hại.
3. L_pred trong paper là KL/discrepancy, không phải answer CE mà pipeline hiện tại đang dùng. Ablation không chứng minh CE + entropy sẽ đạt cùng hiệu quả.
4. Không có ablation chứng minh learned queries tốt hơn pooled queries, shared queries tốt hơn camera-specific queries, hoặc Q + MLP(mean(X)) tốt hơn Q. Đây đều là giả thuyết của mình.
5. Không có ablation cô lập chứng minh riêng omega, một giá trị d_c cụ thể, hay initialization nào là tối ưu trong hệ hiện tại.

Table 5: trên LLaVA-1.5-7B/MT-GQA, MetaCompress 58.43 so với spatial pooling 53.97, ToMe 56.06 và VisionZip 57.60. Đây là bằng chứng cho toàn phương pháp được train trong setup đó, không phải bảo đảm mọi learned pooling đều tốt.

Table 2: trên LLaVA-NeXT-7B, E2E MetaCompress 501 ms, Sample 484 ms, Base 830 ms. Không thể suy ra thêm mạng luôn nhanh hơn các phương pháp chọn token; vị trí nén trước LLM cũng khác Stage 2 của mình.

### Những phần không được bê nguyên

- §5.1 mô tả average pooling; §8.2 nói fractional max pooling và đổi cách dùng ký hiệu kernel/stride. Code chưa công bố nên chưa giải quyết được khác biệt.
- Khởi tạo W_q = W_k của paper gắn với queries tạo từ input; learned query bank không tự kế thừa lập luận gần spatial pooling/PSD đó.
- Learning rate 10, 500 epochs của §8.1 là tối ưu P riêng từng image-text pair, không phải recipe train bộ nén tổng quát.
- Recipe chính có SGD lr 1e-3, clip 1e-2, loss weights 1 trong bối cảnh backbone đóng băng. Không sao chép chúng sang full finetune hiện tại (clip 1.0) rồi coi là được paper bảo chứng.
- Paper đánh giá multi-turn decoder-only LVLM. Không dùng kết quả để tuyên bố attention theo câu hỏi không hữu ích cho DriveLM single-question.

## 4. Đối chiếu Stage 2 hiện tại

| Phần | Bằng chứng code | Xử lý đề xuất |
| --- | --- | --- |
| Stage 1 | `modeling/vlm.py:78`, `modeling/mvpruner.py:286` | Giữ thuật toán và đầu ra |
| Attention trước dropout block 1 | `modeling/t5_split_encoder.py:98`, `:158` | Giữ |
| Relevance câu hỏi → visual | `modeling/t5_split_encoder.py:119` | Giữ |
| Counts, camera relevance, weights và budget | `modeling/mvpruner.py:364`–`:380` | Giữ đúng công thức, tie-break và capacity |
| Gumbel, relaxed selection, STE | `modeling/mvpruner.py:382`–`:406` | Thay ở profile mới |
| Gather mask/bias theo source indices | `modeling/t5_split_encoder.py:171`–`:175` | Phải thích nghi vì output là mixtures |
| T5 block 2–3, final norm, decoder | `modeling/t5_split_encoder.py:180`, `modeling/vlm.py:144` | Giữ trọng số, số block và cách gọi phía sau |
| Train dùng output.loss | `pipeline.py:949` | Cần cộng regularizer rõ ràng nếu bật, không cộng vào evaluation answer loss |
| Architecture validation | `checkpointing.py:40` | Profile/signature riêng; không resume lẫn hai kiến trúc |

Các đường dẫn code trong bảng tương đối với `src/efficient_vlm_ad/`.

IRA hiện tại không chỉ softmax điểm camera. Nó tính mean relevance, min-max chuẩn hóa, so với tỷ lệ token sau Stage 1 rồi điều chỉnh weights theo counts trước khi bounded allocation. Giữ nguyên có nghĩa giữ cả chuỗi tính đó, không thay bằng công thức phân bổ mới đơn giản hơn.

## 5. Đề xuất bộ nén đầu tiên: nhỏ, có thể đối chứng

Với từng camera c, lấy X_c là các visual hidden states thuộc camera đó sau block 1:

\[
Q_c=Q_{bank}[:m_c],\quad K_c=X_cW_k
\]
\[
P_c=\operatorname{softmax}_{row}(Q_c\operatorname{diag}(\omega)K_c^T/\sqrt{d_c}),\quad Z_c=P_cX_c.
\]

Ràng buộc:

- sum m_c = 49; m_min <= m_c <= N_c; ngân sách do IRA cũ quyết định.
- Query chỉ đọc keys/values cùng camera. Ghép output theo thứ tự camera rồi query slot.
- Giữ nguyên text hidden states tại ranh giới nén và text padding mask.
- Dùng một query bank và key projection chung giữa camera trong bản đơn giản; camera identity đã có trong X. Bank riêng theo camera là một ablation nếu shared bank không đủ chuyên biệt.
- d_c = 64 là ứng viên engineering ban đầu, không phải giá trị tốt nhất đã được paper xác nhận. So 32/64/128 sau khi chốt bản cơ sở.
- Với budget 49, sáu camera và minimum 1, camera nhận tối đa 44 queries (còn phụ thuộc N_c). Shared bank chỉ cần 44 hàng cho profile này.
- Với d = 384, d_c = 64, Q_bank 44 × 64, W_k 384 × 64 và omega 64: khoảng 27,456 tham số nếu không thêm bias/module khác. Đây là số tham số, không phải phép đo latency.
- Khởi tạo các query khác nhau, không tất cả bằng 0 hoặc giống nhau. Theo dõi logit scale để tránh P đồng đều ngay kéo dài hoặc quá bão hòa.
- Không thêm Q + MLP(mean(X)), value/output projection, FFN hay nhiều resampler blocks trong bản đầu. Chỉ thêm sau ablation có lý do.
- Chưa thêm spatial positional embedding lần nữa mặc định: X đã nhận spatial/camera/modal embeddings từ trước T5. Có thể đánh giá metadata vị trí riêng trong scoring nếu có bằng chứng cần thiết.

Vì query bank không phụ thuộc input còn keys có phụ thuộc input, phép tính vẫn là data-dependent compression. BLIP-2 cung cấp tiền lệ cho learned-query extraction, nhưng Q-Former lớn và objective khác; không gọi công thức một tầng này là BLIP-2 hoặc bản paper đã kiểm chứng.

### Budget động và gradient

Queries đang active nhận gradient qua loss → Z → P → Q/W_k/omega. Các query cuối bank ít active hơn, nên phải log số lần dùng/gradient theo slot; dùng bank chung giúp tổng hợp lượt cập nhật giữa camera nhưng không bảo đảm giải quyết hết lệch tần suất.

`allocate_bounded_budget` chạy dưới no_grad. Giữ nguyên nó nghĩa là không học trực tiếp quyết định integer budget bằng answer loss.

Một thay đổi quan trọng: Stage 2 cũ có đường surrogate gradient qua p_soft về token relevance scores. Khi bỏ STE và chỉ dùng relevance để tạo budget nguyên, đường gradient đó biến mất. T5 vẫn nhận gradient qua values/keys và đường text, nhưng không được mô tả là giữ nguyên mọi đường gradient cũ. Không thêm STE cho budget mới nếu mục tiêu là giữ nguyên IRA và chỉ thay selector.

## 6. Position bias và mask: bắt buộc xử lý, không thể chỉ đổi một hàm

**Cập nhật sau khảo sát tài liệu bên ngoài:** phép trung bình bias dưới đây chỉ là ứng viên từng đề xuất, không còn là khuyến nghị mặc định. Xem §12 về representative positions, reassignment và giới hạn đối với learned-query slots.

Token mixture không có source index duy nhất. Không giả tạo `selected_indices` để gọi nguyên gather bias rồi coi là đúng.

Một phương án có cấu trúc để thử là đưa P_c vào ma trận nén visual P, thêm identity cho text:

\[
R=\operatorname{blockdiag}(P,I_{text}),\quad B_{new}=R B_{rel} R^T.
\]

Trong đó B_rel là relative position bias thuần, hữu hạn, trước khi cộng padding mask. Sau đó áp dụng key padding mask mới: 49 visual positions hợp lệ + text mask cũ.

Điểm cần kiểm chứng:

- Đây là quy tắc tổng hợp bias đề xuất, không phải tính tương đương attention và không có ablation trong MetaCompress cho T5.
- Không nhân R với bias đã chứa padding -inf hoặc giá trị âm cực lớn; có thể sinh NaN/làm bẩn kết quả. Code hiện tại trả position_bias đã gồm mask, nên phải tách/tính lại B_rel rõ ràng.
- Giữ nguyên text–text bias; xử lý đủ visual–visual, visual–text và text–visual.
- Nếu P là one-hot selector, phép biến đổi phải khớp gather bias cũ. Đây là test hợp đồng mạnh.
- Bias transport có chi phí và thay đổi gradient qua bias nếu không detach P. Phải chọn/log policy và đo riêng.

Phương án rẻ hơn để đối chứng: gán mỗi output vị trí anchor bằng argmax P rồi gather bias. Anchor chỉ là xấp xỉ, có thể trùng; không được gọi token output là token gốc được chọn. Quyết định argmax không có gradient. Không dùng anchor này để âm thầm biến output values trở lại hard pruning.

Chưa chốt policy “tốt nhất”. Bias transport nếu được thử phải là ablation riêng; ưu tiên nghiên cứu các policy có tiền lệ trong §12. Không thay đổi bias policy trong cùng phép so loss mà không ghi nhận.

## 7. Objective và thứ tự thí nghiệm ưu tiên

### Bước A: kiểm tra selector mới có học được

Giữ cùng finetune recipe và answer CE như baseline MVPruner. Train model mới, không chỉ ghép query ngẫu nhiên vào checkpoint rồi đánh giá. Dùng cùng seed/split/generation, token budget và phương thức khởi tạo backbone giữa đối chứng.

### Bước B: regularizer có bằng chứng từ paper

So lần lượt CE; CE + entropy; CE + entropy + anti-collapse + clipping.

- Entropy giảm độ lan rộng của từng hàng P, không có bảo đảm token chỉ tổng hợp một đối tượng.
- Anti-collapse giảm nhiều queries cùng tập trung một nguồn, không ép full coverage hay độc lập tuyệt đối.
- Tính regularizer trên từng camera, chỉ các queries/keys active; không tính padding slots. Tổng hợp với trọng số được định nghĩa rõ, tránh camera có nhiều query vô tình đổi scale loss giữa các mẫu.
- N_c, m_c thay đổi khiến entropy và column mass có scale khác nhau. Báo cả raw/normalized statistics và tune coefficients trên validation. Chuẩn hóa là thay đổi của mình, phải ghi lại thay vì gọi đúng nguyên paper.
- Không sao chép loss weights 1 và clip 1e-2 sang full finetune; pipeline hiện tại clip 1.0. Log norm trước/sau clipping, tỷ lệ steps bị clip.
- Tách answer CE, entropy, anti-collapse, total training loss; evaluate loss vẫn là answer CE có ý nghĩa so sánh.

### Bước C: distillation nếu cần bám sát mục tiêu của paper

Thêm frozen teacher với cùng Stage 1 và bỏ reduction Stage 2 (giữ 196 visual tokens sau boundary), nếu có checkpoint phù hợp hoặc đã train teacher đó. Teacher không có sẵn chỉ nhờ code unpruned 384-token.

Trên cùng decoder input/target positions, dùng teacher logits đã detach, KL teacher || student, bỏ label padding. Nếu dùng temperature thì ghi cả temperature và scaling. Student objective có thể là CE + lambda_KD KL + regularizers; đây là extension của mình, không phải objective nguyên bản paper.

Teacher tạo chi phí train bổ sung, không chạy lúc inference. So CE/KD riêng; không kết luận query compression tốt hơn chỉ vì student được thêm teacher còn đối chứng không được train tương ứng.

## 8. Thời gian: cần sửa nhận định quá sớm trong trao đổi

Stage 2 cũ không chỉ top-k rồi gather:

- `_relaxed_slots` tại `modeling/mvpruner.py:336` lặp k lần, mỗi lần softmax và cập nhật remaining.
- `stage2_from_scores` gọi nó không phụ thuộc train/eval; eval tắt noise nhưng vẫn xây soft slots và p_ste. Với profile hiện tại k = 44.
- Nó còn tạo p_hard, p_soft và nhân với values.

Vì vậy query compression có thêm learned weights nhưng bỏ được một loạt bước tuần tự. Không có cơ sở chắc chắn nói bản mới chậm hơn chỉ vì thêm mạng.

Ngược lại, nếu triển khai bằng vòng lặp từng mẫu/camera, đồng bộ `.item()` để lấy m_c, hoặc tính thừa 44 queries cho cả sáu camera, có thể mất lợi thế. Ưu tiên projection một lần và batch/mask các active query slots, không phụ thuộc Python loop theo từng query. Phân biệt số tương tác hữu ích với số tensor operations dense thực sự tính.

Với nén theo camera, số cặp scoring hữu ích là sum_c m_c N_c, không mặc định bằng 49 × 196. Padding/dense implementation vẫn có thể tính nhiều hơn.

Relevance, IRA, Stage 1 và T5 trước/sau có budget giống nhau vẫn là phần chi phí chung. Bias transport có thể tạo overhead mới. Không hứa hẹn speedup hoặc chênh lệch phần trăm trước GPU benchmark.

Đo riêng:

1. Relevance + allocation.
2. Selector cũ hoặc query compressor mới, gồm packing, projections và bias/mask adaptation.
3. T5 encoder block 2–3.
4. Feature-level prefill và full generate với cùng settings, beam size, length và warmup.
5. Train forward/backward và peak VRAM.

Giữ phạm vi benchmark hiện tại từ features; không cộng DINO extraction offline vào một phía. FLOPs counter hiện tại bỏ một số sort/indexing/elementwise operations, nên không dùng FLOPs đơn lẻ thay latency. Tham khảo `prefill_benchmarking.py:29`, `scripts/benchmark_mvpruner_components.py:21`.

## 9. Profile, checkpoint và tests phải thích nghi khi triển khai

Không sửa hành vi profile `mvpruner_two_stage` hiện tại. Tạo selector mode/profile mới với query dimension, bank size, sharing policy, bias policy và algorithm version trong architecture signature. Không đưa beta/tau không còn hoạt động vào identity của selector mới như thể chúng có tác dụng.

Resume chỉ cho cùng kiến trúc. Nếu warm-start trọng số chung từ MVPruner, phải có đường import tường minh, chỉ cho phép thiếu đúng module mới và reset optimizer/scheduler khi cần; không nới loader dùng chung hoặc ngoại lệ thiếu vision weights hiện tại.

Các diagnostics hiện đang đòi stage2_indices/selected_scores/dropped_scores phải có schema mới mô tả compression P/budgets; không báo summaries như selected source tokens. Train, diagnose, generation và resume phải dùng đúng cùng selector/bias policy.

Đặc biệt, `pipeline.py:1212` và `:1224` đang lấy optimizer_step từ `beta_step`. Nếu selector mới không còn Gumbel/beta schedule, không được xóa counter một cách cơ học rồi làm hỏng resume verification. Profile mới cần optimizer-step counter có semantics rõ, còn profile cũ giữ nguyên hành vi.

Tests cần có:

- Với cùng input, IRA budgets mới khớp cũ tuyệt đối, tổng 49, giới hạn theo camera.
- P không nhận nguồn ngoài camera; hàng active sum 1; không NaN khi N_c hoặc m_c = 1.
- Đủ 49 visual tokens, không mất prompt; batch có padding và budgets khác nhau.
- Queries/keys/omega/projector/T5 có gradients hữu hạn; không bắt relevance gradient phải giống STE cũ.
- Queries cuối bank có usage statistics; không có DDP unused-parameter lỗi do dynamic budget.
- One-hot P khớp output gather và bias gather; identity P khớp split/native T5 ở eval với dropout tắt.
- Train/eval dùng cùng công thức compression (khác dropout chung nếu có), không Gumbel/STE mới.
- Checkpoint không resume lẫn hai selector; thông tin nguồn/geometry/cache cũ vẫn được xác thực.
- MVPruner regression cũ tiếp tục chạy, không viết lại expected results để hợp thức hóa thay đổi.

## 10. Ma trận ablation gọn để tìm thứ thực sự mang lại kết quả

| ID | Selector | Objective | Câu hỏi |
| --- | --- | --- | --- |
| A0 | MVPruner cũ, IRA → 49 | CE hiện tại | Mốc |
| A1 | IRA → camera learned queries → 49 | CE | Thay selection có lợi không? |
| A2 | A1 | CE + entropy | Sparsity regularizer có ích không? |
| A3 | A1 | CE + entropy + anti-collapse, clipping | Giảm collapse có ích không? |
| A4 | Cấu hình tốt nhất A1–A3 | Thêm teacher KD | Distillation có đáng chi phí train không? |
| A5, tùy chọn | Cấu hình tốt nhất | Giữ loss; thêm Delta Q nhỏ | Adaptive queries có lợi ngoài keys thích nghi không? |

Giữ bias policy cố định trong A1–A3. Thử bias policy riêng trước khi kết luận model tốt nhất. Nếu resources cho phép, lặp nhiều seeds; nếu chỉ chạy một seed thì ghi giới hạn, không gọi chênh nhỏ là kết quả chắc chắn.

Theo dõi metric QA hiện có, phân nhóm trái/phải, đếm, vật thể nhỏ nếu có annotation hoặc tập đánh giá được định nghĩa; không tự nhận đã cải thiện an toàn lái xe từ metric QA. Ghi entropy P, column mass, độ trùng output queries, coverage camera và norms đầu ra so với hidden states gốc. Norm sụt do averaging là rủi ro cần đo; output normalization/projection chỉ thêm khi ablation hỗ trợ.

## 11. Đề xuất chốt sau nghiên cứu

Giữ Stage 1 + question relevance + IRA. Bản đầu dùng shared learned-query bank, shared low-dimensional key projection, omega khởi tạo 1, values giữ X, compression có mask theo camera, tổng đầu ra 49. Xử lý position bias/mask minh bạch với kiểm thử one-hot; profile riêng để không tác động baseline cũ.

Ưu tiên xác nhận CE-only học được, rồi entropy, rồi anti-collapse kèm clipping. KD là thí nghiệm tiếp theo nếu mục tiêu muốn bám chặt loss paper. Delta Q, nhiều resampler blocks hoặc mạng học budget chưa có bằng chứng phù hợp để thêm mặc định.

Phần có bằng chứng paper mạnh nhất để ưu tiên là objective/regularization và tính nhẹ của bộ nén, không phải tuyên bố một query bank cụ thể sẽ tốt hơn pooling. Kết quả DriveLM, tốc độ GPU, VRAM và DDP hai GPU đều chưa được xác minh ở thiết kế mới.

## 12. Bổ sung: các công trình khác xử lý vị trí sau merge như thế nào?

Khảo sát theo yêu cầu tiếp theo của người dùng, cùng ngày 2026-10-28. Không sửa model.

### 12.1 MrT5: cần phân biệt cơ chế chính và baseline pooling

[Paper MrT5, §3.1 và Appendix F.2](https://arxiv.org/html/2410.20771v2#A6.SS2).

- Cơ chế chính contextualize rồi xóa token; giữ/gather bias theo các vị trí còn lại, không tạo soft summary bằng query.
- Baseline boundary-prediction mean-pooling trong phụ lục gộp từng span và dùng bias của token đầu span cho cả span. Đây là tiền lệ trực tiếp cho relative bias của T5, nhưng không phải ablation chứng minh policy đó tốt nhất.
- Đã đối chiếu code tác giả: [modeling_bpt5.py](https://github.com/jkallini/mrt5/blob/e79c3e752ef73fb8c375ac3ef61aa2fc2dcff323/models/modeling_bpt5.py#L301) gather hai trục bias; [modeling_mrt5.py](https://github.com/jkallini/mrt5/blob/e79c3e752ef73fb8c375ac3ef61aa2fc2dcff323/models/modeling_mrt5.py#L657) gather cho hard deletion.

### 12.2 VCC: lấy trung bình chỉ số vị trí rồi tính lại bucket/bias

[VCC, NeurIPS 2023, Appendix 7.7, trang PDF 27](https://proceedings.neurips.cc/paper_files/paper/2023/file/4054556fcaa934b0bf76da52cf4f92cb-Paper-Conference.pdf).

Token không nén giữ position index gốc. Token đại diện compressed segment nhận floor(mean(position indices trong segment)); sau đó tính relative bias T5 bằng các indices đó. Tác giả mô tả đây là approximation.

Đây không phải trung bình các phần tử ma trận bias. Do bucket lookup phi tuyến, bias tại vị trí trung bình không bằng trung bình bias nói chung. Phương pháp vẫn có thể ánh xạ nhiều cấu trúc về cùng position; không giải quyết lossless geometry. Các segment của VCC cũng khác learned queries có thể gom nguồn rải rác.

### 12.3 STTM: có ablation rõ giữa merge, survival và reassignment

[STTM, ICCV 2025, §3.4 và Table 8](https://arxiv.org/html/2507.07990v1#S3.SS4).

| Policy vị trí | VNBench R.Acc | VideoMME R.Acc |
| --- | ---: | ---: |
| Trung bình RoPE của nguồn | 96.0 | 95.7 |
| Giữ RoPE của surviving token | 96.5 | 96.0 |
| Gán lại position IDs theo sequence mới | 98.0 | 98.8 |

Đây là relative accuracy theo baseline, không phải accuracy tuyệt đối. Ablation dùng LLaVA-Video-7B. Với Qwen2VL/M-RoPE, tác giả dùng survival vì policy reassignment đó không áp dụng trực tiếp. Không suy ra reassignment luôn tốt nhất hoặc RoPE giống T5 relative bias.

### 12.4 BLIP-2 + Flan-T5: query outputs là sequence slots mới

[Code chính thức LAVIS](https://github.com/salesforce/LAVIS/blob/main/lavis/models/blip2_models/blip2_t5.py#L99) dùng learned queries đọc image features, project output rồi prepend vào text embeddings; gọi native T5 mà không truyền bias trung bình từ image patches.

Điểm liên quan: learned query outputs không nhất thiết phải có vị trí patch đại diện. Chúng có thể là latent slots trong sequence mới. Giới hạn: BLIP-2 nén trước toàn T5; mình nén giữa encoder, nên không phải bản triển khai tương đương.

### 12.5 Điều chỉnh khuyến nghị cho thiết kế hiện tại

Không có một quy tắc chuẩn giữ vị trí đúng tuyệt đối sau merge. Bias/vị trí trùng nhau không tự chứng minh phương pháp không hoạt động; đây vốn là biểu diễn xấp xỉ, phải đánh giá tác động lên task.

Với learned queries đọc nguồn rải rác trong camera, hai policy nên xem xét trước là:

1. **New-slot positions:** thứ tự camera rồi query slot; tính lại relative bias cho sequence 49 + prompt bằng bảng bias học của T5. Tiền lệ gần nhất là latent-prefix/native T5 và STTM reassignment, chưa có bảo đảm cho mid-encoder.
2. **Representative source positions:** gán một anchor nguồn rõ ràng cho mỗi output rồi gather bias. Tiền lệ gần nhất là span-first MrT5 và survival STTM. Argmax P là ứng viên anchor của mình, không phải policy được những paper đó kiểm chứng cho learned queries; phải theo dõi anchor trùng và query đổi nguồn.

Không dùng phép lấy mean bias làm mặc định chỉ vì công thức ngắn. Không để `position_bias=None` tại block 2 rồi giả định T5 tự tái tạo đúng bias: chỉ block encoder đầu giữ learned relative-bias table trong triển khai hiện tại. New-slot policy phải tính lại từ đúng bảng đó, thêm mask đúng một lần và truyền cho các block còn lại.

Nếu không có vị trí nguồn rõ ràng cho learned summaries, new-slot policy có ý nghĩa biểu diễn rõ hơn việc gọi một average position là vị trí thật. Đây vẫn là khuyến nghị thử nghiệm, chưa phải thiết kế được chốt hoặc kết quả đã đo.
