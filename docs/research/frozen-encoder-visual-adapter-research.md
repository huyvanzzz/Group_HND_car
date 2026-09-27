# Frozen Encoder + Visual Adapter / Marker Prompting cho EM-VLM4AD

Ngày research: 2026-09-26

## 1. Tóm tắt ngắn

Vấn đề hiện tại của hướng train end-to-end RepViT là chi phí quá cao. Khi train E2E, mỗi QA sample phải load lại 6 ảnh JPG, resize/normalize, chạy RepViT trong graph, rồi backward qua RepViT. Trong DriveLM, nhiều QA có thể dùng chung một multi-view frame, nên việc load/encode ảnh bị lặp lại rất nhiều.

Hướng cần research ở đây là: **giữ vision encoder frozen, giữ feature cache, nhưng thêm module trainable sau feature** để model vẫn có khả năng thích nghi với domain autonomous driving. Nói đơn giản:

```text
6 anh
-> RepViT frozen, cache feature offline
-> cached visual features [6, 49, 512]
-> visual adapter / prompt / query module trainable
-> GPA / projector
-> T5
-> answer
```

Kết luận research tạm thời: cách này có tiền lệ rõ trong literature. Nhiều VLM mạnh không fine-tune full vision encoder, mà train connector, adapter, prompt tokens, Q-Former, Perceiver Resampler, hoặc detection-aware visual prompts. Với repo hiện tại, những module đặt **sau cached features** là hợp lý nhất nếu mục tiêu là giữ tốc độ train gần với bản freeze/cache.

## 2. Nguồn đã đối chiếu

Nguồn chính:

- MPDrive, CVPR 2025: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_MPDrive_Improving_Spatial_Understanding_with_Marker-Based_Prompt_Learning_for_Autonomous_CVPR_2025_paper.html), [arXiv:2504.00379](https://arxiv.org/abs/2504.00379)
- EM-VLM4AD: [arXiv:2403.19838](https://arxiv.org/abs/2403.19838), [GitHub akshaygopalkr/EM-VLM4AD](https://github.com/akshaygopalkr/EM-VLM4AD)
- BLIP-2: [arXiv:2301.12597](https://arxiv.org/abs/2301.12597)
- Visual Prompt Tuning: [arXiv:2203.12119](https://arxiv.org/abs/2203.12119)
- AdaptFormer: [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/69e2f49ab0837b71b0e0cb7c555990f8-Abstract-Conference.html)
- Set-of-Mark Prompting: [arXiv:2310.11441](https://arxiv.org/abs/2310.11441), [GitHub microsoft/SoM](https://github.com/microsoft/SoM)
- PIVOT visual prompting: [arXiv:2402.07872](https://arxiv.org/abs/2402.07872), [project page](https://pivot-prompt.github.io/)
- MiniDrive: [arXiv:2409.07267](https://arxiv.org/abs/2409.07267)
- DriveVLM: [arXiv:2402.12289](https://arxiv.org/abs/2402.12289)
- DriveLM: [GitHub OpenDriveLab/DriveLM](https://github.com/OpenDriveLab/DriveLM), [arXiv:2312.14150](https://arxiv.org/abs/2312.14150)
- LMDrive: [GitHub opendilab/LMDrive](https://github.com/opendilab/LMDrive), [arXiv:2312.07488](https://arxiv.org/abs/2312.07488)
- FROST-Drive: [arXiv:2601.03460](https://arxiv.org/abs/2601.03460)

Ghi chú về MPDrive code: khi search web, mình chưa thấy official GitHub repo rõ ràng cho MPDrive. Nguồn đáng tin nhất hiện tại là CVF/arXiv/supplemental. Papers With Code cũng hiện MPDrive không có code implementation rõ ràng tại thời điểm research.

## 3. Literature: họ đã giải quyết như thế nào?

### 3.1 Frozen encoder + connector/adaptor

BLIP-2 là ví dụ kinh điển: nó dùng **frozen image encoder** và **frozen LLM**, rồi train một module nhẹ tên Q-Former để nối hai không gian vision-language. Điểm quan trọng với bài của mình: nếu visual features đã có sẵn, module kiểu Q-Former/cross-attention có thể học cách rút gọn và biến đổi visual tokens mà không cần train vision encoder.

LLaVA và các biến thể visual instruction tuning cũng đi theo logic gần giống: vision encoder thường được giữ frozen, sau đó train projector/connector để đưa visual embeddings vào LLM. Điều này ủng hộ ý tưởng: nếu RepViT feature cache đã tốt, ta có thể thêm connector/adaptor trainable thay vì train lại RepViT.

FROST-Drive là nguồn autonomous driving gần với ý này nhất. Paper này lập luận rằng frozen VLM vision encoder có thể giữ được generalization tốt hơn full fine-tuning, và dùng transformer-based adapter để fuse multimodal input. Đây không phải AD-VQA giống DriveLM, nhưng rất liên quan đến câu hỏi: "có cần train vision encoder không?".

### 3.2 Visual Prompt Tuning

Visual Prompt Tuning (VPT) thêm một số prompt tokens trainable vào input/layer của vision transformer, backbone frozen. Kết quả paper cho thấy chỉ thêm ít tham số trainable vẫn có thể cạnh tranh với full fine-tuning trọng nhiều task.

Nhưng với repo hiện tại có feature cache, VPT "đúng nghĩa" bên trong RepViT không giữ được lợi thế cache, vì prompt tokens cần đi qua các layer vision transformer. Tuy vậy, ta có thể mượn biến thể **post-cache prompt tokens**:

```text
cached features [B, 6, 49, 512]
+ learnable visual prompt tokens
-> adapter / attention
-> GPA/projector/T5
```

Biến thể này không phải VPT gốc, nhưng giữ tinh thần: thêm ít tham số trainable để điều hướng visual representation.

### 3.3 AdaptFormer / adapter tuning

AdaptFormer thêm adapter nhẹ vào Transformer và giữ backbone frozen. Nó báo cáo chỉ thêm dưới 2% tham số nhưng cải thiện transfer cho image/video tasks. Nếu chèn adapter **bên trong RepViT**, ta lại cần chạy RepViT mỗi step, không dùng feature cache được.

Với repo hiện tại, cách gần nhất với AdaptFormer mà vẫn giữ cache là đặt adapter **sau cached feature**:

```text
feature = feature + MLP(LayerNorm(feature))
```

Nó không tác động vào các block RepViT, nhưng vẫn học một residual transform trên token feature `[49, 512]` của mỗi view. Đây là hướng đơn giản, nhanh, dễ test nhất.

### 3.4 MPDrive: marker-based prompt learning

MPDrive tập trung vào điểm yếu spatial reasoning của AD-VQA. Thay vì bắt LLM sinh tọa độ dạng text trực tiếp, paper biến spatial coordinate thành **visual marker**: dùng detection expert để vẽ mask/region và number label lên image, rồi model sinh/nhận diện marker index. Cách này làm tọa độ trở thành một reference nhìn-thấy-được trong ảnh.

MPDrive có hai thành phần chính:

- **MCNet (Marker ControlNet)**: xử lý cả ảnh gốc và marker image, đưa thông tin marker vào scene feature nhưng có cơ chế để giữ lại feature gốc.
- **PSPL (Perception-Enhanced Spatial Prompt Learning)**: tạo visual prompts ở hai mức:
  - scene-level prompt từ scene feature
  - instance-level prompt từ detection masks, dùng masked average pooling để lấy object features

MPDrive khác với visual adapter sau cache ở điểm rất quan trọng: MPDrive cần detection expert, marker image, object masks, và thường cần xử lý ảnh/marker image. Nếu muốn áp dụng đầy đủ MPDrive vào repo hiện tại, khả năng cao phải tạo thêm preprocessing/cache riêng cho marker/detection priors.

Phần có thể học từ MPDrive mà vẫn giữ cache:

- Không nhất thiết phải sinh tọa độ text trực tiếp; có thể biến object/coordinate thành marker index.
- Có thể thêm **instance-level visual prompts** nếu ta có detection metadata/mask/bbox được cache sẵn.
- Có thể tạo feature bổ sung từ detection priors offline, rồi train adapter trên cached features + cached object features.

Phần không giữ cache nếu làm naive:

- Vẽ marker image online mỗi step.
- Chạy MCNet trên original + marker image trong training.
- Chạy detection expert online trong training.

### 3.5 Set-of-Mark và PIVOT

Set-of-Mark (SoM) cũng dùng ý tưởng về mark/label lên image để giúp VLM grounding tốt hơn. Nó dùng segmentation/detection để partition image thành regions và overlay marks, giúp model trả lời bằng reference dễ nhìn hơn.

PIVOT dùng visual prompting lặp lại: vẽ candidate proposals lên image, hỏi VLM chọn candidate, rồi refine. PIVOT không phải training architecture cho repo này, nhưng nó ủng hộ insight chung: với spatial tasks, VLM thường làm tốt hơn khi spatial choices được đưa vào visual/text interface dưới dạng marker để dễ tham chiếu.

### 3.6 MiniDrive / DriveVLM / DriveLM / LMDrive

MiniDrive rất liên quan vì nó cũng hướng tới VLM nhẹ cho autonomous driving. Nó đề xuất FE-MoE để map 2D features thành visual token embeddings và DI-Adapter để visual tokens thay đổi theo instruction. Điểm đáng học: visual representation không nên hoàn toàn static với mỗi câu hỏi; có thể có module instruction-aware nằm sau feature extractor.

DriveVLM và DriveVLM-Dual nhấn mạnh VLM có hạn chế spatial reasoning và compute cost, nên kết hợp VLM với pipeline/priors truyền thống. Điều này ủng hộ hướng dùng detection priors/markers nếu cần spatial reasoning mạnh hơn.

DriveLM là dataset/task nền, không phải giải pháp adapter. Nhưng nó cho thấy bài toán có graph QA, multi-view, perception/prediction/planning/behavior/motion. Nếu adapter chỉ train trên QA shuffled theo record, nó không tận dụng trục graph/frame-level; đây là điểm có thể cải tiến sau.

LMDrive là closed-loop E2E driving với language instruction, không trùng trực tiếp với DriveLM QA. Nó dùng multi-modal sensor + language, nhưng training phức tạp hơn và không phải hướng giữ feature cache đơn giản.

## 4. Các hướng có thể áp dụng vào repo hiện tại

### Hướng A: Visual Adapter sau cache

Đặt module trainable ngay sau cached RepViT features:

```text
cached_features [B, 6, 49, 512]
-> LayerNorm
-> MLP 512 -> bottleneck -> 512
-> residual add
-> GPA
```

Bản chất: giống adapter/residual tuning trên latent features. Không cần load ảnh khi train. Đây là hướng đơn giản nhất.

Ưu điểm:

- Giữ feature cache.
- Gần như không tăng I/O.
- Dễ debug shape/numerics.
- Có thể bật/tắt bằng config.
- Checkpoint nhỏ.

Nhược điểm:

- Không sửa được feature extraction của RepViT thật sự.
- Nếu cached feature thiếu thông tin object/marker, adapter không tự tạo được thông tin đó.

### Hướng B: Prompt tokens trên visual features

Thêm learnable tokens vào visual sequence:

```text
[prompt tokens] + [49 patch/grid tokens mỗi view]
-> adapter attention / transformer block
-> GPA/projector
```

Bản chất: biến thể post-cache của VPT. Nó không cần chạy RepViT, nhưng cần một attention block nhỏ để prompt tokens interact với cached tokens.

Ưu điểm:

- Vẫn giữ cache.
- Nhẹ hơn Q-Former đầy đủ.
- Có khả năng học task-specific visual context.

Nhược điểm:

- Tăng sequence length hoặc cần logic pooling mới.
- Không phải VPT gốc trong vision backbone, nên cần ablation riêng.

### Hướng C: Q-Former / cross-attention visual queries

Dùng learnable query tokens attend vào cached visual tokens:

```text
queries [Nq, d]
cross-attend to cached features [6*49, 512]
-> compact visual tokens [Nq, d]
-> T5
```

Bản chất: gần BLIP-2/Q-Former hoặc Perceiver Resampler. Đây là hướng mạnh hơn adapter MLP vì nó có thể chọn/rút gọn tokens theo attention.

Ưu điểm:

- Vẫn giữ feature cache.
- Có thể giảm token count nếu cần.
- Phù hợp nếu 6*49 tokens quá nhiều/nhiều nhiều.

Nhược điểm:

- Code phức tạp hơn MLP adapter.
- Cần quyết định query count, hidden dim, attention layers.
- Training có thể nhạy cảm hơn MLP adapter.

### Hướng D: Marker/detection-aware features kiểu MPDrive

Làm offline detection/marker preprocessing, rồi cache thêm object priors:

```text
cached RepViT scene features
+ cached object bbox/mask/centroid/marker-id features
-> scene adapter + instance adapter
-> GPA/projector/T5
```

Có hai mức:

- Bản nhẹ: chỉ dùng bbox/centroid/category làm text/object tokens.
- Bản gần MPDrive hơn: tạo marker images offline, cache marker-image features hoặc object pooled features.

Ưu điểm:

- Đánh trúng điểm yếu spatial reasoning.
- Gần với MPDrive nhất.
- Có thể cải thiện các câu hỏi có `<c4,CAM_BACK,x,y>` hoặc object reference.

Nhược điểm:

- Cần detection expert hoặc metadata object đáng tin.
- Nếu tạo marker image/MCNet online thì mất lợi thế cache.
- Nếu cache offline thì pipeline phức tạp hơn và tốn dung lượng hơn.

### Hướng E: E2E partial unfreeze

Chỉ unfreeze vài block cuối RepViT hoặc thêm adapter bên trong RepViT.

Ưu điểm:

- Thật sự thay đổi encoder representation.
- Có thể tốt hơn nếu domain gap lớn.

Nhược điểm:

- Bắt buộc load ảnh và chạy RepViT trong training.
- Chậm hơn cache rất nhiều.
- Có nguy cơ quay lại 6-7 giờ/epoch trên Kaggle T4 x2.

## 5. Bảng tradeoff

| Hướng | Cần load ảnh khi train? | Dùng feature cache? | Chi phí train | Độ khó | Lợi ích spatial reasoning | Phù hợp repo hiện tại |
| --- | --- | --- | --- | --- | --- | --- |
| Visual Adapter sau cache | Không | Có | Thấp | Thấp | Vừa, phụ thuộc feature có sẵn | Rất phù hợp |
| Prompt tokens trên visual features | Không | Có | Thấp-vừa | Vừa | Vừa | Phù hợp |
| Q-Former / cross-attention queries | Không | Có | Vừa | Vừa-cao | Vừa-cao, vì query có thể chọn token liên quan | Phù hợp nếu chấp nhận refactor |
| Marker/detection-aware features offline | Không, nếu cache offline | Có, kèm cache mới | Vừa | Cao | Cao với spatial/object QA | Phù hợp sau khi có detection priors |
| MPDrive đầy đủ với MCNet online | Có | Không hoặc cache phức tạp | Cao | Cao | Cao | Chưa phù hợp ngay |
| E2E partial unfreeze RepViT | Có | Không | Cao | Vừa | Vừa-cao | Chỉ nên là later experiment |
| Full E2E RepViT | Có | Không | Rất cao | Vừa | Không đảm bảo hơn adapter | Không nên ưu tiên trên Kaggle |

## 6. Cái gì giữ được cache, cái gì bắt buộc quay lại image path?

Giữ được cache:

- MLP visual adapter sau cached features.
- Residual adapter trên `[B, 6, 49, 512]`.
- Prompt tokens interact với cached features.
- Q-Former / Perceiver-style cross-attention trên cached features.
- Instruction-aware adapter dùng question embedding để modulate cached visual tokens.
- Detection/object priors nếu bbox/mask/object features được tính và cache offline.

Bắt buộc quay lại image path:

- Train/unfreeze RepViT weights.
- Chèn LoRA/adapters bên trong RepViT.
- VPT đúng nghĩa chèn prompt vào layer RepViT.
- Marker image online mỗi step.
- MCNet online xử lý original + marker image.
- Detection expert online trọng training.

Vùng trung gian:

- MPDrive-style marker có thể giữ cache nếu làm offline: tạo marker images/object masks trước, cache marker-image features hoặc instance features. Nhưng lúc đó pipeline data/cache sẽ phức tạp hơn bản hiện tại.

## 7. Shortlist đang thảo luận tiếp

Không chốt giải pháp cuối ở đây, nhưng có 3 hướng đáng để thảo luận tiếp nhất:

1. **Visual Adapter sau cache**
   - Nên làm baseline đầu tiên.
   - Đơn giản, nhanh, ít rủi ro.
   - Trả lời đúng mong muốn: không train full encoder nhưng có thêm visual module trainable.

2. **Instruction-aware visual adapter**
   - Học từ MiniDrive DI-Adapter.
   - Thay vì visual feature static cho mỗi câu hỏi, dùng question embedding để gate/modulate visual features.
   - Vẫn giữ cache, nhưng hợp với VQA hơn adapter static.

3. **MPDrive-lite offline object/marker priors**
   - Nếu mục tiêu là spatial reasoning, đây là hướng có ý nghĩa nhất.
   - Không nên làm MCNet online ngay.
   - Nên research dataset có sẵn object annotations/detection outputs không; nếu có, cache object-level tokens để đưa vào adapter.

## 8. Ghi chú cho repo hiện tại

Mapping với pipeline hiện tại:

- Bản freeze/cache hiện tại đã có cached RepViT features `[N, 6, 49, 512]`.
- GPA hiện tại fuse across 6 views.
- Vị trí hợp lý nhất để chèn module mới là:

```text
CachedVLMDataset
-> visual_features
-> NEW visual_adapter / prompt_adapter
-> GPA
-> projector
-> T5
```

Training stage nếu thêm adapter:

- `align`: train visual adapter + GPA + projector + modal/spatial embeddings, freeze T5.
- `finetune`: train visual adapter + GPA + projector + modal/spatial embeddings + T5.
- RepViT vẫn frozen và không nằm trong training graph.

Acceptance research quan trọng:

- Nếu thêm adapter sau cache, tốc độ train nên gần với bản freeze/cache, không gần với bản E2E.
- Nếu thêm marker/detection online, tốc độ sẽ quay lại chậm do xử lý ảnh/detection.
- Vì vậy, bắt đầu từ adapter sau cache là cách ít rủi ro nhất để kiểm tra lợi ích.

## 9. Kết luận

Ý tưởng "freeze encoder, không load 6 ảnh khi train, thêm visual adapter" là có cơ sở trong literature. BLIP-2, LLaVA-style connectors, VPT, AdaptFormer, MiniDrive, và FROST-Drive đều ủng hộ một mẫu chung: **không nhất thiết phải fine-tune full vision encoder để cải thiện downstream VLM**.

MPDrive không phải là "visual adapter sau cache" đơn thuần. MPDrive là một hướng spatial prompting mạnh hơn, đưa marker/detection priors vào model để biến coordinate reasoning thành marker reasoning. Nếu muốn học từ MPDrive mà vẫn giữ tốc độ, nên làm bản "MPDrive-lite offline": cache object/marker priors, rồi đưa vào adapter sau feature.

Hướng nên thảo luận tiếp:

- Baseline nhanh: visual adapter sau cache.
- Bản VQA hơn: instruction-aware adapter sau cache.
- Bản spatial hơn: MPDrive-lite với detection/object priors offline.

Chưa nên chốt implementation cuối trước khi quyết định mục tiêu ưu tiên: tốc độ train, metric ngon hơn, spatial reasoning tốt hơn, hay thiết kế research để ablation nhiều hướng.
