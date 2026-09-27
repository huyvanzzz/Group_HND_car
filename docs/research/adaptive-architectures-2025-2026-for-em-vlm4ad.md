# Adaptive Architectures 2025/2026 cho EM-VLM4AD

Ngay research: 2026-09-27

## 1. Tom tat ban chat

Bai toan cua minh hien tai khong phai chi la "them adapter nao". Cai can giu la **toc do cua pipeline feature cache**:

```text
RepViT frozen
-> cache visual features [N, 6, 49, 512]
-> train module nhe sau cache
-> GPA/projector/T5
```

Vay nen mot huong adaptive chi that su phu hop neu no **khong bat training loop load lai 6 anh va chay RepViT moi step**. Theo research 2025/2026, cac huong adaptive dang noi bat co the chia thanh hai nhom:

- **Phu hop voi minh**: module dat sau cached visual features, vi du visual adapter, instruction-aware adapter, query/token selector tren cached tokens.
- **Khong giu duoc loi the cache neu lam dung nguyen ban**: adapter/LoRA/VPT chen ben trong vision encoder, vi luc do phai chay encoder trong training.

Ket luan ngan: huong dang gia nhat cho repo nay la **post-cache instruction-aware visual adapter**. No nam giua adapter MLP don gian va Q-Former nang, van giu cache nhung co kha nang lam visual features thay doi theo cau hoi.

## 2. Nguon chinh da doi chieu

- pFedMMA, ICLR 2026 / arXiv 2025: [arXiv:2507.05394](https://arxiv.org/abs/2507.05394), [GitHub sajjad-ucsb/pFedMMA](https://github.com/sajjad-ucsb/pFedMMA)
- Multi-Modal Adapter goc cho CLIP: [arXiv:2409.02958](https://arxiv.org/abs/2409.02958), [CVPR 2024 Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_MMA_Multi-Modal_Adapter_for_Vision-Language_Models_CVPR_2024_paper.html)
- MiniDrive: [arXiv:2409.07267](https://arxiv.org/abs/2409.07267)
- TS-VLM: [arXiv:2505.12670](https://arxiv.org/abs/2505.12670)
- FROST-Drive: [arXiv:2601.03460](https://arxiv.org/abs/2601.03460), [WACV 2026 PDF](https://openaccess.thecvf.com/content/WACV2026W/LLVM-AD/papers/Dong_FROST-Drive_Scalable_and_Efficient_End-to-End_Driving_with_a_Frozen_Vision_WACVW_2026_paper.pdf)
- MPDrive: [arXiv:2504.00379](https://arxiv.org/abs/2504.00379), [CVPR 2025 Open Access](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_MPDrive_Improving_Spatial_Understanding_with_Marker-Based_Prompt_Learning_for_Autonomous_CVPR_2025_paper.html)
- VLA4AD survey 2025: [arXiv:2506.24044](https://arxiv.org/abs/2506.24044), [ICCVW 2025 PDF](https://openaccess.thecvf.com/content/ICCV2025W/WDFM-AD/papers/Jiang_A_Survey_on_Vision-Language-Action_Models_for_Autonomous_Driving_ICCVW_2025_paper.pdf)
- PEFT/VLM surveys 2025/2026: [PEFT A2Z arXiv:2504.14117](https://arxiv.org/abs/2504.14117), [ScienceDirect VLM survey 2025](https://www.sciencedirect.com/science/article/pii/S1566253525006955)
- Adaptive visual token pruning 2025: [ATP-LLaVA CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Ye_ATP-LLaVA_Adaptive_Token_Pruning_for_Large_Vision_Language_Models_CVPR_2025_paper.html), [AdaptInfer arXiv:2508.06084](https://arxiv.org/abs/2508.06084), [GlimpsePrune arXiv:2508.01548](https://arxiv.org/abs/2508.01548)

Ghi chu: mot so paper nen nhu BLIP-2, VPT, AdaptFormer cu hon 2025 nhung van la goc kien truc. File nay chi nhac khi can, con trong tam la cac huong 2025/2026 va autonomous driving.

## 3. Taxonomy cac huong adaptive

| Nhom | Y tuong | Giu feature cache? | Phu hop voi minh? |
| --- | --- | --- | --- |
| Post-cache visual adapter | Residual MLP/adapter tren `[B, 6, 49, 512]` | Co | Rat phu hop |
| pFedMMA-inspired shared bottleneck | Visual/text branch rieng, shared projection de align modal | Co neu dat sau cache | Phu hop sau khi don gian hoa |
| Instruction-aware adapter | Cau hoi/T5 embedding gate visual features | Co | Rat dang thu |
| Prompt/query adapter | Learnable prompts hoac mini query tokens attend vao visual tokens | Co | Phu hop nhung nang hon adapter |
| Adaptive token pruning/selection | Chon/gop visual tokens theo cau hoi | Co neu lam tren cached tokens | Phu hop de tang toc/loc nhieu |
| Marker/object-aware offline priors | Detector/marker features chay offline, cache lai | Co neu frozen offline | Phu hop later, phuc tap |
| Encoder-internal adapter | Adapter/LoRA/VPT chen trong RepViT/T5 encoder | Khong voi vision encoder | Khong uu tien |
| Full/partial E2E unfreeze | Train RepViT hoac block cuoi RepViT | Khong | Khong uu tien tren Kaggle |

## 4. Phan tich cac paper/hướng

### 4.1 pFedMMA / Multi-Modal Adapter

pFedMMA de xuat personalized federated fine-tuning cho CLIP. Diem quan trong khong nam o federated learning, ma nam o **multi-modal adapter**:

```text
visual hidden -> visual down -> shared projection -> visual up -> residual
text hidden   -> text down   -> shared projection -> text up   -> residual
```

Paper noi moi adapter co modality-specific down/up projection va globally shared projection de align cross-modal features. Repo chinh thuc cung co `visual_adapter`, `text_adapter`, `shared_adapter`.

Can sua lai mot hieu nham quan trong: pFedMMA freeze backbone weights, nhung adapter cua no nam **trong forward cua CLIP image/text encoders**, khong phai chi dat ngoai sau final embedding. Bang chung code: `encode_image()` truyen `visual_adapter_func` vao image encoder, `encode_text()` truyen `text_adapter_func` vao text encoder.

Ap dung cho minh:

- Khong nen chen y het pFedMMA vao RepViT, vi se phai chay RepViT trong training.
- Nen muon **cau truc shared bottleneck** cua pFedMMA, nhung dat sau cached RepViT features.
- Ban hop ly:

```text
cached visual features
-> visual_down
-> shared_bottleneck
-> visual_up
-> residual
-> GPA
```

Neu muon multi-modal hon:

```text
T5 question embeddings
-> text_down
-> shared_bottleneck
-> text_up
-> residual
```

Nhung ban dau nen lam visual branch truoc de giam rui ro.

### 4.2 MiniDrive / Dynamic Instruction Adapter

MiniDrive rat sat voi bai cua minh vi cung la VLM nhe cho autonomous driving. Paper de xuat:

- FE-MoE: map 2D features thanh visual token embeddings.
- DI-Adapter: lam visual token embeddings thay doi theo instruction text embeddings.

Insight quan trong: trong VQA, cung mot anh nhung cau hoi khac nhau can tap trung vao thong tin khac nhau. Neu visual tokens static cho moi cau hoi, model co the kem hon. Day la dung diem yeu cua pipeline cache hien tai: cached RepViT feature la static theo frame.

Ap dung cho minh:

```text
question input_ids
-> T5 embedding / pooled question embedding
-> gate vector hoac FiLM scale-shift
cached visual features
-> visual adapter duoc modulate boi question
-> GPA/projector/T5
```

Huong nay giu cache vi no khong train RepViT. No chi lam cached features **question-conditioned**. Day co le la huong "adaptive" phu hop nhat voi AD-VQA.

Rui ro:

- Can thiet ke pooling question embedding don gian, tranh dung T5 encoder them lan rieng.
- Neu gate qua manh co the gay unstable loss; nen dung residual gate nhe.

### 4.3 TS-VLM / text-guided soft pooling

TS-VLM 2025 de xuat text-guided pooling cho multi-view driving reasoning, nham giam compute va tap trung vao regions lien quan. Theo abstract, TS-VLM co the giam compute dang ke va co ban nho chi khoang 20.1M parameters.

Ap dung cho minh:

- Hien EM-VLM4AD/GPA dang fuse view theo visual tokens, nhung khong that su question-conditioned.
- Co the them text-guided scoring de chon/gom visual tokens truoc GPA:

```text
question embedding
cached visual tokens [6, 49, 512]
-> score/token importance
-> weighted pooling hoac top-k soft pooling
-> GPA/projector
```

Huong nay giu cache va rat hop voi multi-view VQA. No la cau noi giua adapter va token pruning.

Rui ro:

- Neu top-k hard selection som qua, co the mat spatial detail.
- Nen bat dau bang soft weights/residual, chua nen hard prune.

### 4.4 Adaptive token pruning / token selection 2025

Nhieu paper 2025 tap trung giam visual token redundancy:

- ATP-LLaVA: adaptive token pruning cho LVLM.
- AdaptInfer: dynamic text-guided pruning.
- GlimpsePrune: dynamic pruning theo do phuc tap scene.

Diem chung: visual tokens qua nhieu, va nen chon token theo context/query thay vi fixed ratio.

Ap dung cho minh:

- Minh chi co `6*49 = 294` visual tokens, khong qua lon nhu LVLM high-res, nen token pruning khong phai uu tien de giam compute.
- Nhung token selection co the dung de **tang reasoning**: chon token lien quan cau hoi, giam nhieu cho T5.
- Nen coi la module phu:

```text
instruction-aware adapter
-> optional soft token weighting
-> GPA
```

Khong nen lam hard pruning dau tien, vi AD-VQA can spatial context 6 camera.

### 4.5 FROST-Drive / frozen vision encoder cho autonomous driving

FROST-Drive 2026 lap luan rang full fine-tune vision encoder trong driving co the lam mat generalization. Ho giu vision encoder frozen va them transformer-based adapter cho multimodal fusion, roi decoder waypoint.

Diem co ich cho minh:

- No ung ho chien luoc **frozen vision encoder + train adapter** trong autonomous driving.
- No khong phai AD-VQA, nhung cung chung rang buoc real-time/robustness.

Khac biet:

- FROST-Drive van co the chay vision encoder trong forward cho task driving; minh muon train tren cached features.
- Vi vay minh chi muon principle: frozen encoder + adapter, khong muon y het data path.

### 4.6 MPDrive / marker-based prompt learning

MPDrive 2025 giai quyet spatial understanding bang visual markers. No dung detection expert/marker image va co cac thanh phan nhu MCNet, PSPL, scene-level va instance-level prompts.

Ap dung cho minh:

- Neu lam day du, MPDrive se can detector/marker encoder, rat co kha nang lam mat loi the cache.
- Chi phu hop neu tao object/marker priors offline:

```text
offline detector / marker preprocessing
-> cache object bbox/category/marker features
-> train adapter doc cache
```

Day la huong later experiment, khong nen lam dau tien.

### 4.7 Encoder-internal PEFT: LoRA, DoRA, VPT, AdaptFormer

Survey PEFT 2025/2026 chia cac huong nhu adapter tuning, prompt tuning, LoRA/DoRA, token-level/representation-level tuning. Nhung voi vision encoder:

- LoRA vao RepViT: can chay RepViT trong training.
- AdaptFormer/VPT trong RepViT: can chay RepViT trong training.
- pFedMMA nguyen ban trong image encoder: can chay image encoder trong training.

Ket luan: nhung huong nay co the tot ve parameter efficiency, nhung **khong giai quyet bottleneck load/encode 6 anh** cua minh.

Voi T5 thi LoRA/DoRA van co the dung, vi T5 dang nam trong training graph san. Nhung day la text-side PEFT, khong thay the visual adaptive module.

## 5. Shortlist cho repo hien tai

| Huong | Toc do | Do kho | Rui ro NaN/OOM | Spatial reasoning | Phu hop Kaggle | Ghi chu |
| --- | --- | --- | --- | --- | --- | --- |
| Visual-only adapter sau cache | Rat nhanh | Thap | Thap | Vua | Rat phu hop | Baseline nen lam dau tien |
| pFedMMA-inspired shared bottleneck sau cache | Nhanh | Vua | Thap-vua | Vua | Phu hop | Manh hon MLP vi co shared alignment idea |
| Instruction-aware visual adapter | Nhanh | Vua | Vua | Vua-cao | Rat dang thu | Sat AD-VQA nhat vi visual feature phu thuoc question |
| Soft text-guided token weighting | Nhanh | Vua | Thap-vua | Cao hon GPA thuong | Phu hop | Hoc tu TS-VLM/token pruning nhung khong hard prune |
| Mini Q-Former/query adapter | Vua | Cao | Vua-cao | Cao | Thu sau | Co the nang va nhay hyperparam |
| MPDrive-lite offline object priors | Vua | Cao | Vua | Cao | Later | Can detector/marker cache offline |
| Encoder-internal adapter/LoRA RepViT | Cham | Vua-cao | Vua | Co the cao | Khong uu tien | Pha loi the cache |

## 6. Huong dang de thao luan tiep

### Uu tien 1: Instruction-aware visual adapter sau cache

Day la huong minh danh gia hop nhat. No giu cache, nhung giai quyet diem yeu "visual feature static cho moi cau hoi".

Kien truc goi y:

```text
question embeddings
-> question pooler
-> gate / FiLM params

cached visual features
-> LayerNorm
-> MLP adapter
-> modulate bang gate/FiLM
-> residual add
-> GPA
```

Stage training:

- Align: train adapter + GPA + projector + embeddings, freeze T5.
- Finetune: train adapter + GPA + projector + embeddings + T5.
- RepViT van frozen va khong vao graph.

### Uu tien 2: pFedMMA-inspired shared bottleneck

Dung khi muon multi-modal alignment ro hon:

```text
visual branch: visual_down -> shared -> visual_up
text branch:   text_down   -> shared -> text_up
```

Nhung nen lam sau visual-only/instruction-aware adapter, vi text branch can can than de khong lam hong T5 input embeddings.

### Uu tien 3: Soft text-guided token weighting

Thay vi Q-Former nang, dung question embedding score tung visual token:

```text
score = f(visual_token, question_embedding)
weighted_visual = visual_token * sigmoid(score)
```

No nhe hon Q-Former, van adaptive theo cau hoi, va co the giai thich duoc token/camera nao duoc tap trung.

## 7. Nhung huong khong nen uu tien ngay

- Full E2E RepViT: qua cham tren Kaggle T4 x2.
- LoRA/adapter trong RepViT: tiet kiem tham so nhung khong tiet kiem load/encode anh.
- Q-Former day du: co the tot nhung train phuc tap, khong nen la buoc dau.
- MPDrive day du: hay ve spatial reasoning, nhung can detection/marker pipeline phuc tap.
- Hard token pruning: co nguy co bo mat spatial detail trong multi-view AD-VQA.

## 8. Ket luan

Adaptive architecture 2025/2026 khong chi co adapter MLP. Co nhieu nhom: multi-modal shared adapters, instruction-aware adapters, text-guided pooling, query adapters, adaptive token pruning, marker/object-aware prompts, frozen-encoder driving adapters.

Nhung voi rang buoc cua minh, **duong phu hop nhat la cac module dat sau cached RepViT features**. Trong nhom do, huong dang gia nhat la:

1. Instruction-aware visual adapter sau cache.
2. pFedMMA-inspired shared bottleneck sau cache.
3. Soft text-guided token weighting/pooling.

MPDrive-lite va object priors nen de sau, khi minh da co baseline adapter va can tang spatial reasoning. Encoder-internal PEFT chi nen dung neu chap nhan quay lai image path cham.
