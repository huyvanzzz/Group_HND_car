# T5 Internal MVPruner-Style Ablation Design

Ngay cap nhat: 2026-10-05

Branch thuc hien: `nguyet-t5-internal-mvpruner`

## 1. Muc tieu

Tai branch nay, muc tieu la them mot path thuc nghiem moi cho huong 3:

```text
visual_features [B, 6, 49, C]
  -> spatial/view embeddings
  -> projector sang T5 d_model
  -> flatten thanh [V1 ... V6 | Question]
  -> T5 encoder layers truoc pruning layer
  -> lay self-attention text-query -> visual-key
  -> chon visual tokens theo ablation policy
  -> compact sequence [selected visual tokens | text]
  -> T5 encoder layers con lai + decoder
```

Path nay khong dung GPA truoc T5. GPA baseline duoc giu rieng o branch `nguyet`; branch nay chi giu logic phuc vu A1-A3.

Trong 3 ablations A1-A3, cac tham so prune chinh duoc giu giong nhau:

```yaml
architecture: t5_internal_pruning
pruning:
  layer_policy: middle
  keep_ratio: 0.5
  min_keep_per_view: 2
```

Khac biet duy nhat can so sanh la `selection_policy`.

## 2. Diem chung cua A1-A3

Tat ca A1-A3 bat dau voi 294 visual tokens:

```text
6 views * 49 patch tokens = 294 visual tokens
```

Voi `keep_ratio=0.5`, so visual tokens duoc giu lai la:

```text
round(294 * 0.5) = 147 tokens
```

Pruning layer duoc resolve theo `layer_policy: middle`, nen A1-A3 dung cung mot layer neu dung cung T5 backbone. Tai pruning layer, model lay self-attention:

```text
A: [B, heads, sequence, sequence]
visual_slice = [0 : 294]
text_slice   = [294 : 294 + T]
```

Visual relevance score:

```text
score(v_j) = mean_over_heads_and_valid_text_tokens A[text_i -> visual_j]
```

Sau khi chon token, sequence duoc compact:

```text
X_next = [selected_visual_tokens | text_tokens]
```

T5 encoder tiep tuc tinh relative position bias tren compacted sequence. Ban dau khong co gang preserve original relative distance.

## 3. Ba ablation configs

### A1: global_topk

Config:

```text
configs/repvit_t5_efficient_mini_internal_pruning_a1_global_topk.yaml
```

Policy:

```text
reserve min_keep_per_view token cho moi view
phan budget con lai chon top-k tren toan bo 294 visual tokens
```

Y nghia:

```text
Token cua 6 cameras canh tranh truc tiep voi nhau.
View nao co text->visual score cao co the giu nhieu token hon.
```

Rui ro:

```text
Global top-k co the bo qua mot camera neu attention score cua camera do thap.
```

### A2: per_view_uniform

Config:

```text
configs/repvit_t5_efficient_mini_internal_pruning_a2_per_view_uniform.yaml
```

Policy:

```text
chia 147-token budget gan deu cho 6 views
trong tung view, chon top-k token theo text->visual score
```

Y nghia:

```text
Moi camera duoc giu coverage gan bang nhau.
Text attention chi quyet dinh token nao duoc giu trong cung mot view.
```

Rui ro:

```text
Co the lang phi token cho view it lien quan va khong cho view quan trong giu them token.
```

### A3: IRA

Config:

```text
configs/repvit_t5_efficient_mini_internal_pruning_a3_ira.yaml
```

Policy:

```text
token_score[v, p] = text->visual score cua patch p trong view v
view_score[v] = mean_p token_score[v, p]
view_importance = softmax(view_score)
chia 147-token budget theo view_importance
trong tung view, chon top-k token theo token_score
```

Y nghia:

```text
Question quyet dinh camera nao duoc giu nhieu token hon.
Day la ablation gan voi y tuong IRA cua MVPruner nhat trong ban v1.
```

Khac voi MVPruner goc:

```text
V1 nay khong dung first_stage_share vi pipeline hien tai chua co Stage 1 DRA/CCTS.
Budget IRA duoc chia truc tiep tu attention-derived view_importance.
```

## 4. Cach chay thuc nghiem

Feature cache dung chung cho A1-A3:

```text
outputs/repvit_t5_efficient_mini_internal_pruning_cache/cache
```

Train align stage:

```bash
python -m efficient_vlm_ad train --config configs/repvit_t5_efficient_mini_internal_pruning_a1_global_topk.yaml --stage align
python -m efficient_vlm_ad train --config configs/repvit_t5_efficient_mini_internal_pruning_a2_per_view_uniform.yaml --stage align
python -m efficient_vlm_ad train --config configs/repvit_t5_efficient_mini_internal_pruning_a3_ira.yaml --stage align
```

Fine-tune stage:

```bash
python -m efficient_vlm_ad train --config configs/repvit_t5_efficient_mini_internal_pruning_a1_global_topk.yaml --stage finetune --resume outputs/repvit_t5_efficient_mini_internal_pruning_a1_global_topk/checkpoints/align_best.pt
python -m efficient_vlm_ad train --config configs/repvit_t5_efficient_mini_internal_pruning_a2_per_view_uniform.yaml --stage finetune --resume outputs/repvit_t5_efficient_mini_internal_pruning_a2_per_view_uniform/checkpoints/align_best.pt
python -m efficient_vlm_ad train --config configs/repvit_t5_efficient_mini_internal_pruning_a3_ira.yaml --stage finetune --resume outputs/repvit_t5_efficient_mini_internal_pruning_a3_ira/checkpoints/align_best.pt
```

Debug numerics nen chay truoc moi experiment dai:

```bash
python -m efficient_vlm_ad train --config configs/repvit_t5_efficient_mini_internal_pruning_a1_global_topk.yaml --stage align --max-steps 1 --debug --debug-numerics
```

## 5. File/folder can giu trong branch nay

Can giu cho A1-A3:

```text
src/efficient_vlm_ad/modeling/t5_internal_pruning.py
src/efficient_vlm_ad/modeling/factory.py
src/efficient_vlm_ad/modeling/multimodal.py
src/efficient_vlm_ad/pipeline.py
src/efficient_vlm_ad/config.py
src/efficient_vlm_ad/cli.py
configs/repvit_t5_efficient_mini_internal_pruning_a1_global_topk.yaml
configs/repvit_t5_efficient_mini_internal_pruning_a2_per_view_uniform.yaml
configs/repvit_t5_efficient_mini_internal_pruning_a3_ira.yaml
tests/test_config.py
tests/test_modeling.py
```

Khong nen xoa:

```text
outputs/repvit_t5_efficient_mini_internal_pruning_cache/cache
```

Ba config A1-A3 dang tro ve cache nay de tai su dung RepViT features. Neu xoa cache, can chay lai `prepare-features`.

## 6. Dieu can doc ket qua

Neu A1 tot hon A2:

```text
Global competition giua cac camera co ich; cau hoi thuong chi can mot so views noi bat.
```

Neu A2 tot hon A1:

```text
Coverage deu giua cameras quan trong hon viec de attention tu do bo view.
```

Neu A3 tot hon A1/A2:

```text
Adaptive view budget theo IRA co ich; question-conditioned camera allocation nen duoc phat trien tiep.
```

Neu A3 kem A2:

```text
View importance tu attention co the chua on dinh o pruning layer hien tai; can sweep layer hoac them warmup/training schedule.
```
