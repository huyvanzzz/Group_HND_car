# MVPruner Adaptation Notes For Efficient VLM AD

Ngay nghien cuu: 2026-09-27

Nguon doc:
- Local repo hien tai: `C:\23020407\Group_HND_car`
- MVPruner local reference: `C:\23020407\Group_HND_car\MVPruner`
- MVPruner implementation note: `MVPruner/transformers/src/transformers/models/llama/modeling_llama_mvpruner.py`

## 1. Vi sao khong copy truc tiep MVPruner

MVPruner prune visual tokens trong LLaMA/LLaVA decoder. No giu tung block visual tokens cua 6 camera trong cung mot sequence ngon ngu, lay attention text-to-vision o layer sau, roi remove token va cap nhat `position_ids`, `cache_position`, attention mask.

Pipeline hien tai cua `src/efficient_vlm_ad` khac hon:

```text
visual_features [B, 6, 49, C]
  -> GPA fuse 6 views
  -> fused visual [B, 49, C]
  -> projector
  -> concat voi T5 text embeddings
  -> T5 encoder-decoder
```

Sau GPA, dimension view `6` da bien mat. Vi vay khong con text-to-view attention sau nhieu layer nhu MVPruner. Neu muon copy y nguyen MVPruner, can doi kien truc de giu 6 view tokens vao sau trong T5, va can can thiep vao internals cua T5 encoder. Do la mot thay doi lon, khong phu hop cho experiment dau tien.

## 2. Ba huong adaptation

### Huong 1: question-conditioned GPA

Giu nguyen kien truc hien tai, chi them cau hoi vao buoc tinh score cua GPA. Day la huong nhe nhat va hop voi code hien co nhat.

V1 dung multiplicative gate:

```text
visual_hidden = Z(flat_visual) * G(flat_visual)  # [B, 6, H]
gate = sigmoid(Wq q).unsqueeze(1)                # [B, 1, H]
conditioned_hidden = visual_hidden * gate        # [B, 6, H]
scores = w(conditioned_hidden).squeeze(-1)       # [B, 6]
weights = softmax(scores, dim=1)
fused = sum(weights * visual_features, dim=1)
```

Hypothesis:

```text
question decides which dimensions of each view representation matter
```

Uu diem:
- it doi code
- khong doi shape output cua model
- van dung cache visual features hien co
- de so sanh voi baseline visual-only GPA

Gioi han:
- day la text-conditioned view fusion, chua phai token pruning
- speedup khong tang vi so visual tokens sau GPA van la 49
- attention khong den tu layer sau cua T5 nhu MVPruner

### Huong 2: text-guided token pruning before GPA

Van giu feature cache `[B, 6, 49, C]`, nhung tinh score giua text summary va tung visual token de giu top-k token truoc khi GPA fuse.

Uu diem:
- bat dau cham vao token pruning
- van khong can chay lai visual encoder
- co the do trade-off giua accuracy va so token truoc T5

Gioi han:
- dynamic top-k co the lam batching phuc tap neu so token moi sample khac nhau
- score chi dua tren text summary, khong phai attention noi bo cua T5
- can policy ro rang cho padding/mask neu token count bien doi
- phai lua chon k

### Huong 3: T5-internal MVPruner-style pruning

Dua ca 6 view token blocks vao T5, lay attention text-to-vision o mot encoder layer, roi prune tokens va chay tiep cac layer sau.

Uu diem:
- gan MVPruner nhat
- co tiem nang speedup that su neu visual token count lon
- co the hoc/cat token theo dynamic information requirement cua cau hoi

Gioi han:
- can rewrite/wrap T5 encoder internals
- can quan ly attention mask va position/relative bias cua T5 can than
- thay doi lon hon ve architecture va checkpoint compatibility
- rui ro cao hon cho experiment dau tien

## 3. Lua chon v1

Chon Huong 1: question-conditioned GPA voi multiplicative gate. Khong dung concat trong scoring MLP va khong dung additive conditioning trong pass dau tien.

Cong thuc muc tieu:

```text
s_v = w^T [ h_v * sigmoid(W_q q) ] + b
```

Trong do:
- `h_v` la hidden representation cua view `v` tu GPA hien tai
- `q` la masked mean pooling cua question token embeddings
- `W_q q` tao gate theo dimension hidden cua GPA
- softmax tren 6 scores tao view weights

Additive alternative:

```text
s_v = w^T tanh(W_v h_v + W_q q)
```

Alternative nay co interpretation sach nhu query-conditioned attention over six camera views, nhung de lai cho experiment sau.

## 4. Feature cache note

Co the va nen luu visual encoder features khi train vi vision encoder dang freeze. Repo hien tai da co flow nay:

```text
prepare-data
  -> outputs/.../prepared_data/*.jsonl

prepare-features
  -> outputs/.../cache/features.float16.memmap
  -> outputs/.../cache/index.json
  -> outputs/.../cache/manifest.json

train/evaluate/benchmark
  -> CachedVLMDataset doc cached features
```

`CachedVLMDataset` validate manifest theo `model_id`, `image_size`, va `view_order` truoc khi doc memmap. Vi vay training khong can chay lai visual encoder neu cache da ton tai va config khop.

Recommended workflow cho question-conditioned GPA:

```bash
python -m efficient_vlm_ad prepare-data --config configs/repvit_t5_efficient_mini_question_gpa.yaml
python -m efficient_vlm_ad prepare-features --config configs/repvit_t5_efficient_mini_question_gpa.yaml
python -m efficient_vlm_ad train --config configs/repvit_t5_efficient_mini_question_gpa.yaml --stage align
python -m efficient_vlm_ad train --config configs/repvit_t5_efficient_mini_question_gpa.yaml --stage finetune --resume CHECKPOINT
```

Luu y: config experiment moi dung output/cache dir rieng de khong tron voi baseline. Neu muon tai su dung cache baseline, co the tro `cache.dir` ve cung cache baseline khi vision config, image size, view order, va prepared data root giong nhau.
