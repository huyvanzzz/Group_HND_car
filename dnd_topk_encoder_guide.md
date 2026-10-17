# Hướng dẫn áp dụng DND (Top-K) vào T5 Encoder — Group_HND_car

## Kiến trúc hiện tại (đã xác nhận từ code)

```
visual_features [B, 6, 49, 512]
    │
    ▼
DynamicInstructionAdapter (cross-attn visual ← text)
    │
    ▼
add_spatial_embeddings (row_emb + col_emb)
    │
    ▼
GatedPoolingAttention → fused [B, 49, 512], weights [B, 6]
    │
    ▼
MultiModalProjector (Linear 512→384) → visual_tokens [B, 49, 384]
    │
    ▼
modal_embeddings + concat [visual_tokens, text_tokens]
    │
    ▼
T5 Encoder (8 blocks, d_model=384)  ◄── DND sẽ được chèn ở đây
    │
    ▼
T5 Decoder → output logits → cross-entropy loss
```

**Text model:** `google/t5-efficient-mini` (8 encoder blocks, 4 decoder blocks, d_model=384)  
**Config YAML:** `repvit_t5_efficient_mini_kaggle_2gpu_di_adapter.yaml`  
**Adapter:** `DynamicInstructionAdapter` (minidrive_di) — luôn trainable  

---

## Tổng quan 6 bước

| Bước | Nội dung | File |
|------|----------|------|
| 1 | Tạo module DND (Router + Block wrapper + Loss) | `modeling/dnd.py` (MỚI) |
| 2 | Thêm DND config | `config.py` |
| 3 | Chèn DND vào T5 encoder blocks | `factory.py` |
| 4 | Đăng ký DND modules vào freeze/trainable logic | `multimodal.py` + `pipeline.py` |
| 5 | Sửa training loop — thêm DND loss | `pipeline.py` |
| 6 | Cập nhật exports + YAML config | `__init__.py` + YAML |

---

## Bước 1: Tạo file `modeling/dnd.py`

**Vị trí:** `src/efficient_vlm_ad/modeling/dnd.py` (file mới)

File này chứa 3 class:

### 1.1 `DNDRouter`
- 1 `nn.Linear(d_model, 1)` + `sigmoid`
- Init weights = 0, bias = 0 (theo paper)
- Input: hidden_states `(B, S, D)` → Output: scores `(B, S)` ∈ (0, 1)

### 1.2 `DNDT5Block`
Wrapper bọc ngoài 1 T5 encoder block gốc. Logic forward:

```
1. Chạy block gốc: vanilla_out = original_block(hidden_states, ...)
   → X_v = vanilla_out[0]  # (B, S, D)

2. Router scoring: scores = router(X_v)  # (B, S)

3. Top-K selection: 
   K = int(k_target * S)   # ví dụ 20% * S
   top_indices = topk(scores, K, dim=1).indices  # (B, K)
   
4. Pack: 
   X_selected = gather(X_v, top_indices)  # (B, K, D)

5. Nested pass (re-forward qua CÙNG block gốc):
   - Tạo position_bias mới cho subsequence K tokens
   - nested_out = original_block(X_selected, ...)
   - X_d_packed = nested_out[0]  # (B, K, D)

6. Unpack + Fusion:
   - X_d = zeros_like(X_v)
   - scatter X_d_packed vào X_d tại top_indices
   - gate = β * scores.unsqueeze(-1)  # (B, S, 1)
   - X_fused = (1 - gate) * X_v + gate * X_d

7. Lưu self._last_scores = scores (để tính loss)
8. Return (X_fused, ...) — cùng format với block gốc
```

**Lưu ý quan trọng về T5Block:**
- T5Block trả về tuple: `(hidden_states, present_key_value_state, position_bias)`
  hoặc `(hidden_states, None, position_bias, None)` tùy cấu hình
- `position_bias` được tính ở block đầu tiên rồi truyền xuống các block sau
- Khi tạo nested pass, cần truyền `position_bias=None` để T5 tự tính lại
  relative position cho subsequence ngắn K tokens

### 1.3 `DNDScoreDispersionLoss`
- Input: list of scores tensors `[(B, S), (B, S), ...]` từ mỗi DND block
- Với mỗi scores tensor:
  - Normalize thành distribution: `p' = p / sum(p)`
  - Entropy: `H = sum(p' * log(p'))` (âm → minimize = maximize entropy)
- Return: `lambda_sd * sum(H_per_layer)`

---

## Bước 2: Thêm DND config

**File:** `src/efficient_vlm_ad/config.py`

### 2.1 Tạo `DNDConfig` dataclass (thêm sau `AdapterConfig`, ~dòng 68)

```python
@dataclass(frozen=True)
class DNDConfig:
    enabled: bool = False
    L_s: int = 2          # bỏ 2 block encoder đầu
    L_e: int = -2         # bỏ 2 block encoder cuối (với T5-mini 8 blocks → L_e=6)
    k_target: float = 0.2 # Top-K ratio: 20%
    beta_init: float = 0.1
    lambda_sd: float = 3e-4
```

> [!NOTE]
> T5-efficient-mini chỉ có 8 encoder blocks (index 0-7).
> Với L_s=2, L_e=6 → DND áp dụng cho blocks 2, 3, 4, 5 (4 blocks).
> Mỗi block thêm 1 Linear(384, 1) + 1 scalar β = 385 params.
> Tổng thêm: 4 × 385 = **1,540 params** — không đáng kể.

### 2.2 Thêm `dnd` field vào `ModelConfig` (dòng ~71)

```python
class ModelConfig:
    profile: str
    vision: VisionConfig
    text: TextConfig
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    dnd: DNDConfig = field(default_factory=DNDConfig)    # ← THÊM
```

### 2.3 Sửa `load_config()` — parse DND section (dòng ~148+)

Trong hàm `load_config()`, thêm đoạn đọc `dnd` config từ YAML:
- Đọc `dnd_raw = model_raw.get("dnd", {})`
- Tạo `DNDConfig(enabled=bool(dnd_raw.get("enabled", False)), ...)`
- Truyền vào `ModelConfig(..., dnd=dnd_config)`

---

## Bước 3: Chèn DND vào T5 encoder blocks

**File:** `src/efficient_vlm_ad/modeling/factory.py`

### 3.1 Import DND modules (dòng 1-8)

Thêm: `from .dnd import DNDT5Block, DNDRouter, DNDScoreDispersionLoss`

### 3.2 Sửa `build_text_and_tokenizer()` (dòng 129-136)

Hiện tại hàm này chỉ load T5 model và trả về. Cần thêm logic chèn DND:

```python
def build_text_and_tokenizer(cfg):
    ...
    model = AutoModelForSeq2SeqLM.from_pretrained(...)
    
    # ===== THÊM: Inject DND vào encoder blocks =====
    if cfg.model.dnd.enabled:
        num_blocks = len(model.encoder.block)  # = 8 với t5-efficient-mini
        L_s = cfg.model.dnd.L_s
        L_e = cfg.model.dnd.L_e if cfg.model.dnd.L_e > 0 else num_blocks + cfg.model.dnd.L_e
        d_model = cfg.model.text.d_model  # = 384
        
        for i in range(L_s, L_e):
            original_block = model.encoder.block[i]
            model.encoder.block[i] = DNDT5Block(
                original_block=original_block,
                d_model=d_model,
                k_target=cfg.model.dnd.k_target,
                beta_init=cfg.model.dnd.beta_init,
            )
    
    return model, tokenizer
```

> [!IMPORTANT]
> `model.encoder.block` là `nn.ModuleList` của T5. Mỗi phần tử là 1 `T5Block`.
> Khi replace bằng `DNDT5Block`, PyTorch tự động đăng ký parameters mới
> (router linear + beta) vào model's parameter tree.

---

## Bước 4: Đăng ký DND vào freeze/trainable logic

### 4.1 File `src/efficient_vlm_ad/multimodal.py` (dòng 28-41)

Hiện tại `set_trainable_for_stage()` freeze/unfreeze theo stage:
- `align`: text frozen, adapter/gpa/projector trainable
- `finetune`: tất cả trainable

**DND modules** (router, beta) nằm BÊN TRONG `text_model.encoder.block[i]` nên:
- Ở stage `align`: text bị freeze → DND parameters cũng bị freeze ❌
- Ở stage `finetune`: text unfreeze → DND parameters cũng unfreeze ✅

**Cần sửa:** Sau khi freeze/unfreeze text, đảm bảo DND params luôn trainable ở cả 2 stages:

```python
def set_trainable_for_stage(modules, cfg, stage):
    ...
    _set_module_trainable(modules["text"], stage == "finetune")
    
    # ===== THÊM: DND params luôn trainable =====
    if hasattr(cfg.model, 'dnd') and cfg.model.dnd.enabled:
        text = modules["text"]
        for block in text.encoder.block:
            if hasattr(block, 'router'):  # là DNDT5Block
                _set_module_trainable(block.router, True)
                block.beta.requires_grad = True
    
    for name in ("gpa", "projector", ...):
        ...
```

### 4.2 File `src/efficient_vlm_ad/pipeline.py` — `_module_dict_for_freezing()` (dòng 482-493)

Hàm này tạo dict modules để freeze. Hiện tại DND params nằm trong `text_model`
nên không cần thêm entry mới. Logic ở 4.1 đã xử lý.

---

## Bước 5: Sửa training loop — thêm DND loss

**File:** `src/efficient_vlm_ad/pipeline.py`

### 5.1 Tạo helper function thu thập DND scores (thêm trước `train_stage()`)

```python
def _collect_dnd_scores(model) -> list[torch.Tensor]:
    """Thu thập routing scores từ tất cả DNDT5Block trong encoder."""
    scores = []
    text_model = model.text_model
    if not hasattr(text_model, 'encoder'):
        return scores
    for block in text_model.encoder.block:
        if hasattr(block, '_last_scores') and block._last_scores is not None:
            scores.append(block._last_scores)
    return scores
```

### 5.2 Sửa training loop (dòng 811-814)

**Hiện tại:**
```python
with accelerator.autocast():
    output = model(**_batch_model_inputs(batch, cfg))
    raw_loss = output.loss
    loss = raw_loss / cfg.training.gradient_accumulation_steps
```

**Sửa thành:**
```python
with accelerator.autocast():
    output = model(**_batch_model_inputs(batch, cfg))
    raw_loss = output.loss
    
    # ===== THÊM: DND Router Loss =====
    dnd_loss = torch.tensor(0.0, device=raw_loss.device)
    if cfg.model.dnd.enabled:
        unwrapped = accelerator.unwrap_model(model)
        dnd_scores = _collect_dnd_scores(unwrapped)
        if dnd_scores:
            dnd_loss_fn = DNDScoreDispersionLoss(lambda_sd=cfg.model.dnd.lambda_sd)
            dnd_loss = dnd_loss_fn(dnd_scores)
    
    total_raw_loss = raw_loss + dnd_loss
    loss = total_raw_loss / cfg.training.gradient_accumulation_steps
```

### 5.3 Cập nhật loss logging (dòng 851-853)

Sửa để log cả 2 loss components:
```python
loss_value = float(raw_loss.detach().cpu().item())
dnd_loss_value = float(dnd_loss.detach().cpu().item()) if cfg.model.dnd.enabled else 0.0
```

Thêm `dnd_loss_value` vào progress logging payload.

### 5.4 Import DNDScoreDispersionLoss (đầu file pipeline.py)

Thêm import: `from .modeling.dnd import DNDScoreDispersionLoss`

---

## Bước 6: Cập nhật exports + YAML config

### 6.1 File `src/efficient_vlm_ad/modeling/__init__.py`

Thêm:
```python
from .dnd import DNDT5Block, DNDRouter, DNDScoreDispersionLoss
```

và thêm vào `__all__`.

### 6.2 YAML config

Trong `repvit_t5_efficient_mini_kaggle_2gpu_di_adapter.yaml`, thêm section `dnd`:

```yaml
model:
  profile: repvit_t5_efficient_mini_kaggle_2gpu_di_adapter
  vision: ...
  text: ...
  adapter: ...
  dnd:                    # ← THÊM
    enabled: true
    L_s: 2                # bỏ 2 block đầu
    L_e: -2               # bỏ 2 block cuối  
    k_target: 0.2         # Top-K 20%
    beta_init: 0.1
    lambda_sd: 0.0003
```

---

## Luồng dữ liệu sau khi áp dụng DND

```
visual_features [B, 6, 49, 512]
    │
    ▼
DynamicInstructionAdapter
    │
    ▼
GPA → fused [B, 49, 512]
    │
    ▼
Projector → [B, 49, 384]
    │
    ▼
concat [visual_tokens, text_tokens] → inputs_embeds [B, S_total, 384]
    │
    ▼
T5 Encoder:
    Block 0: forward bình thường
    Block 1: forward bình thường
    Block 2: DNDT5Block  ← vanilla pass → router → Top-K → nested pass → fusion
    Block 3: DNDT5Block  ← vanilla pass → router → Top-K → nested pass → fusion
    Block 4: DNDT5Block  ← vanilla pass → router → Top-K → nested pass → fusion
    Block 5: DNDT5Block  ← vanilla pass → router → Top-K → nested pass → fusion
    Block 6: forward bình thường
    Block 7: forward bình thường
    │
    ▼
T5 Decoder → logits
    │
    ▼
Loss = CrossEntropy(logits, labels) + λ_sd * ScoreDispersionLoss(all_DND_scores)
```

---

## Lưu ý quan trọng

### Checkpoint compatibility
- DND thêm params mới (router, beta) vào `text_model.encoder.block`
- Checkpoint cũ (không có DND) sẽ thiếu keys khi load
- Cần dùng `strict=False` khi resume từ checkpoint không có DND
- Hoặc tạo hàm migrate checkpoint

### Inference mode
- Khi eval/generate, DND vẫn chạy (router vẫn chọn Top-K và re-forward)
- Điều này tốt vì cải thiện quality
- Nếu cần tắt DND khi inference, thêm flag `self.dnd_enabled` trong DNDT5Block
  và skip nested pass khi `dnd_enabled=False`

### Hyperparameters khuyến nghị
| Param | Giá trị | Lý do |
|-------|---------|-------|
| L_s | 2 | Giữ 2 block đầu xây dựng representation cơ bản |
| L_e | -2 (=6) | Giữ 2 block cuối ổn định output cho decoder |
| k_target | 0.2 | 20% tokens được review, cân bằng compute/quality |
| beta_init | 0.1 | Nhỏ để không phá representation pretrained ban đầu |
| lambda_sd | 3e-4 | Theo paper, đủ để router phân biệt tokens |

### So sánh overhead
- T5-efficient-mini: 8 encoder blocks, d_model=384
- DND áp cho 4 blocks, mỗi block re-process 20% tokens
- Nested pass attention cost: $(0.2 \times S)^2 = 0.04 \times S^2$
- Overhead per DND block: ~4% FLOPs attention + MLP cost cho 20% tokens
- Tổng overhead encoder: 4 blocks × ~4% ≈ **~16% FLOPs encoder thêm**
- Encoder chỉ chiếm ~50% total compute → overhead tổng model ≈ **~8%**
