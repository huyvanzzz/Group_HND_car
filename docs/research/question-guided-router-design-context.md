# Question-Guided Router Design Context

Date: 2026-10-02

## Current Project Context

This repository is a refactor of EM-VLM4AD for Kaggle training on DriveLM-nuScenes style multi-view VQA.

The current stable training path uses cached RepViT-M1.5 visual features and a T5-Efficient-Mini text model:

```text
6 camera images
-> frozen RepViT feature cache
-> visual features [B, 6, 49, 512]
-> multimodal fusion
-> T5-Efficient-Mini
-> generated answer text
```

The cache-based design is important because DriveLM has many QA records that reuse the same multi-camera frame. Training end-to-end through RepViT requires repeatedly loading and encoding six images per QA sample, which was much slower on Kaggle T4 x2.

The current code already supports:

- cached-feature training,
- DI-adapter training,
- two-stage training: `align` then `finetune`,
- multi-GPU training through Accelerate,
- checkpoint verification for T5 tied weights and optimizer duplicate storage,
- progress/debug JSONL logs for Kaggle.

The new design should preserve these operational advantages and only change the visual fusion module.

## Problem With The Current GPA Fusion

The current GPA fusion follows the original EM-VLM4AD design.

Input visual features have shape:

```text
[B, 6, 49, 512]
```

where:

- `6` is the number of camera views,
- `49` is the `7 x 7` RepViT spatial feature grid per camera,
- `512` is the visual feature dimension.

GPA computes one scalar weight per camera and fuses the six views by weighted summation:

```text
[B, 6, 49, 512] -> [B, 49, 512]
```

This is lightweight, but it collapses all six views into the same 49 spatial positions. For example, a token from the front camera and a token from the back camera at the same `7 x 7` grid position are mixed into one token. This can blur view-specific evidence and makes it harder for the model to reason about direction-sensitive questions such as front/back/left/right objects.

This is not a code bug. It is a design tradeoff inherited from EM-VLM4AD. The new router design is intended to test whether a more selective fusion strategy can preserve useful multi-view evidence while keeping the number of visual tokens small.

## Design Goal

Replace GPA with a question-guided token router for a new experimental profile.

The router should:

- keep the RepViT feature cache,
- avoid re-running the vision encoder during training,
- keep the visual token count comparable to GPA by default,
- use the question to select relevant visual tokens,
- preserve camera identity through learnable camera embeddings,
- optionally regularize selected tokens with a small diversity loss.

The original GPA path must remain available as a baseline.

## Proposed Visual Flow

The new fusion path is:

```text
cached RepViT features [B, 6, 49, 512]
-> add camera ID embeddings
-> add row/column spatial embeddings
-> flatten 6 x 49 tokens into 294 visual tokens
-> compute a question representation from T5 input embeddings
-> FiLM-modulate visual tokens with the question representation
-> score all 294 visual tokens
-> select top-K visual tokens
-> sort selected tokens by original camera/patch order
-> project selected tokens to T5 hidden size
-> concatenate visual tokens and text tokens
-> T5 answer generation
```

Default:

```text
K = 49
```

The default is intentionally equal to the current GPA output length, so the first comparison is fair:

```text
GPA:    294 visual tokens -> 49 fused tokens
Router: 294 visual tokens -> 49 selected tokens
```

Later ablations can try larger values such as `K = 96` or `K = 128`.

## Camera-ID Embedding

The current model has row/column spatial embeddings and visual/text modality embeddings, but it does not have camera identity embeddings.

The router design adds six learnable camera vectors:

```text
Front
Front-Left
Front-Right
Back
Back-Left
Back-Right
```

Each camera embedding is added to all 49 tokens of that camera:

```text
features [B, 6, 49, 512]
+ camera_embedding [1, 6, 1, 512]
```

This gives every visual token explicit view-direction information before token scoring and selection.

## Row/Column Spatial Embedding

RepViT does not produce ViT-style patch tokens, but its cached output is still a spatial feature map:

```text
224 x 224 image -> RepViT -> 7 x 7 feature grid -> 49 tokens
```

Therefore, row/column spatial embeddings are still useful. They tell the model where each token is located inside the camera view:

```text
visual token = visual content + camera identity + row position + column position
```

This is not patch-specific logic. It is spatial-grid positional encoding.

## Question-Guided Token Router

The router receives:

```text
visual_features: [B, 6, 49, 512]
text_tokens:     [B, L, d_model]
attention_mask:  [B, L]
```

It computes a question representation by masked mean pooling over T5 input embeddings:

```text
question_repr = masked_mean(text_tokens, attention_mask)
```

Because the visual dimension and text dimension can differ, the question representation is projected into the visual feature dimension.

Then FiLM modulation is applied:

```text
gamma, beta = MLP(question_repr)
visual_modulated = visual_tokens * (1 + gamma) + beta
```

Each modulated visual token receives a scalar score:

```text
scores = scorer(LayerNorm(visual_modulated))
```

The router selects the top-K tokens. After selection, indices should be sorted by original token order, not by score order. This keeps the final token sequence stable and preserves camera/patch ordering.

## Diversity Loss

The diversity loss is computed directly on the K visual tokens selected by the router.

For this project:

```text
initial visual tokens:  [B, 294, 512]
selected visual tokens: [B, 49, 512]
```

The loss is not computed on:

- dropped visual tokens,
- text tokens,
- answer tokens,
- T5 logits.

The intended formula is:

```python
normed = F.normalize(selected_tokens, p=2, dim=-1)      # [B, K, D]
sim = torch.einsum("bkd,bqd->bkq", normed, normed)     # [B, K, K]

eye = torch.eye(K, dtype=torch.bool, device=sim.device)
mask = ~eye.unsqueeze(0)

loss_div_raw = sim[mask.expand_as(sim)].pow(2).mean()
loss_div = diversity_weight * loss_div_raw
```

This is conceptually the same as the ColaVLA-style diversity regularizer: selected tokens are L2-normalized, pairwise cosine similarities are computed, and off-diagonal similarities are penalized.

The implementation should use a boolean off-diagonal mask instead of subtracting the identity matrix and averaging over the diagonal. This avoids including diagonal zeros in the denominator.

The total training loss is:

```text
loss_total = loss_vqa + diversity_weight * loss_div_raw
```

Default:

```text
diversity_weight = 0.001
```

The loss should be treated as a weak regularizer. It should not dominate the T5 answer-generation cross entropy.

## Required Debug Signals

Router experiments must log enough information to verify whether the router is behaving sensibly.

At minimum, debug logs should include:

```text
selected_token_count
selected_camera_histogram
selected_score_min
selected_score_max
selected_score_mean
loss_vqa
loss_div_raw
loss_div_weighted
diversity_ratio = loss_div_weighted / loss_vqa
```

The most important sanity check is `selected_camera_histogram`.

Example:

```json
{
  "Front": 10,
  "Front-Left": 6,
  "Front-Right": 7,
  "Back": 18,
  "Back-Left": 4,
  "Back-Right": 4
}
```

For questions about rear traffic, it would be encouraging to see more selected tokens from rear-facing cameras. The histogram should not be used as a hard correctness metric, but it is useful for debugging collapse and question sensitivity.

## Experiment Strategy

The first implementation should not remove or rewrite the GPA path.

Recommended ablations:

```text
A. GPA baseline
B. GPA + camera embedding
C. Router top_k=49, diversity_weight=0.0
D. Router top_k=49, diversity_weight=0.001
E. Router top_k=96, diversity_weight=0.001
```

The first router profile should default to:

```yaml
fusion:
  type: question_guided_router
  top_k: 49
  camera_embedding: true
  film: true
  diversity_weight: 0.001
```

`top_k=49` is chosen for a fair comparison against GPA. Larger `top_k` values can be explored after the 49-token router is stable.

## What Is Not Included In V1

The first router implementation should not include:

- Gumbel Top-K,
- task-type auxiliary loss,
- question-type classifier,
- ego-state input,
- end-to-end RepViT training,
- object detector or marker prior.

These ideas can be explored later, but adding them now would make it harder to isolate whether the router itself improves over GPA.

## Success Criteria

The router design is considered technically ready if:

- cached-feature training still works,
- `align` and `finetune` both run with finite loss,
- checkpoint verify/resume still passes,
- evaluation and benchmark still work,
- debug logs show router-specific stats,
- selected token shape is `[B, 49, 512]` by default,
- the old GPA and DI-adapter configs remain usable.

The design is considered useful only if downstream metrics or qualitative predictions improve against GPA/DI-adapter baselines. The implementation itself should make this comparison easy and reproducible.
