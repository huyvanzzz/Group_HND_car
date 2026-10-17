# MVPruner two-stage profile

Profile triển khai: `configs/repvit_t5_efficient_mini_kaggle_2gpu_mvpruner.yaml`.
Pipeline router cũ và GPA không bị thay đổi; profile mới chỉ được kích hoạt khi
`model.fusion.name: mvpruner_two_stage`.

## Luồng model

1. Feature cache chứa `6 x 49 x 512` token RepViT cho mỗi sample.
2. Giữ nguyên spatial embedding của repo, cộng camera embedding, flatten thành
   294 token rồi projector `512 -> 384`.
3. Stage 1 chạy DRA + CCTS hard selection. DRA chia ngân sách chính xác 196 token
   theo diversity từng camera. CCTS dùng uniqueness nhân shifted max-cosine
   relevance với các question token hợp lệ. Biến thể `diversity_only` dành cho
   ablation.
4. Cộng visual/text modal embeddings, sau đó chạy T5 encoder blocks 0 và 1.
5. Attention probability trước dropout của block 1 tạo visual relevance. IRA
   chuẩn hóa độc lập từng sample trên 6 camera và chia đúng 49 token, có capacity
   constraint.
6. Khi train, Stage 2 dùng hard Gumbel Top-K ở forward và slot-wise Relaxed Top-K
   ở backward: `P_STE = P_hard - stopgrad(P_soft) + P_soft`,
   `V_out = P_STE @ V`. Khi eval, selection là deterministic Top-K.
7. Gather cả hai chiều của position bias và attention mask theo retained indices,
   rồi chạy T5 blocks 2, 3, final layer norm và decoder/generation như pipeline cũ.

Stage 1 giữ 196 token và Stage 2 giữ 49 token theo mặc định. Camera, spatial và
modal embeddings đều được giữ. Loss baseline vẫn chỉ là T5 cross-entropy.

## Chính sách số học và compatibility

- Chỉ Transformers `4.44.2` đã được kiểm thử; phiên bản khác fail fast. Cài bằng
  `pip install -e ".[mvpruner]"`.
- T5 attention dùng scale 1.0. Stage 2 lấy probability trước attention dropout;
  dropout của nhánh hidden-state vẫn hoạt động bình thường.
- `-inf` chỉ hợp lệ ở attention logits tại key bị mask. NaN, `+inf`, giá trị
  không hữu hạn ở key hợp lệ, relevance, allocation weight hoặc Gumbel score đều
  báo lỗi thay vì fallback âm thầm.
- Position bias batch-size 1 được broadcast; bias batch-size B được gather riêng
  từng sample. Mask đã nằm trong bias của Transformers 4.44.2 và không được cộng
  lần hai.
- Custom split encoder hỗ trợ gradient checkpointing bằng đúng helper/interface
  của T5 4.44.2.

## Schedule và checkpoint

`beta` giảm tuyến tính từ 0.2 về 0 theo optimizer step; `tau=1.0`. Schedule reset
khi chuyển `align -> finetune`, nhưng được khôi phục khi resume cùng stage.
Checkpoint metadata chứa budget, beta/tau và optimizer step. Mỗi rank ghi RNG
sidecar riêng (`<checkpoint>.rng/rank_XXXXX.pt`) gồm Python/NumPy/Torch/CUDA RNG,
sampler state và cursor; manifest xác nhận world size trước khi resume.

## Kiểm thử và benchmark

Các invariant chính nằm ở:

- `tests/test_mvpruner.py`: allocation, shifted cosine, STE/gradient, numerics.
- `tests/test_t5_split_encoder.py`: attention trước dropout, position bias,
  batch invariance, gradient checkpointing và identity `294 -> 294 -> 294`.
- `tests/test_modeling.py`: forward/generate offline qua model hoàn chỉnh.
- `tests/test_smoke_pipeline.py`: prepare, align, finetune, evaluate, benchmark
  với `fake_vision`/`fake_t5`, không mạng.

Benchmark end-to-end dùng CLI `benchmark`. Breakdown component dùng:

```powershell
python scripts/benchmark_mvpruner_components.py `
  --config configs/repvit_t5_efficient_mini_kaggle_2gpu_mvpruner.yaml `
  --checkpoint outputs/repvit_t5_efficient_mini_kaggle_2gpu_mvpruner/checkpoints/finetune_best.pt `
  --mode eval
```

Chạy lại với `--mode train` để đo overhead Relaxed Top-K và backward. JSON kết
quả tách projector, Stage 1, blocks 0-1, Stage 2 và blocks 2-3; so sánh tổng thể
với cùng lệnh benchmark trên profile router cũ.
