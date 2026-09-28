# EM-VLM4AD Research Brief

Ngay nghien cuu: 2026-09-24

Nguon doc:
- Paper local: `D:\AD\Paper\em_vlm4ad.pdf`
- Paper online doi chieu: [arXiv:2403.19838v2](https://arxiv.org/abs/2403.19838), ban v2 ngay 2024-05-09
- Upstream repo: [akshaygopalkr/EM-VLM4AD](https://github.com/akshaygopalkr/EM-VLM4AD), `main` tai commit `ebfb59f43538f173bb6a83455e4c76a9acfa91d2` ngay 2024-11-15
- Local repo hien tai: `D:\AD\Efficient_VLM_For_Autonomous_Driving`, moi co `README.md`

## 1. Tom tat paper theo huong engineering

EM-VLM4AD la mot VLM nhe cho Visual Question Answering trong autonomous driving. Bai toan nhan 6 anh multi-view cua nuScenes/DriveLM va mot cau hoi dang text, sinh cau tra loi text cho cac tac vu perception, planning, prediction, va ego-vehicle behavior.

Dong gop chinh cua paper:
- Dung backbone ngon ngu nho hon nhom AD-VLM dung LLaMA/GPT/BLIP-2: T5-Base hoac T5-Large 8-bit + LoRA.
- Dung ViT-B/32 pretrained ImageNet nhu patch embedder, nhung chi lay phan patch projection/positional embedding thay vi chay full image encoder nang.
- Gop 6 camera views bang Gated Pooling Attention (GPA), sau do concat embedding anh voi text embedding va dua vao T5.
- Bao cao hieu qua tinh toan tot hon nhieu baseline: EM-VLM4AD Base co 235M params, 9.47B FLOPs, 0.94GB; Q-Large co 769M params, 31.5B FLOPs, 0.77GB. Paper claim it nhat 10x nhe hon cac AD-VLM baseline ve memory/FLOPs.

Kien truc muc cao:

```text
DriveLM JSON
  -> question text + 6 image paths
  -> read/resize/normalize images
  -> ViT-B/32 patch projection per camera view
  -> flatten each view embedding
  -> Gated Pooling Attention across 6 views
  -> optional projection to T5 hidden size
  -> add image/text modal embeddings
  -> concat [text embeddings, image embeddings]
  -> T5 encoder-decoder
  -> generated answer text
```

Training theo paper:
- Stage 1: freeze image patch encoder va LM; train GPA + projection layer de align image embeddings voi khong gian T5.
- Stage 2: image patch encoder van freeze; fine-tune LM, GPA va projection layer tiep tuc train.
- Moi stage 6 epochs; hyperparameters chinh: learning rate `1e-4`, weight decay `0.05`, exponential scheduler, batch size `4`, GPA hidden size `128`.
- T5-Base full fine-tune tot nhat trong thiet lap cua paper. T5-Large dung 8-bit quantization + LoRA/LoftQ vi full fine-tune quantized LM gay mode collapse theo paper.

Ket qua paper can nam:
- EM-VLM4AD Base: BLEU-4 `45.36`, METEOR `34.49`, ROUGE-L `71.98`, CIDEr `3.20`.
- EM-VLM4AD Q-Large: BLEU-4 `40.11`, METEOR `34.34`, ROUGE-L `70.72`, CIDEr `3.10`.
- DriveLM-Agent baseline co BLEU/METEOR cao hon nhung ROUGE-L/CIDEr thap hon: BLEU-4 `53.09`, METEOR `36.19`, ROUGE-L `66.79`, CIDEr `2.79`.
- Diem yeu duoc paper neu ro: cau hoi lien quan du doan hanh vi ego vehicle can temporal/video context, nen single multi-view frame chua du.

## 2. Mapping paper sang upstream repo

| Paper concept | Upstream implementation | Ghi chu |
| --- | --- | --- |
| EM-VLM4AD model wrapper | `modules/multi_frame_model.py::DriveVLMT5` | Wrapper `nn.Module` quanh T5 va `MultiViewProcessor`. |
| T5 backbone | `DriveVLMT5.__init__` | `T5-Base` load `google-t5/t5-base`; option con lai load `google-t5/t5-large`. |
| T5-Large LoRA/LoftQ | `DriveVLMT5.__init__` | Dung `LoraConfig(r, alpha, dropout, target_modules=['q', 'v'])` va `LoftQConfig(loftq_bits=8)`. |
| Image patch embedder | `MultiViewProcessor.img_model = vit_b_32(weights='DEFAULT')` | Dung torchvision ViT-B/32 pretrained. |
| Freeze image encoder | `MultiViewProcessor(..., freeze=True)` | Set `requires_grad=False` cho `img_model.parameters()`. |
| ViT patch extraction | `get_img_embedding()` | Goi private API `self.img_model._process_input(img)`, them class token va positional embedding, roi bo class token. |
| 6-view fusion/GPA | `MultiViewProcessor.gpa()` | `Z`, `G`, `w` tinh attention weights theo gated pooling attention; `gpa_hidden_size` default `128`. |
| Projection layer | `self.img_projection_layer` | Chi tao khi `lm != 'T5-Base'`; T5-Base hidden size bang 768 nen khong can projection tu ViT hidden size 768. |
| Modal embeddings | `self.modal_embeddings = nn.Embedding(2, hidden_size)` | Cong embedding loai modality vao text/image truoc khi concat. |
| Multimodal concat | `MultiViewProcessor.forward()` | Lay text input embeddings tu T5, cong modal embedding, concat voi image embeddings theo sequence dimension. |
| Forward loss | `DriveVLMT5.forward()` | Goi T5 voi `inputs_embeds=merged_embedding, labels=labels`. |
| Generation | `DriveVLMT5.generate()` | Tao `attention_mask`, `decoder_input_ids`, generate voi `max_length=512`. |
| Dataset format | `modules/multi_frame_dataset.py::MultiFrameDataset` | Moi item gom `qa` dict va `img_path` dict; cau hoi format thanh `Question: ... Answer:`. |
| Image loading | `MultiFrameDataset.__getitem__` | `read_image(p).float()`, transform, `.to(device)`, stack 6 images. |
| Training loop | `train.py::custom_train()` | AdamW, ExponentialLR gamma `0.9`, loss logging, checkpoint `latest_model.pth`, stats/loss plot. |
| Eval metrics | `eval.py` | Generate predictions JSON roi dung `pycocotools.COCO` + `pycocoevalcap.COCOEvalCap`. |
| Colab reproduction | `colab/train_T5_Base.ipynb`, `colab/train_T5_Large.ipynb`, `colab/eval.ipynb` | Notebook chua hyperparameter gan voi paper va Google Drive paths. |

## 3. Paper-code gap audit

Nhung diem khop:
- Model dung dung truc T5 + ViT-B/32 patch embedding + GPA + modal embedding.
- Dataset dung 6 camera views va prompt format `Question: {Q} Answer:`.
- Metrics dung BLEU-4, METEOR, ROUGE-L, CIDEr thong qua `pycocoevalcap`.
- README yeu cau data split DriveLM/NuScenes va checkpoint folder `multi_frame_results`.

Nhung diem lech hoac can doc ky truoc khi ke thua:
- Paper mo ta 2-stage training ro rang, con `train.py` chi expose `--freeze-lm`; khong co orchestration chinh thuc kieu `stage=align` / `stage=finetune`.
- Script `train.py` co default `--epochs 15`, trong khi paper noi moi stage 6 epochs. Notebook T5-Base dat `epochs=6`, T5-Large dat `epochs=12` khi resume checkpoint, nen logic stage co ve duoc thao tac thu cong qua checkpoint/freeze flags.
- `DriveVLMT5.__init__` trong script luon ap LoRA cho `T5-Large` branch, bat ke `config.lora`; trong `train.py` van co flag `--lora`, tao cam giac option nay khong that su dieu khien model script.
- `config.freeze_lm` ton tai trong CLI/stat CSV, nhung `modules/multi_frame_model.py` khong dung flag nay de freeze T5. Notebook T5-Base cu co logic freeze LM, nhung module script hien tai khong co.
- README inference noi dung `--checkpoint-file`, nhung `eval.py` dung arg `--model-name`; day la mismatch command/docs.
- Paper noi Q-Large la 8-bit quantized T5-Large, nhung script load `T5ForConditionalGeneration.from_pretrained('google-t5/t5-large')` roi dung `LoftQConfig`; can verify thuc te co quantize weights dung nhu ky vong tren dependency version hien tai hay khong.

## 4. Technical debt va rui ro khi ke thua code

- Device coupling trong Dataset: `MultiFrameDataset` day images/token ids len global `device` ngay trong `__getitem__` va `collate_fn`. Dieu nay kho scale voi `num_workers > 0`, CPU-only test, distributed training, pinned memory, va bat loi path/image.
- Private torchvision API: `self.img_model._process_input(img)` la private method. Khi torchvision thay doi internals, code co the vo ma khong bao truoc.
- Shape assumptions hard-code: `VIT_HIDDEN_STATE = 768`, `VIT_SEQ_LENGTH = 49`, 6 camera views ngam dinh. Neu doi resolution, patch size, ViT variant, hoac so camera, GPA/projection can refactor.
- Projection asymmetry: T5-Base khong co projection layer vi hidden size 768; T5-Large co projection 768 -> 1024. Neu them backbone moi, can abstraction hidden-size ro rang.
- Path assumptions: dataset JSON chua image paths duoc read truc tiep; Colab notebook lai prepend `DriveLM`. Can chuan hoa root path/dataset manifest de chay local/cluster/Colab nhat quan.
- Checkpoint metadata yeu: save/load `state_dict` vao `latest_model.pth`, stats rieng `stats.json`, chua luu model config/version/tokenizer transform/source commit day du.
- Eval side effects va deps: `pycocoevalcap` can Java neu chay SPICE; README khuyen comment SPICE trong third-party package. Day la dau hieu can mot eval wrapper co metric selection.
- Reproducibility thieu: chua thay seed control, deterministic flags, run config snapshot, dataset hash, split provenance trong checkpoint.
- Trainer path chua chac dung: `train()` dung HuggingFace `Trainer` voi custom model/dataset nhung parameter ten `config=training_config` co kha nang sai voi API hien tai; duong nay khong nen la baseline thiet ke moi neu chua test.
- Label masking chua co: labels tu tokenizer padding duoc dua thang vao T5 loss, khong thay replace pad token thanh `-100`; can verify anh huong loss khi refactor.
- Generation config hard-code: `max_length=512` trong model generate, trong eval lai co `--max-len` chi filter output sau generate.

## 5. Adaptation checklist cho code moi

Nen giu lai:
- Y tuong kien truc: T5 encoder-decoder nhan `inputs_embeds` da concat text/image.
- GPA fusion across camera views vi nhe va dung voi contribution cua paper.
- Prompt format `Question: ... Answer:` neu muon reproduce behavior/checkpoints.
- Metric pipeline COCO caption metrics, nhung nen boc lai co config de bat/tat SPICE.
- Hai backbone profile ban dau: T5-Base full fine-tune va T5-Large LoRA/LoftQ.

Nen refactor truoc khi mo rong:
- Tach config thanh dataclass/YAML: dataset, model, training stage, optimizer, eval, checkpoint.
- Tach module thanh cac lop ro boundary: `DriveLMDataset`, `ImagePatchEmbedder`, `GatedViewPooler`, `MultimodalProjector`, `DriveVLMT5`.
- Khong move tensors len GPU trong Dataset; de training/eval loop hoac collator/device manager lam viec nay.
- Thay private `_process_input` bang wrapper kiem soat patch projection, hoac pin torchvision version va test shape ro.
- Them explicit training stage API: `stage1_align` freeze LM + image encoder; `stage2_finetune` unfreeze dung phan can train.
- Chuan hoa checkpoint: luu `model_state_dict`, `optimizer_state_dict`, scheduler, epoch, stage, config, tokenizer name, source commit, metric summary.
- Chuan hoa dataset root/image path resolver, de cung JSON chay duoc local, Colab, Linux/Windows.
- Them evaluation runner co `metrics: [bleu, meteor, rouge_l, cider]`, optional skip SPICE khong can sua package ben ngoai.

Can test truoc khi them feature:
- Dataset parses sample JSON item dung contract: output question string, image path order, 6 image tensors, answer text.
- Collator tao `input_ids`, `labels`, `images` voi shapes mong doi va device do training loop quyet dinh.
- GPA weights sum to 1 theo view dimension va output shape la `(batch, 49, hidden)`.
- T5-Base path concat text/image embeddings dung hidden size 768 khong can projection.
- T5-Large path projection 768 -> `d_model` va LoRA target modules dung.
- Freeze policy theo stage: stage 1 chi GPA/projection/modal embeddings trainable; stage 2 LM trainable theo mode full/LoRA, image encoder van frozen.
- Checkpoint round-trip load duoc config va state dict.
- Eval prediction writer loai duplicate theo `image_id`, tao JSON dung COCO expected shape.

Can chuan hoa interface toi thieu:
- `train.py --config configs/em_vlm4ad_t5_base.yaml --stage align|finetune --resume PATH`
- `eval.py --config configs/em_vlm4ad_t5_base.yaml --checkpoint PATH --metrics bleu meteor rouge_l cider`
- Dataset config gom `annotation_file`, `image_root`, `image_id_file`, `split`, `num_views`, `view_order`.
- Model config gom `lm_name`, `image_encoder`, `gpa_hidden_size`, `lora`, `quantization`, `max_generation_length`.
- Output run folder gom `config.yaml`, `checkpoint.pt`, `metrics.json`, `predictions.json`, `loss.csv`, `run_metadata.json`.

## 6. Cau hoi thiet ke can chot sau khi co yeu cau moi

- Muc tieu chinh la reproduce paper, cai tien toc do, them modality/video, hay bien thanh framework nghien cuu de thay backbone?
- Co can tuong thich checkpoint upstream `latest_model.pth` khong, hay duoc phep doi checkpoint schema?
- Muon uu tien T5-Base, T5-Large LoRA, hay them backbone moi nhu Flan-T5/Small-VLM?
- Dataset se dung DriveLM/NuScenes goc, sample subset, hay dataset noi bo co schema khac?
- Can chay tren Windows local, Colab, Linux server, hay tat ca?
- Tieu chi thanh cong cua code moi la training reproduce metrics, inference demo, benchmark compute, hay clean architecture de research tiep?

## 7. Validation notes

- Da doi chieu paper local voi arXiv v2 metadata va noi dung online.
- Da doi chieu repo upstream bang GitHub API, raw source files, README va notebook summaries.
- Khong chay training/eval nang, khong tai dataset/checkpoints.
- Nhung ket luan ve code script dua tren upstream `main` tai commit `ebfb59f43538f173bb6a83455e4c76a9acfa91d2`.
