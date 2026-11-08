"""Genuine, self-contained Idefics3 fixtures; no pretrained network downloads."""
from pathlib import Path
import json

import torch
import yaml
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from transformers import (
    Idefics3Config, Idefics3ForConditionalGeneration, Idefics3ImageProcessor,
    Idefics3Processor, PreTrainedTokenizerFast,
)

from efficient_vlm_ad.config import CAMERA_ORDER

CHAT_TEMPLATE = "<|im_start|>{% for message in messages %}{{message['role'] | capitalize}}{% if message['content'][0]['type'] == 'image' %}{{':'}}{% else %}{{': '}}{% endif %}{% for line in message['content'] %}{% if line['type'] == 'text' %}{{line['text']}}{% elif line['type'] == 'image' %}{{ '<image>' }}{% endif %}{% endfor %}<end_of_utterance>\n{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"


def tiny_components(*, dropout=0.0):
    special = ['<|im_start|>', '<|im_end|>', '<end_of_utterance>', '<image>',
               '<fake_token_around_image>', '<global-img>', '<unk>']
    backend = Tokenizer(models.BPE(unk_token='<unk>'))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        [' '.join(CAMERA_ORDER), 'User: Assistant: What is visible? A road. A car.',
         'What changed? A person. <c1,CAM_FRONT,1>'] * 3,
        trainers.BpeTrainer(vocab_size=360, special_tokens=special,
                            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()),
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, bos_token=special[0], pad_token=special[1],
        eos_token=special[2], unk_token=special[-1], chat_template=CHAT_TEMPLATE,
        model_max_length=512,
    )
    processor = Idefics3Processor(
        Idefics3ImageProcessor(do_image_splitting=False, size={'longest_edge': 16},
                              max_image_size={'longest_edge': 16}),
        tokenizer=tokenizer, image_seq_len=4, chat_template=CHAT_TEMPLATE,
    )
    config = Idefics3Config(
        vision_config=dict(hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                           num_attention_heads=2, image_size=16, patch_size=4),
        text_config=dict(model_type='llama', hidden_size=16, intermediate_size=32,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         vocab_size=len(tokenizer), pad_token_id=tokenizer.pad_token_id,
                         bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
                         attention_dropout=dropout, max_position_embeddings=512),
        scale_factor=2, image_token_id=tokenizer.convert_tokens_to_ids('<image>'),
        pad_token_id=tokenizer.pad_token_id,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = Idefics3ForConditionalGeneration._from_config(
            config, attn_implementation={'vision_config': 'eager', 'text_config': 'sdpa'})
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.bos_token_id = tokenizer.bos_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    return model, processor


def create_local_run(root, *, rows=3, dropout=0.0):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    native, processor = tiny_components(dropout=dropout)
    model_dir = root / 'native'
    native.save_pretrained(model_dir)
    processor.save_pretrained(model_dir)
    data = root / 'dataset'
    cameras = []
    for scene in range(2):
        mapping = {}
        for view, name in enumerate(CAMERA_ORDER):
            path = data / 'images' / f'{scene}_{name}.png'
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (20, 12), (20 + scene * 70, 30 + view * 20, 80)).save(path)
            mapping[name] = path.relative_to(data).as_posix()
        cameras.append(mapping)
    for split in ('train', 'val', 'test'):
        path = data / 'data' / 'multi_frame' / f'multi_frame_{split}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        records = [[{'Q': 'What is visible?' if i % 2 == 0 else 'What changed?',
                     'A': 'A road.' if i % 2 == 0 else 'A person.'}, cameras[i % 2]]
                   for i in range(rows)]
        path.write_text(json.dumps(records), encoding='utf-8')
    output = root / 'output'
    raw = dict(
        project={'output_dir': output.as_posix()},
        data={'hf_repo_id': 'local/fake', 'local_dir': data.as_posix(),
              'view_order': CAMERA_ORDER, 'expected_counts': dict.fromkeys(('train', 'val', 'test'), rows)},
        cache={'dir': (output / 'cache').as_posix()},
        runtime={'device': 'cpu', 'precision': 'fp32'},
        model={'model_id': model_dir.as_posix(), 'revision': None},
        training={'batch_size': 1, 'gradient_accumulation_steps': 2, 'align_epochs': 0,
                  'finetune_epochs': 1, 'gradient_checkpointing': True, 'num_workers': 0,
                  'seed': 42, 'max_grad_norm': 1., 'learning_rate': .001},
        generation={'max_new_tokens': 3, 'num_beams': 1},
        evaluation={'eval_batch_size': 2, 'benchmark_max_samples': 2},
    )
    config = root / 'config.yaml'
    config.write_text(yaml.safe_dump(raw), encoding='utf-8')
    return config
