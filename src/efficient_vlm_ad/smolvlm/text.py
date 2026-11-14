"""Six-camera native chat formatting for raw-feature cache batches."""
import torch
from transformers.models.idefics3.processing_idefics3 import get_image_prompt_string

from .model import CAMERA_ORDER
from .config import validate_prompt_camera_order


def chat_messages(question, answer=None, *, prompt_camera_order=CAMERA_ORDER):
    """Retain native image markers while spelling out the requested camera order."""
    content=[]
    for camera in validate_prompt_camera_order(prompt_camera_order):
        content.extend([{'type':'text','text':camera+':'}, {'type':'image'},
                        {'type':'text','text':'\n'}])
    content.append({'type':'text','text':str(question)})
    messages=[{'role':'user','content':content}]
    if answer is not None:
        messages.append({'role':'assistant','content':[{'type':'text','text':str(answer)}]})
    return messages


class CachedBatchCollator:
    def __init__(self, processor, *, image_seq_len, max_length, generation=False,
                 max_new_tokens=512, prompt_camera_order=CAMERA_ORDER):
        self.prompt_camera_order=validate_prompt_camera_order(prompt_camera_order)
        self.view_indices=[CAMERA_ORDER.index(camera) for camera in self.prompt_camera_order]
        self.processor=processor
        self.image_seq_len=image_seq_len
        self.max_length=max_length
        self.generation=generation
        self.max_new_tokens=max_new_tokens
        if image_seq_len <= 0 or max_length <= 0 or max_new_tokens < 0:
            raise ValueError('token budgets must be positive')
        if not processor.tokenizer.is_fast:
            raise ValueError('native supervision requires a fast tokenizer with offsets')

    def _expand(self, text):
        token=str(self.processor.image_token)
        if text.count(token) != 6:
            raise ValueError('chat must contain exactly six image placeholders')
        expanded=get_image_prompt_string(0,0,self.image_seq_len,
            str(self.processor.fake_image_token),token,self.processor.global_image_tag)
        return text.replace(token,expanded)

    def __call__(self, items):
        if not items:
            raise ValueError('cannot collate an empty batch')
        tok=self.processor.tokenizer
        image_token=str(self.processor.image_token)
        sequences=[]
        label_sequences=[]
        for item in items:
            question,answer=str(item['question']),str(item['answer'])
            if image_token in question or image_token in answer:
                raise ValueError('question and answer cannot contain image placeholders')
            prefix=self.processor.apply_chat_template(chat_messages(question,prompt_camera_order=self.prompt_camera_order),tokenize=False,
                                                       add_generation_prompt=True)
            prefix=self._expand(prefix)
            if self.generation:
                encoded=tok(prefix,add_special_tokens=True,return_offsets_mapping=True)
                labels=None
            else:
                full=self.processor.apply_chat_template(chat_messages(question,answer,prompt_camera_order=self.prompt_camera_order),tokenize=False,
                                                        add_generation_prompt=False)
                full=self._expand(full)
                if not full.startswith(prefix):
                    raise ValueError('native full chat must begin with generation prefix')
                eos=tok.eos_token
                if not eos:
                    raise ValueError('native tokenizer must define an EOS token')
                eos_start=full.rfind(eos)
                if eos_start < len(prefix) or full[eos_start+len(eos):].strip():
                    raise ValueError('native chat must end assistant continuation with EOS')
                end=eos_start+len(eos)
                encoded=tok(full,add_special_tokens=True,return_offsets_mapping=True)
                # Offsets identify continuation characters even when BPE merges across
                # the prefix boundary. Token-count subtraction loses that information.
                labels=[token if finish>len(prefix) and start<end and finish>start else -100
                        for token,(start,finish) in zip(encoded['input_ids'],encoded['offset_mapping'])]
                label_sequences.append(labels)
            ids=encoded['input_ids']
            extra=self.max_new_tokens if self.generation else 0
            if len(ids)+extra > self.max_length:
                raise ValueError('expanded chat exceeds context budget; truncation is disabled')
            if ids.count(tok.convert_tokens_to_ids(image_token)) != 6*self.image_seq_len:
                raise ValueError('native chat has incorrect expanded image token count')
            sequences.append(ids)
        width=max(map(len,sequences))
        if tok.pad_token_id is None:
            raise ValueError('native tokenizer must define padding')
        input_ids=torch.full((len(items),width),tok.pad_token_id,dtype=torch.long)
        attention_mask=torch.zeros_like(input_ids)
        labels=torch.full_like(input_ids,-100) if not self.generation else None
        for row,ids in enumerate(sequences):
            start=width-len(ids) if self.generation else 0
            stop=start+len(ids)
            input_ids[row,start:stop]=torch.tensor(ids,dtype=torch.long)
            attention_mask[row,start:stop]=1
            if labels is not None:
                labels[row,start:stop]=torch.tensor(label_sequences[row],dtype=torch.long)
        # Stack individual view references directly into the final allocation.
        # Reordering a stacked batch would make a second full feature copy.
        raw_features=[item['visual_features'] for item in items]
        if any(feature.ndim != 3 or feature.shape[0] != 6 for feature in raw_features):
            raise ValueError('visual_features must stack to [B,6,N,D]')
        features=torch.stack([feature[index] for feature in raw_features
                              for index in self.view_indices]).reshape(len(items),6,*raw_features[0].shape[1:])
        if features.ndim!=4 or features.shape[1]!=6:
            raise ValueError('visual_features must stack to [B,6,N,D]')
        result=dict(input_ids=input_ids,attention_mask=attention_mask,visual_features=features,
                    answers=[item['answer'] for item in items],
                    eval_ids=[item['eval_id'] for item in items],
                    sample_ids=[item['sample_id'] for item in items])
        if labels is not None:
            result['labels']=labels
        return result
