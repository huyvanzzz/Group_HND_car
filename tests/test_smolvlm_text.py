import numpy as np
from PIL import Image
import pytest
import torch
from smolvlm_helpers import tiny_components


def item(answer='car', question='What is here?', sample_id='a'):
    return dict(question=question,answer=answer,visual_features=torch.randn(6,16,8),sample_id=sample_id,eval_id=7)


@pytest.mark.parametrize('answer',['car','','a café!'])
def test_native_processor_parity_and_offset_labels(answer):
    from efficient_vlm_ad.smolvlm.text import CachedBatchCollator, chat_messages
    _,processor=tiny_components()
    row=item(answer=answer)
    collator=CachedBatchCollator(processor,image_seq_len=4,max_length=1024)
    batch=collator([row])
    messages=chat_messages(row['question'],row['answer'])
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
    pixels=[[Image.fromarray(np.full((16,16,3),100,dtype=np.uint8)) for _ in range(6)]]
    native=processor(images=pixels,text=[text],return_tensors='pt',padding=True)
    assert torch.equal(batch['input_ids'],native.input_ids)
    assert batch['labels'].dtype==torch.long
    labels=batch['labels'][0]; ids=batch['input_ids'][0]
    target=processor.tokenizer.decode(ids[labels!=-100],skip_special_tokens=False)
    assert target==' '+answer+processor.tokenizer.eos_token
    assert batch['visual_features'].shape==(1,6,16,8)
    assert batch['answers']==[answer] and batch['sample_ids']==['a'] and batch['eval_ids']==[7]
    assert (labels[ids==processor.tokenizer.convert_tokens_to_ids('<image>')]==-100).all()


def test_padding_generation_and_context_rejection():
    from efficient_vlm_ad.smolvlm.text import CachedBatchCollator
    _,processor=tiny_components()
    processor.tokenizer.padding_side='right'
    rows=[item(),item(question='A longer question about a truck',sample_id='b')]
    generation=CachedBatchCollator(processor,image_seq_len=4,max_length=1024,generation=True,max_new_tokens=20)
    batch=generation(rows)
    assert 'labels' not in batch and processor.tokenizer.padding_side=='right'
    assert batch['attention_mask'][0,0]==0 and batch['attention_mask'][:,-1].all()
    assert processor.tokenizer.decode(batch['input_ids'][0],skip_special_tokens=False).endswith('Assistant:')
    training=CachedBatchCollator(processor,image_seq_len=4,max_length=1024)
    batch=training(rows)
    assert batch['attention_mask'][0,0]==1 and batch['attention_mask'][0,-1]==0
    assert (batch['labels'][batch['attention_mask']==0]==-100).all()
    with pytest.raises(ValueError,match='context'):
        CachedBatchCollator(processor,image_seq_len=4,max_length=12)(rows)
    with pytest.raises(ValueError,match='context'):
        CachedBatchCollator(processor,image_seq_len=4,max_length=1024,generation=True,max_new_tokens=1024)(rows)


def test_reject_placeholder_in_answer_or_question():
    from efficient_vlm_ad.smolvlm.text import CachedBatchCollator
    _,processor=tiny_components()
    collator=CachedBatchCollator(processor,image_seq_len=4,max_length=1024)
    for row in [item(answer='<image>'),item(question='<image>')]:
        with pytest.raises(ValueError,match='image'):
            collator([row])
