from dataclasses import replace
import numpy as np
from PIL import Image
import pytest
import torch
from smolvlm_helpers import tiny_components, create_local_run
from efficient_vlm_ad.config import CAMERA_ORDER
from efficient_vlm_ad.smolvlm.config import load_config
from efficient_vlm_ad.smolvlm.text import CachedBatchCollator, chat_messages
from efficient_vlm_ad.smolvlm.model import CachedSmolVLM, processor_policy

REVERSED = CAMERA_ORDER[::-1]

def test_config_precision_and_prompt_order(tmp_path):
    cfg=load_config(create_local_run(tmp_path))
    assert cfg.model.prompt_camera_order == CAMERA_ORDER
    assert replace(cfg, runtime=replace(cfg.runtime,precision='fp16')).runtime.precision=='fp16'
    assert replace(cfg,model=replace(cfg.model,prompt_camera_order=REVERSED)).data.view_order==CAMERA_ORDER
    for order in [CAMERA_ORDER[:-1],CAMERA_ORDER[:-1]+['Front'], ['bad']*6]:
        with pytest.raises(ValueError,match='prompt_camera_order'):
            replace(cfg.model,prompt_camera_order=order)
    with pytest.raises(ValueError,match='precision'):
        replace(cfg,runtime=replace(cfg.runtime,precision='bf16'))

def test_reverse_native_processor_parity_features_and_gradients():
    native,processor=tiny_components()
    model=CachedSmolVLM(native,processor,gradient_checkpointing=False,prompt_camera_order=REVERSED).eval()
    canonical=CachedSmolVLM(native,processor,gradient_checkpointing=False)
    assert 'prompt_camera_order' not in canonical.architecture_signature
    assert model.architecture_signature==dict(canonical.architecture_signature,prompt_camera_order=REVERSED)
    assert model.architecture_signature['processor']==processor_policy(processor)
    images=[Image.fromarray(np.full((16,16,3),20+i*30,dtype=np.uint8)) for i in range(6)]
    pixels=processor.image_processor(images=[images],return_tensors='pt')
    raw=model.extract_features(pixels.pixel_values)[0]
    row=dict(question='What is visible?',answer='A road.',visual_features=raw,sample_id='a',eval_id=1)
    collate=CachedBatchCollator(processor,image_seq_len=4,max_length=512,prompt_camera_order=REVERSED)
    batch=collate([row])
    torch.testing.assert_close(batch['visual_features'][0],raw.flip(0),rtol=0,atol=0)
    text=processor.apply_chat_template(chat_messages(row['question'],row['answer'],prompt_camera_order=REVERSED),tokenize=False)
    assert [text.index(name+':') for name in REVERSED]==sorted(text.index(name+':') for name in REVERSED)
    direct=processor(images=[images[::-1]],text=[text],return_tensors='pt')
    assert torch.equal(batch['input_ids'],direct.input_ids)
    expected=native(**{k:v for k,v in direct.items() if k!='token_type_ids'},labels=batch['labels'],use_cache=False)
    result=model(**{k:batch[k] for k in ['input_ids','attention_mask','visual_features','labels']})
    torch.testing.assert_close(result.logits,expected.logits)
    result.loss.backward()
    assert all(p.dtype==torch.float32 for p in model.parameters())
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in native.model.connector.parameters())

def test_dataset_dtype_keeps_same_valid_cache(tmp_path):
    from efficient_vlm_ad.hf_data import prepare_data
    from efficient_vlm_ad.smolvlm.cache import prepare_features,validate_cache,FEATURE_FILE
    from efficient_vlm_ad.smolvlm.data import CachedVLMDataset
    from efficient_vlm_ad.smolvlm.model import build_model_and_processor
    from pathlib import Path
    cfg=load_config(create_local_run(tmp_path))
    prepare_data(cfg,disable_progress=True)
    model,processor=build_model_and_processor(cfg)
    manifest=prepare_features(cfg,model,processor,disable_progress=True)
    stamp=(Path(cfg.cache.dir)/FEATURE_FILE).stat().st_mtime_ns
    other=replace(cfg,model=replace(cfg.model,prompt_camera_order=REVERSED),runtime=replace(cfg.runtime,precision='fp16'))
    assert validate_cache(other,processor,model.geometry)==manifest
    a=CachedVLMDataset(cfg,'train',processor=processor,geometry=model.geometry)
    b=CachedVLMDataset(other,'train',processor=processor,geometry=model.geometry,output_dtype=np.float16)
    assert a._features is None and b._features is None
    assert b[0]['visual_features'].dtype==torch.float16
    torch.testing.assert_close(a[0]['visual_features'],b[0]['visual_features'].float(),rtol=0,atol=0)
    b[0]['visual_features'].fill_(999)
    assert not (b[0]['visual_features']==999).any()
    assert (Path(cfg.cache.dir)/FEATURE_FILE).stat().st_mtime_ns==stamp

@pytest.mark.skipif(not torch.cuda.is_available(),reason='requires CUDA autocast')
def test_cuda_project_keeps_half_input_and_fp32_master_weights():
    native,processor=tiny_components()
    model=CachedSmolVLM(native,processor,gradient_checkpointing=False).cuda()
    seen=[]
    handle=native.model.connector.register_forward_pre_hook(lambda module,args:seen.append(args[0].dtype))
    with torch.autocast('cuda',dtype=torch.float16):
        result=model._project(torch.randn(1,6,16,8,device='cuda',dtype=torch.float16),1)
        result.float().sum().backward()
    handle.remove()
    assert seen==[torch.float16]
    assert all(p.dtype==torch.float32 for p in model.parameters())
    assert any(p.grad is not None for p in native.model.connector.parameters())

