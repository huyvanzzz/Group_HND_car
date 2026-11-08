import copy
import torch
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from transformers import (Idefics3Config, Idefics3ForConditionalGeneration,
                          Idefics3Processor, Idefics3ImageProcessor, PreTrainedTokenizerFast)


CHAT = "{% for message in messages %}{{message['role']|capitalize}}:{% for item in message['content'] %}{% if item['type']=='image' %}{{'<image>'}}{% else %}{{' ' + item['text']}}{% endif %}{% endfor %}{{'<end_of_utterance>\\n'}}{% endfor %}{% if add_generation_prompt %}{{'Assistant:'}}{% endif %}"


def tiny_components():
    special = ['<pad>', '<bos>', '<end_of_utterance>', '<image>', '<fake_token_around_image>', '<global-img>']
    tok = Tokenizer(models.BPE(unk_token='<pad>'))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(['Front Front-Left Front-Right Back Back-Left Back-Right User: Assistant: What is here? car truck answer hello\n'], trainers.BpeTrainer(vocab_size=300, special_tokens=special, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token='<pad>', bos_token='<bos>', eos_token='<end_of_utterance>', additional_special_tokens=special[3:])
    processor = Idefics3Processor(Idefics3ImageProcessor(do_image_splitting=False, size={'longest_edge':8}, max_image_size={'longest_edge':8}), tokenizer, image_seq_len=4, chat_template=CHAT)
    cfg = Idefics3Config(image_token_id=tokenizer.convert_tokens_to_ids('<image>'), pad_token_id=0, scale_factor=2,
        vision_config=dict(hidden_size=16, intermediate_size=24, num_hidden_layers=1, num_attention_heads=2, image_size=8, patch_size=2),
        text_config=dict(model_type='llama', vocab_size=len(tokenizer), hidden_size=16, intermediate_size=24, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=1024, pad_token_id=0, bos_token_id=1, eos_token_id=2))
    cfg._attn_implementation = {'vision_config':'eager','text_config':'sdpa'}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        native = Idefics3ForConditionalGeneration(cfg)
    native.generation_config.pad_token_id=0
    native.generation_config.bos_token_id=1
    native.generation_config.eos_token_id=2
    return native, processor


def test_cached_native_logits_loss_and_freeze():
    from efficient_vlm_ad.smolvlm.model import CachedSmolVLM
    native, processor = tiny_components()
    model = CachedSmolVLM(native, processor, gradient_checkpointing=False).eval()
    pixels = torch.randn(2, 6, 3, 8, 8)
    mask = torch.ones(2, 6, 8, 8, dtype=torch.bool)
    mask[:, :, -2:, :] = False
    raw = model.extract_features(pixels, mask)
    assert raw.shape == (2,6,16,16) and not raw.requires_grad
    ids = torch.tensor([[3]*24+[20,21,2], [3]*24+[22,23,2]])
    attention = torch.ones_like(ids)
    labels=ids.clone(); labels[:,:24]=-100
    direct=native(input_ids=ids, attention_mask=attention, pixel_values=pixels, pixel_attention_mask=mask, labels=labels, use_cache=False)
    cached=model(input_ids=ids,attention_mask=attention,visual_features=raw,labels=labels)
    torch.testing.assert_close(cached.logits,direct.logits)
    torch.testing.assert_close(cached.loss,direct.loss)
    quant=model(input_ids=ids,attention_mask=attention,visual_features=raw.half(),labels=labels)
    native_patch_mask=torch.ones(12,4,4,dtype=torch.bool)
    native_patch_mask[:,-1,:]=False
    independent_raw=native.model.vision_model(pixel_values=pixels.flatten(0,1),
        patch_attention_mask=native_patch_mask).last_hidden_state.detach().half().float()
    quant_reference=native(input_ids=ids,attention_mask=attention,
        image_hidden_states=native.model.connector(independent_raw),labels=labels,use_cache=False)
    torch.testing.assert_close(quant.logits,quant_reference.logits)
    torch.testing.assert_close(quant.loss,quant_reference.loss)
    model.train()
    assert not native.model.vision_model.training
    cached=model(input_ids=ids,attention_mask=attention,visual_features=raw,labels=labels)
    cached.loss.backward()
    assert all(p.grad is None and not p.requires_grad for p in native.model.vision_model.parameters())
    assert any(p.grad is not None for p in native.model.connector.parameters())
    assert any(p.grad is not None for p in native.model.text_model.parameters())
    assert len(list(model.parameters())) == len({id(p) for p in model.parameters()})


@pytest.mark.parametrize('beams',[1,3])
def test_generation_batch_matches_single_and_continuation(beams):
    from efficient_vlm_ad.smolvlm.model import CachedSmolVLM
    native, processor=tiny_components()
    model=CachedSmolVLM(native,processor,gradient_checkpointing=False).eval()
    raw=model.extract_features(torch.randn(2,6,3,8,8))
    ids=torch.tensor([[0,0]+[3]*24+[20], [3]*24+[21,22,23]])
    mask=(ids!=0).long()
    counts={'connector':0,'merger':0}
    def count_connector(module,args,result):
        counts['connector']+=1
    handle=native.model.connector.register_forward_hook(count_connector)
    merger=native.model.inputs_merger
    def counting_merger(*args,**kwargs):
        counts['merger']+=1
        return merger(*args,**kwargs)
    native.model.inputs_merger=counting_merger
    result=model.generate_from_features(input_ids=ids,attention_mask=mask,visual_features=raw,max_new_tokens=3,min_new_tokens=3,num_beams=beams)
    handle.remove()
    native.model.inputs_merger=merger
    assert counts==dict(connector=1,merger=1)
    assert result.shape == (2,3)
    direct=native.generate(input_ids=ids,attention_mask=mask,pixel_values=None,
        inputs_embeds=native.model.inputs_merger(input_ids=ids,
            inputs_embeds=native.get_input_embeddings()(ids),
            image_hidden_states=native.model.connector(raw.flatten(0,1))),
        max_new_tokens=3,min_new_tokens=3,num_beams=beams,use_cache=True)
    assert torch.equal(result,direct[:,ids.shape[1]:])
    for row in range(2):
        trimmed=ids[row][mask[row].bool()].unsqueeze(0)
        single=model.generate_from_features(input_ids=trimmed,attention_mask=torch.ones_like(trimmed),visual_features=raw[row:row+1],max_new_tokens=3,min_new_tokens=3,num_beams=beams)
        assert torch.equal(result[row],single[0])


def test_signature_policy_and_context():
    from efficient_vlm_ad.smolvlm.model import CachedSmolVLM, processor_policy
    native,processor=tiny_components()
    model=CachedSmolVLM(native,processor,gradient_checkpointing=True)
    signature=model.architecture_signature
    other=CachedSmolVLM(copy.deepcopy(native),processor,gradient_checkpointing=False)
    assert signature==other.architecture_signature
    assert model.geometry==dict(num_views=6,vision_tokens=16,vision_dim=16,visual_tokens_per_view=4,text_dim=16)
    assert model.visual_token_len==24
    assert processor_policy(processor)['image_processor']['do_image_splitting'] is False
    ids=torch.full((1,1024),3)
    with pytest.raises(ValueError,match='context'):
        model.generate_from_features(input_ids=ids,attention_mask=torch.ones_like(ids),visual_features=torch.randn(1,6,16,16),max_new_tokens=1)


def test_factory_round_trip_and_runtime_gc(tmp_path):
    from types import SimpleNamespace
    from smolvlm_helpers import tiny_components as shared_components
    from efficient_vlm_ad.smolvlm.model import build_model_and_processor
    native,processor=shared_components()
    native.save_pretrained(tmp_path)
    processor.save_pretrained(tmp_path)
    cfg=SimpleNamespace(model=SimpleNamespace(model_id=str(tmp_path),revision=None),
                        training=SimpleNamespace(gradient_checkpointing=False))
    model,loaded=build_model_and_processor(cfg)
    assert not model.native.model.text_model.is_gradient_checkpointing
    assert model.geometry['vision_dim']==8
    for key,value in native.state_dict().items():
        torch.testing.assert_close(value,model.native.state_dict()[key])
    assert loaded.image_processor.size=={'longest_edge':16}
    assert loaded.image_processor.max_image_size=={'longest_edge':16}
    assert loaded.image_processor.do_image_splitting is False
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    optimized=[p for group in optimizer.param_groups for p in group['params']]
    assert len(optimized)==len({id(p) for p in optimized})
    assert {id(p) for p in optimized}=={id(p) for p in model.native.parameters() if p.requires_grad}
    assert not {id(p) for p in optimized} & {id(p) for p in model.native.model.vision_model.parameters()}


def test_metadata_only_processor_geometry_matches_factory_without_weights(tmp_path):
    from types import SimpleNamespace
    from smolvlm_helpers import tiny_components as shared_components
    from efficient_vlm_ad.smolvlm.model import (build_model_and_processor,
        load_processor_and_geometry,processor_policy)
    native,processor=shared_components()
    native.save_pretrained(tmp_path)
    processor.save_pretrained(tmp_path)
    cfg=SimpleNamespace(model=SimpleNamespace(model_id=str(tmp_path),revision=None),
                        training=SimpleNamespace(gradient_checkpointing=False))
    model,loaded=build_model_and_processor(cfg)
    for weights in tmp_path.glob('*.safetensors'):
        weights.unlink()
    metadata,geometry=load_processor_and_geometry(cfg)
    assert geometry==model.geometry
    assert processor_policy(metadata)==processor_policy(loaded)


def test_signature_ignores_tokenizer_batch_padding_state():
    from efficient_vlm_ad.smolvlm.model import CachedSmolVLM
    native,processor=tiny_components()
    model=CachedSmolVLM(native,processor)
    before=model.architecture_signature
    processor.tokenizer(['long answer','car'],padding=True,return_tensors='pt')
    assert model.architecture_signature==before


def test_gradient_checkpointing_matches_gradients_and_native_shift():
    from efficient_vlm_ad.smolvlm.model import CachedSmolVLM
    native,processor=tiny_components()
    model=CachedSmolVLM(native,processor,gradient_checkpointing=True).train()
    other=CachedSmolVLM(copy.deepcopy(native),processor,gradient_checkpointing=False).train()
    raw=model.extract_features(torch.randn(1,6,3,8,8))
    ids=torch.tensor([[3]*24+[20,21,2]])
    labels=ids.clone(); labels[:,:24]=-100
    args=dict(input_ids=ids,attention_mask=torch.ones_like(ids),visual_features=raw,labels=labels)
    a=model(**args); b=other(**args)
    expected=torch.nn.functional.cross_entropy(a.logits[:,:-1].reshape(-1,a.logits.shape[-1]),labels[:,1:].reshape(-1))
    torch.testing.assert_close(a.loss,expected)
    a.loss.backward(); b.loss.backward()
    for (name,p),(_,q) in zip(model.named_parameters(),other.named_parameters()):
        if p.requires_grad:
            assert p.grad is not None, name
            torch.testing.assert_close(p.grad,q.grad)


def test_generation_config_default_context_budget():
    from efficient_vlm_ad.smolvlm.model import CachedSmolVLM
    native,processor=tiny_components()
    model=CachedSmolVLM(native,processor).eval()
    native.generation_config.max_new_tokens=1024
    ids=torch.tensor([[3]*24+[20]])
    with pytest.raises(ValueError,match='context'):
        model.generate_from_features(input_ids=ids,attention_mask=torch.ones_like(ids),visual_features=torch.randn(1,6,16,16))


def test_factory_checks_attention_modules_not_just_config(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from transformers.models.llama.modeling_llama import LlamaAttention
    from efficient_vlm_ad.smolvlm.model import build_model_and_processor
    native,processor=tiny_components()
    processor.save_pretrained(tmp_path)
    native.model.text_model.layers[0].self_attn=LlamaAttention(native.config.text_config,layer_idx=0)
    monkeypatch.setattr(Idefics3ForConditionalGeneration,'from_pretrained',
        lambda *args,**kwargs:(native,{}))
    cfg=SimpleNamespace(model=SimpleNamespace(model_id=str(tmp_path),revision=None))
    with pytest.raises(ValueError,match='attention'):
        build_model_and_processor(cfg)
