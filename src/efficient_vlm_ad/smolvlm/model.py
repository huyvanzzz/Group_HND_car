"""Native SmolVLM with frozen, cacheable vision encoder outputs."""
from pathlib import Path
import json

import torch
from torch import nn
from transformers import AutoConfig, AutoProcessor, Idefics3ForConditionalGeneration

MODEL_ID = 'HuggingFaceTB/SmolVLM-256M-Instruct'
MODEL_REVISION = '7e3e67edbbed1bf9888184d9df282b700a323964'
CAMERA_ORDER = ['Front', 'Front-Left', 'Front-Right', 'Back', 'Back-Left', 'Back-Right']


def _canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _geometry(config):
    vision=config.vision_config
    side=vision.image_size // vision.patch_size
    if vision.image_size % vision.patch_size or side % config.scale_factor:
        raise ValueError('vision geometry must divide evenly into connector patches')
    return dict(num_views=6,vision_tokens=side**2,vision_dim=vision.hidden_size,
                visual_tokens_per_view=side**2 // config.scale_factor**2,
                text_dim=config.text_config.hidden_size)


def _configure_processor(processor,config):
    image=processor.image_processor
    image.do_image_splitting=False
    image.size={'longest_edge':config.vision_config.image_size}
    image.max_image_size={'longest_edge':config.vision_config.image_size}
    processor.image_seq_len=_geometry(config)['visual_tokens_per_view']


def _model_identity(cfg):
    model_id=str(cfg.model.model_id)
    revision=cfg.model.revision
    if not Path(model_id).is_dir() and (model_id != MODEL_ID or revision != MODEL_REVISION):
        raise ValueError('production SmolVLM model and revision must be pinned')
    return model_id,revision


def load_processor_and_geometry(cfg):
    """Read native config and processor metadata without allocating model weights."""
    model_id,revision=_model_identity(cfg)
    config=AutoConfig.from_pretrained(model_id,revision=revision)
    if config.model_type != 'idefics3':
        raise ValueError('native SmolVLM requires Idefics3 configuration')
    processor=AutoProcessor.from_pretrained(model_id,revision=revision)
    _configure_processor(processor,config)
    return processor,_geometry(config)


def processor_policy(processor):
    """Content identity of preprocessing, vocabulary, and native chat policy."""
    tok = processor.tokenizer
    image = processor.image_processor.to_dict()
    image.pop('_processor_class', None)
    backend=json.loads(tok.backend_tokenizer.to_str())
    backend.pop('padding',None)
    backend.pop('truncation',None)
    return _canonical(dict(image_processor=image, chat_template=processor.chat_template,
        tokenizer=dict(vocab=tok.get_vocab(), backend=backend,
                       special_tokens=tok.special_tokens_map, model_max_length=tok.model_max_length),
        image_seq_len=processor.image_seq_len, camera_order=CAMERA_ORDER,
        fake_image_token=str(processor.fake_image_token), image_token=str(processor.image_token),
        global_image_tag=processor.global_image_tag, image_splitting=False,
        policy='native_six_camera_chat_v1'))


class CachedSmolVLM(nn.Module):
    def __init__(self, native, processor, *, gradient_checkpointing=True,
                 model_id=MODEL_ID, model_revision=None):
        super().__init__()
        self.native = native
        self.processor = processor
        cfg = native.config
        self.geometry = _geometry(cfg)
        self.visual_token_len = 6 * self.geometry['visual_tokens_per_view']
        self.model_id, self.model_revision = str(model_id), model_revision
        _configure_processor(processor,cfg)
        self.native.model.vision_model.requires_grad_(False).eval()
        text = self.native.model.text_model
        if gradient_checkpointing:
            text.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        else:
            text.gradient_checkpointing_disable()

    @property
    def architecture_signature(self):
        # Configs contain a runtime attention backend but no checkpoint identity paths.
        config = self.native.config.to_dict()
        def clean(obj):
            if isinstance(obj, dict):
                return {k: clean(v) for k,v in obj.items() if k not in
                        {'_name_or_path', 'transformers_version', 'torch_dtype', '_attn_implementation_autoset', 'gradient_checkpointing'}}
            if isinstance(obj, list):
                return [clean(v) for v in obj]
            return obj
        return _canonical(dict(model_id=self.model_id, model_revision=self.model_revision,
            native_config=clean(config), geometry=self.geometry, fusion='no_pruner',
            processor=processor_policy(self.processor), generation_special_tokens={
                key:getattr(self.native.generation_config,key) for key in
                ('bos_token_id','eos_token_id','pad_token_id')}))

    def train(self, mode=True):
        super().train(mode)
        self.native.model.vision_model.eval()
        return self

    @torch.no_grad()
    def extract_features(self, pixel_values, pixel_attention_mask=None):
        size = self.native.config.vision_config.image_size
        if pixel_values.ndim != 5 or tuple(pixel_values.shape[1:]) != (6,3,size,size):
            raise ValueError('pixel_values must have shape [B,6,3,image_size,image_size]')
        batch = pixel_values.shape[0]
        vision = self.native.model.vision_model
        pixels = pixel_values.flatten(0,1).to(device=next(vision.parameters()).device,
                                            dtype=next(vision.parameters()).dtype)
        if (pixels == 0).all(dim=(1,2,3)).any():
            raise ValueError('all-zero images are native padding, not six real camera views')
        if pixel_attention_mask is None:
            mask = torch.ones((batch*6,size,size), device=pixels.device, dtype=torch.bool)
        else:
            if tuple(pixel_attention_mask.shape) != (batch,6,size,size):
                raise ValueError('pixel_attention_mask shape must match six views')
            mask = pixel_attention_mask.flatten(0,1).to(device=pixels.device)
        patch = self.native.config.vision_config.patch_size
        patch_mask = mask.unfold(1,patch,patch).unfold(2,patch,patch).sum((-1,-2)) > 0
        vision.eval()
        raw = vision(pixel_values=pixels,patch_attention_mask=patch_mask).last_hidden_state
        return raw.reshape(batch,6,self.geometry['vision_tokens'],self.geometry['vision_dim'])

    def _project(self, visual_features, batch):
        shape=(batch,6,self.geometry['vision_tokens'],self.geometry['vision_dim'])
        if tuple(visual_features.shape) != shape:
            raise ValueError(f'visual_features must have shape {shape}')
        connector=self.native.model.connector
        weight=next(connector.parameters())
        return connector(visual_features.flatten(0,1).to(device=weight.device,dtype=weight.dtype))

    def _context(self, input_ids, extra=0):
        if input_ids.shape[1]+extra > self.native.config.text_config.max_position_embeddings:
            raise ValueError('expanded prompt exceeds native context budget')

    def forward(self, *, input_ids, attention_mask, visual_features, labels=None):
        self._context(input_ids)
        projected=self._project(visual_features,input_ids.shape[0])
        return self.native(input_ids=input_ids,attention_mask=attention_mask,
                           image_hidden_states=projected,labels=labels,use_cache=False)

    forward_from_features = forward

    @torch.no_grad()
    def generate_from_features(self, *, input_ids, attention_mask, visual_features, **kwargs):
        kwargs = dict(kwargs)
        generation_config=kwargs.get('generation_config',self.native.generation_config)
        budget=kwargs.get('max_new_tokens',generation_config.max_new_tokens)
        if budget is None:
            budget=max(0,kwargs.get('max_length',generation_config.max_length)-input_ids.shape[1])
        self._context(input_ids,budget)
        projected=self._project(visual_features,input_ids.shape[0])
        embeddings=self.native.get_input_embeddings()(input_ids)
        fused=self.native.model.inputs_merger(input_ids=input_ids,inputs_embeds=embeddings,
                                              image_hidden_states=projected)
        kwargs['use_cache']=True
        output=self.native.generate(input_ids=input_ids,inputs_embeds=fused,
                                    attention_mask=attention_mask,**kwargs)
        if hasattr(output,'sequences'):
            return output.sequences[:,input_ids.shape[1]:]
        return output[:,input_ids.shape[1]:]


def build_model_and_processor(cfg):
    model_id,revision=_model_identity(cfg)
    processor=AutoProcessor.from_pretrained(model_id,revision=revision)
    native,info=Idefics3ForConditionalGeneration.from_pretrained(model_id,revision=revision,
        torch_dtype=torch.float32,attn_implementation={'vision_config':'eager','text_config':'sdpa'},
        output_loading_info=True)
    if any(info.get(key) for key in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise ValueError(f'incomplete native checkpoint: {info}')
    if native.model.vision_model.config._attn_implementation != 'eager' or native.model.text_model.config._attn_implementation != 'sdpa':
        raise ValueError('native attention backends must be vision eager and text sdpa')
    vision_attention=[type(layer.self_attn).__name__ for layer in native.model.vision_model.encoder.layers]
    text_attention=[type(layer.self_attn).__name__ for layer in native.model.text_model.layers]
    if (not vision_attention or not text_attention
            or any(name != 'Idefics3VisionAttention' for name in vision_attention)
            or any(not name.endswith('SdpaAttention') for name in text_attention)):
        raise ValueError('actual native attention modules must be vision eager and text sdpa')
    gc=getattr(getattr(cfg,'training',None),'gradient_checkpointing',True)
    return CachedSmolVLM(native,processor,gradient_checkpointing=gc,
                        model_id=model_id,model_revision=revision),processor
