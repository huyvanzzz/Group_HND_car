import os
os.environ.update(HF_HOME='/workspace/.hf_home',HF_HUB_DISABLE_XET='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',PYTORCH_ALLOC_CONF='expandable_segments:True',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
import argparse,json,types,contextlib
from pathlib import Path
from dataclasses import replace
import torch,numpy as np
from accelerate import Accelerator,DataLoaderConfiguration
from accelerate.utils import GradScalerKwargs
from efficient_vlm_ad.smolvlm import pipeline as original,checkpoint as ck,model as modeling
from efficient_vlm_ad.smolvlm.config import load_config
from efficient_vlm_ad.smolvlm.data import CachedVLMDataset
from efficient_vlm_ad.smolvlm.cache import FEATURE_FILE,record_key
ROOT=Path('/workspace/smolvlm-vast');OPTIONS=json.loads((ROOT/'optimization.json').read_text())
# Copy lifecycle code into a private module; original source files stay untouched.
cp=types.ModuleType('efficient_vlm_ad.smolvlm._vast_checkpoint');cp.__package__='efficient_vlm_ad.smolvlm'
source=Path(ck.__file__).read_text()
source=source.replace("'metadata': metadata}","'metadata': metadata, 'grad_scaler_state_dict': accelerator.scaler.state_dict() if accelerator.scaler is not None else None}")
exec(compile(source,str(ROOT/'optimized_checkpoint_runtime.py'),'exec'),cp.__dict__)
base_recipe=cp.recipe
cp.recipe=lambda cfg:{**base_recipe(cfg),'vast_optimization_v1':OPTIONS}
base_read=cp.read
def read(path,model,cfg=None,**kw):
 payload=base_read(path,model,cfg,**kw)
 if kw.get('training') and not payload.get('grad_scaler_state_dict'):raise ValueError('FP16 resume requires GradScaler state')
 return payload
cp.read=read
p=types.ModuleType('efficient_vlm_ad.smolvlm._vast_pipeline');p.__package__='efficient_vlm_ad.smolvlm'
source=Path(original.__file__).read_text()
old="""            raw = model(**inputs(batch, accelerator.device)).loss
            if not bool(torch.isfinite(raw)):
                raise FloatingPointError('Nonfinite native CE')
            size = window_size(index, remaining, A)
            accelerator.backward(loss_for_backward(raw, A, size))"""
new="""            size = window_size(index, remaining, A)
            boundary = window_boundary(index, remaining, A)
            with accelerator.no_sync(model) if not boundary else contextlib.nullcontext():
                raw = model(**inputs(batch, accelerator.device)).loss
                if not bool(torch.isfinite(raw)):
                    raise FloatingPointError('Nonfinite native CE')
                accelerator.backward(loss_for_backward(raw, A, size))"""
assert old in source;source=source.replace(old,new)
source=source.replace("        scheduler.load_state_dict(payload['scheduler_state_dict'])","        scheduler.load_state_dict(payload['scheduler_state_dict'])\n        accelerator.scaler.load_state_dict(payload['grad_scaler_state_dict'])")
source=source.replace("                if not bool(torch.isfinite(norm)):","                if accelerator.scaler is None and not bool(torch.isfinite(norm)):")
source=source.replace("'gradient_norm_before_clipping': float(norm)","'gradient_norm_before_clipping': float(norm) if bool(torch.isfinite(norm)) else None")
old="        value = validation(native, val_batches, accelerator.device, disable_progress=disable_progress) if accelerator.is_main_process else 0."
assert old in source;source=source.replace(old,"        with accelerator.autocast():\n    "+old)
source=source.replace('value = validation(native, val_batches, accelerator.device, disable_progress=disable_progress) if accelerator.is_main_process else 0.', 'value = validation(native, val_batches, accelerator.device, disable_progress=disable_progress)')
exec(compile(source,str(ROOT/'optimized_pipeline_runtime.py'),'exec'),p.__dict__)
p.checkpoint=cp;p.contextlib=contextlib
class HalfFeatures(CachedVLMDataset):
 def __getitem__(self,index):
  if self._features is None:self._features=np.memmap(Path(self.cfg.cache.dir)/FEATURE_FILE,mode='r',dtype=np.float16,shape=self.shape)
  rec=self.records[index];feat=np.array(self._features[self.index[record_key(rec,self.cfg.data.view_order)]],dtype=np.float16,copy=True)
  return dict(question=rec.question,answer=rec.answer,visual_features=torch.from_numpy(feat),sample_id=rec.sample_id,eval_id=rec.eval_id)
p.CachedVLMDataset=HalfFeatures
# loader() resolves DataLoader here, preserving seedable sampler and existing collator.
def dataloader(*args,**kwargs):
 kwargs['persistent_workers']=kwargs.get('num_workers',0)>0
 return torch.utils.data.DataLoader(*args,**kwargs)
p.DataLoader=dataloader
base_loader=p.loader
SMOKE=False
def loader(cfg,model,processor,split,**kwargs):
 if SMOKE:kwargs['max_samples']=108 if split=='train' else 6
 data=base_loader(cfg,model,processor,split,**kwargs)
 if split=='val' and torch.distributed.is_initialized():
  rank=torch.distributed.get_rank();world=torch.distributed.get_world_size()
  subset=torch.utils.data.Subset(data.dataset,range(rank,len(data.dataset),world))
  data=dataloader(subset,batch_size=cfg.evaluation.eval_batch_size,collate_fn=data.collate_fn,num_workers=cfg.training.num_workers,pin_memory=cfg.training.pin_memory,generator=torch.Generator().manual_seed(cfg.training.seed))
 return data
p.loader=loader
@torch.no_grad()
def distributed_validation(model,batches,device,disable_progress=False):
 model.eval();totals=torch.zeros(2,device=device,dtype=torch.float64)
 for batch in batches:
  values=p.inputs(batch,device);count=(values['labels'][:,1:]!=-100).sum()
  loss=model(**values).loss
  if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite validation CE')
  totals[0]+=loss.double()*count;totals[1]+=count
 if torch.distributed.is_initialized():torch.distributed.all_reduce(totals)
 if not bool(totals[1]>0):raise ValueError('Validation has no target tokens')
 return float(totals[0]/totals[1])
p.validation=distributed_validation

def accelerator_for(cfg):
 return Accelerator(mixed_precision='fp16',gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,step_scheduler_with_optimizer=False,dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True,data_seed=cfg.training.seed),kwargs_handlers=[GradScalerKwargs(init_scale=256.)])
p.accelerator_for=accelerator_for
base_build=p.build_model_and_processor
def build(cfg):
 model,processor=base_build(cfg)
 def project(self,features,batch):
  expected=(batch,6,self.geometry['vision_tokens'],self.geometry['vision_dim'])
  if tuple(features.shape)!=expected:raise ValueError('invalid features geometry')
  connector=self.native.model.connector
  # Preserve FP16 under AMP; FP32 inference remains supported.
  dtype=features.dtype if torch.is_autocast_enabled('cuda') else next(connector.parameters()).dtype
  return connector(features.flatten(0,1).to(device=next(connector.parameters()).device,dtype=dtype))
 model._project=types.MethodType(project,model)
 return model,processor
p.build_model_and_processor=build
base_meta=p.metadata
p.metadata=lambda cfg,model:{**base_meta(cfg,model),'compute_dtype':'mixed_fp16_fp32','feature_input_dtype':'float16','parameter_dtype':'float32','optimizer_state_dtype':'float32','persistent_workers':True,'no_sync':True,'allocator':OPTIONS['allocator'],'grad_scaler_initial_scale':256.}
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('command',choices=['train','verify','evaluate','benchmark']);ap.add_argument('--config',default=str(ROOT/'optimized.yaml'));ap.add_argument('--checkpoint');ap.add_argument('--smoke',action='store_true');ap.add_argument('--output');args=ap.parse_args()
 cfg=load_config(args.config);SMOKE=args.smoke
 if args.output:cfg=replace(cfg,project=replace(cfg.project,output_dir=args.output))
 if SMOKE:cfg=replace(cfg,training=replace(cfg.training,finetune_epochs=2,finetune_max_steps=18 if args.checkpoint else 9))
 if args.command=='train':p.train(cfg,resume=args.checkpoint,debug=SMOKE,debug_numerics=True,disable_progress=SMOKE)
 elif args.command=='verify':
  model,_=build(cfg);payload=cp.read(args.checkpoint,model,cfg,training=True,world_size=2);print('RESUME_IDENTITY_AND_SCALER_OK',payload['grad_scaler_state_dict'])
 else:
  if args.command=='evaluate':
   from distributed_evaluate import evaluate
   evaluate(cfg,args.checkpoint,smoke=SMOKE)
  else:original.inference(cfg,args.checkpoint,benchmark=True)
