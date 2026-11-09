import os,time,json
from pathlib import Path
from dataclasses import asdict,replace
import torch
from torch.utils.data import DataLoader,Subset
from efficient_vlm_ad.smolvlm import pipeline as native
from efficient_vlm_ad.smolvlm import checkpoint
from efficient_vlm_ad.smolvlm.data import CachedVLMDataset
from efficient_vlm_ad.smolvlm.text import CachedBatchCollator
from efficient_vlm_ad.smolvlm.model import build_model_and_processor
from efficient_vlm_ad.progress_logging import MinimalProgressWriter
from efficient_vlm_ad.evaluation import caption_metrics,write_predictions,metric_display_values,write_generation_diagnostics
@torch.no_grad()
def evaluate(cfg,path,smoke=False):
 rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
 device=torch.device('cuda',local);torch.cuda.set_device(device)
 if world>1:torch.distributed.init_process_group('nccl')
 native.seed(cfg.training.seed)
 model,processor=build_model_and_processor(cfg)
 payload=checkpoint.read(path,model);model.load_state_dict(payload['model_state_dict'],strict=True);checkpoint_metadata=payload['metadata'];del payload
 if not smoke:
  if rank==0:native.validate_cache(cfg,processor,model.geometry,full=True)
  if world>1:torch.distributed.barrier()
 model.to(device).eval()
 ds=CachedVLMDataset(cfg,'test',validate=False)
 total=min(len(ds),5) if smoke else len(ds)
 indices=list(range(rank,total,world))
 collate=CachedBatchCollator(processor,image_seq_len=model.geometry['visual_tokens_per_view'],max_length=model.native.config.text_config.max_position_embeddings,generation=True,max_new_tokens=4 if smoke else cfg.generation.max_new_tokens)
 data=DataLoader(Subset(ds,indices),batch_size=cfg.evaluation.eval_batch_size,collate_fn=collate,num_workers=cfg.training.num_workers,pin_memory=cfg.training.pin_memory,persistent_workers=cfg.training.num_workers>0)
 kwargs=asdict(cfg.generation)
 if smoke:kwargs['max_new_tokens']=4
 if rank==0:native.event(cfg,'INFERENCE',{**native.metadata(cfg,model),'stage':'evaluate','checkpoint':str(path),'num_samples':total,'batch_per_gpu':cfg.evaluation.eval_batch_size,'world_size':world,'sharding':'strided_no_padding_no_duplicates','checkpoint_gradient_checkpointing':checkpoint_metadata.get('gradient_checkpointing')})
 writer=MinimalProgressWriter(cfg.project.output_dir,'eval_progress.jsonl') if rank==0 else None
 started=time.perf_counter();rows=[];offset=0
 if writer is not None:native._minimal_progress(writer,'evaluate',0,total,started)
 for batch in data:
  values=native.inputs(batch,device);output=model.generate_from_features(**values,**kwargs);texts=processor.tokenizer.batch_decode(output,skip_special_tokens=True)
  for j,text in enumerate(texts):
   rows.append((indices[offset+j],dict(eval_id=batch['eval_ids'][j],sample_id=batch['sample_ids'][j],reference=batch['answers'][j],prediction=text)))
  offset+=len(texts)
  if writer is not None and offset%cfg.evaluation.eval_progress_log_every_samples==0:native._minimal_progress(writer,'evaluate',min(offset*world,total),total,started)
 if world>1:
  gathered=[None]*world;torch.distributed.all_gather_object(gathered,rows);rows=[row for part in gathered for row in part]
 rows.sort(key=lambda row:row[0])
 if [i for i,_ in rows]!=list(range(total)):raise AssertionError('Distributed evaluation sample coverage mismatch')
 if rank==0:
  rows=[row for _,row in rows];out=Path(cfg.project.output_dir);write_predictions(out/'predictions.jsonl',rows)
  diagnostics=write_generation_diagnostics(out/'generation_diagnostics.json',rows)
  metrics=caption_metrics(rows,skip_meteor=smoke);native.write_json(out/'metrics.json',metrics);display=metric_display_values(metrics);native.write_json(out/'metrics_display.json',display)
  native._minimal_progress(writer,'evaluate',total,total,started)
  native.event(cfg,'EVAL',dict(prediction_count=len(rows),metrics=metrics,metrics_display=display,metric_names=list(metrics),generation_diagnostics=diagnostics,batch_per_gpu=cfg.evaluation.eval_batch_size,world_size=world))
  native.write_json(out/'distributed_evaluation.json',dict(sample_count=total,world_size=world,batch_per_gpu=cfg.evaluation.eval_batch_size,unique_source_indices=True))
 if world>1:torch.distributed.barrier();torch.distributed.destroy_process_group()
