import os,json
from pathlib import Path
import torch
import optimized_entry as opt
from efficient_vlm_ad.smolvlm.config import load_config
from efficient_vlm_ad.smolvlm import pipeline as native
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.distributed.init_process_group('nccl')
cfg=load_config('/workspace/smolvlm-vast/optimized.yaml');model,processor=opt.build(cfg);device=torch.device('cuda',rank);model.to(device).eval()
data=opt.loader(cfg,model,processor,'val',max_samples=5)
with torch.autocast('cuda',dtype=torch.float16):value=opt.distributed_validation(model,data,device)
indices=list(data.dataset.indices);parts=[None,None];torch.distributed.all_gather_object(parts,indices)
assert sorted(sum(parts,[]))==list(range(5))
reference=None
if rank==0:
 full=native.loader(cfg,model,processor,'val',max_samples=5)
 with torch.autocast('cuda',dtype=torch.float16):reference=native.validation(model,full,device,disable_progress=True)
 assert abs(value-reference)<1e-3,(value,reference)
 Path('/workspace/smolvlm-vast/measurements/distributed_validation_check.json').write_text(json.dumps(dict(ok=True,samples=5,rank_indices=parts,distributed_loss=value,single_gpu_reference=reference,batch_per_gpu=32),indent=2))
 print('VALIDATION_SHARD_AND_WEIGHTED_LOSS_OK',value,reference,flush=True)
torch.distributed.barrier();torch.distributed.destroy_process_group()
