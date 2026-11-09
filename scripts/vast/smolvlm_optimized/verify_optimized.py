import subprocess
from pathlib import Path
r=Path('/workspace/smolvlm-vast');py='/workspace/smolvlm-venv/bin/python';output=str(r/'optimized_smoke');entry=str(r/'optimized_entry.py')
cmd=[py,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',entry,'train','--smoke','--output',output]
subprocess.run(cmd,check=True,cwd='/workspace/huy-smolvlm-256m-no-pruner')
checkpoint=str(r/'optimized_smoke/checkpoints/finetune_latest.pt')
subprocess.run([py,entry,'verify','--smoke','--output',output,'--checkpoint',checkpoint],check=True)
subprocess.run(cmd+['--checkpoint',checkpoint],check=True,cwd='/workspace/huy-smolvlm-256m-no-pruner')
print('OPTIMIZED_TRAIN_RESUME_SMOKE_PASSED',flush=True)
