import subprocess
from pathlib import Path
r=Path('/workspace/smolvlm-vast');py='/workspace/smolvlm-venv/bin/python'
base=[py,'-m','torch.distributed.run','--standalone','--nproc_per_node=2']
subprocess.run(base+[str(r/'check_validation_shards.py')],check=True)
subprocess.run(base+[str(r/'optimized_entry.py'),'evaluate','--smoke','--output',str(r/'optimized_eval_smoke'),'--checkpoint',str(r/'optimized_smoke/checkpoints/finetune_latest.pt')],check=True)
print('VALIDATION_EVALUATION_CHANGES_PASSED',flush=True)
