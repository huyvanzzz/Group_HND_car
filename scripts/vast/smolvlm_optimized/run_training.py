from pathlib import Path
import os,subprocess,json,tarfile
os.environ.update(HF_HOME='/workspace/.hf_home',HF_HUB_DISABLE_XET='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',HF_HUB_DOWNLOAD_TIMEOUT='120',PYTORCH_ALLOC_CONF='expandable_segments:True',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',OMP_NUM_THREADS='1',PYTHONUNBUFFERED='1')
work=Path('/workspace/smolvlm-vast')
if not (work/'OPTIMIZED_READY.json').is_file():raise RuntimeError('Optimized preflight not ready; inspect optimized_verify.log')
import hashlib
ready=json.loads((work/'OPTIMIZED_READY.json').read_text())
for name,digest in ready['file_sha256'].items():
 if hashlib.sha256((work/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Optimized configuration/code changed since verification: '+name)
python='/workspace/smolvlm-venv/bin/python'
config=str(work/'optimized.yaml')
repo=Path('/workspace/huy-smolvlm-256m-no-pruner')
profile=repo/'outputs/smolvlm_256m_kaggle_2gpu_no_pruner'
latest=profile/'checkpoints/finetune_latest.pt'
resume=[]
if latest.is_file():
 subprocess.run([python,str(work/'optimized_entry.py'),'verify','--config',config,'--checkpoint',str(latest)],cwd=repo,check=True)
 resume=['--checkpoint',str(latest)]
# No epochs-per-run: continuous training to the configured total, never a new recipe.
subprocess.run([python,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(work/'optimized_entry.py'),'train','--config',config,*resume],cwd=repo,check=True)
status=json.loads((profile/'training_status.json').read_text())
if not status['training_complete']:raise RuntimeError('Training stopped before the configured total')
best=profile/'checkpoints/finetune_best.pt'
checkpoint=best if best.is_file() else latest
for command in ('evaluate','benchmark'):
 prefix=[python,'-m','torch.distributed.run','--standalone','--nproc_per_node=2'] if command=='evaluate' else [python]
 subprocess.run(prefix+[str(work/'optimized_entry.py'),command,'--config',config,'--checkpoint',str(checkpoint)],cwd=repo,check=True)
archive=work/'smolvlm_finetune_artifacts.tar.gz'
with tarfile.open(archive,'w:gz') as bundle:
 bundle.add(config,arcname='optimized.yaml')
 for name in ('optimization.json','optimized_entry.py','distributed_evaluate.py','OPTIMIZED_READY.json'):
  bundle.add(work/name,arcname=name)
 for name in ('checkpoints','training_status.json','prepared_data','debug','metrics.json','metrics_display.json','generation_diagnostics.json','predictions.jsonl','benchmark.json'):
  item=profile/name
  if item.exists():bundle.add(item,arcname=profile.name+'/'+name)
print('ALL_DONE',archive,flush=True)
