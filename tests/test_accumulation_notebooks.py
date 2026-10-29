import ast
import json
from pathlib import Path

import pytest
import yaml

PROFILES = ('unpruned', 'mvpruner', 'camera_query', 'camera_query_qaware')


@pytest.mark.parametrize('profile', PROFILES)
def test_accumulation_runtime_is_fresh_and_keeps_cache_and_profile(profile, tmp_path, monkeypatch):
    from efficient_vlm_ad.config import load_config
    import os
    config = Path(f'configs/dinov2_t5_efficient_mini_kaggle_2gpu_{profile}.yaml')
    nb = json.loads(Path(f'notebooks/kaggle_train_dinov2_{profile}_finetune.ipynb').read_text(encoding='utf-8'))
    sources = [''.join(c['source']) for c in nb['cells']]
    assert 'RESUME_TRAINING = False' in sources[1]
    source = next(s for s in sources if 'RUNTIME_CONFIG = Path(' in s)
    snippet = source[source.index('import yaml\n'):]
    tracked = tmp_path / 'tracked.yaml'
    tracked.write_bytes(config.read_bytes())
    runtime = tmp_path / 'runtime.yaml'
    snippet = snippet.replace(config.as_posix(), tracked.as_posix())
    snippet = snippet.replace(f'/kaggle/working/dinov2_{profile}_runtime.yaml', runtime.as_posix())
    original = yaml.safe_load(tracked.read_text())
    namespace = dict(Path=Path, os=os, json=json, load_config=load_config, GRADIENT_CHECKPOINTING=False)
    exec(compile(snippet, '<runtime>', 'exec'), namespace)
    raw = yaml.safe_load(runtime.read_text())
    assert raw['project']['output_dir'] == original['project']['output_dir']
    assert os.environ['PROFILE'] == raw['project']['output_dir']
    assert raw['cache'] == original['cache']
    assert Path(raw['cache']['dir']) == Path(raw['project']['output_dir']) / 'cache'
    assert raw['model'] == original['model']
    assert raw['training']['gradient_accumulation_steps'] == 2
    assert tracked.read_bytes() == config.read_bytes()
    train = next(s for s in sources if 'RESUME_PATH=' in s)
    assert 'if [ "$RESUME_TRAINING" = "true" ]; then' in train
    assert 'Checkpoint directory is nonempty' in train
    assert train.index('verify-resume-checkpoint') < train.index('accelerate.commands.launch')


@pytest.mark.parametrize('profile', PROFILES)
@pytest.mark.parametrize('nonempty', [False, True])
def test_fresh_run_ignores_hf_resume_and_rejects_existing_checkpoint(profile, nonempty, tmp_path):
    import os
    import shutil
    import subprocess
    bash = shutil.which('bash')
    git_bash = Path('C:/Program Files/Git/bin/bash.exe')
    if git_bash.exists():
        bash = str(git_bash)
    if not bash:
        pytest.skip('bash required for notebook shell guard')
    nb = json.loads(Path(f'notebooks/kaggle_train_dinov2_{profile}_finetune.ipynb').read_text(encoding='utf-8'))
    source = next(''.join(c['source']) for c in nb['cells'] if 'RESUME_PATH=' in ''.join(c['source']))
    script = source.split('RESUME=()')[0].replace('%%bash\n', '')
    script = script.replace('cd /kaggle/working/Efficient_VLM_For_Autonomous_Driving\n', '')
    checkpoints = tmp_path / 'checkpoints'
    checkpoints.mkdir()
    if nonempty:
        (checkpoints / 'finetune_latest.pt').write_bytes(b'old checkpoint')
    env = dict(os.environ, PROFILE=tmp_path.as_posix(), RESUME_TRAINING='false',
               FINETUNE_CHECKPOINT_HF_REPO_ID='must-not-download', PYTHON_EXECUTABLE='must-not-run')
    result = subprocess.run([bash, '-c', script], env=env, capture_output=True, text=True)
    if nonempty:
        assert result.returncode != 0
        assert 'Checkpoint directory is nonempty' in result.stderr
        assert (checkpoints / 'finetune_latest.pt').read_bytes() == b'old checkpoint'
    else:
        assert result.returncode == 0, result.stderr
        assert 'Fresh accumulation-v2' in result.stdout
        assert list(checkpoints.iterdir()) == []


@pytest.mark.parametrize('profile', PROFILES)
def test_resume_reuses_configured_local_checkpoint_before_download(profile):
    nb = json.loads(Path(f'notebooks/kaggle_train_dinov2_{profile}_finetune.ipynb').read_text(encoding='utf-8'))
    source = next(''.join(c['source']) for c in nb['cells'] if 'RESUME_PATH=' in ''.join(c['source']))
    assert source.index('elif [ -f "$PROFILE/checkpoints/$FINETUNE_CHECKPOINT_FILENAME" ]') < source.index('elif [ -n "$FINETUNE_CHECKPOINT_HF_REPO_ID" ]')
