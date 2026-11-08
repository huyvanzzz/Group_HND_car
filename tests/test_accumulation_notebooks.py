"""Execute notebook configuration and checkpoint selection with offline boundaries."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest
import yaml
from runtime_helpers import preserved_runtime

PROFILES = ('unpruned', 'mvpruner', 'camera_query', 'camera_query_qaware',
            'camera_pool_qaware', 'mvpruner_qaware', 'mvpruner_stage2_only',
            'mvpruner_ccts_context', 'mvpruner_three_stage_qaware', 'mvpruner_direct_camera')


def sources(profile):
    notebook = Path(f'notebooks/kaggle_train_dinov2_{profile}_finetune.ipynb')
    return [''.join(c['source']) for c in json.loads(notebook.read_text(encoding='utf-8'))['cells']]


@pytest.mark.parametrize('profile', PROFILES)
def test_accumulation_runtime_preserves_configured_architecture_and_cache(profile, tmp_path):
    from efficient_vlm_ad.config import load_config
    config = Path(f'configs/dinov2_t5_efficient_mini_kaggle_2gpu_{profile}.yaml')
    source = next(s for s in sources(profile) if 'RUNTIME_CONFIG = Path(' in s)
    snippet = source[source.index('TRACKED_CONFIG ='):]
    tracked = tmp_path / 'tracked.yaml'
    runtime = tmp_path / 'runtime.yaml'
    tracked.write_bytes(config.read_bytes())
    original = yaml.safe_load(tracked.read_text(encoding='utf-8'))
    snippet = snippet.replace(config.as_posix(), tracked.as_posix())
    snippet = snippet.replace(f'/kaggle/working/dinov2_{profile}_runtime.yaml', runtime.as_posix())
    namespace = dict(Path=Path, os=os, json=json, yaml=yaml, load_config=load_config,
                     GRADIENT_CHECKPOINTING=False)
    with preserved_runtime():
        exec(compile(snippet, '<runtime>', 'exec'), namespace)
        raw = yaml.safe_load(runtime.read_text(encoding='utf-8'))
        assert os.environ['PROFILE'] == original['project']['output_dir']
        assert os.environ['CONFIG'] == str(runtime)
    assert raw['project'] == original['project']
    assert raw['cache'] == original['cache']
    assert Path(raw['cache']['dir']) == Path(raw['project']['output_dir']) / 'cache'
    assert raw['model'] == original['model']
    assert namespace['cfg'].training.entropy_weight == original['training'].get('entropy_weight', 0.)
    assert raw['training']['gradient_accumulation_steps'] == 2
    assert tracked.read_bytes() == config.read_bytes()


@pytest.mark.parametrize('profile', PROFILES)
@pytest.mark.parametrize('available,expected,remote', [
    ((), '', False), ((), 'custom.pt', True),
    (('custom.pt',), 'custom.pt', True),
    (('finetune_latest.pt', 'custom.pt'), 'finetune_latest.pt', True),
    (('unrelated.pt',), '', False),
])
def test_checkpoint_selection_prefers_latest_then_configured_then_download(profile, available, expected, remote, tmp_path):
    bash_path = Path('C:/Program Files/Git/bin/bash.exe')
    bash = str(bash_path) if bash_path.is_file() else shutil.which('bash')
    assert bash, 'Notebook lifecycle tests require Bash'
    source = next(s for s in sources(profile) if 'RESUME_PATH=' in s)
    script = source.split('RESUME=()')[0].replace('%%bash\n', '')
    script = script.replace('cd /kaggle/working/Efficient_VLM_For_Autonomous_Driving\n', '')
    checkpoints = tmp_path / 'checkpoints'
    checkpoints.mkdir()
    for filename in available:
        (checkpoints / filename).write_bytes(b'preserved local checkpoint')
    # These functions replace only remote checkpoint transport. All Bash selection
    # branches and file checks are the notebook's actual code.
    boundary = """hf() { printf 'DOWNLOAD\\n'; }
python_download() { cat >/dev/null; printf 'DOWNLOAD\\n'; }
"""
    script += '\nprintf "SELECTED=%s\\n" "$RESUME_PATH"\n'
    env = dict(os.environ, PROFILE=tmp_path.as_posix(), FINETUNE_CHECKPOINT_FILENAME='custom.pt',
               FINETUNE_CHECKPOINT_HF_REPO_ID='example/checkpoint' if remote else '', FINETUNE_CHECKPOINT_HF_REPO_TYPE='model',
               PYTHON_EXECUTABLE='python_download')
    result = subprocess.run([bash, '-c', boundary + script], env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    chosen = f'{tmp_path.as_posix()}/checkpoints/{expected}' if expected else ''
    assert f'SELECTED={chosen}\n' in result.stdout
    assert ('DOWNLOAD' in result.stdout) == (remote and not available)
    for filename in available:
        assert (checkpoints / filename).read_bytes() == b'preserved local checkpoint'
