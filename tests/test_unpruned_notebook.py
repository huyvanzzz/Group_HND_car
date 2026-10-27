import ast
import json
from pathlib import Path

import pytest
import yaml


CONFIG = Path('configs/dinov2_t5_efficient_mini_kaggle_2gpu_unpruned.yaml')
NOTEBOOK = Path('notebooks/kaggle_train_dinov2_unpruned_finetune.ipynb')


def cells():
    return json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells']


def test_unpruned_config_default_and_matching_main_settings():
    raw = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    main = yaml.safe_load(CONFIG.with_name(CONFIG.name.replace('unpruned', 'mvpruner')).read_text())
    assert raw['training']['gradient_checkpointing'] is False
    assert raw['model']['fusion'] == {'name': 'unpruned'}
    for section in ('data', 'runtime', 'generation', 'evaluation'):
        assert raw[section] == main[section]
    for key, value in main['training'].items():
        if key != 'align_epochs':
            assert raw['training'][key] == value


def test_allocator_and_strict_flag_precede_torch():
    sources = [''.join(c['source']) for c in cells()]
    first = sources[1]
    assert 'GRADIENT_CHECKPOINTING = False' in first
    assert 'type(GRADIENT_CHECKPOINTING) is not bool' in first
    for key in ('PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF'):
        assert f"os.environ['{key}'] = 'expandable_segments:True'" in first
    text = '\n'.join(sources)
    assert text.index("os.environ['PYTORCH_ALLOC_CONF']") < text.index('import os, torch')
    tree = ast.parse(first)
    for value in (False, True, 'false', 0, None):
        flag_nodes = [node for node in tree.body if isinstance(node, ast.If)
                      and 'type(GRADIENT_CHECKPOINTING)' in ast.unparse(node.test)]
        assert len(flag_nodes) == 1
        code = compile(ast.Module(body=flag_nodes, type_ignores=[]), '<flag>', 'exec')
        if type(value) is bool:
            exec(code, {'GRADIENT_CHECKPOINTING': value})
        else:
            with pytest.raises(TypeError):
                exec(code, {'GRADIENT_CHECKPOINTING': value})


def test_runtime_copy_finetune_resume_and_archive_contract():
    text = '\n'.join(''.join(c['source']) for c in cells())
    for expected in ["raw['training']['gradient_checkpointing'] = GRADIENT_CHECKPOINTING",
                     "RUNTIME_CONFIG = Path('/kaggle/working/dinov2_unpruned_runtime.yaml')",
                     "os.environ['CONFIG'] = str(RUNTIME_CONFIG)",
                     'cfg = load_config(RUNTIME_CONFIG)', "'visual_tokens': 384",
                     "'gradient_checkpointing': cfg.training.gradient_checkpointing",
                     'outputs/dinov2_t5_efficient_mini_kaggle_2gpu_unpruned',
                     '--stage finetune', 'verify-resume-checkpoint',
                     'rank_00000.pt', 'rank_00001.pt', 'validate_cache(cfg)',
                     'expected_counts=cfg.data.expected_counts',
                     'dinov2_unpruned_finetune_artifacts.tar.gz', '"$CONFIG" "$PROFILE/checkpoints"']:
        assert expected in text
    assert text.index("os.environ['CONFIG'] = str(RUNTIME_CONFIG)") < text.index('prepared_files =')
    assert '--stage align' not in text and '--max-steps' not in text
    assert 'batch_size\'] =' not in text
    for cell in cells():
        if cell['cell_type'] == 'code':
            assert cell['outputs'] == [] and cell['execution_count'] is None
            code = ''.join(cell['source'])
            if not code.startswith('%%bash'):
                ast.parse(code)


@pytest.mark.parametrize('enabled', [False, True])
def test_runtime_copy_applies_flag_without_modifying_tracked_yaml(tmp_path, monkeypatch, enabled):
    from efficient_vlm_ad.config import load_config

    source = next(''.join(c['source']) for c in cells()
                  if 'RUNTIME_CONFIG = Path(' in ''.join(c['source']))
    snippet = source[source.index('import yaml\n'):]
    tracked = tmp_path / 'tracked.yaml'
    runtime = tmp_path / 'runtime.yaml'
    original = CONFIG.read_bytes()
    tracked.write_bytes(original)
    snippet = snippet.replace(str(CONFIG).replace('\\', '/'), tracked.as_posix())
    snippet = snippet.replace('/kaggle/working/dinov2_unpruned_runtime.yaml', runtime.as_posix())
    import os
    monkeypatch.setenv('CONFIG', str(tracked))
    namespace = dict(Path=Path, os=os, json=json, load_config=load_config,
                     GRADIENT_CHECKPOINTING=enabled)
    exec(compile(snippet, '<runtime-copy>', 'exec'), namespace)
    assert tracked.read_bytes() == original
    assert yaml.safe_load(runtime.read_text())['training']['gradient_checkpointing'] is enabled
    assert namespace['cfg'].training.gradient_checkpointing is enabled
    assert os.environ['CONFIG'] == str(runtime)
