"""Dependency validation must isolate unrelated Kaggle packages, not hide failures."""
from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace
import ast
import json
from pathlib import Path

import pytest

from efficient_vlm_ad.smolvlm.environment import validate_dependencies


def installed(packages):
    def lookup(name):
        if name not in packages:
            raise PackageNotFoundError(name)
        version, requires = packages[name]
        return SimpleNamespace(version=version, requires=requires)
    return lookup


def test_unrelated_conflicts_do_not_block_required_tree():
    lookup = installed({'app': ('1', ['hub<1']), 'hub': ('0.36.2', []),
                        'gradio': ('6', ['hub>=1'])})
    assert validate_dependencies(['app'], distribution=lookup) == {'app': '1', 'hub': '0.36.2'}


@pytest.mark.parametrize('packages', [
    {'app': ('1', ['child']), 'child': ('1', ['hub>=1']), 'hub': ('0.36', [])},
    {'app': ('1', ['missing'])},
])
def test_missing_or_incompatible_transitive_dependency_stops(packages):
    with pytest.raises(RuntimeError, match='dependency'):
        validate_dependencies(['app'], distribution=installed(packages))


def test_markers_extras_and_cycles():
    packages = {'app': ('1', ['child[fast]', 'absent; python_version < "2"']),
                'child': ('1', ['app', 'speed>=2; extra == "fast"',
                                'absent; extra == "unused"']), 'speed': ('2', [])}
    assert set(validate_dependencies(['app'], distribution=installed(packages))) == {'app', 'child', 'speed'}


def test_root_pin_is_checked():
    with pytest.raises(RuntimeError, match='dependency'):
        validate_dependencies(['app==2'], distribution=installed({'app': ('1', [])}))


@pytest.mark.parametrize('broken_workflow', [False, True])
def test_notebook_reports_global_conflicts_but_enforces_workflow(broken_workflow):
    notebook = Path(__file__).resolve().parents[1] / 'notebooks/kaggle_train_smolvlm_256m_no_pruner_finetune.ipynb'
    source = ''.join(json.loads(notebook.read_text(encoding='utf-8'))['cells'][2]['source'])
    start = source.index('workflow_requirements =')
    stop = source.index('from transformers import AutoProcessor', start)
    checks = ast.parse(source[start:stop])
    calls = []
    def validate(requirements):
        calls.append(requirements)
        if broken_workflow:
            raise RuntimeError('SmolVLM dependency validation failed')
        return {'transformers': '4.46.3'}
    namespace = dict(sys=SimpleNamespace(executable='python'),
                     subprocess=SimpleNamespace(run=lambda *a, **kw: SimpleNamespace(
                         returncode=1, stdout='gradio requires huggingface-hub>=1', stderr='')),
                     validate_dependencies=validate)
    if broken_workflow:
        with pytest.raises(RuntimeError, match='dependency'):
            exec(compile(checks, 'notebook_setup', 'exec'), namespace)
    else:
        exec(compile(checks, 'notebook_setup', 'exec'), namespace)
    assert len(calls) == 1
    assert 'efficient-vlm-ad' in calls[0] and 'pycocoevalcap' in calls[0]
