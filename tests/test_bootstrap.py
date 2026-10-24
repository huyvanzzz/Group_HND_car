import ast
from pathlib import Path


def test_bootstrap_references_retained_profiles_and_finetune_only():
    source = Path('scripts/kaggle_bootstrap.py').read_text(encoding='utf-8')
    constants = [n.value for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    profiles = [s for s in constants if s.startswith('configs/') and s.endswith('.yaml')]
    assert profiles and all(Path(path).is_file() for path in profiles)
    assert 'huy-v2drop' in constants
    assert 'align' not in constants
    assert 'export PYTORCH_ALLOC_CONF=expandable_segments:True' in source
