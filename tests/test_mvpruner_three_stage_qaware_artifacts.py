"""Protocol and executable-cell contracts for the isolated three-stage artifacts."""
import ast
import json
import re
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
STEM = 'dinov2_t5_efficient_mini_kaggle_2gpu_mvpruner_three_stage_qaware'
NB = ROOT / 'notebooks/kaggle_train_dinov2_mvpruner_three_stage_qaware_finetune.ipynb'


def test_three_stage_config_preserves_qaware_protocol():
    raw = yaml.safe_load((ROOT / 'configs' / (STEM + '.yaml')).read_text())
    base = yaml.safe_load((ROOT / 'configs/dinov2_t5_efficient_mini_kaggle_2gpu_camera_query_qaware.yaml').read_text())
    assert raw['project']['output_dir'] == 'outputs/' + STEM
    assert raw['cache']['dir'] == 'outputs/' + STEM + '/cache'
    assert raw['model']['profile'] == STEM
    fusion = raw['model']['fusion']
    assert (fusion['stage1_top_k'], fusion['stage2_top_k'], fusion['stage3_top_k']) == (196, 98, 49)
    assert fusion['name'] == 'mvpruner_three_stage_qaware'
    assert (fusion['beta_start'], fusion['beta_end'], fusion['tau']) == (.2, .1, 1.)
    assert fusion['beta_schedule'] == 'linear' and fusion['stage2_layer'] == 2
    assert fusion['query'] == base['model']['fusion']['query']
    assert fusion['question_scoring'] == base['model']['fusion']['question_scoring']
    for key in ('vision', 'text', 'projector'):
        assert raw['model'][key] == base['model'][key]
    assert raw['training']['align_epochs'] == 0
    assert {k: v for k, v in raw['training'].items() if k != 'align_epochs'} == base['training']
    for key in ('data', 'runtime', 'generation', 'evaluation'):
        assert raw[key] == base[key]
    assert raw['training']['entropy_weight'] == 0.
    assert raw['training']['gradient_checkpointing'] is False
    assert raw['training']['beta_decay_updates'] is None


def test_three_stage_notebook_cells_and_pipeline_contract():
    nb = json.loads(NB.read_text(encoding='utf-8'))
    sources = [''.join(c['source']) for c in nb['cells']]
    text = '\n'.join(sources)
    first = next(s for c,s in zip(nb['cells'], sources) if c['cell_type'] == 'code')
    for key in ('PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF'):
        assert f"os.environ['{key}'] = 'expandable_segments:True'" in first
        assert text.index(key) < text.index('import os, torch')
    for cell, source in zip(nb['cells'], sources):
        if cell['cell_type'] != 'code':
            continue
        assert cell['outputs'] == [] and cell['execution_count'] is None
        if source.startswith('%%bash'):
            for script in re.findall(r"<<'PY'\n(.*?)\nPY", source, re.S):
                ast.parse(script)
        else:
            ast.parse(source)
    for expected in ('stage2_prune', 'stage3_merge', 'text_pool_entropy', 'text_pool_max_weight_mean',
                     'diagnose_forward', 'diagnose_backward', 'mvpruner_gradients', 'default_debug_jsonl',
                     'FINETUNE_CHECKPOINT', 'rank_00000.pt', 'rank_00001.pt', '$RESUME_PATH.rng/manifest.json',
                     'TRACKED_CONFIG.read_text', 'RUNTIME_CONFIG.write_text', 'validate_prepared(prepared, full=True',
                     'validate_cache(cfg)', 'dinov2_manifest_fields', '--num_processes 2', '--mixed_precision no',
                     'transformers==4.46.3', 'tokenizers==0.20.4', 'accelerate==1.15.0'):
        assert expected in text
    ordered = ('validate_cache(cfg)\n', 'diagnose-train --config', '-m efficient_vlm_ad.accumulation_debug',
               'verify-resume-checkpoint --config', '-m accelerate.commands.launch', 'evaluate --config', 'benchmark --config')
    positions = [text.index(term) for term in ordered]
    assert positions == sorted(positions)
    assert 'RESUME_TRAINING' not in text and 'TRACKED_CONFIG.write_text' not in text
    assert '--stage align' not in text and 'mvpruner_camera_query_qaware' not in text
    assert '_qaware_v2' not in text
    assert '(196, 98, 49)' in text and '(0.2, 0.1, 1.0)' in text


def test_three_stage_docs_explain_boundaries_and_measurement_limits():
    text = (ROOT / 'docs/research/mvpruner-three-stage-qaware.md').read_text(encoding='utf-8')
    for term in ('196', '98', '49', '[44,64]', 'stage2_prune', 'stage3_merge',
                 '53,350', 'accelerator.prepare', 'entropy_weight: 0.0',
                 'native softmax', 'Kiểm chứng', 'GPU'):
        assert term in text
