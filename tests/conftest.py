"""Suite selection only: default pytest still executes every supported group."""
from pathlib import Path

import pytest


def pytest_collection_modifyitems(items):
    integration_files = {
        'test_smoke_pipeline.py', 'test_verify_align_checkpoint.py',
    }
    integration_cases = {
        'test_module_entrypoint_help', 'test_prepare_data_no_progress_runs',
        'test_debug_sample_cli_outputs_sections_and_jsonl',
        'test_diagnose_train_cli_reports_boundaries_and_batch',
        'test_diagnose_train_cli_reports_numerics_when_requested',
        'test_historical_checkpoint_full_lifecycle',
    }
    for item in items:
        name = Path(str(item.path)).name
        if name.endswith('_distributed.py'):
            item.add_marker(pytest.mark.distributed)
        elif (name.endswith(('_smoke.py', '_lifecycle.py')) or name in integration_files
              or item.originalname in integration_cases):
            item.add_marker(pytest.mark.integration)
