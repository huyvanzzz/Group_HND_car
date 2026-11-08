"""In-process CLI tests must not contaminate the next test's runtime."""
import os
import random
import sys

import numpy as np
import pytest
import torch

from runtime_helpers import preserved_runtime, run_cli


def test_preserved_runtime_restores_state_even_on_error(tmp_path):
    environment, directory, argv = dict(os.environ), os.getcwd(), sys.argv[:]
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    with pytest.raises(RuntimeError, match='fixture failure'):
        with preserved_runtime({'TEST_RUNTIME_SENTINEL': 'active'}, tmp_path):
            assert os.environ['TEST_RUNTIME_SENTINEL'] == 'active'
            assert os.getcwd() == str(tmp_path)
            os.environ['TEST_RUNTIME_ADDED'] = 'temporary'
            sys.argv[:] = ['changed']
            random.random()
            np.random.rand()
            torch.rand(3)
            raise RuntimeError('fixture failure')
    assert dict(os.environ) == environment
    assert os.getcwd() == directory and sys.argv == argv
    assert random.getstate() == python_rng
    restored_numpy = np.random.get_state()
    assert restored_numpy[0] == numpy_rng[0]
    np.testing.assert_array_equal(restored_numpy[1], numpy_rng[1])
    assert restored_numpy[2:] == numpy_rng[2:]
    torch.testing.assert_close(torch.get_rng_state(), torch_rng, rtol=0, atol=0)


def test_cli_capture_preserves_real_parser_exit_codes():
    valid = run_cli(['--help'])
    assert valid.returncode == 0 and 'diagnose-train' in valid.stdout
    invalid = run_cli(['not-a-command'])
    assert invalid.returncode == 2 and 'invalid choice' in invalid.stderr
    again = run_cli(['--help'])
    assert again.stdout == valid.stdout and again.stderr == ''
