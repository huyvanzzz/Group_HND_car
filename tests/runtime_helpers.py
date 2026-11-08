"""Real CLI execution for tests without repeatedly starting Python/Torch."""
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from io import StringIO
import importlib
import os
from pathlib import Path
import random
import subprocess
import sys
import traceback

import numpy as np
import torch


@contextmanager
def preserved_runtime(env=None, cwd=None):
    environment = dict(os.environ)
    directory = Path.cwd()
    argv = sys.argv[:]
    python_rng, numpy_rng, cpu_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        if env is not None:
            os.environ.update({str(k): str(v) for k, v in env.items()})
        if cwd is not None:
            os.chdir(cwd)
        yield
    finally:
        os.environ.clear()
        os.environ.update(environment)
        os.chdir(directory)
        sys.argv[:] = argv
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def run_module(module, args, *, env=None, cwd=None, check=False):
    """Capture parser output/errors; do not replace production computation."""
    if module in ('efficient_vlm_ad', 'efficient_vlm_ad.__main__'):
        module = 'efficient_vlm_ad.cli'
    out, err, status = StringIO(), StringIO(), 0
    arguments = [str(arg) for arg in args]
    with preserved_runtime(env, cwd), redirect_stdout(out), redirect_stderr(err):
        sys.argv[:] = [module, *arguments]
        try:
            importlib.import_module(module).main(arguments)
        except SystemExit as exc:
            if isinstance(exc.code, int):
                status = exc.code
            elif exc.code is not None:
                status = 1
                print(str(exc.code), file=sys.stderr)
        except Exception:
            status = 1
            traceback.print_exc()
    result = subprocess.CompletedProcess([module, *arguments], status, out.getvalue(), err.getvalue())
    if check:
        result.check_returncode()
    return result


def run_cli(args, **kwargs):
    return run_module('efficient_vlm_ad.cli', args, **kwargs)


def checked_cli(args, **kwargs):
    result = run_cli(args, **kwargs)
    assert result.returncode == 0, result.stdout + '\n' + result.stderr
    return result


def run_isolated_workflow(test_file, test_name):
    """One child imports Torch once and executes the complete real workflow."""
    root = Path(__file__).resolve().parents[1]
    node = Path(test_file).resolve().relative_to(root).as_posix() + '::' + test_name
    if os.environ.get('HUY_TEST_WORKFLOW_CHILD') == node:
        return False
    env = dict(os.environ, HUY_TEST_WORKFLOW_CHILD=node,
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    result = subprocess.run([sys.executable, '-m', 'pytest', node, '-q'], cwd=root,
                            env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + '\n' + result.stderr
    return True
