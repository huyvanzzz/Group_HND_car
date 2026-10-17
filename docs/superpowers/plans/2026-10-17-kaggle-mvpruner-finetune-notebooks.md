# Kaggle MVPruner Finetune Notebooks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one debug and one full Kaggle notebook for direct MVPruner finetuning without an align stage.

**Architecture:** Both notebooks are self-contained Kaggle entrypoints built around the tracked MVPruner profile. The debug notebook isolates artifacts with a copied runtime config; the full notebook uses the tracked config and optionally resumes a same-stage finetune checkpoint.

**Tech Stack:** Jupyter notebook JSON, Bash cells, Python, Accelerate, Transformers 4.46.3, pytest.

## Global Constraints

- Keep `runtime.precision: fp32` and launch Accelerate with `--mixed_precision no`.
- Never run `--stage align` or require an align checkpoint.
- Install `.[mvpruner]` and fail fast unless Transformers is exactly 4.46.3.
- Load `HF_TOKEN` only from Kaggle Secrets.

---

### Task 1: Notebook contract tests

**Files:**
- Modify: `tests/test_notebooks.py`

**Interfaces:**
- Consumes: `_notebook_source(path)`.
- Produces: assertions for both new notebook entrypoints.

- [ ] Add tests checking config name, FP32 launch, finetune-only commands, smoke limits, full-run absence of limits, and optional same-stage resume.
- [ ] Run the two tests and confirm they fail because the notebooks do not exist.

### Task 2: Kaggle notebooks

**Files:**
- Create: `notebooks/kaggle_debug_mvpruner_finetune.ipynb`
- Create: `notebooks/kaggle_train_mvpruner_finetune.ipynb`

**Interfaces:**
- Consumes: `efficient_vlm_ad` CLI and the tracked MVPruner YAML profile.
- Produces: executable Kaggle debug and full-training workflows.

- [ ] Add environment, clone/install, version validation, data/cache, diagnose/train, evaluation/benchmark and artifact cells.
- [ ] Validate both files as JSON and run the notebook tests.
- [ ] Run the complete notebook test module while excluding only the three intentionally deleted legacy notebooks.
