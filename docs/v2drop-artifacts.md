# v2drop artifact inventory

## Removed artifacts

- `configs/legacy_vit_t5_base.yaml`
- `configs/repvit_t5_efficient_mini.yaml`
- `configs/repvit_t5_efficient_mini_kaggle_2gpu.yaml`
- `configs/repvit_t5_efficient_mini_kaggle_2gpu_di_adapter.yaml`
- `configs/repvit_t5_efficient_mini_kaggle_2gpu_end2end.yaml`
- `configs/repvit_t5_efficient_mini_kaggle_2gpu_router.yaml`
- `configs/repvit_t5_efficient_mini_kaggle_2gpu_router_vsa_k49.yaml`
- `configs/repvit_t5_efficient_mini_kaggle_2gpu_router_vsa_k49_safe.yaml`
- `configs/repvit_t5_efficient_mini_kaggle_2gpu_router_vsa_nonorm_k49.yaml`
- `configs/repvit_t5_efficient_mini_kaggle_2gpu_router_vsa_nonorm_k49_safe.yaml`
- `configs/repvit_t5_efficient_tiny_smoke.yaml`
- `notebooks/kaggle_debug_repvit_t5_efficient_mini_2gpu_di_adapter.ipynb`
- `notebooks/kaggle_debug_repvit_t5_efficient_mini_2gpu_router.ipynb`
- `notebooks/kaggle_debug_repvit_t5_efficient_mini_2gpu_router_vsa_k49.ipynb`
- `notebooks/kaggle_debug_repvit_t5_efficient_mini_2gpu_router_vsa_nonorm_k49.ipynb`
- `notebooks/kaggle_finetune_only_router_vsa_k49.ipynb`
- `notebooks/kaggle_finetune_only_router_vsa_nonorm_k49.ipynb`
- `notebooks/kaggle_full_train_repvit_t5_efficient_mini_2gpu.ipynb`
- `notebooks/kaggle_full_train_repvit_t5_efficient_mini_2gpu_di_adapter.ipynb`
- `notebooks/kaggle_full_train_repvit_t5_efficient_mini_2gpu_end2end.ipynb`
- `notebooks/kaggle_full_train_repvit_t5_efficient_mini_2gpu_router.ipynb`
- `notebooks/kaggle_full_train_repvit_t5_efficient_mini_2gpu_router_vsa_k49.ipynb`
- `notebooks/kaggle_full_train_repvit_t5_efficient_mini_2gpu_router_vsa_nonorm_k49.ipynb`

## Coverage changes

Removed only profile/notebook assertions and dedicated legacy, GPA, DI adapter, end-to-end and router/VSA model cases from the shared test modules. Obsolete cases referencing already-missing safe profiles and notebooks were removed rather than skipped. No source implementation, data, cache or output directory was removed.

Retained shared projector, feature-cache freezing, numerical reporting, architecture masks/delegation, data/cache, generation, CLI and training infrastructure checks. Shared config references now target a retained MVPruner profile or offline attention smoke profile.

Added both-mode profile checks and parameterized existing MVPruner forward/generation and notebook contract coverage. Offline CLI smoke now covers both attention and l2_variation: temporary local data, feature caching, diagnose, finetune, checkpoint verification, same-stage resume, evaluation and benchmarking. It asserts saved stage-2 score mode and RNG sidecar presence. All fake-model work is CPU FP32, with a four-block fake T5 encoder, six camera views, four tokens per camera, stage-1 budget 18 and stage-2 budget 6.

Retained the two attention notebooks and added matching L2 debug/full notebooks. Both full notebooks export the allocator setting before shell Python/accelerate processes; branch defaults are huy-v2drop. Original bootstrap, prepared/cache separate repository download checks and checkpoint RNG sidecar handling are preserved.

## Retained and added entrypoints

The legacy `scripts/kaggle_bootstrap.py` now targets the retained L2 config and
finetune-only commands, creating a separate debug runtime config/output. Its Git
update uses fast-forward pull instead of discarding local changes. Notebook
bootstrap/interpreter setup is unchanged.

- `configs/mvpruner_attention_smoke.yaml`
- `configs/mvpruner_l2_smoke.yaml`
- `configs/repvit_t5_efficient_mini_kaggle_2gpu_mvpruner.yaml`
- `configs/repvit_t5_efficient_mini_kaggle_2gpu_mvpruner_l2.yaml`
- `notebooks/kaggle_debug_mvpruner_finetune.ipynb`
- `notebooks/kaggle_debug_mvpruner_l2_finetune.ipynb`
- `notebooks/kaggle_train_mvpruner_finetune.ipynb`
- `notebooks/kaggle_train_mvpruner_l2_finetune.ipynb`

## Removed dedicated test cases

### `tests/test_config.py`

- `test_loads_repvit_mini_profile`
- `test_loads_kaggle_2gpu_profiles`
- `test_loads_kaggle_di_adapter_profiles`
- `test_loads_kaggle_router_profile`
- `test_loads_kaggle_router_vsa_profile`
- `test_loads_kaggle_router_vsa_nonorm_profile`
- `test_default_fusion_config_keeps_visual_self_attention_off`
- `test_loads_kaggle_end_to_end_profile`

### `tests/test_notebooks.py`

- `test_smoke_notebook_is_debug_only`
- `test_full_train_notebook_has_no_smoke_limits_by_default`
- `test_refactor_notebook_was_removed_to_avoid_stale_entrypoint`
- `test_router_base_notebooks_are_no_norm_entrypoints`
- `test_end_to_end_debug_notebook_exercises_raw_image_training_path`
- `test_end_to_end_full_notebook_runs_full_epochs_without_smoke_limits`
- `test_di_adapter_debug_notebook_exercises_cached_feature_adapter_path`
- `test_di_adapter_full_notebook_runs_full_epochs_without_smoke_limits`
- `test_router_debug_notebook_exercises_cached_feature_router_path`
- `test_router_vsa_nonorm_debug_notebook_uses_nonorm_config`
- `test_router_full_notebook_runs_full_epochs_without_smoke_limits`
- `test_router_vsa_nonorm_full_notebook_uses_nonorm_config`
- `test_router_vsa_nonorm_finetune_only_notebook_has_debug_and_no_align_train`
- `test_router_vsa_finetune_only_notebook_targets_score_norm_vsa_config`
- `test_resume_verify_notebook_checks_same_stage_resume_cases`

### `tests/test_modeling.py`

- `test_gpa_preserves_token_grid_and_returns_view_weights`
- `test_legacy_profile_uses_identity_projector`
- `test_stage_freeze_policy_for_repvit_mini`
- `test_stage_freeze_policy_trains_visual_adapter_when_present`
- `test_stage_policy_trains_router_instead_of_gpa_when_present`
- `test_dynamic_instruction_adapter_preserves_visual_shape_and_masks_text_padding`
- `test_router_uses_fp32_relaxed_khot_ste_and_native_gather`
- `test_router_without_layer_norms_preserves_raw_selected_token_scale`
- `test_visual_self_attention_block_preserves_shape_and_uses_layerscale`
- `test_router_vsa_uses_score_norm_only_and_gather_first_selection`
- `test_router_eval_is_deterministic_and_tau_decay_uses_optimizer_steps`
- `test_router_model_forward_reports_router_numerics`
- `test_end_to_end_model_forward_runs_vision_encoder_and_reports_numerics`

### `tests/test_smoke_pipeline.py`

- `test_router_offline_smoke_cli_sequence`

