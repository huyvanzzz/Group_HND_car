import json
import pytest

from efficient_vlm_ad.benchmarking import summarize_component_events, summarize_latencies
from efficient_vlm_ad.evaluation import generation_diagnostics, metric_display_values, write_generation_diagnostics, write_predictions


def test_write_predictions_rejects_duplicate_eval_ids(tmp_path):
    path = tmp_path / "predictions.jsonl"

    try:
        write_predictions(
            path,
            [
                {"eval_id": 1, "prediction": "a", "reference": "a"},
                {"eval_id": 1, "prediction": "b", "reference": "b"},
            ],
        )
    except ValueError as exc:
        assert "Duplicate eval_id" in str(exc)
    else:
        raise AssertionError("duplicate eval_id should fail")


def test_write_predictions_outputs_jsonl(tmp_path):
    path = tmp_path / "predictions.jsonl"

    write_predictions(path, [{"eval_id": 1, "prediction": "go", "reference": "go"}])

    assert json.loads(path.read_text(encoding="utf-8").strip())["prediction"] == "go"


def test_summarize_latencies_computes_generation_ms_per_output_token():
    summary = summarize_latencies(
        e2e_seconds=[0.2, 0.4],
        generation_seconds=[0.1, 0.3],
        output_tokens=[5, 15],
        prefill_seconds=[0.01, 0.03],
        prefill_flops=[1e9, 3e9],
    )

    assert summary["e2e_mean_ms"] == 300.0
    assert summary["prefill_mean_ms"] == 20.0
    assert summary["prefill_flops_mean_gflops"] == 2.0
    assert summary["prefill_p50_ms"] == 10.0
    assert summary["prefill_p95_ms"] == 30.0
    assert summary["prefill_flops_p50_gflops"] == 1.0
    assert summary["prefill_flops_p95_gflops"] == 3.0
    assert not any(key in summary for key in ("generation_mean_ms", "generation_p50_ms", "generation_p95_ms"))
    assert summary["average_generation_ms_per_output_token"] == 20.0


def test_summarize_component_events_reports_per_iteration_cpu_and_cuda_time():
    event = type(
        "Event",
        (),
        {
            "key": "mvpruner.stage1_dra_ccts",
            "cpu_time_total": 4000.0,
            "device_time_total": 2000.0,
        },
    )()

    summary = summarize_component_events([event], iterations=2)

    assert summary["mvpruner.stage1_dra_ccts"]["cpu_mean_ms"] == 2.0
    assert summary["mvpruner.stage1_dra_ccts"]["cuda_mean_ms"] == 1.0


def test_summary_rejects_empty_or_mismatched_samples():
    with pytest.raises(ValueError, match="nonempty"):
        summarize_latencies(e2e_seconds=[], generation_seconds=[], output_tokens=[])
    with pytest.raises(ValueError, match="same benchmark samples"):
        summarize_latencies(e2e_seconds=[.1], generation_seconds=[.1], output_tokens=[1],
                            prefill_seconds=[], prefill_flops=[100])


def test_summary_synthetic_metrics_are_null():
    summary = summarize_latencies(e2e_seconds=[.1], generation_seconds=[.05], output_tokens=[2])
    assert all(summary[key] is None for key in (
        "prefill_mean_ms", "prefill_p50_ms", "prefill_p95_ms",
        "prefill_flops_mean_gflops", "prefill_flops_p50_gflops", "prefill_flops_p95_gflops"))


def test_metric_display_values_scales_caption_metrics():
    display = metric_display_values({"BLEU-4": 0.25, "CIDEr": 1.2, "exact_match": 0.5})

    assert display == {"BLEU-4": 25.0, "CIDEr": 120.0, "exact_match": 0.5}


def test_generation_diagnostics_detects_missing_open_object_tags(tmp_path):
    rows = [
        {
            "eval_id": 1,
            "prediction": "c2,CAM_FRONT,1,2>, c3,CAM_BACK,3,4>, c3,CAM_BACK,3,4> 438 438 438 438",
            "reference": "<c2,CAM_FRONT,1,2>",
            "generated_token_count": 512,
        },
        {"eval_id": 2, "prediction": "the ego car the ego car the ego car", "reference": "go", "generated_token_count": 8},
    ]

    diagnostics = generation_diagnostics(rows)

    assert diagnostics["prediction_count"] == 2
    assert diagnostics["missing_open_object_tag_count"] >= 1
    removed_keys = [
        "duplicate_object" + "_tag_count",
        "repeated" + "_trigram_rate_mean",
        "repeated" + "_fourgram_rate_mean",
        "numeric" + "_repetition_sequence_count",
        "max_new_token" + "_hit_count",
    ]
    assert not any(key in diagnostics for key in removed_keys)

    path = tmp_path / "generation_diagnostics.json"
    assert write_generation_diagnostics(path, rows) == diagnostics
    assert path.exists()
