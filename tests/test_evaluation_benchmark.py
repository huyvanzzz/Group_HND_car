import json

from efficient_vlm_ad.benchmarking import summarize_latencies
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
    )

    assert summary["e2e_mean_ms"] == 300.0
    assert summary["generation_mean_ms"] == 200.0
    assert summary["average_generation_ms_per_output_token"] == 20.0


def test_metric_display_values_scales_caption_metrics():
    display = metric_display_values({"BLEU-4": 0.25, "CIDEr": 1.2, "exact_match": 0.5})

    assert display == {"BLEU-4": 25.0, "CIDEr": 120.0, "exact_match": 0.5}


def test_generation_diagnostics_detects_tag_and_repetition_problems(tmp_path):
    rows = [
        {
            "eval_id": 1,
            "prediction": "c2,CAM_FRONT,1,2>, c3,CAM_BACK,3,4>, c3,CAM_BACK,3,4> 438 438 438 438",
            "reference": "<c2,CAM_FRONT,1,2>",
            "generated_token_count": 512,
        },
        {"eval_id": 2, "prediction": "the ego car the ego car the ego car", "reference": "go", "generated_token_count": 8},
    ]

    diagnostics = generation_diagnostics(rows, max_new_tokens=512)

    assert diagnostics["prediction_count"] == 2
    assert diagnostics["missing_open_object_tag_count"] >= 1
    assert diagnostics["duplicate_object_tag_count"] >= 1
    assert diagnostics["repeated_trigram_rate_mean"] > 0
    assert diagnostics["numeric_repetition_sequence_count"] == 1
    assert diagnostics["max_new_token_hit_count"] == 1

    path = tmp_path / "generation_diagnostics.json"
    assert write_generation_diagnostics(path, rows, max_new_tokens=512) == diagnostics
    assert path.exists()
