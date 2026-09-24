import json

from efficient_vlm_ad.benchmarking import summarize_latencies
from efficient_vlm_ad.evaluation import metric_display_values, write_predictions


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
