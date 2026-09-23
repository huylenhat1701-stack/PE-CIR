"""Metric edge cases independent of model inference."""

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "export_b5_verifier", Path(__file__).parents[1] / "scripts/export_b5_verifier.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_auc_ties_and_reversed_order():
    assert module.auc([0, 1], [0.5, 0.5]) == 0.5
    assert module.auc([0, 1], [0.1, 0.9]) == 1
    assert module.auc([0, 1], [0.9, 0.1]) == 0
    assert module.auc([1, 1], [0.2, 0.8]) is None
    assert module.auc([0, 1, 0, 1], [0.1, 0.5, 0.5, 0.9]) == 0.875


def test_confusion_matrix_threshold_and_f1():
    m = module.summarize([0, 0, 1, 1], [0.1, 0.5, 0.4, 0.9])
    assert m["confusion_matrix"] == [[1, 1], [1, 1]]
    assert m["f1"] == m["precision"] == m["recall"] == 0.5
    assert module.summarize([0, 1], [0.1, 0.2])["precision"] == 0


def test_risk_retains_ties_and_ends_at_error_rate():
    curve = module.risk_curve([0, 1, 1, 0], [0.0, 1.0, 0.5, 0.5])
    assert len(curve) == 2
    assert curve[0]["coverage"] == 0.5
    assert curve[0]["risk"] == 0
    assert curve[-1]["coverage"] == 1
    assert curve[-1]["risk"] == 0.25


def test_export_counts_and_rejects_invalid_input(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    row = {
        "modification": "test",
        "positive_score": 0.8,
        "cf_a_score": 0.3,
        "cf_b_score": 0.2,
        "verifier": {k: list(v) for k, v in module.LABELS.items()},
    }
    source = run / "constraint_scores.json"
    source.write_text(json.dumps([row]))
    with pytest.raises(ValueError, match="exactly"):
        module.export(run, tmp_path / "bad_count")
    result = module.export(run, tmp_path / "good", expected_count=1)
    assert result["metadata"]["binary_decisions"] == 9
    assert result["metrics"]["micro"]["accuracy"] == 1
    assert result["metrics"]["micro"]["n"] == 9
    assert (tmp_path / "good/predictions_5000.json").read_bytes() == source.read_bytes()
    row["verifier"]["positive"][0] = float("nan")
    source.write_text(json.dumps([row]))
    with pytest.raises(ValueError, match="Invalid verifier"):
        module.export(run, tmp_path / "bad_score", expected_count=1)
    assert not (tmp_path / "bad_score").exists()
