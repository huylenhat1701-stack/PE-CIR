import json
import runpy
from pathlib import Path

import pytest
import yaml

PREPARE = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                            "scripts/train_b5_exploratory.py"))["prepare"]


def runs(root):
    for name, score in (("B1", 77.54), ("B4", 76.58)):
        directory = root / name
        directory.mkdir()
        config = dict(model=dict(variant=name), data=dict(train_manifest="train.csv", image_root="images"),
                      training=dict(seed=0, epochs=5, batch_size_per_device=16,
                                    loss_weights=dict(counterfactual=1.0, factorization=0.2, verifier=0)),
                      output=dict(run_dir=str(directory)))
        (directory / "config.yaml").write_text(yaml.safe_dump(config))
        (directory / "metrics.json").write_text(json.dumps(dict(variant=name,
            counterfactual_evaluation=dict(counterfactual_query_count=5000,
                both_win_percent=score, hard_negative_error_percent=100-score))))


def test_override_does_not_falsify_gate_or_change_source(tmp_path):
    runs(tmp_path)
    before = (tmp_path / "B4/config.yaml").read_bytes()
    config, protocol = PREPARE(tmp_path / "B1", tmp_path / "B4", tmp_path / "B5", True)
    assert protocol["training_enabled"] and protocol["manual_go_override"]
    assert not protocol["measured_metric_gate_passed"]
    assert not protocol["validation_manifest_identity_verified"]
    assert config["model"]["variant"] == "B5"
    assert config["training"]["loss_weights"] == dict(factorization=0.2, counterfactual=1.0, verifier=0.5)
    assert (tmp_path / "B4/config.yaml").read_bytes() == before
    assert not (tmp_path / "B5").exists()


def test_override_is_explicit(tmp_path):
    runs(tmp_path)
    with pytest.raises(ValueError, match="override-go"):
        PREPARE(tmp_path / "B1", tmp_path / "B4", tmp_path / "B5", False)
