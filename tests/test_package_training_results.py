import json
import runpy
from pathlib import Path
from zipfile import ZipFile

import pytest
import yaml

API = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/package_training_results.py"))


def test_inventory_does_not_fake_missing_artifacts(tmp_path):
    run = tmp_path / "B5"
    run.mkdir()
    (run / "per_query_predictions.json").write_text("[]")
    (run / "git_commit.txt").write_text("unknown")
    (run / "best_checkpoint.pt").write_bytes(b"test checkpoint payload")
    (run / "run_protocol.json").write_text('{"manual_go_override":true}')
    mining = tmp_path / "mining"
    mining.mkdir()
    original = '{"seed":0,"max_tuples":50000}'
    (mining / "mining_config.json").write_text(original)
    output = tmp_path / "results.zip"
    report = API["package"]([run], mining, output)
    files = report["runs"]["B5"]["files"]
    assert files["per_query_predictions.json"]["status"] == "empty_placeholder"
    assert files["git_commit.txt"]["status"] == "unknown_or_invalid_commit"
    assert files["metrics.json"]["status"] == "missing"
    with ZipFile(output) as archive:
        assert archive.testzip() is None
        assert "runs/B5/metrics.json" not in archive.namelist()
        assert yaml.safe_load(archive.read("mining/mining_config.yaml")) == json.loads(original)
        assert json.loads(archive.read("runs/B5/run_protocol.json"))["manual_go_override"]
    assert (mining / "mining_config.json").read_text() == original
    assert not (mining / "mining_config.yaml").exists()
    with pytest.raises(FileExistsError):
        API["package"]([run], mining, output)
