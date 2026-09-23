import csv
import hashlib
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "package_handoff_data", Path(__file__).parents[1] / "scripts/package_handoff_data.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_package_inspection_preserves_count_hash_and_deduplicates_images(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"a")
    manifest = tmp_path / "train.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*module.KEYS, "modification"])
        writer.writeheader()
        row = {key: "a.jpg" for key in module.KEYS}
        row["modification"] = "change color"
        writer.writerows([row, row])
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    protocol = {"samples": 2, "train_manifest_sha256": digest}
    names, actual, size = module.inspect_data(manifest, tmp_path, protocol)
    assert names == ["a.jpg"] and actual == digest and size == 1
    with pytest.raises(ValueError, match="row count"):
        module.inspect_data(manifest, tmp_path, {**protocol, "samples": 3})
    with pytest.raises(ValueError, match="hash differs"):
        module.inspect_data(manifest, tmp_path, {**protocol, "train_manifest_sha256": "wrong"})
