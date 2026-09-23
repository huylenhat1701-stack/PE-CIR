"""Full-manifest inference guards, ordering and final partial batch."""

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("evaluate_b5_train", SCRIPTS / "evaluate_b5_train.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_manifest_requires_all_rows_and_safe_images(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "a.jpg").write_bytes(b"placeholder")
    manifest = tmp_path / "train.csv"
    row = {key: "a.jpg" for key in module.IMAGE_KEYS}
    row["modification"] = "change color"
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        w.writeheader()
        w.writerow(row)
    rows, count = module.read_manifest(manifest, images, 1)
    assert rows == [row] and count == 1
    with pytest.raises(ValueError, match="no truncation"):
        module.read_manifest(manifest, images, 2)
    (images / "a.jpg").unlink()
    with pytest.raises(ValueError, match="Missing/outside"):
        module.read_manifest(manifest, images, 1)


def test_predict_retains_tail_and_image_identity(tmp_path):
    class Backbone:
        def encode_image(self, images, normalize):
            return images

        def encode_text(self, texts, normalize):
            return torch.ones(len(texts), 2)

    class Model:
        def __call__(self, reference, modification):
            return SimpleNamespace(query=reference)

        def verify(self, candidates, output, modification):
            assert not torch.is_grad_enabled()
            p = torch.full((len(candidates),), 0.8)
            return SimpleNamespace(preserve=p, edit=p, violation=1 - p)

    rows = [
        {**{key: f"{i}_{key}.jpg" for key in module.IMAGE_KEYS}, "modification": f"text {i}"}
        for i in range(3)
    ]
    batches = [
        {
            **{key: torch.ones(count, 2) for key in module.IMAGE_KEYS},
            "modification": [r["modification"] for r in rows[start : start + count]],
        }
        for start, count in [(0, 2), (2, 1)]
    ]
    journal = tmp_path / "partial.jsonl"
    records, metrics = module.predict(Backbone(), Model(), batches, rows, "cpu", journal)
    assert len(records) == metrics["counterfactual_query_count"] == 3
    assert records[-1]["sample_index"] == 3
    assert records[-1]["images"]["positive"] == "2_positive.jpg"
    assert len(journal.read_text().splitlines()) == 3
    assert json.loads(journal.read_text().splitlines()[-1]) == records[-1]
    assert metrics["both_win_percent"] == 0  # similarity ties are not wins
    with pytest.raises(ValueError, match="Incomplete inference"):
        module.predict(Backbone(), Model(), batches[:1], rows, "cpu", journal)
