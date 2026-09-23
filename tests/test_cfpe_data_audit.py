import csv
import json
import runpy
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_cfpe_data.py"
API = runpy.run_path(str(SCRIPT))


def row(reference, positive, cf_a, cf_b):
    return dict(reference=reference, positive=positive, cf_a=cf_a, cf_b=cf_b,
                modification="change red to blue", attribute_type="color")


def test_cross_role_overlap_and_aliases(tmp_path):
    train = [row("a.jpg", "b.jpg", "c.jpg", "d.jpg")]
    val = [row("e.jpg", "f.jpg", "./a.jpg", "g.jpg")]
    result = API["diagnose"](train, val, tmp_path.resolve())
    assert result["cross_split"]["shared_image_paths"] == 1
    assert result["cross_split"]["overlap_by_role"]["reference"]["cf_a"] == 1
    assert result["cross_split"]["validation_tuples_with_any_train_image_percent"] == 100
    assert result["missing_images"] == 7


def test_temporal_flag_is_target_specific():
    sample = row("a", "b", "c", "d")
    sample.update(after="long", positive_caption="a dog can pose long enough")
    assert "positive_possible_temporal_size_word" in API["flags"](sample)
    sample["after"] = "blue"
    assert not API["flags"](sample)


def test_cli_preserves_inputs_and_requires_new_output(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    for name in "abcdefgh":
        (images / name).touch()
    for name, sample in (("train", row("a", "b", "c", "d")),
                         ("val", row("e", "f", "g", "h"))):
        with (tmp_path / f"{name}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(sample))
            writer.writeheader()
            writer.writerow(sample)
    before = (tmp_path / "train.csv").read_bytes()
    command = [sys.executable, str(SCRIPT), "--data-dir", str(tmp_path),
               "--image-root", str(images), "--output-dir", str(tmp_path / "audit")]
    subprocess.run(command, check=True, capture_output=True)
    report = json.loads((tmp_path / "audit/report.json").read_text())
    assert report["cross_split"]["shared_image_paths"] == 0
    assert report["missing_images"] == 0
    assert (tmp_path / "train.csv").read_bytes() == before
    assert subprocess.run(command, capture_output=True).returncode != 0
