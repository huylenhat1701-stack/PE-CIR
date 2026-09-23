import csv
import json
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

from PIL import Image


def test_pack_keeps_sources_and_outputs_valid_previews(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    for name in ("r", "p", "a", "b"):
        Image.new("RGB", (800, 400), "red").save(root / f"{name}.png")
    row = dict(reference="r.png", positive="p.png", cf_a="a.png", cf_b="b.png",
               object="dress", split="train", source_csv_line=2)
    source = tmp_path / "audit.csv"
    with source.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    output = tmp_path / "pack.zip"
    script = Path(__file__).resolve().parents[1] / "scripts/pack_cfpe_visual_audit.py"
    command = [sys.executable, str(script), "--audit", str(source),
               "--image-root", str(root), "--output", str(output)]
    subprocess.run(command, check=True, capture_output=True)
    with ZipFile(output) as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read("manifest.json"))
        assert len(manifest["tuples"]) == 1
        for name in manifest["tuples"][0]["preview_paths"].values():
            with archive.open(name) as f, Image.open(f) as image:
                assert image.size == (384, 192)
    with Image.open(root / "r.png") as image:
        assert image.size == (800, 400)
    assert subprocess.run(command, capture_output=True).returncode != 0
