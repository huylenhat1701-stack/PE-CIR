import csv
import json
import runpy
import subprocess
import sys
from collections import Counter
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
API = runpy.run_path(str(SCRIPTS / "prepare_cfpe_v3.py"))


def test_real_caption_examples():
    annotate = API["annotate_object"]
    accepted = {
        "white bus driving along the asphalt road at sunset": ("bus", "white", "", ""),
        "elegant : person was wearing a plunging black dress for an evening out": ("dress", "black", "", ""),
        "a red tulip in close up": ("tulip", "red", "", ""),
        "handmade leather earrings created for person": ("earring", "", "", "leather"),
        "an old wooden sleigh waits for the new snow fall against a barn": ("sleigh", "", "", "wooden"),
        "i love a red door on a farmhouse !": ("door", "red", "", ""),
        "wooden bucket above the well and sunlight on the water": ("bucket", "", "", "wooden"),
    }
    for caption, expected in accepted.items():
        assert annotate(caption) == expected, caption
    for caption in (
        "frog on a white background",
        "old violin , isolated on a white background",
        "a wild mushroom on a blue background",
        "a black and white photo of a female worked in an industrial factory",
        "a yellow and brown turtle on a rock",
        "a herd of horses white yellow blue and orange",
        "a dog can pose long enough",
        "a red dog collar",
        "a red dress with blue shoes",
        "not a red bus",
    ):
        assert annotate(caption) is None, caption


def test_synonyms_and_explicit_preserved_attributes():
    assert API["annotate_object"]("a black puppy")[0] == "dog"
    assert API["annotate_object"]("a red bike")[0] == "bicycle"
    records = []
    for obj in ("dress", "shirt"):
        for color in ("red", "blue"):
            for pattern in ("", "floral"):
                for i in range(3):
                    caption = f"a {color} {pattern} {obj}"
                    records.append(dict(image=f"{obj}_{color}_{pattern}_{i}.jpg", caption=caption,
                                        signature=API["annotate_object"](caption)))
    rows = API["mine_object"](records, 0, 1000, 2)
    assert rows == API["mine_object"](records, 0, 1000, 2)
    lookup = {r["image"]: r["signature"] for r in records}
    assert rows and max(Counter(r["reference"] for r in rows).values()) <= 2
    for row in rows:
        ref, pos, cfa, cfb = [lookup[row[r]] for r in API["ROLES"]]
        assert len({row[r] for r in API["ROLES"]}) == 4
        assert ref[0] == pos[0] == cfa[0] != cfb[0]
        assert ref[1] == cfa[1] != pos[1] == cfb[1]
        if ref[2]:
            assert pos[2] == cfa[2] == ref[2]


def test_cli_mines_disjoint_single_attribute_pilot(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    source = tmp_path / "captions.csv"
    with source.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "caption"])
        writer.writeheader()
        for i in range(256):
            obj = ("dress", "shirt")[(i // 128) % 2]
            color = ("red", "blue")[(i // 64) % 2]
            suffix = chr(97 + i // 26) + chr(97 + i % 26)
            name = f"{i}.jpg"
            (images / name).write_bytes(f"image-{i}".encode())
            writer.writerow(dict(image=name, caption=f"a {color} {obj} view {suffix}"))
    command = [sys.executable, str(SCRIPTS / "prepare_cfpe_v3.py"),
               "--source-manifest", str(source), "--image-root", str(images),
               "--output-dir", str(tmp_path / "out")]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "out/report.json").read_text())
    assert report["dataset_profile"] == "object_type_preservation_v3"
    assert report["status"] == "CANDIDATES_ONLY_MANUAL_AUDIT_REQUIRED"
    assert all(report["splits"][s]["tuples"] > 0 for s in ("train", "val"))
    sets = {}
    for split in ("train", "val"):
        with (tmp_path / f"out/{split}_candidates.csv").open() as f:
            sets[split] = {r[k] for r in csv.DictReader(f) for k in API["ROLES"]}
    assert not sets["train"] & sets["val"]
    assert not (tmp_path / "out/train.csv").exists()
    assert subprocess.run(command, capture_output=True).returncode != 0
