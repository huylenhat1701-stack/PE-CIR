import csv
import json
import runpy
import subprocess
import sys
from collections import Counter
from pathlib import Path

API = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/prepare_cfpe_v2.py"))


def test_attribute_binding_and_ambiguous_captions():
    annotate = API["annotate"]
    assert annotate("a blue floral dress on display") == ("dress", "blue", "floral", "")
    assert annotate("red leather bags") == ("bag", "red", "", "leather")
    for caption in (
        "a dog wearing glasses and a wide dog collar",
        "a well trained dog poses long enough",
        "a red dog with a floral collar",
        "a red floral dog collar",
        "a red and blue floral dress",
        "a red floral dress without sleeves",
        "a red floral dress beside a blue bag",
    ):
        assert annotate(caption) is None, caption


def records():
    return [dict(image=f"{color}_{pattern}_{i}.jpg", caption=f"a {color} {pattern} dress view {i}",
                 signature=API["annotate"](f"a {color} {pattern} dress"))
            for color in ("red", "blue") for pattern in ("floral", "plain") for i in range(4)]


def test_mining_semantics_diversity_and_determinism():
    source = records()
    rows = API["mine"](source, 0, 100, 2)
    assert rows and rows == API["mine"](source, 0, 100, 2)
    lookup = {r["image"]: r["signature"] for r in source}
    assert max(Counter(r["reference"] for r in rows).values()) <= 2
    for row in rows:
        sigs = [lookup[row[k]] for k in API["ROLES"]]
        assert len({row[k] for k in API["ROLES"]}) == 4
        ref, pos, cfa, cfb = sigs
        edit = API["KINDS"].index(row["attribute_type"]) + 1
        assert ref == cfa
        assert all(s[0] == ref[0] for s in sigs)
        assert pos[edit] == cfb[edit] != ref[edit]
        assert all(pos[j] == ref[j] for j in range(1, 4) if j != edit)
        assert sum(cfb[j] != ref[j] for j in range(1, 4) if j != edit) == 1


def test_duplicate_bytes_removed_before_split(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    rows = []
    for i in range(100):
        name = f"{i}.jpg"
        (images / name).write_bytes(str(i // 2).encode())
        rows.append(dict(image=name, caption=f"a red floral dress view {i}"))
    manifest = tmp_path / "source.csv"
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "caption"])
        w.writeheader()
        w.writerows(rows)
    pools, counts = API["read_sources"](manifest, images.resolve(), 0, 0.5)
    # Captions with only numeric suffixes normalize identically: deliberately
    # conservative caption deduplication removes further duplicates.
    assert counts["duplicate_image_bytes"] == 50
    assert len(pools["train"]) + len(pools["val"]) == 1
    assert not {r["sha256"] for r in pools["train"]} & {r["sha256"] for r in pools["val"]}
    assert {API["split_for"](str(i), 0, 0.5) for i in range(100)} == {"train", "val"}


def test_cli_reports_low_yield_and_does_not_overwrite(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "a.jpg").write_bytes(b"fixture image content")
    source = tmp_path / "source.csv"
    source.write_text("image,caption\na.jpg,a blue floral dress\n")
    cmd = [sys.executable, API["__file__"], "--source-manifest", str(source),
           "--image-root", str(images), "--output-dir", str(tmp_path / "output")]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "output/report.json").read_text())
    assert not report["screening_count_target_met"]
    assert report["status"] == "CANDIDATES_ONLY_MANUAL_AUDIT_REQUIRED"
    assert not (tmp_path / "output/train.csv").exists()
    assert subprocess.run(cmd, capture_output=True).returncode != 0


def test_nonempty_mined_splits_are_disjoint(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    rows = []
    for i in range(256):
        color = ("red", "blue")[(i // 64) % 2]
        pattern = ("floral", "plain")[(i // 128) % 2]
        suffix = chr(97 + i // 26) + chr(97 + i % 26)
        name = f"{i}.jpg"
        (images / name).write_bytes(f"unique-content-{i}".encode())
        rows.append(dict(image=name, caption=f"a {color} {pattern} dress view {suffix}"))
    source = tmp_path / "source.csv"
    with source.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "caption"])
        writer.writeheader()
        writer.writerows(rows)
    pools, _ = API["read_sources"](source, images.resolve(), 0, 0.3)
    mined = {s: API["mine"](pool, 0, 500, 2) for s, pool in pools.items()}
    assert all(len(rows) > 0 for rows in mined.values())
    role_images = {s: {r[k] for r in rows for k in API["ROLES"]} for s, rows in mined.items()}
    assert not role_images["train"] & role_images["val"]
