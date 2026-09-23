"""Package the unchanged training manifest, referenced images and runtime metadata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
import zipfile
from pathlib import Path

KEYS = ("reference", "positive", "cf_a", "cf_b")


def inspect_data(manifest, image_root, protocol):
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if digest != protocol["train_manifest_sha256"]:
        raise ValueError("Manifest hash differs from original B5 training record")
    with manifest.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not {*KEYS, "modification"}.issubset(reader.fieldnames or []):
            raise ValueError("Missing pseudo-edit columns")
        rows = list(reader)
    if len(rows) != protocol["samples"]:
        raise ValueError("Training row count differs from original run")
    if any(not all(row.get(k, "").strip() for k in (*KEYS, "modification")) for row in rows):
        raise ValueError("Incomplete row; cannot silently drop training samples")
    names = sorted({row[k] for row in rows for k in KEYS})
    for name in names:
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"Image must use a safe relative path: {name}")
        path = (image_root / name).resolve()
        if not path.is_relative_to(image_root) or not path.is_file():
            raise ValueError(f"Invalid/missing image: {name}")
    return names, digest, sum((image_root / name).stat().st_size for name in names)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--b5-run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--preflight-only", action="store_true")
    a = p.parse_args()
    manifest, root = a.manifest.resolve(), a.image_root.resolve()
    protocol = json.loads((a.b5_run / "run_protocol.json").read_text(encoding="utf-8"))
    names, digest, total = inspect_data(manifest, root, protocol)
    print(
        f"Training samples: {protocol['samples']}; unique image paths: {len(names)}; "
        f"image bytes: {total:,} ({total / 2**30:.2f} GiB)",
        flush=True,
    )
    packages = {
        d.metadata["Name"]: d.version
        for d in importlib.metadata.distributions()
        if d.metadata.get("Name")
    }
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "capture_note": "Environment at handoff time; not proof of historical training versions",
        "packages": dict(sorted(packages.items())),
    }
    try:
        import torch

        info.update(
            torch=str(torch.__version__),
            cuda_runtime=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        )
    except ImportError:
        info["torch"] = "unavailable in the active environment"
    if a.preflight_only:
        print(json.dumps({k: v for k, v in info.items() if k != "packages"}, indent=2))
        return 0
    if a.output.exists():
        p.error("Output already exists; choose a new file")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    partial = a.output.with_name(a.output.name + ".partial")
    with zipfile.ZipFile(partial, "x", zipfile.ZIP_STORED, allowZip64=True) as z:
        z.write(manifest, "PE-CIR_B4_B5/data/train.csv")
        inventory = []
        for i, name in enumerate(names, 1):
            path = root / name
            archive_name = "PE-CIR_B4_B5/data/images/" + Path(name).as_posix()
            # Hash exactly the bytes included in the archive.
            h = hashlib.sha256()
            with path.open("rb") as source, z.open(archive_name, "w", force_zip64=True) as dest:
                while chunk := source.read(1024 * 1024):
                    h.update(chunk)
                    dest.write(chunk)
            inventory.append({"image": name, "sha256": h.hexdigest(), "bytes": path.stat().st_size})
            if i % 1000 == 0 or i == len(names):
                print(f"Packed {i}/{len(names)} image files", flush=True)
        if hashlib.sha256(manifest.read_bytes()).hexdigest() != digest:
            raise ValueError("Manifest changed while packaging")
        z.writestr("PE-CIR_B4_B5/environment_server.json", json.dumps(info, indent=2))
        # Pin core runtime versions without leaking index URLs or credentials.
        names_to_pin = (
            "numpy",
            "Pillow",
            "PyYAML",
            "torch",
            "torchvision",
            "open-clip-torch",
            "timm",
            "ftfy",
            "regex",
            "huggingface-hub",
            "safetensors",
            "tqdm",
        )
        pins = []
        for name in names_to_pin:
            try:
                pins.append(f"{name}=={importlib.metadata.version(name)}")
            except importlib.metadata.PackageNotFoundError:
                pass
        z.writestr("PE-CIR_B4_B5/constraints-server.txt", "\n".join(pins) + "\n")
        z.writestr(
            "PE-CIR_B4_B5/data/DATA_INVENTORY.json",
            json.dumps(
                {
                    "split": "train",
                    "samples": protocol["samples"],
                    "train_manifest_sha256": digest,
                    "image_files": inventory,
                    "data_changed": False,
                },
                indent=2,
            ),
        )
    partial.rename(a.output)
    print(f"DONE: {a.output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
