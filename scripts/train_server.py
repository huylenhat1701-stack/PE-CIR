"""Validate a single-GPU server run and launch CF-PE-CIR training."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parents[1]
COLUMNS = ("reference", "positive", "cf_a", "cf_b", "modification")


def validate_data(manifest: Path, root: Path, batch_size: int) -> int:
    if batch_size < 2:
        raise ValueError("Batch size must be at least 2 for contrastive training")
    if not root.is_dir():
        raise FileNotFoundError(f"Image root not found: {root}")
    root = root.resolve()
    checked = set()
    count = 0
    with manifest.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(COLUMNS).issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest requires columns: {', '.join(COLUMNS)}")
        for line, row in enumerate(reader, 2):
            if any(not (row.get(key) or "").strip() for key in COLUMNS):
                raise ValueError(f"Incomplete pseudo-edit at CSV line {line}")
            for key in COLUMNS[:-1]:
                path = (root / row[key]).resolve()
                if not path.is_relative_to(root):
                    raise ValueError(f"Image escapes image root at line {line}: {row[key]}")
                if path not in checked:
                    if not path.is_file():
                        raise FileNotFoundError(f"Missing image at line {line}: {path}")
                    checked.add(path)
            count += 1
    if count < batch_size:
        raise ValueError(f"Only {count} samples for batch size {batch_size}: zero training batches")
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("B1", "B4"), default="B1")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 0 or args.epochs < 1 or (
        args.max_steps is not None and args.max_steps < 1
    ):
        parser.error("workers must be nonnegative; epochs and max-steps must be positive")
    manifest, root, run_dir = (
        args.manifest.resolve(), args.image_root.resolve(), args.run_dir.resolve()
    )
    if run_dir.exists():
        raise FileExistsError(f"Choose a new run directory; already exists: {run_dir}")
    samples = validate_data(manifest, root, args.batch_size)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Install GPU-compatible PyTorch and check nvidia-smi")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Select exactly one GPU using CUDA_VISIBLE_DEVICES before launching")
    # Exercise the CUDA runtime before allocating a run directory or downloading CLIP.
    torch.ones(1, device="cuda").add_(1)
    torch.cuda.synchronize()
    report = {
        "gpu": torch.cuda.get_device_name(0),
        "vram_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "samples": samples,
        "batches_per_epoch": samples // args.batch_size,
        "batch_size": args.batch_size,
        "variant": args.variant,
    }
    print(json.dumps(report, indent=2), flush=True)
    if args.preflight_only:
        return 0
    config = yaml.safe_load(
        (PROJECT / "configs" / f"cfpe_{args.variant.lower()}.yaml").read_text(encoding="utf-8")
    )
    config["training"].update(
        batch_size_per_device=args.batch_size, num_workers=args.workers,
        epochs=args.epochs, seed=args.seed,
    )
    if args.max_steps is not None:
        config["training"]["max_steps"] = args.max_steps
    config["data"].update(train_manifest=str(manifest), image_root=str(root))
    config["output"]["run_dir"] = str(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    config_path = run_dir / "server_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (run_dir / "server_preflight.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    return subprocess.run(
        [sys.executable, "-u", str(PROJECT / "scripts" / "train_cfpe.py"),
         "--config", str(config_path), "--device", "cuda"],
        cwd=PROJECT, env=env, check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
