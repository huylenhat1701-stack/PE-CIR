"""Run one existing B1/B4/B5 checkpoint on all 45,000 training tuples."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluate_b5_train import read_manifest
from export_b5_verifier import sha256

RUNS = {
    "B1": "B1_server_20260921-155218",
    "B4": "B4_server_20260921-175002",
    "B5": "B5_override_20260922-085644",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=RUNS)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.batch_size < 1:
        parser.error("Batch size must be positive")
    run = (args.run_dir or ROOT / "runs" / RUNS[args.variant]).resolve()
    output = (args.output or ROOT / "reports" / f"{args.variant}_train45000").resolve()
    if output.exists():
        parser.error("Output already exists. Choose a new --output directory.")
    import torch

    checkpoint_path = run / "best_checkpoint.pt"
    checkpoint_hash = sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["variant"] != args.variant:
        raise ValueError("Checkpoint variant does not match requested variant")
    data = checkpoint["config"]["data"]
    if data.get("max_samples") is not None:
        raise ValueError("Checkpoint training used max_samples; inspect its training scope first")
    manifest = (args.manifest or Path(data["train_manifest"])).resolve()
    image_root = (args.image_root or Path(data["image_root"])).resolve()
    rows, unique_images = read_manifest(manifest, image_root, 45000)
    manifest_hash = sha256(manifest)
    source_path = run / "run_protocol.json"
    source = json.loads(source_path.read_text()) if source_path.exists() else {}
    recorded_hash = source.get("train_manifest_sha256")
    if recorded_hash and recorded_hash != manifest_hash:
        raise ValueError("Manifest hash differs from the training protocol")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Activate the training environment first.")
    protocol = {
        "variant": args.variant, "split": "train", "samples": len(rows),
        "unique_image_paths": unique_images, "manifest": str(manifest),
        "manifest_sha256": manifest_hash,
        "training_manifest_hash_verified": bool(recorded_hash),
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_hash,
        "batch_size": args.batch_size, "gpu": torch.cuda.get_device_name(0),
        "scope": "Training-set Stage-1 counterfactual evaluation, before rerank; not held-out validation.",
    }
    print(json.dumps(protocol, indent=2), flush=True)
    if args.variant == "B5":
        command = [sys.executable, "-u", str(ROOT / "scripts/evaluate_b5_train.py"),
                   "--run-dir", str(run), "--output", str(output),
                   "--manifest", str(manifest), "--image-root", str(image_root),
                   "--batch-size", str(args.batch_size)]
        if args.preflight_only:
            command.append("--preflight-only")
        subprocess.run(command, check=True)
        return
    if args.preflight_only:
        print("PREFLIGHT OK; no inference performed.")
        return
    output.mkdir(parents=True, exist_ok=False)
    protocol_path = output / "evaluation_protocol.json"
    def save():
        protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    protocol.update(status="running", started_utc=datetime.now(UTC).isoformat())
    save()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    command = [sys.executable, "-u", str(ROOT / "scripts/evaluate_counterfactual.py"),
               "--manifest", str(manifest), "--image-root", str(image_root),
               "--checkpoint", str(checkpoint_path), "--run-dir", str(output),
               "--batch-size", str(args.batch_size), "--device", "cuda"]
    try:
        print("Running 45,000 samples. B1/B4 print metrics only when finished.", flush=True)
        subprocess.run(command, check=True, env=env)
        metrics = json.loads((output / "metrics.json").read_text())["counterfactual_evaluation"]
        if metrics["counterfactual_query_count"] != 45000:
            raise ValueError("Incomplete evaluation")
        if sha256(manifest) != manifest_hash or sha256(checkpoint_path) != checkpoint_hash:
            raise ValueError("Input files changed during inference")
        protocol.update(status="completed", finished_utc=datetime.now(UTC).isoformat(),
                        input_hashes_unchanged=True)
        save()
        print(f"DONE: {output}", flush=True)
    except BaseException as error:
        protocol.update(status="failed", error=str(error))
        save()
        raise


if __name__ == "__main__":
    main()
