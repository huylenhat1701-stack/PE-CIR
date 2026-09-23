"""Run B5 with an explicit manual GO override while preserving evaluation evidence."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import yaml


def read_run(path, expected):
    config = yaml.safe_load((path / "config.yaml").read_text(encoding="utf-8"))
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    if config["model"]["variant"] != expected or metrics.get("variant") != expected:
        raise ValueError(f"Expected a {expected} run: {path}")
    return config, metrics["counterfactual_evaluation"]


def prepare(b1_run, b4_run, run_dir, override):
    c1, m1 = read_run(b1_run, "B1")
    c4, m4 = read_run(b4_run, "B4")
    for metrics in (m1, m4):
        if int(metrics["counterfactual_query_count"]) <= 0:
            raise ValueError("Evaluation has no samples")
        for key in ("both_win_percent", "hard_negative_error_percent"):
            value = float(metrics[key])
            if not math.isfinite(value) or not 0 <= value <= 100:
                raise ValueError(f"Invalid evaluation metric: {key}")
    settings_match = c1["data"] == c4["data"] and all(
        c1["training"].get(k) == c4["training"].get(k)
        for k in ("seed", "epochs", "batch_size_per_device"))
    # Existing evaluator does not record validation manifest identity; preserve
    # that limitation instead of claiming verified comparable validation.
    gate = (m4["both_win_percent"] > m1["both_win_percent"] and
            m4["hard_negative_error_percent"] < m1["hard_negative_error_percent"])
    if not override:
        raise ValueError("Exploratory launcher requires explicit --override-go")
    config = copy.deepcopy(c4)
    config["model"]["variant"] = "B5"
    config["training"].setdefault("loss_weights", {})["verifier"] = 0.5
    config["output"]["run_dir"] = str(run_dir.resolve())
    protocol = dict(
        training_enabled=True, manual_go_override=True, measured_metric_gate_passed=gate,
        purpose="Exploratory B5 requested by user; not a claim that B4 passed the scientific GO gate.",
        initialization="fresh B5 model; no B4 checkpoint warm start",
        source_B1=str(b1_run.resolve()), source_B4=str(b4_run.resolve()),
        B1_counterfactual_evaluation=m1, B4_counterfactual_evaluation=m4,
        recorded_train_settings_match=settings_match,
        validation_manifest_identity_verified=False,
        data_note="Reuses B4 data. Previously identified label noise and train/val image overlap are not fixed by this run.",
    )
    return config, protocol


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--b1-run", type=Path, required=True)
    p.add_argument("--b4-run", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--override-go", action="store_true")
    p.add_argument("--preflight-only", action="store_true")
    a = p.parse_args()
    if a.run_dir.exists():
        p.error("Choose a new run directory")
    config, protocol = prepare(a.b1_run, a.b4_run, a.run_dir, a.override_go)
    from train_server import validate_data
    import torch
    manifest = Path(config["data"]["train_manifest"]).resolve()
    image_root = Path(config["data"]["image_root"]).resolve()
    batch = int(config["training"]["batch_size_per_device"])
    samples = validate_data(manifest, image_root, batch)
    if config["data"].get("max_samples") is not None:
        samples = min(samples, int(config["data"]["max_samples"]))
    if samples < batch:
        raise ValueError("No full training batches")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Select one working GPU with CUDA_VISIBLE_DEVICES=0")
    torch.ones(1, device="cuda").add_(1)
    torch.cuda.synchronize()
    epochs = int(config["training"]["epochs"])
    expected_steps = (samples // batch) * epochs
    if config["training"].get("max_steps"):
        expected_steps = min(expected_steps, int(config["training"]["max_steps"]))
    protocol.update(train_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                    gpu=torch.cuda.get_device_name(0), samples=samples, batch_size=batch,
                    epochs=epochs, expected_steps=expected_steps)
    print(json.dumps(protocol, indent=2), flush=True)
    if a.preflight_only:
        return 0
    a.run_dir.mkdir(parents=True, exist_ok=False)
    config_path = a.run_dir.resolve() / "b5_override_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (a.run_dir / "run_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    script = Path(__file__).resolve().parent / "train_cfpe.py"
    return subprocess.run([sys.executable, "-u", str(script), "--config", str(config_path),
                           "--device", "cuda"], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
