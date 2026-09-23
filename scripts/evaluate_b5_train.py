"""Evaluate every training tuple using an existing B5 checkpoint; never train."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

# Works after copying these scripts into an existing checkout, without reinstalling.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from export_b5_verifier import export, sha256

IMAGE_KEYS = ("reference", "positive", "cf_a", "cf_b")


def read_manifest(manifest, image_root, expected_count):
    with manifest.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = (*IMAGE_KEYS, "modification")
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError("Training manifest is missing required columns")
        rows = list(reader)
    if len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} rows, found {len(rows)}; no truncation")
    for number, row in enumerate(rows, 1):
        if not all(row.get(key, "").strip() for key in required):
            raise ValueError(f"Incomplete training row {number}; no skipping allowed")
    images = {row[key] for row in rows for key in IMAGE_KEYS}
    for name in images:
        path = (image_root / name).resolve()
        if not path.is_relative_to(image_root) or not path.is_file():
            raise ValueError(f"Missing/outside image root: {name}")
    return rows, len(images)


def predict(backbone, model, loader, rows, device, destination):
    import torch

    total = 0
    a_wins = b_wins = both_wins = 0
    records = []
    started = time.monotonic()
    with torch.no_grad(), destination.open("w", encoding="utf-8") as journal:
        for step, batch in enumerate(loader, 1):
            texts = list(batch["modification"])
            values = {
                key: backbone.encode_image(batch[key].to(device), normalize=True)
                for key in IMAGE_KEYS
            }
            modification = backbone.encode_text(texts, normalize=True)
            output = model(values["reference"], modification)
            scores = {
                key: (output.query * values[key]).sum(dim=-1).cpu().tolist()
                for key in IMAGE_KEYS[1:]
            }
            probs = {}
            for key in IMAGE_KEYS[1:]:
                v = model.verify(values[key], output, modification)
                probs[key] = torch.stack((v.preserve, v.edit, v.violation), dim=-1).cpu().tolist()
            for i, text in enumerate(texts):
                row = rows[total]
                if row["modification"] != text:
                    raise ValueError("Manifest order changed during inference")
                win_a = scores["positive"][i] > scores["cf_a"][i]
                win_b = scores["positive"][i] > scores["cf_b"][i]
                record = {
                    "sample_index": total + 1,
                    "manifest_record_number": total + 1,
                    "images": {key: row[key] for key in IMAGE_KEYS},
                    "modification": text,
                    "attribute_type": row.get("attribute_type", "unknown"),
                    **{f"{key}_score": scores[key][i] for key in IMAGE_KEYS[1:]},
                    "cf_a_win": win_a,
                    "cf_b_win": win_b,
                    "verifier": {key: probs[key][i] for key in IMAGE_KEYS[1:]},
                }
                journal.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                records.append(record)
                total += 1
                a_wins += win_a
                b_wins += win_b
                both_wins += win_a and win_b
            if step == 1 or step % 25 == 0 or total == len(rows):
                journal.flush()
                elapsed = time.monotonic() - started
                print(
                    f"Evaluated {total}/{len(rows)} samples | elapsed {elapsed / 60:.1f} min | "
                    f"ETA {(elapsed / total) * (len(rows) - total) / 60:.1f} min",
                    flush=True,
                )
    if total != len(rows):
        raise ValueError(f"Incomplete inference: {total}/{len(rows)}")
    metrics = {
        "counterfactual_query_count": total,
        "cf_a_win_percent": 100 * a_wins / total,
        "cf_b_win_percent": 100 * b_wins / total,
        "both_win_percent": 100 * both_wins / total,
        "hard_negative_error_percent": 100 * (total - both_wins) / total,
    }
    return records, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=45000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--manifest", type=Path, help="Relocated copy of the SAME train CSV; hash must match"
    )
    parser.add_argument(
        "--image-root", type=Path, help="Relocated image folder; keep original relative paths"
    )
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.expected_count <= 0 or args.batch_size <= 0:
        parser.error("Counts must be positive")
    if args.output.exists():
        parser.error("Choose a NEW output directory; existing results are never overwritten")

    import torch

    from pic2word.data import build_pseudo_edit_dataloader
    from pic2word.models import CFPECIRModel, FrozenCLIPBackbone

    run = args.run_dir.resolve()
    checkpoint_path = run / "best_checkpoint.pt"
    checkpoint_hash = sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["variant"] != "B5":
        raise ValueError("The checkpoint must be B5")
    config = checkpoint["config"]
    manifest = (args.manifest or Path(config["data"]["train_manifest"])).resolve()
    image_root = (args.image_root or Path(config["data"]["image_root"])).resolve()
    if config["data"].get("max_samples") is not None:
        raise ValueError(
            "Checkpoint uses max_samples; full-manifest training identity is ambiguous"
        )
    manifest_hash = sha256(manifest)
    original_protocol = json.loads((run / "run_protocol.json").read_text(encoding="utf-8"))
    if manifest_hash != original_protocol["train_manifest_sha256"]:
        raise ValueError("Training manifest differs from the hash recorded during B5 training")
    if args.expected_count != original_protocol["samples"]:
        raise ValueError("Expected count must equal the original training sample count")
    rows, unique_images = read_manifest(manifest, image_root, args.expected_count)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required; activate the environment used to train B5")
    seed = int(config["training"]["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    protocol = {
        "status": "preflight_passed",
        "split": "train",
        "variant": "B5",
        "samples": len(rows),
        "unique_image_paths": unique_images,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_step": checkpoint.get("global_step"),
        "training_seed": seed,
        "manifest": str(manifest),
        "original_training_manifest": config["data"]["train_manifest"],
        "manifest_sha256": manifest_hash,
        "training_manifest_hash_verified": True,
        "image_root": str(image_root),
        "batch_size": args.batch_size,
        "shuffle": False,
        "drop_last": False,
        "num_workers": 0,
        "gradients": False,
        "training_enabled": False,
        "inference_precision": "FP32; same default as evaluate_counterfactual.py; no autocast",
        "device": torch.cuda.get_device_name(0),
        "torch_version": str(torch.__version__),
        "source_run_protocol": original_protocol,
        "source_script_sha256": sha256(Path(__file__)),
        "export_script_sha256": sha256(Path(__file__).with_name("export_b5_verifier.py")),
        "scope": "Training-set evaluation. Not held-out validation or generalization evidence.",
    }
    print(json.dumps(protocol, indent=2), flush=True)
    if args.preflight_only:
        return 0

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inference = output / "inference"
    inference.mkdir()
    protocol_path = inference / "evaluation_protocol.json"

    def save_protocol():
        protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    protocol.update(status="running", started_utc=datetime.now(UTC).isoformat())
    save_protocol()
    try:
        model_cfg = config["model"]
        backbone = FrozenCLIPBackbone.from_pretrained(
            model_name=model_cfg["backbone"],
            pretrained=model_cfg.get("pretrained", "openai"),
            cache_dir=model_cfg.get("cache_dir", "checkpoints/clip"),
            device="cuda",
        )
        model = CFPECIRModel(
            backbone.image_embedding_dim,
            variant="B5",
            slots=int(model_cfg.get("slots", 8)),
            hidden_dim=int(model_cfg.get("hidden_dim", 512)),
        ).to("cuda")
        model.load_state_dict(checkpoint["model"])
        model.eval()
        backbone.eval()
        loader = build_pseudo_edit_dataloader(
            manifest,
            image_root,
            backbone.preprocess,
            batch_size=args.batch_size,
            num_workers=0,
            shuffle=False,
            drop_last=False,
        )
        if loader.dataset.rows != rows:
            raise ValueError("Dataset loader changed manifest rows")
        records, metrics = predict(
            backbone, model, loader, rows, "cuda", inference / "predictions.partial.jsonl"
        )
        if sha256(checkpoint_path) != checkpoint_hash or sha256(manifest) != manifest_hash:
            raise ValueError("Checkpoint or manifest changed during inference")
        scores_path = inference / "constraint_scores.json"
        scores_path.write_text(
            json.dumps(records, ensure_ascii=False, allow_nan=False), encoding="utf-8"
        )
        (inference / "predictions.partial.jsonl").rename(inference / "predictions.jsonl")
        (inference / "metrics.json").write_text(
            json.dumps({"counterfactual_evaluation": metrics}, indent=2), encoding="utf-8"
        )
        protocol.update(
            status="completed",
            finished_utc=datetime.now(UTC).isoformat(),
            predictions_sha256=sha256(scores_path),
            input_hashes_unchanged=True,
        )
        save_protocol()
        result = export(inference, output / "export", expected_count=args.expected_count)
        for head, values in result["metrics"].items():
            print(
                head,
                " ".join(f"{k}={values[k]:.6f}" for k in ("precision", "recall", "f1", "auroc")),
                flush=True,
            )
        package = output / "B5_train45000_results.zip"
        with zipfile.ZipFile(package, "x", zipfile.ZIP_DEFLATED) as archive:
            for path in (output / "export").iterdir():
                if path.is_file():
                    archive.write(path, f"export/{path.name}")
            archive.write(protocol_path, "evaluation_protocol.json")
            archive.write(inference / "metrics.json", "counterfactual_metrics.json")
        print(f"DONE: {package}", flush=True)
        print(
            "TRAIN-set results: 45,000 tuples / 135,000 candidates / 405,000 binary decisions.",
            flush=True,
        )
    except BaseException as error:
        # Preserve a completed inference if only postprocessing fails.
        if protocol["status"] != "completed":
            protocol["status"] = "failed"
        protocol["error"] = str(error)
        save_protocol()
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
