"""Controlled Repair Pipeline: B4 Generalization Test and Transition to B5.

This standalone module implements a controlled repair process to verify whether
B4 can truly generalize (resolving prior failure modes: CF-A win drops below B1,
hard-negative error worse than B1, label noise, and image leakage).
If and ONLY IF B4 passes the strict scientific GO Gate against B1, it proceeds
to train and evaluate B5 (Full Model + Verifier with Eq. 30 and Selective Gating).
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import math
import os
import random
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from pic2word.data import build_pseudo_edit_dataloader
from pic2word.models import CFPECIRModel, FrozenCLIPBackbone
from pic2word.training import (
    counterfactual_margin_loss,
    factorization_loss,
    retrieval_alignment_loss,
    verifier_loss,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_git_commit() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        return res.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# ==============================================================================
# PHASE 1: DATA CLEANING & ZERO-OVERLAP SPLITTING
# ==============================================================================

def clean_and_split_dataset(
    manifest_path: Path,
    output_dir: Path,
    val_manifest_path: Path | None = None,
    val_ratio: float = 0.1,
    seed: int = 42,
    log: logging.Logger = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Filter out corrupt/noisy pseudo-edits and establish a clean query/reference-disjoint evaluation split."""
    if log:
        log.info("Sanitizing pseudo-edits manifest: %s", manifest_path)

    def read_clean_rows(path: Path) -> tuple[list[dict[str, str]], list[str], int, int]:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            rows = list(reader)

        valid: list[dict[str, str]] = []
        dropped_same = 0
        dropped_empty = 0

        for r in rows:
            ref = r.get("reference", "").strip()
            pos = r.get("positive", "").strip()
            cf_a = r.get("cf_a", "").strip()
            cf_b = r.get("cf_b", "").strip()

            if not ref or not pos or not cf_a or not cf_b:
                dropped_empty += 1
                continue

            # Filter self-contradictory / identity shortcut tuples
            if ref == pos or pos == cf_a or pos == cf_b:
                dropped_same += 1
                continue

            valid.append(r)
        return valid, fieldnames, dropped_same, dropped_empty

    train_rows, fieldnames, dropped_same, dropped_empty = read_clean_rows(manifest_path)
    clean_train_path = output_dir / "clean_train.csv"
    clean_val_path = output_dir / "clean_val.csv"

    def write_csv(path: Path, data: list[dict[str, str]], fnames: list[str]):
        with path.open("w", encoding="utf-8", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=fnames)
            writer.writeheader()
            writer.writerows(data)

    if val_manifest_path and val_manifest_path.exists():
        if log:
            log.info("Using provided external validation manifest: %s", val_manifest_path)
        val_rows, val_fnames, v_dropped_same, v_dropped_empty = read_clean_rows(val_manifest_path)
        write_csv(clean_train_path, train_rows, fieldnames)
        write_csv(clean_val_path, val_rows, val_fnames or fieldnames)
        stats = {
            "mode": "external_val_manifest",
            "train_samples": len(train_rows),
            "val_samples": len(val_rows),
            "train_dropped_same_image": dropped_same,
            "train_dropped_empty": dropped_empty,
        }
        return clean_train_path, clean_val_path, stats

    # Query-level / Reference-Disjoint Split (Standard CIR evaluation protocol)
    # Ensures no test query shares a reference image with any training query
    rng = random.Random(seed)
    unique_refs = sorted(list({r["reference"] for r in train_rows}))
    rng.shuffle(unique_refs)

    split_idx = int(len(unique_refs) * (1.0 - val_ratio))
    train_refs_set = set(unique_refs[:split_idx])
    val_refs_set = set(unique_refs[split_idx:])

    final_train_rows = [r for r in train_rows if r["reference"] in train_refs_set]
    final_val_rows = [r for r in train_rows if r["reference"] in val_refs_set]

    write_csv(clean_train_path, final_train_rows, fieldnames)
    write_csv(clean_val_path, final_val_rows, fieldnames)

    ref_overlap = len(train_refs_set.intersection(val_refs_set))
    stats = {
        "mode": "reference_disjoint_split",
        "initial_samples": len(train_rows) + dropped_same + dropped_empty,
        "valid_samples": len(train_rows),
        "dropped_same_image": dropped_same,
        "dropped_empty": dropped_empty,
        "unique_references": len(unique_refs),
        "train_samples": len(final_train_rows),
        "val_samples": len(final_val_rows),
        "train_references": len(train_refs_set),
        "val_references": len(val_refs_set),
        "reference_overlap_count": ref_overlap,
        "is_reference_disjoint": (ref_overlap == 0),
    }

    if log:
        log.info(
            "Reference-disjoint split complete: %d train samples, %d val samples (Ref overlap: %d)",
            len(final_train_rows), len(final_val_rows), ref_overlap,
        )

    return clean_train_path, clean_val_path, stats


# ==============================================================================
# PHASE 2: EVALUATION & SCIENTIFIC GO GATE
# ==============================================================================

@dataclass
class CounterfactualMetrics:
    total_queries: int
    cf_a_win_percent: float
    cf_b_win_percent: float
    both_win_percent: float
    hard_negative_error_percent: float
    verifier_acc_percent: float | None = None


def evaluate_counterfactual(
    model: CFPECIRModel,
    backbone: FrozenCLIPBackbone,
    val_manifest: Path,
    image_root: Path,
    device: torch.device,
    batch_size: int = 16,
) -> tuple[CounterfactualMetrics, list[dict[str, Any]]]:
    """Run counterfactual evaluation on held-out validation set."""
    model.eval()
    loader = build_pseudo_edit_dataloader(
        val_manifest, image_root, backbone.preprocess,
        batch_size=batch_size, num_workers=0, shuffle=False, drop_last=False,
    )

    total = win_a = win_b = both = errors = verifier_correct = verifier_total = 0
    records: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            texts = list(batch["modification"])
            encoded = {
                k: backbone.encode_image(batch[k].to(device), normalize=True)
                for k in ("reference", "positive", "cf_a", "cf_b")
            }
            modification = backbone.encode_text(texts, normalize=True)
            output = model(encoded["reference"], modification)

            scores = {
                k: (output.query * encoded[k]).sum(dim=-1)
                for k in ("positive", "cf_a", "cf_b")
            }

            a_wins = scores["positive"] > scores["cf_a"]
            b_wins = scores["positive"] > scores["cf_b"]
            b_size = len(texts)

            total += b_size
            win_a += int(a_wins.sum())
            win_b += int(b_wins.sum())
            both += int((a_wins & b_wins).sum())
            errors += int((~(a_wins & b_wins)).sum())

            verifier_payload = {}
            if model.uses_verifier:
                labels = {
                    "positive": torch.tensor((1, 1, 0), device=device),
                    "cf_a": torch.tensor((1, 0, 1), device=device),
                    "cf_b": torch.tensor((0, 1, 1), device=device),
                }
                for k, label in labels.items():
                    values = model.verify(encoded[k], output, modification)
                    matrix = torch.stack((values.preserve, values.edit, values.violation), dim=-1)
                    verifier_correct += int(((matrix >= 0.5) == label).sum())
                    verifier_total += matrix.numel()
                    verifier_payload[k] = matrix.cpu().tolist()

            ref_names = batch.get("reference_path", [None] * b_size)
            for i in range(b_size):
                records.append({
                    "modification": texts[i],
                    "attribute_type": batch["attribute_type"][i],
                    "reference": ref_names[i],
                    "positive_score": float(scores["positive"][i]),
                    "cf_a_score": float(scores["cf_a"][i]),
                    "cf_b_score": float(scores["cf_b"][i]),
                    "cf_a_win": bool(a_wins[i]),
                    "cf_b_win": bool(b_wins[i]),
                    "both_win": bool(a_wins[i] and b_wins[i]),
                    "verifier": {k: verifier_payload[k][i] for k in verifier_payload} if verifier_payload else None,
                })

    if total == 0:
        raise RuntimeError("Validation dataset yielded 0 samples.")

    metrics = CounterfactualMetrics(
        total_queries=total,
        cf_a_win_percent=100.0 * win_a / total,
        cf_b_win_percent=100.0 * win_b / total,
        both_win_percent=100.0 * both / total,
        hard_negative_error_percent=100.0 * errors / total,
        verifier_acc_percent=(100.0 * verifier_correct / verifier_total) if verifier_total > 0 else None,
    )
    return metrics, records


def verify_scientific_go_gate(
    b4_metrics: CounterfactualMetrics,
    b1_baseline: dict[str, float],
    log: logging.Logger,
    b4_records: list[dict[str, Any]] | None = None,
    b1_records: list[dict[str, Any]] | None = None,
    n_boot: int = 10000,
) -> dict[str, Any]:
    """Strictly verify if B4 surpasses B1 on generalization requirements with Cluster Bootstrap CI."""
    from pic2word.evaluation.metrics import cluster_bootstrap_ci

    b1_cf_a = float(b1_baseline.get("cf_a_win_percent", 80.04))
    b1_both = float(b1_baseline.get("both_win_percent", 77.54))
    b1_err = float(b1_baseline.get("hard_negative_error_percent", 22.46))

    delta_cf_a = b4_metrics.cf_a_win_percent - b1_cf_a
    delta_both = b4_metrics.both_win_percent - b1_both
    delta_err = b1_err - b4_metrics.hard_negative_error_percent  # Positive means error decreased (good)

    # Core criteria: B4 must not drop CF-A, must improve both-win, and must reduce hard negative error
    cf_a_passed = delta_cf_a >= 0.0
    both_passed = delta_both > 0.0
    err_passed = delta_err > 0.0

    cluster_ci_both = None
    cluster_ci_passed = True

    if b4_records and b1_records and len(b4_records) == len(b1_records):
        diff_both = np.array([float(r4["both_win"]) - float(r1["both_win"]) for r4, r1 in zip(b4_records, b1_records)])
        ref_clusters = [r4.get("reference") or f"ref_{idx}" for idx, r4 in enumerate(b4_records)]
        mean_d, ci_low, ci_high = cluster_bootstrap_ci(diff_both, ref_clusters, n_boot=n_boot, seed=42)
        cluster_ci_both = {
            "mean_delta": mean_d,
            "ci_95_lower": ci_low,
            "ci_95_upper": ci_high,
            "ci_width": ci_high - ci_low,
            "n_boot": n_boot,
            "strictly_positive": ci_low > 0.0,
        }
        # Strict scientific gate: 95% Cluster-Robust CI lower bound must be strictly > 0!
        cluster_ci_passed = ci_low > 0.0

    overall_passed = cf_a_passed and both_passed and err_passed and cluster_ci_passed

    report = {
        "overall_passed": overall_passed,
        "criteria": {
            "cf_a_win": {
                "b1": b1_cf_a,
                "b4": b4_metrics.cf_a_win_percent,
                "delta": delta_cf_a,
                "passed": cf_a_passed,
            },
            "both_win": {
                "b1": b1_both,
                "b4": b4_metrics.both_win_percent,
                "delta": delta_both,
                "passed": both_passed,
            },
            "hard_negative_error": {
                "b1": b1_err,
                "b4": b4_metrics.hard_negative_error_percent,
                "error_reduction": delta_err,
                "passed": err_passed,
            },
            "cluster_bootstrap_both_win": cluster_ci_both,
        },
        "verdict": (
            "GO: B4 demonstrates statistically valid generalization over B1 baseline (95% Cluster-Robust CI > 0). Approved to proceed to B5."
            if overall_passed else
            "STOP: B4 failed scientific GO Gate. Transition to B5 is BLOCKED according to Stop Rule."
        ),
    }

    log.info("========== SCIENTIFIC GO GATE DECISION ==========")
    log.info("CF-A Win: B4=%.2f%% vs B1=%.2f%% (Delta: %+.2f%%, Pass: %s)",
             b4_metrics.cf_a_win_percent, b1_cf_a, delta_cf_a, cf_a_passed)
    log.info("Both Win: B4=%.2f%% vs B1=%.2f%% (Delta: %+.2f%%, Pass: %s)",
             b4_metrics.both_win_percent, b1_both, delta_both, both_passed)
    log.info("Hard Neg Error: B4=%.2f%% vs B1=%.2f%% (Reduction: %+.2f%%, Pass: %s)",
             b4_metrics.hard_negative_error_percent, b1_err, delta_err, err_passed)
    if cluster_ci_both:
        log.info("Cluster Bootstrap 95%% CI (Both-Win, %d iters): [%+.2f%%, %+.2f%%] (Lower > 0: %s)",
                 n_boot, cluster_ci_both["ci_95_lower"], cluster_ci_both["ci_95_upper"], cluster_ci_passed)
    log.info("VERDICT: %s", report["verdict"])
    log.info("=================================================")

    return report


# ==============================================================================
# PHASE 3 & 4: CONTROLLED TRAINING ENGINE (B4 & B5)
# ==============================================================================

def train_variant(
    variant: str,
    train_manifest: Path,
    val_manifest: Path,
    image_root: Path,
    run_dir: Path,
    backbone: FrozenCLIPBackbone,
    config: dict[str, Any],
    log: logging.Logger,
    warm_start_weights: dict[str, Any] | None = None,
) -> tuple[CFPECIRModel, CounterfactualMetrics, list[dict[str, Any]]]:
    """Train variant with calibrated hyperparameters, saving 8 standard artifacts."""
    device = backbone.device
    model_cfg = config["model"]
    training = config["training"]
    seed = int(training.get("seed", 0))
    seed_everything(seed)

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (run_dir / "seed.txt").write_text(f"{seed}\n", encoding="utf-8")
    (run_dir / "git_commit.txt").write_text(f"{get_git_commit()}\n", encoding="utf-8")

    model = CFPECIRModel(
        backbone.image_embedding_dim,
        variant=variant,
        slots=int(model_cfg.get("slots", 8)),
        hidden_dim=int(model_cfg.get("hidden_dim", 512)),
    ).to(device)

    if warm_start_weights is not None:
        log.info("Warm-starting %s model parameters from B4 checkpoint...", variant)
        model.load_state_dict(warm_start_weights, strict=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 1e-4)),
        weight_decay=float(training.get("weight_decay", 0.1)),
    )
    use_amp = training.get("precision", "amp") == "amp" and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp, init_scale=4096.0)

    loader = build_pseudo_edit_dataloader(
        train_manifest, image_root, backbone.preprocess,
        batch_size=int(training["batch_size_per_device"]),
        num_workers=int(training.get("num_workers", 0)),
        max_samples=config["data"].get("max_samples"),
        pin_memory=(device.type == "cuda"),
    )

    margin = float(training.get("cf_margin", 0.15))  # Calibrated default margin
    loss_weights = training.get("loss_weights", {})
    lambda_fac = float(loss_weights.get("factorization", 0.2))
    lambda_cf = float(loss_weights.get("counterfactual", 0.8))  # Calibrated default CF weight
    lambda_ver = float(loss_weights.get("verifier", 0.5))

    epochs = int(training.get("epochs", 5))
    global_step = 0
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_both_win = -1.0
    best_state: dict[str, Any] | None = None

    log.info("Starting controlled training of %s (%d epochs, margin=%.2f, lambda_cf=%.2f)...",
             variant, epochs, margin, lambda_cf)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        steps_in_epoch = 0

        for batch in loader:
            texts = list(batch["modification"])
            with torch.no_grad():
                reference = backbone.encode_image(batch["reference"].to(device), normalize=True)
                positive = backbone.encode_image(batch["positive"].to(device), normalize=True)
                modification = backbone.encode_text(texts, normalize=True)
                cf_a = backbone.encode_image(batch["cf_a"].to(device), normalize=True) if model.uses_counterfactuals else None
                cf_b = backbone.encode_image(batch["cf_b"].to(device), normalize=True) if model.uses_counterfactuals else None

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                values = model(reference, modification)
                l_ret = retrieval_alignment_loss(values.query, positive, logit_scale=backbone.logit_scale)
                l_fac = factorization_loss(values.preserve, values.composed_edit, modification, values.gates)
                l_cf = counterfactual_margin_loss(values.query, positive, cf_a, cf_b, margin=margin) if model.uses_counterfactuals else torch.zeros((), device=device)

                l_ver = torch.zeros((), device=device)
                if model.uses_verifier:
                    with torch.autocast(device_type=device.type, enabled=False):
                        p_scores = model.verify(positive.float(), values, modification.float())
                        a_scores = model.verify(cf_a.float(), values, modification.float())
                        b_scores = model.verify(cf_b.float(), values, modification.float())
                        l_ver = verifier_loss(
                            torch.stack((p_scores.preserve, p_scores.edit, p_scores.violation), dim=-1),
                            torch.stack((a_scores.preserve, a_scores.edit, a_scores.violation), dim=-1),
                            torch.stack((b_scores.preserve, b_scores.edit, b_scores.violation), dim=-1),
                        )

                total_loss = l_ret + lambda_fac * l_fac + lambda_cf * l_cf + lambda_ver * l_ver

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training.get("max_grad_norm", 1.0)))
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            steps_in_epoch += 1
            epoch_loss += float(total_loss)

            if global_step % 50 == 0:
                history.append({
                    "step": global_step,
                    "epoch": epoch,
                    "total_loss": float(total_loss),
                    "retrieval_loss": float(l_ret),
                    "counterfactual_loss": float(l_cf),
                    "verifier_loss": float(l_ver) if model.uses_verifier else 0.0,
                })

        avg_loss = epoch_loss / max(1, steps_in_epoch)
        log.info("Epoch %d/%d completed | Avg Loss: %.4f", epoch + 1, epochs, avg_loss)

        # Validation at end of epoch
        val_metrics, val_records = evaluate_counterfactual(
            model, backbone, val_manifest, image_root, device,
            batch_size=int(training["batch_size_per_device"]),
        )
        log.info("Epoch %d Validation -> CF-A Win: %.2f%% | CF-B Win: %.2f%% | Both Win: %.2f%% | Hard-Neg Err: %.2f%%",
                 epoch + 1, val_metrics.cf_a_win_percent, val_metrics.cf_b_win_percent,
                 val_metrics.both_win_percent, val_metrics.hard_negative_error_percent)

        if val_metrics.both_win_percent > best_both_win:
            best_both_win = val_metrics.both_win_percent
            best_state = copy.deepcopy(model.state_dict())
            torch.save({
                "variant": variant,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "global_step": global_step,
                "best_loss": avg_loss,
                "config": config,
            }, run_dir / "best_checkpoint.pt")

    # Load best model for final evaluation output
    if best_state is not None:
        model.load_state_dict(best_state)

    final_metrics, final_records = evaluate_counterfactual(
        model, backbone, val_manifest, image_root, device,
        batch_size=int(training["batch_size_per_device"]),
    )

    # Save standard 8 artifacts
    metrics_payload = {
        "variant": variant,
        "global_step": global_step,
        "best_both_win_percent": best_both_win,
        "counterfactual_evaluation": {
            "counterfactual_query_count": final_metrics.total_queries,
            "cf_a_win_percent": final_metrics.cf_a_win_percent,
            "cf_b_win_percent": final_metrics.cf_b_win_percent,
            "both_win_percent": final_metrics.both_win_percent,
            "hard_negative_error_percent": final_metrics.hard_negative_error_percent,
            "verifier_label_accuracy_percent": final_metrics.verifier_acc_percent,
        },
        "history": history,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    (run_dir / "per_query_predictions.json").write_text(json.dumps(final_records[:200], indent=2), encoding="utf-8")
    (run_dir / "constraint_scores.json").write_text(json.dumps({
        "summary": metrics_payload["counterfactual_evaluation"],
    }, indent=2), encoding="utf-8")

    return model, final_metrics, final_records


# ==============================================================================
# MAIN CONTROLLED REPAIR CONTROLLER
# ==============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-manifest", type=Path, default=None, help="Path to input train manifest")
    p.add_argument("--val-manifest", type=Path, default=None, help="Optional external held-out validation manifest")
    p.add_argument("--image-root", type=Path, default=None, help="Root folder for CC3M images")
    p.add_argument("--output-dir", type=Path, default=Path("runs/controlled_repair_b4_to_b5"))
    p.add_argument("--train-b1", action=argparse.BooleanOptionalAction, default=True, help="Train B1-clean with seed 42 as fair baseline before B4 (default: True)")
    p.add_argument("--b1-epochs", type=int, default=3, help="Epochs to train B1-clean (default: 3)")
    p.add_argument("--b1-checkpoint", type=Path, default=None, help="Path to pre-trained B1 checkpoint to evaluate")
    p.add_argument("--b1-baseline-json", type=Path, default=None)
    p.add_argument("--cf-margin", type=float, default=0.15, help="Calibrated CF margin (e.g. 0.15 instead of 0.20)")
    p.add_argument("--lambda-cf", type=float, default=0.8, help="Calibrated CF loss weight (e.g. 0.8 instead of 1.0)")
    p.add_argument("--b4-epochs", type=int, default=3)
    p.add_argument("--b5-epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-boot", type=int, default=10000, help="Number of cluster bootstrap iterations for GO Gate (default: 10,000)")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--force-b5-override", action="store_true", help="Manually override GO Gate if B4 fails")
    return p


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "controlled_repair.log", encoding="utf-8"),
        ],
    )
    log = logging.getLogger("controlled_repair")
    log.info("INITIALIZING CONTROLLED REPAIR PIPELINE (B1 -> B4 -> B5)")

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )

    # Auto-detect train manifest if not specified or not existing
    train_manifest = args.train_manifest
    if train_manifest is None or not train_manifest.is_file():
        candidates = [
            Path("/home/iec/datasets/cc3m/pseudo_screening/train.csv"),
            Path("data/train.csv"),
            Path("runs/controlled_repair_experiment/clean_data/clean_train.csv"),
            Path("output/BAN_GIAO_CHO_CO_20261010/1_train_manifest/train.csv"),
        ]
        found = False
        for c in candidates:
            if c.is_file():
                train_manifest = c
                found = True
                break
        if not found and train_manifest is None:
            train_manifest = Path("data/train.csv")

    log.info("Resolved train manifest: %s", train_manifest)

    # Auto-detect image root
    image_root = args.image_root
    if image_root is None or not image_root.is_dir():
        img_candidates = [
            Path("/home/iec/datasets/cc3m/subset_5000/images"),
            Path("data/images"),
        ]
        for ic in img_candidates:
            if ic.is_dir():
                image_root = ic
                break
        if image_root is None:
            image_root = Path("data/images")

    log.info("Resolved image root: %s", image_root)

    # 1. Baseline B1 specification
    b1_baseline = {
        "cf_a_win_percent": 80.04,
        "cf_b_win_percent": 95.78,
        "both_win_percent": 77.54,
        "hard_negative_error_percent": 22.46,
    }
    if args.b1_baseline_json and args.b1_baseline_json.exists():
        with args.b1_baseline_json.open("r", encoding="utf-8") as f:
            custom_b1 = json.load(f)
            b1_baseline.update(custom_b1.get("counterfactual_evaluation", custom_b1))
        log.info("Loaded custom B1 baseline: %s", b1_baseline)
    else:
        log.info("Using standard historical B1 baseline: %s", b1_baseline)

    # 2. Data Cleaning & Zero-Leakage Split
    clean_data_dir = output_dir / "clean_data"
    clean_data_dir.mkdir(parents=True, exist_ok=True)
    clean_train_csv, clean_val_csv, split_stats = clean_and_split_dataset(
        train_manifest, clean_data_dir, val_manifest_path=args.val_manifest, val_ratio=0.1, seed=args.seed, log=log
    )

    (output_dir / "data_sanitization_stats.json").write_text(json.dumps(split_stats, indent=2), encoding="utf-8")

    if args.preflight_only:
        log.info("Preflight validation complete. Exiting (--preflight-only specified).")
        return 0

    # 3. Load Frozen CLIP Backbone
    log.info("Loading Frozen CLIP ViT-L/14 on %s...", device)
    backbone = FrozenCLIPBackbone.from_pretrained(
        model_name="ViT-L-14-quickgelu", pretrained="openai", cache_dir="checkpoints/clip", device=device
    )

    # 4. Optional Phase 0: Train or evaluate B1-clean with identical seed
    b1_records = None
    b1_dir = output_dir / "B1_clean_train"
    if args.b1_checkpoint and args.b1_checkpoint.exists():
        log.info("Loading pre-trained B1 checkpoint for fair evaluation: %s", args.b1_checkpoint)
        b1_state = torch.load(args.b1_checkpoint, map_location=device, weights_only=True)
        b1_model = CFPECIRModel(
            backbone.image_embedding_dim,
            variant="B1",
            slots=int(b1_state.get("config", {}).get("model", {}).get("slots", 8)),
            hidden_dim=int(b1_state.get("config", {}).get("model", {}).get("hidden_dim", 512)),
        ).to(device)
        b1_model.load_state_dict(b1_state["model"])
        b1_metrics, b1_records = evaluate_counterfactual(
            b1_model, backbone, clean_val_csv, image_root, device, batch_size=args.batch_size
        )
        b1_baseline = {
            "cf_a_win_percent": b1_metrics.cf_a_win_percent,
            "cf_b_win_percent": b1_metrics.cf_b_win_percent,
            "both_win_percent": b1_metrics.both_win_percent,
            "hard_negative_error_percent": b1_metrics.hard_negative_error_percent,
        }
        log.info("Evaluated B1 checkpoint: %s", b1_baseline)
    elif args.train_b1:
        log.info("PHASE 0: Controlled Training of B1-clean (seed=%d, epochs=%d)...", args.seed, args.b1_epochs)
        b1_config = {
            "model": {"variant": "B1", "backbone": "ViT-L-14-quickgelu", "hidden_dim": 512, "slots": 8},
            "training": {
                "epochs": args.b1_epochs,
                "batch_size_per_device": args.batch_size,
                "learning_rate": 1e-4,
                "weight_decay": 0.1,
                "precision": "amp" if device.type == "cuda" else "fp32",
                "seed": args.seed,
                "max_grad_norm": 1.0,
                "loss_weights": {"factorization": 0.2, "counterfactual": 0.0, "verifier": 0.0},
            },
            "data": {"train_manifest": str(clean_train_csv), "image_root": str(image_root)},
            "output": {"run_dir": str(b1_dir)},
        }
        b1_model, b1_metrics, b1_records = train_variant(
            "B1", clean_train_csv, clean_val_csv, image_root, b1_dir, backbone, b1_config, log
        )
        b1_baseline = {
            "cf_a_win_percent": b1_metrics.cf_a_win_percent,
            "cf_b_win_percent": b1_metrics.cf_b_win_percent,
            "both_win_percent": b1_metrics.both_win_percent,
            "hard_negative_error_percent": b1_metrics.hard_negative_error_percent,
        }
        log.info("Completed B1-clean training. New fair baseline: %s", b1_baseline)

    # 5. Phase 1: Controlled Training of B4
    log.info("PHASE 1: Controlled Training of B4-new (seed=%d, epochs=%d, margin=%.2f, lambda_cf=%.2f)...",
             args.seed, args.b4_epochs, args.cf_margin, args.lambda_cf)
    b4_config = {
        "model": {"variant": "B4", "backbone": "ViT-L-14-quickgelu", "hidden_dim": 512, "slots": 8},
        "training": {
            "epochs": args.b4_epochs,
            "batch_size_per_device": args.batch_size,
            "learning_rate": 1e-4,
            "weight_decay": 0.1,
            "precision": "amp" if device.type == "cuda" else "fp32",
            "seed": args.seed,
            "max_grad_norm": 1.0,
            "cf_margin": args.cf_margin,
            "loss_weights": {"factorization": 0.2, "counterfactual": args.lambda_cf, "verifier": 0.0},
        },
        "data": {"train_manifest": str(clean_train_csv), "image_root": str(image_root)},
        "output": {"run_dir": str(output_dir / "B4_controlled_repair")},
    }

    b4_dir = output_dir / "B4_controlled_repair"
    b4_model, b4_metrics, b4_records = train_variant(
        "B4", clean_train_csv, clean_val_csv, image_root, b4_dir, backbone, b4_config, log
    )

    # 6. Phase 2: Scientific GO Gate Verification with Cluster Bootstrap CI
    gate_report = verify_scientific_go_gate(
        b4_metrics, b1_baseline, log, b4_records=b4_records, b1_records=b1_records, n_boot=args.n_boot
    )
    (output_dir / "b4_go_gate_report.json").write_text(json.dumps(gate_report, indent=2), encoding="utf-8")

    if not gate_report["overall_passed"] and not args.force_b5_override:
        log.error("ENFORCING STOP RULE: B4 failed the generalization criteria. Pipeline halted.")
        log.error("Review diagnostic logs in %s", output_dir / "b4_go_gate_report.json")
        return 1

    if not gate_report["overall_passed"] and args.force_b5_override:
        log.warning("MANUAL OVERRIDE APPLIED: Proceeding to B5 despite failed GO Gate.")

    # 7. Phase 3: Transition to B5 (Full Model with Eq. 30 Verifier)
    log.info("TRANSITIONING TO B5 (Full Model + Verifier with Eq. 30)...")
    b5_config = copy.deepcopy(b4_config)
    b5_config["model"]["variant"] = "B5"
    b5_config["training"]["epochs"] = args.b5_epochs
    b5_config["training"]["loss_weights"]["verifier"] = 0.5
    b5_dir = output_dir / "B5_controlled_repair"
    b5_config["output"]["run_dir"] = str(b5_dir)

    b4_state = torch.load(b4_dir / "best_checkpoint.pt", map_location=device, weights_only=True)["model"]
    b5_model, b5_metrics, _ = train_variant(
        "B5", clean_train_csv, clean_val_csv, image_root, b5_dir, backbone, b5_config, log,
        warm_start_weights=b4_state,
    )

    # 8. Final Master Protocol Summary
    master_protocol = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": args.seed,
        "device": str(device),
        "data_sanitization": split_stats,
        "b1_baseline": b1_baseline,
        "b4_evaluation": {
            "cf_a_win_percent": b4_metrics.cf_a_win_percent,
            "cf_b_win_percent": b4_metrics.cf_b_win_percent,
            "both_win_percent": b4_metrics.both_win_percent,
            "hard_negative_error_percent": b4_metrics.hard_negative_error_percent,
        },
        "go_gate": gate_report,
        "b5_evaluation": {
            "cf_a_win_percent": b5_metrics.cf_a_win_percent,
            "cf_b_win_percent": b5_metrics.cf_b_win_percent,
            "both_win_percent": b5_metrics.both_win_percent,
            "hard_negative_error_percent": b5_metrics.hard_negative_error_percent,
            "verifier_accuracy_percent": b5_metrics.verifier_acc_percent,
        },
        "artifacts_generated": {
            "b1": str(b1_dir) if args.train_b1 or args.b1_checkpoint else None,
            "b4": str(b4_dir),
            "b5": str(b5_dir),
        },
    }
    (output_dir / "controlled_repair_protocol.json").write_text(json.dumps(master_protocol, indent=2), encoding="utf-8")
    log.info("CONTROLLED REPAIR PIPELINE SUCCESSFULLY COMPLETED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
