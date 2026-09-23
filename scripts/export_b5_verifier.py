"""Export existing B5 counterfactual predictions; standard library only, no training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

HEADS = ("preserve", "edit", "violation")
LABELS = {"positive": (1, 1, 0), "cf_a": (1, 0, 1), "cf_b": (0, 1, 1)}


def auc(labels, scores):
    """Mann-Whitney AUROC, with half credit for tied scores."""
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    negatives_below = 0
    wins = 0.0
    for _, group in itertools.groupby(sorted(zip(scores, labels)), key=lambda x: x[0]):
        ys = [y for _, y in group]
        pos = sum(ys)
        neg = len(ys) - pos
        wins += pos * (negatives_below + 0.5 * neg)
        negatives_below += neg
    return wins / (positives * negatives)


def summarize(labels, scores):
    predicted = [int(p >= 0.5) for p in scores]
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predicted))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predicted))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predicted))
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predicted))
    return {
        "n": len(labels),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "auroc": auc(labels, scores),
        "accuracy": (tp + tn) / len(labels),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def risk_curve(labels, scores):
    """Selective binary error; retain whole confidence ties, never use labels to rank."""
    items = sorted(
        ((max(p, 1 - p), int((p >= 0.5) != y)) for y, p in zip(labels, scores)), reverse=True
    )
    kept = errors = 0
    points = []
    for confidence, group in itertools.groupby(items, key=lambda x: x[0]):
        group = list(group)
        kept += len(group)
        errors += sum(error for _, error in group)
        points.append(
            {
                "confidence_threshold": confidence,
                "retained": kept,
                "coverage": kept / len(items),
                "errors": errors,
                "risk": errors / kept,
            }
        )
    return points


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_plot(path, curves):
    # Vector plot with fixed 0..1 axes; CSV retains every threshold point.
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="570" viewBox="0 0 900 570">',
        '<rect width="900" height="570" fill="white"/>',
        '<g font-family="sans-serif" font-size="15" fill="#222">',
        '<text x="75" y="30" font-size="22">B5 verifier: selective classification risk</text>',
    ]
    for i in range(6):
        value = i / 5
        x, y = 75 + 720 * value, 465 - 400 * value
        parts += [
            f'<path d="M75 {y} H795 M{x} 65 V465" stroke="#ddd" fill="none"/>',
            f'<text x="{x - 12}" y="490">{value:.1f}</text>',
            f'<text x="30" y="{y + 5}">{value:.1f}</text>',
        ]
    colors = ("#2563eb", "#ea580c", "#16a34a", "#9333ea")
    for i, (head, curve) in enumerate(curves.items()):
        # Collapse only points occupying the same plot pixel for a compact SVG.
        buckets = {}
        for point in curve:
            buckets[round(720 * point["coverage"])] = point
        coordinates = " ".join(
            f"{75 + 720 * p['coverage']:.2f},{465 - 400 * p['risk']:.2f}" for p in buckets.values()
        )
        parts += [
            f'<polyline points="{coordinates}" fill="none" stroke="{colors[i]}" stroke-width="2"/>',
            f'<text x="{80 + i * 180}" y="545" fill="{colors[i]}">{head}</text>',
        ]
    parts += [
        '<text x="385" y="515">Coverage (retained fraction)</text>',
        '<text x="18" y="285" transform="rotate(-90 18 285)">Risk (error rate)</text>',
        "</g></svg>",
    ]
    path.write_text("\n".join(parts), encoding="utf-8")


def export(run, output, expected_count=5000):
    source = run / "constraint_scores.json"
    records = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != expected_count:
        raise ValueError(f"Expected exactly {expected_count} records; no truncation allowed")
    predictions = []
    by_head = {head: ([], []) for head in HEADS}
    for index, record in enumerate(records, 1):
        for candidate, labels in LABELS.items():
            probabilities = record["verifier"][candidate]
            if len(probabilities) != 3 or any(
                not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1
                for p in probabilities
            ):
                raise ValueError(f"Invalid verifier probabilities at record {index}, {candidate}")
            score = record[f"{candidate}_score"]
            if not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError(f"Invalid retrieval score at record {index}")
            row = {
                "sample_index": index,
                "modification": record["modification"],
                "attribute_type": record.get("attribute_type", "unknown"),
                "candidate": candidate,
                "stage1_score": score,
            }
            for head, y, p in zip(HEADS, labels, probabilities):
                row.update(
                    {
                        f"{head}_label": y,
                        f"{head}_probability": p,
                        f"{head}_prediction": int(p >= 0.5),
                    }
                )
                by_head[head][0].append(y)
                by_head[head][1].append(p)
            predictions.append(row)
    by_head["micro"] = (
        [y for ys, _ in by_head.values() for y in ys],
        [p for _, ps in by_head.values() for p in ps],
    )
    metrics = {head: summarize(*values) for head, values in by_head.items()}
    curves = {head: risk_curve(*values) for head, values in by_head.items()}
    old_metrics_path = run / "metrics.json"
    if old_metrics_path.exists():
        old = json.loads(old_metrics_path.read_text(encoding="utf-8"))
        old_accuracy = old.get("counterfactual_evaluation", {}).get(
            "verifier_label_accuracy_percent"
        )
        if old_accuracy is not None and not math.isclose(
            old_accuracy / 100, metrics["micro"]["accuracy"], abs_tol=1e-8
        ):
            raise ValueError("Computed accuracy differs from original evaluation")
    metadata = {
        "mode": "postprocess_existing_predictions_no_new_inference",
        "source_run": str(run.resolve()),
        "samples": len(records),
        "candidate_predictions": len(predictions),
        "binary_decisions": len(predictions) * 3,
        "threshold": 0.5,
        "labels": LABELS,
        "confusion_matrix_axes": "rows=true [0,1]; columns=predicted [0,1]",
        "label_source": "Existing pseudo-edit evaluation labels, not human-verified labels",
        "risk_definition": "Binary classification error among retained decisions, per head and micro",
        "confidence_definition": "max(p, 1-p); retain confidence >= threshold; ties retained together",
        "risk_scope": "Verifier classification only; not full-gallery retrieval risk",
        "sample_index_definition": "1-based source JSON position; original image IDs were not saved",
        "checkpoint_linkage": "Checkpoint co-located in source run; original export did not record its hash",
        "model_dataset_training_changed": False,
        "source_files": {
            name: sha256(run / name)
            for name in (
                "constraint_scores.json",
                "best_checkpoint.pt",
                "config.yaml",
                "run_protocol.json",
            )
            if (run / name).is_file()
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "predictions_5000.json").write_bytes(source.read_bytes())
    write_csv(output / "predictions_15000_candidates.csv", predictions)
    result = {
        "metadata": metadata,
        "metrics": metrics,
        "macro": {
            key: sum(metrics[h][key] for h in HEADS) / 3
            for key in ("precision", "recall", "f1", "auroc", "accuracy")
        },
    }
    (output / "verifier_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_csv(
        output / "verifier_metrics.csv",
        [
            dict(head=h, **{k: v for k, v in m.items() if k != "confusion_matrix"})
            for h, m in metrics.items()
        ],
    )
    write_csv(
        output / "confusion_matrix.csv",
        [
            {"head": h, "actual": y, "predicted": p, "count": m["confusion_matrix"][y][p]}
            for h, m in metrics.items()
            for y in (0, 1)
            for p in (0, 1)
        ],
    )
    write_csv(
        output / "risk_coverage.csv",
        [dict(head=h, **point) for h, curve in curves.items() for point in curve],
    )
    write_plot(output / "risk_coverage.svg", curves)
    lines = [
        "# B5 verifier evaluation",
        "",
        f"Samples: {len(records):,}; candidates: {len(predictions):,}; binary decisions: {len(predictions) * 3:,}.",
        "",
        "Postprocessing of existing B5 probabilities. No new inference or training.",
        "",
        "| Head | Precision | Recall | F1 | AUROC | Accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for head, values in list(metrics.items()) + [("macro", result["macro"])]:
        lines.append(
            "| "
            + head
            + " | "
            + " | ".join(
                f"{values[k]:.6f}" for k in ("precision", "recall", "f1", "auroc", "accuracy")
            )
            + " |"
        )
    lines += [
        "",
        "## Definitions and provenance",
        "",
        "Predicted positive when p >= 0.5. Undefined precision/recall/F1 use zero. AUROC uses probabilities with half credit for ties.",
        "Labels (preserve, edit, violation): positive=(1,1,0), CF-A=(1,0,1), CF-B=(0,1,1).",
        "Each head evaluates 15,000 candidates. Micro pools 45,000 binary decisions; macro averages three heads.",
        "Confusion matrices: rows = actual 0/1; columns = predicted 0/1.",
        "",
        "Risk–Coverage: confidence=max(p,1-p); risk=classification errors/retained decisions. Whole confidence ties are retained together. Zero coverage has undefined risk and is omitted. This is verifier classification risk, not gallery retrieval risk.",
        "",
        "## Limitations",
        "",
        "Labels come from the existing pseudo-edit protocol. Existing label noise and train/validation overlap are not repaired by this export.",
        "The original scores omit image IDs, evaluation manifest identity, and inference-time checkpoint hash. Row indices identify source positions only. Current checkpoint/source hashes are recorded for traceability, not proof of historical linkage.",
        "The historical per_query_predictions.json placeholder is not overwritten. This export covers the stored three-candidate counterfactual evaluation, not CIRR/Fashion-IQ gallery rankings.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export(args.run_dir, args.output)
    for head, values in result["metrics"].items():
        print(
            head, " ".join(f"{k}={values[k]:.6f}" for k in ("precision", "recall", "f1", "auroc"))
        )
    print("DONE:", args.output.resolve())


if __name__ == "__main__":
    main()
