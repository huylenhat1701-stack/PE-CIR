"""Measure CF-A/CF-B wins and verifier quality on held-out pseudo-edit tuples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pic2word.data import build_pseudo_edit_dataloader
from pic2word.models import CFPECIRModel, FrozenCLIPBackbone


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    model_cfg = checkpoint["config"]["model"]
    backbone = FrozenCLIPBackbone.from_pretrained(
        model_name=model_cfg["backbone"], pretrained=model_cfg.get("pretrained", "openai"),
        cache_dir=model_cfg.get("cache_dir", "checkpoints/clip"), device=args.device,
    )
    model = CFPECIRModel(
        backbone.image_embedding_dim, variant=checkpoint["variant"],
        slots=int(model_cfg.get("slots", 8)), hidden_dim=int(model_cfg.get("hidden_dim", 512)),
    ).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    loader = build_pseudo_edit_dataloader(
        args.manifest, args.image_root, backbone.preprocess,
        batch_size=args.batch_size, num_workers=0, shuffle=False, drop_last=False,
    )
    total = win_a = win_b = both = errors = verifier_correct = verifier_total = 0
    records: list[dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            texts = list(batch["modification"])
            encoded = {
                key: backbone.encode_image(batch[key].to(args.device), normalize=True)
                for key in ("reference", "positive", "cf_a", "cf_b")
            }
            modification = backbone.encode_text(texts, normalize=True)
            output = model(encoded["reference"], modification)
            scores = {
                key: (output.query * encoded[key]).sum(dim=-1)
                for key in ("positive", "cf_a", "cf_b")
            }
            a_wins = scores["positive"] > scores["cf_a"]
            b_wins = scores["positive"] > scores["cf_b"]
            batch_size = len(texts)
            total += batch_size
            win_a += int(a_wins.sum())
            win_b += int(b_wins.sum())
            both += int((a_wins & b_wins).sum())
            errors += int((~(a_wins & b_wins)).sum())
            verifier_payload = {}
            if model.uses_verifier:
                labels = {
                    "positive": torch.tensor((1, 1, 0), device=args.device),
                    "cf_a": torch.tensor((1, 0, 1), device=args.device),
                    "cf_b": torch.tensor((0, 1, 1), device=args.device),
                }
                for key, label in labels.items():
                    values = model.verify(encoded[key], output, modification)
                    matrix = torch.stack((values.preserve, values.edit, values.violation), dim=-1)
                    verifier_correct += int(((matrix >= 0.5) == label).sum())
                    verifier_total += matrix.numel()
                    verifier_payload[key] = matrix.cpu().tolist()
            for index in range(batch_size):
                records.append({
                    "modification": texts[index],
                    "attribute_type": batch["attribute_type"][index],
                    "positive_score": float(scores["positive"][index]),
                    "cf_a_score": float(scores["cf_a"][index]),
                    "cf_b_score": float(scores["cf_b"][index]),
                    "cf_a_win": bool(a_wins[index]), "cf_b_win": bool(b_wins[index]),
                    "verifier": {
                        key: verifier_payload[key][index] for key in verifier_payload
                    },
                })
    if total == 0:
        raise RuntimeError("No complete evaluation batches; lower --batch-size")
    metrics = {
        "counterfactual_query_count": total,
        "cf_a_win_percent": 100.0 * win_a / total,
        "cf_b_win_percent": 100.0 * win_b / total,
        "both_win_percent": 100.0 * both / total,
        "hard_negative_error_percent": 100.0 * errors / total,
        "verifier_label_accuracy_percent": (
            100.0 * verifier_correct / verifier_total if verifier_total else None
        ),
    }
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"
    existing = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    existing["counterfactual_evaluation"] = metrics
    metrics_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    (run_dir / "constraint_scores.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
