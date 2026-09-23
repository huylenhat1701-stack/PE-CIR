"""Evaluate a B1/B4/B5 checkpoint on Fashion-IQ validation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from pic2word.data import load_fashioniq_split
from pic2word.evaluation.metrics import recall_at_k
from pic2word.models import CFPECIRModel, FrozenCLIPBackbone
from pic2word.retrieval import CandidateIndex, build_candidate_index, search_cfpe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, choices=(50, 100), default=100)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset = load_fashioniq_split(args.dataset_root)
    missing = dataset.missing_images()
    if missing:
        raise FileNotFoundError(f"Fashion-IQ is incomplete: {len(missing)} images are missing")
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    cfg = checkpoint["config"]["model"]
    backbone = FrozenCLIPBackbone.from_pretrained(
        model_name=cfg["backbone"], pretrained=cfg.get("pretrained", "openai"),
        cache_dir=cfg.get("cache_dir", "checkpoints/clip"), device=args.device,
    )
    model = CFPECIRModel(
        backbone.image_embedding_dim, variant=checkpoint["variant"],
        slots=int(cfg.get("slots", 8)), hidden_dim=int(cfg.get("hidden_dim", 512)),
    ).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    indexes: dict[str, CandidateIndex] = {}
    args.index_dir.mkdir(parents=True, exist_ok=True)
    for category in dataset.image_paths:
        path = args.index_dir / f"fashioniq_{category}_val.pt"
        if path.is_file():
            indexes[category] = CandidateIndex.load(path)
        else:
            indexes[category] = build_candidate_index(backbone, dataset.candidates(category), batch_size=8, progress=lambda n, total: print(f"Indexed {n}/{total}", flush=True) if n % 512 == 0 or n == total else None)
            indexes[category].save(path)
        if set(indexes[category].paths) != {p.resolve() for p in dataset.candidates(category)}:
            raise ValueError(f"Incomplete or incompatible index for {category}")
    stage1_rankings = defaultdict(list)
    rankings: dict[str, list[list[str]]] = defaultdict(list)
    targets: dict[str, list[str]] = defaultdict(list)
    predictions: list[dict[str, object]] = []
    constraints: list[dict[str, object]] = []
    for number, query in enumerate(dataset.queries, start=1):
        index = indexes[query.category]
        id_for_path = {path.resolve(): key for key, path in dataset.image_paths[query.category].items()}
        results = search_cfpe(
            model, backbone, index, dataset.path_for(query.category, query.reference_id),
            query.modification, top_k=args.top_k, return_k=args.top_k,
        )
        ranking = [id_for_path[item.path] for item in results]
        stage1_rankings[query.category].append([id_for_path[item.path] for item in sorted(results, key=lambda item: item.stage1_score, reverse=True)])
        rankings[query.category].append(ranking)
        targets[query.category].append(query.target_id)
        predictions.append({
            "category": query.category, "reference_id": query.reference_id,
            "target_id": query.target_id, "captions": query.captions, "ranking": ranking,
        })
        constraints.append({
            "category": query.category, "reference_id": query.reference_id,
            "top_k": [{"image_id": id_for_path[item.path], "stage1_score": item.stage1_score,
                "final_score": item.final_score, "p_P": item.preserve_probability,
                "p_E": item.edit_probability, "p_V": item.violation_probability}
                for item in results],
        })
        if number == 1 or number % 100 == 0 or number == len(dataset.queries):
            print(f"Evaluated {number}/{len(dataset.queries)}")
    per_category = {
        category: recall_at_k(rankings[category], targets[category], (10, 50))
        for category in rankings
    }
    metrics = {
        "stage1_per_category_recall": {c: recall_at_k(stage1_rankings[c], targets[c], (10, 50)) for c in rankings},
        "top_k_rerank": args.top_k if model.uses_verifier else None,
        "dataset": "Fashion-IQ", "variant": checkpoint["variant"],
        "query_count": len(dataset.queries), "per_category_recall": per_category,
        "mean_recall": {
            k: sum(values[k] for values in per_category.values()) / len(per_category)
            for k in (10, 50)
        }, "split": "val",
    }
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "fashioniq_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "fashioniq_per_query_predictions.json").write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "fashioniq_constraint_scores.json").write_text(
        json.dumps(constraints, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
