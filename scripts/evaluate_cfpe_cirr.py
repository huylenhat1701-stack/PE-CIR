"""Evaluate B1/B4/B5 on CIRR with optional B5 Top-K constraint reranking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pic2word.data import load_cirr_split
from pic2word.evaluation.metrics import recall_at_k
from pic2word.models import CFPECIRModel, FrozenCLIPBackbone
from pic2word.retrieval import CandidateIndex, build_candidate_index, search_cfpe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, choices=(50, 100), default=50)
    parser.add_argument("--index-batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset = load_cirr_split(args.dataset_root)
    missing = dataset.missing_images()
    if missing:
        raise FileNotFoundError(f"CIRR is incomplete: {len(missing)} images are missing")
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
    if args.index.is_file():
        index = CandidateIndex.load(args.index)
    else:
        index = build_candidate_index(backbone, dataset.candidate_paths, batch_size=args.index_batch_size, progress=lambda n, total: print(f"Indexed {n}/{total}", flush=True) if n % 512 == 0 or n == total else None)
        index.save(args.index)
    if set(index.paths) != {path.resolve() for path in dataset.candidate_paths}:
        raise ValueError("Candidate index does not match CIRR; delete it and rebuild")
    path_to_id = {path.resolve(): key for key, path in dataset.image_paths.items()}
    rankings: list[list[str]] = []
    group_rankings: list[list[str]] = []
    stage1_rankings = []
    targets: list[str] = []
    predictions: list[dict[str, object]] = []
    constraints: list[dict[str, object]] = []
    for number, query in enumerate(dataset.queries, start=1):
        results = search_cfpe(
            model, backbone, index, dataset.path_for(query.reference_id), query.caption,
            top_k=args.top_k, return_k=len(index.paths),
        )
        ranking = [path_to_id[item.path] for item in results]
        group_ids = set(query.group_members)
        group = [item for item in ranking if item in group_ids and item != query.reference_id]
        stage1_rankings.append([path_to_id[item.path] for item in sorted(results, key=lambda item: item.stage1_score, reverse=True)])
        rankings.append(ranking)
        group_rankings.append(group[:3])
        targets.append(query.target_id)
        predictions.append({
            "pair_id": query.pair_id, "reference_id": query.reference_id,
            "target_id": query.target_id, "caption": query.caption, "ranking": ranking[:args.top_k], "group_ranking": group,
        })
        constraints.append({
            "pair_id": query.pair_id,
            "top_k": [{
                "image_id": path_to_id[item.path], "stage1_score": item.stage1_score,
                "final_score": item.final_score, "p_P": item.preserve_probability,
                "p_E": item.edit_probability, "p_V": item.violation_probability,
            } for item in results[:args.top_k]],
        })
        if number == 1 or number % 100 == 0 or number == len(dataset.queries):
            print(f"Evaluated {number}/{len(dataset.queries)}")
    metrics = {
        "dataset": "CIRR", "variant": checkpoint["variant"], "query_count": len(targets),
        "top_k_rerank": args.top_k if model.uses_verifier else None,
        "stage1_global_recall": recall_at_k(stage1_rankings, targets, (1, 5, 10, 50)),
        "global_recall": recall_at_k(rankings, targets, (1, 5, 10, 50)),
        "group_recall": recall_at_k(group_rankings, targets, (1, 2, 3)),
        "split": "val",
        "group_protocol": "Filter full gallery after global Top-K reranking; preserve Stage-1 tail",
    }
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "per_query_predictions.json").write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "constraint_scores.json").write_text(
        json.dumps(constraints, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
