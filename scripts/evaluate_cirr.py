"""Evaluate a trained Pic2Word Mapping Network on the CIRR validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from pic2word.config import MappingNetworkConfig
from pic2word.data import CIRRSplit, load_cirr_split
from pic2word.evaluation import evaluate_cirr
from pic2word.models import FrozenCLIPBackbone, MappingNetwork, Pic2WordModel
from pic2word.retrieval import CandidateIndex, build_candidate_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/cirr"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--version", default="rc2")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/pic2word/pic2word_official_converted.pt"),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/train_local.yaml"))
    parser.add_argument("--index", type=Path, default=Path("indexes/cirr_val.pt"))
    parser.add_argument("--index-batch-size", type=int, default=4)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Evaluate only complete query groups when some CIRR images are unavailable",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/cirr_val_metrics.json"),
        help="JSON file for the acceptance report",
    )
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise TypeError(f"Configuration must be a YAML mapping: {path}")
    return config


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but this PyTorch installation cannot detect a GPU")
    return torch.device(requested)


def load_model(
    config: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
) -> Pic2WordModel:
    model_config = config["model"]
    backbone = FrozenCLIPBackbone.from_pretrained(
        model_name=model_config["backbone"],
        pretrained=model_config["pretrained"],
        cache_dir="checkpoints/clip",
        device=device,
    )
    mapping_network = MappingNetwork(
        MappingNetworkConfig(
            image_embedding_dim=backbone.image_embedding_dim,
            token_embedding_dim=backbone.token_embedding_dim,
            hidden_dim=int(model_config["hidden_dim"]),
            hidden_layers=int(model_config["hidden_layers"]),
            dropout=float(model_config["dropout"]),
        )
    )
    checkpoint = torch.load(checkpoint_path.resolve(), map_location=device, weights_only=True)
    mapping_network.load_state_dict(checkpoint["mapping_network"])
    return Pic2WordModel(backbone, mapping_network).to(device).eval()


def select_available_subset(dataset: CIRRSplit) -> tuple[CIRRSplit, int, int]:
    """Keep available candidates and queries whose entire six-image group exists."""

    available_paths = {
        image_id: path for image_id, path in dataset.image_paths.items() if path.is_file()
    }
    available_ids = set(available_paths)
    queries = tuple(
        query
        for query in dataset.queries
        if query.reference_id in available_ids
        and query.target_id in available_ids
        and set(query.group_members).issubset(available_ids)
    )
    if not queries:
        raise RuntimeError("No complete CIRR query groups are available for partial evaluation")
    partial = CIRRSplit(
        root=dataset.root,
        split=dataset.split,
        version=dataset.version,
        queries=queries,
        image_paths=available_paths,
    )
    return partial, len(dataset.image_paths) - len(available_paths), len(dataset.queries) - len(queries)


def main() -> int:
    args = build_parser().parse_args()
    dataset = load_cirr_split(args.dataset_root, split=args.split, version=args.version)
    total_query_count = len(dataset.queries)
    total_candidate_count = len(dataset.candidate_ids)
    print(f"Loaded {len(dataset.queries)} CIRR queries and {len(dataset.candidate_ids)} images")

    missing_images = dataset.missing_images()
    if missing_images:
        if not args.allow_partial:
            examples = "\n".join(f"  - {path}" for path in missing_images[:5])
            raise FileNotFoundError(
                f"CIRR is missing {len(missing_images)} raw images under "
                f"{(args.dataset_root / 'img_raw').resolve()}. Examples:\n{examples}"
            )
        dataset, missing_image_count, excluded_query_count = select_available_subset(dataset)
        print(
            "PARTIAL EVALUATION: "
            f"{missing_image_count} images missing; {excluded_query_count} queries excluded"
        )
        if args.index == Path("indexes/cirr_val.pt"):
            args.index = Path("indexes/cirr_val_partial.pt")
        if args.output == Path("reports/cirr_val_metrics.json"):
            args.output = Path("reports/cirr_val_metrics_partial.json")
    else:
        missing_image_count = 0
        excluded_query_count = 0

    device = resolve_device(args.device)
    print(f"Loading frozen CLIP and Mapping Network on {device}")
    model = load_model(load_yaml(args.config), args.checkpoint, device)

    if args.index.is_file() and not args.rebuild_index:
        index = CandidateIndex.load(args.index)
        print(f"Loaded CIRR candidate index with {len(index.paths)} images")
    else:
        print(f"Encoding {len(dataset.candidate_paths)} CIRR candidate images")

        def report_index_progress(completed: int, total: int) -> None:
            if completed == 1 or completed % 100 == 0 or completed == total:
                print(f"Encoded {completed}/{total} candidate images")

        index = build_candidate_index(
            model.backbone,
            dataset.candidate_paths,
            batch_size=args.index_batch_size,
            progress=report_index_progress,
        )
        index.save(args.index)
        print(f"Saved CIRR candidate index: {args.index.resolve()}")

    def report_progress(completed: int, total: int) -> None:
        if completed == 1 or completed % 100 == 0 or completed == total:
            print(f"Evaluated {completed}/{total} queries")

    result = evaluate_cirr(
        model,
        index,
        dataset,
        max_queries=args.max_queries,
        progress=report_progress,
    )
    print(f"\nCIRR {args.split}: {result.query_count} queries")
    for k, value in result.global_recall.items():
        print(f"Recall@{k}: {value:.2f}")
    for k, value in result.group_recall.items():
        print(f"Group Recall@{k}: {value:.2f}")
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "dataset": "CIRR",
                "split": args.split,
                "version": args.version,
                "query_count": result.query_count,
                "total_query_count": total_query_count,
                "query_coverage_percent": 100.0 * result.query_count / total_query_count,
                "candidate_count": len(dataset.candidate_ids),
                "total_candidate_count": total_candidate_count,
                "candidate_coverage_percent": (
                    100.0 * len(dataset.candidate_ids) / total_candidate_count
                ),
                "checkpoint": str(args.checkpoint.resolve()),
                "partial": bool(missing_image_count),
                "officially_comparable": not bool(missing_image_count),
                "missing_image_count": missing_image_count,
                "excluded_query_count": excluded_query_count,
                "global_recall": result.global_recall,
                "group_recall": result.group_recall,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved metrics: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
