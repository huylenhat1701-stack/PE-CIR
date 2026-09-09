"""Run a local Pic2Word Top-K retrieval demo."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml

from pic2word.config import MappingNetworkConfig
from pic2word.models import FrozenCLIPBackbone, MappingNetwork, Pic2WordModel
from pic2word.retrieval import (
    CandidateIndex,
    build_candidate_index,
    find_candidate_images,
    search_pic2word,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-image", type=Path, required=True)
    parser.add_argument("--text", required=True, help="Modification sentence")
    parser.add_argument("--candidate-dir", type=Path, default=Path("data/cc_data/train"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/pic2word/last.pt"))
    parser.add_argument("--config", type=Path, default=Path("configs/train_local.yaml"))
    parser.add_argument("--index", type=Path, default=Path("indexes/cc3m_local.pt"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-candidates", type=int, default=100)
    parser.add_argument("--index-batch-size", type=int, default=4)
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


def main() -> int:
    args = build_parser().parse_args()
    config = load_yaml(args.config)
    model_config = config["model"]
    device = resolve_device(args.device)

    print(f"Loading frozen CLIP on {device}")
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
    checkpoint = torch.load(args.checkpoint.resolve(), map_location=device, weights_only=True)
    mapping_network.load_state_dict(checkpoint["mapping_network"])
    model = Pic2WordModel(backbone, mapping_network).to(device).eval()

    if args.index.is_file() and not args.rebuild_index:
        index = CandidateIndex.load(args.index)
        print(f"Loaded candidate index with {len(index.paths)} images")
    else:
        paths = find_candidate_images(args.candidate_dir, limit=args.max_candidates)
        print(f"Encoding {len(paths)} candidate images")
        index = build_candidate_index(backbone, paths, batch_size=args.index_batch_size)
        index.save(args.index)
        print(f"Saved candidate index: {args.index.resolve()}")

    results, prompt = search_pic2word(
        model,
        index,
        args.query_image,
        args.text,
        top_k=args.top_k,
    )
    print(f"Prompt: {prompt}")
    for rank, result in enumerate(results, start=1):
        print(f"{rank}. score={result.score:.6f} path={result.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
