"""Convert the official Pic2Word ViT-L/14 checkpoint to this project's format."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from pic2word.checkpoints import load_official_mapping_checkpoint, sha256_file
from pic2word.config import MappingNetworkConfig
from pic2word.models import MappingNetwork


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("checkpoints/pic2word/pic2word_official.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/pic2word/pic2word_official_converted.pt"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    state, metadata = load_official_mapping_checkpoint(args.input)
    mapping_network = MappingNetwork(
        MappingNetworkConfig(
            image_embedding_dim=768,
            token_embedding_dim=768,
            hidden_dim=512,
            hidden_layers=2,
            dropout=0.1,
        )
    )
    mapping_network.load_state_dict(state, strict=True)

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(
        {
            "mapping_network": mapping_network.state_dict(),
            "source": {
                **metadata,
                "format": "google-research/composed_image_retrieval",
                "sha256": sha256_file(args.input),
            },
        },
        temporary_path,
    )
    temporary_path.replace(output_path)
    print(f"Imported official epoch {metadata['original_epoch']} Mapping Network")
    print(f"Saved converted checkpoint: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
