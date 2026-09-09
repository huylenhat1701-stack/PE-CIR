"""Download and verify the CLIP backbone used by Pic2Word."""

from __future__ import annotations

import argparse
from pathlib import Path

import open_clip
import torch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="ViT-L-14-quickgelu")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--cache-dir", default="checkpoints/clip")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="cpu",
        help="Use CPU by default so downloading does not require GPU memory.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot detect an NVIDIA GPU")

    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.model} ({args.pretrained}) to {cache_dir}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=args.model,
        pretrained=args.pretrained,
        device=device,
        cache_dir=str(cache_dir),
    )
    tokenizer = open_clip.get_tokenizer(args.model)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    print("CLIP is ready")
    print(f"Device: {device}")
    print(f"Image embedding dimension: {getattr(model.visual, 'output_dim', 'unknown')}")
    print(f"Token embedding dimension: {model.token_embedding.embedding_dim}")
    print(f"Tokenizer output shape: {tuple(tokenizer(['a photo of *']).shape)}")
    print(f"Preprocess: {preprocess}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
