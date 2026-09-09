"""Small command-line entry point for the initial Pic2Word package."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from pic2word.retrieval.prompts import (
    build_domain_prompt,
    build_object_prompt,
    build_sentence_prompt,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pic2word")
    commands = parser.add_subparsers(dest="command", required=True)

    prompt = commands.add_parser("prompt", help="Build a Pic2Word prompt")
    prompt.add_argument("mode", choices=("domain", "objects", "sentence"))
    prompt.add_argument("--text", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "domain":
        result = build_domain_prompt(args.text)
    elif args.mode == "objects":
        result = build_object_prompt(part for part in args.text.split(",") if part.strip())
    else:
        result = build_sentence_prompt(args.text)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
