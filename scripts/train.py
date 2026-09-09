"""Train the Pic2Word Mapping Network on local CC3M images."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from pic2word.config import MappingNetworkConfig
from pic2word.data import build_cc3m_dataloader
from pic2word.models import FrozenCLIPBackbone, MappingNetwork, Pic2WordModel
from pic2word.training import Pic2WordTrainer, TrainerConfig, TrainingStepMetrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train_local.yaml"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--resume", type=Path)
    return parser


def load_config(path: Path) -> dict[str, Any]:
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    model_config = config["model"]
    training_config = config["training"]
    data_config = config["data"]
    output_config = config["output"]
    device = resolve_device(args.device)
    set_seed(int(training_config["seed"]))

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
    model = Pic2WordModel(backbone, mapping_network)
    trainer = Pic2WordTrainer(
        model,
        TrainerConfig(
            learning_rate=float(training_config["learning_rate"]),
            weight_decay=float(training_config["weight_decay"]),
            warmup_steps=int(training_config["warmup_steps"]),
            precision=str(training_config["precision"]),
            max_grad_norm=training_config.get("max_grad_norm", 1.0),
        ),
        device=device,
    )
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)
        print(f"Resumed from step {trainer.state.global_step}: {args.resume.resolve()}")

    batch_size = args.batch_size or int(training_config["batch_size_per_device"])
    max_samples = args.max_samples or data_config.get("max_samples")
    max_steps = args.max_steps or training_config.get("max_steps")
    dataloader = build_cc3m_dataloader(
        data_config["train_manifest"],
        data_config["image_root"],
        backbone.preprocess,
        image_column=data_config.get("image_column"),
        max_samples=max_samples,
        batch_size=batch_size,
        num_workers=int(training_config["num_workers"]),
        shuffle=True,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )

    checkpoint_dir = Path(output_config["checkpoint_dir"]).resolve() / "pic2word"
    log_dir = Path(output_config["log_dir"]).resolve() / "pic2word"
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = log_dir / "training_metrics.csv"
    fieldnames = list(TrainingStepMetrics._fields)
    if args.resume is None or not metrics_path.is_file():
        with metrics_path.open("w", encoding="utf-8", newline="") as stream:
            csv.DictWriter(stream, fieldnames=fieldnames).writeheader()

    def report(metrics: TrainingStepMetrics) -> None:
        print(
            f"step={metrics.step} loss={metrics.total_loss:.6f} "
            f"i2t={metrics.image_to_text_loss:.6f} "
            f"t2i={metrics.text_to_image_loss:.6f} lr={metrics.learning_rate:.2e}"
        )
        with metrics_path.open("a", encoding="utf-8", newline="") as stream:
            csv.DictWriter(stream, fieldnames=fieldnames).writerow(metrics._asdict())

    epochs = int(training_config["epochs"])
    started_at = time.perf_counter()
    last_metrics: TrainingStepMetrics | None = None
    for epoch in range(trainer.state.epoch, epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        metrics = trainer.train_epoch(
            dataloader,
            prompt=model_config["prompt"],
            max_steps=max_steps,
            on_step=report,
        )
        if not metrics:
            break
        last_metrics = metrics[-1]
        trainer.state.epoch = epoch + 1
        if trainer.state.epoch % int(training_config["save_every_epochs"]) == 0:
            trainer.save_checkpoint(checkpoint_dir / f"epoch_{trainer.state.epoch:03d}.pt")
        trainer.save_checkpoint(checkpoint_dir / "last.pt")
        if max_steps is not None and trainer.state.global_step >= int(max_steps):
            break

    latest_checkpoint = checkpoint_dir / "last.pt"
    summary_path = log_dir / "training_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "config": str(args.config.resolve()),
                "device": str(device),
                "epochs_completed": trainer.state.epoch,
                "global_step": trainer.state.global_step,
                "training_samples": len(dataloader.dataset),
                "batch_size": batch_size,
                "elapsed_seconds": time.perf_counter() - started_at,
                "last_metrics": last_metrics._asdict() if last_metrics else None,
                "checkpoint": str(latest_checkpoint),
                "metrics_csv": str(metrics_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Training stopped at step {trainer.state.global_step}")
    print(f"Latest checkpoint: {latest_checkpoint}")
    print(f"Training metrics: {metrics_path}")
    print(f"Training summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
