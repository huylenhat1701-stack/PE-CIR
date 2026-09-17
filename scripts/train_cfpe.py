"""Train B1, B4, or B5 on pseudo-edits mined from CC3M captions."""

from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from pic2word.data import build_pseudo_edit_dataloader
from pic2word.models import CFPECIRModel, FrozenCLIPBackbone
from pic2word.training import (
    counterfactual_margin_loss,
    factorization_loss,
    retrieval_alignment_loss,
    verifier_loss,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    result.add_argument("--resume", type=Path)
    result.add_argument("--max-steps", type=int)
    return result


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Configuration must be a YAML mapping")
    return payload


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def score_tensor(scores: Any) -> torch.Tensor:
    return torch.stack((scores.preserve, scores.edit, scores.violation), dim=-1)


def main() -> int:
    args = parser().parse_args()
    config = load_config(args.config)
    model_cfg, training, data, output = (
        config["model"], config["training"], config["data"], config["output"]
    )
    variant = str(model_cfg["variant"]).upper()
    run_dir = Path(output["run_dir"]).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / "train.log", encoding="utf-8")],
    )
    log = logging.getLogger("cfpe")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU is available")
    seed = int(training.get("seed", 0))
    seed_everything(seed)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (run_dir / "seed.txt").write_text(f"{seed}\n", encoding="utf-8")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    (run_dir / "git_commit.txt").write_text(f"{revision or 'unknown'}\n", encoding="utf-8")

    log.info("Loading frozen CLIP %s on %s", model_cfg["backbone"], device)
    backbone = FrozenCLIPBackbone.from_pretrained(
        model_name=model_cfg["backbone"],
        pretrained=model_cfg.get("pretrained", "openai"),
        cache_dir=model_cfg.get("cache_dir", "checkpoints/clip"),
        device=device,
    )
    model = CFPECIRModel(
        backbone.image_embedding_dim,
        variant=variant,
        slots=int(model_cfg.get("slots", 8)),
        hidden_dim=int(model_cfg.get("hidden_dim", 512)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 1e-4)),
        weight_decay=float(training.get("weight_decay", 0.1)),
    )
    use_amp = training.get("precision", "amp") == "amp" and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp, init_scale=4096.0)
    loader = build_pseudo_edit_dataloader(
        data["train_manifest"],
        data["image_root"],
        backbone.preprocess,
        batch_size=int(training["batch_size_per_device"]),
        num_workers=int(training.get("num_workers", 0)),
        max_samples=data.get("max_samples"),
        pin_memory=device.type == "cuda",
    )
    global_step = 0
    best_loss = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        if checkpoint.get("variant") != variant:
            raise ValueError("Checkpoint variant does not match config")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = int(checkpoint.get("global_step", 0))
        best_loss = float(checkpoint.get("best_loss", best_loss))
        log.info("Resumed %s from step %d", variant, global_step)
    max_steps = int(args.max_steps or training.get("max_steps", 0)) or None
    margin = float(training.get("cf_margin", 0.2))
    weights = training.get("loss_weights", {})
    lambda_fac = float(weights.get("factorization", 0.2))
    lambda_cf = float(weights.get("counterfactual", 1.0))
    lambda_ver = float(weights.get("verifier", 0.5))
    history: list[dict[str, float | int | str]] = []
    started = time.perf_counter()

    def checkpoint_payload() -> dict[str, Any]:
        return {
            "variant": variant,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": global_step,
            "best_loss": best_loss,
            "embedding_dim": backbone.image_embedding_dim,
            "config": config,
        }

    stop = False
    for epoch in range(int(training.get("epochs", 1))):
        model.train()
        for batch in loader:
            if max_steps is not None and global_step >= max_steps:
                stop = True
                break
            texts = list(batch["modification"])
            with torch.no_grad():
                reference = backbone.encode_image(batch["reference"].to(device), normalize=True)
                positive = backbone.encode_image(batch["positive"].to(device), normalize=True)
                modification = backbone.encode_text(texts, normalize=True)
                if model.uses_counterfactuals:
                    cf_a = backbone.encode_image(batch["cf_a"].to(device), normalize=True)
                    cf_b = backbone.encode_image(batch["cf_b"].to(device), normalize=True)
                else:
                    cf_a = cf_b = None
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                values = model(reference, modification)
                loss_ret = retrieval_alignment_loss(values.query, positive, logit_scale=backbone.logit_scale)
                loss_fac = factorization_loss(
                    values.preserve, values.composed_edit, modification, values.gates
                )
                loss_cf = torch.zeros((), device=device)
                loss_ver = torch.zeros((), device=device)
                if model.uses_counterfactuals:
                    assert cf_a is not None and cf_b is not None
                    loss_cf = counterfactual_margin_loss(
                        values.query, positive, cf_a, cf_b, margin=margin
                    )
                if model.uses_verifier:
                    p = score_tensor(model.verify(positive, values, modification))
                    a = score_tensor(model.verify(cf_a, values, modification))
                    b = score_tensor(model.verify(cf_b, values, modification))
                    loss_ver = verifier_loss(p, a, b)
                total = loss_ret + lambda_fac * loss_fac + lambda_cf * loss_cf + lambda_ver * loss_ver
            if not torch.isfinite(total):
                raise FloatingPointError(f"Non-finite loss at step {global_step + 1}")
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training.get("max_grad_norm", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            record = {
                "step": global_step,
                "variant": variant,
                "total_loss": float(total.detach()),
                "retrieval_loss": float(loss_ret.detach()),
                "factorization_loss": float(loss_fac.detach()),
                "counterfactual_loss": float(loss_cf.detach()),
                "verifier_loss": float(loss_ver.detach()),
            }
            history.append(record)
            log.info(
                "step=%d total=%.5f ret=%.5f fac=%.5f cf=%.5f ver=%.5f",
                global_step, record["total_loss"], record["retrieval_loss"],
                record["factorization_loss"], record["counterfactual_loss"],
                record["verifier_loss"],
            )
            if record["total_loss"] < best_loss:
                best_loss = float(record["total_loss"])
                torch.save(checkpoint_payload(), run_dir / "best_checkpoint.pt")
            interval = int(training.get("save_every_steps", 100))
            if interval and global_step % interval == 0:
                torch.save(checkpoint_payload(), run_dir / "last_checkpoint.pt")
        if stop:
            break
    torch.save(checkpoint_payload(), run_dir / "last_checkpoint.pt")
    metrics = {
        "variant": variant,
        "global_step": global_step,
        "best_train_loss": best_loss,
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
        "status": "trained_not_yet_benchmark_evaluated",
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    for name in ("per_query_predictions.json", "constraint_scores.json"):
        path = run_dir / name
        if not path.exists():
            path.write_text("[]\n", encoding="utf-8")
    log.info("Completed %s at step %d; artifacts: %s", variant, global_step, run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
