"""Training loop for the Pic2Word Mapping Network."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from pic2word.models.pic2word_model import Pic2WordModel
from pic2word.training.losses import symmetric_contrastive_loss


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    """Optimizer and numerical settings for Pic2Word training."""

    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    warmup_steps: int = 10_000
    precision: str = "amp"
    max_grad_norm: float | None = 1.0

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps cannot be negative")
        if self.precision not in {"fp32", "amp"}:
            raise ValueError("precision must be either 'fp32' or 'amp'")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be greater than zero when provided")


@dataclass(slots=True)
class TrainerState:
    """Progress values stored inside every checkpoint."""

    epoch: int = 0
    global_step: int = 0


class TrainingStepMetrics(NamedTuple):
    """Scalar values reported after one optimizer update."""

    step: int
    total_loss: float
    image_to_text_loss: float
    text_to_image_loss: float
    learning_rate: float


def build_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linearly warm the learning rate, then keep it at its configured value."""

    def scale(step: int) -> float:
        if warmup_steps == 0:
            return 1.0
        return min(1.0, (step + 1) / warmup_steps)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


class Pic2WordTrainer:
    """Optimize only the Mapping Network while CLIP stays frozen."""

    def __init__(
        self,
        model: Pic2WordModel,
        config: TrainerConfig,
        *,
        device: str | torch.device,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.train()
        self.config = config
        self.state = TrainerState()
        self.optimizer = torch.optim.AdamW(
            self.model.mapping_network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = build_warmup_scheduler(self.optimizer, config.warmup_steps)
        self.use_amp = config.precision == "amp" and self.device.type == "cuda"
        # ViT-L/14's text-path gradients overflow the generic 65536 initial
        # scale on 4 GB consumer GPUs. 4096 avoids several skipped first steps
        # while retaining AMP's dynamic growth/backoff behavior.
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp,
            init_scale=4096.0,
        )

    def _autocast_context(self) -> Any:
        if not self.use_amp:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    def train_step(self, images: Tensor, prompt: str) -> TrainingStepMetrics:
        """Run forward, contrastive loss, backward, and one AdamW update."""

        if images.ndim != 4 or images.shape[0] < 2:
            raise ValueError("Training requires an image batch shaped [batch>=2, channels, H, W]")

        images = images.to(self.device, non_blocking=True)
        prompts = [prompt] * images.shape[0]
        self.optimizer.zero_grad(set_to_none=True)

        with self._autocast_context():
            output = self.model(images, prompts)
            losses = symmetric_contrastive_loss(
                output.image_features,
                output.text_features,
                logit_scale=self.model.backbone.logit_scale.detach(),
            )

        if not torch.isfinite(losses.total):
            raise FloatingPointError(f"Non-finite training loss: {losses.total.item()}")

        self.scaler.scale(losses.total).backward()
        if self.config.max_grad_norm is not None:
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.mapping_network.parameters(), self.config.max_grad_norm)
        scale_before_update = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        optimizer_step_skipped = self.use_amp and self.scaler.get_scale() < scale_before_update
        if not optimizer_step_skipped:
            self.scheduler.step()
            self.state.global_step += 1

        return TrainingStepMetrics(
            step=self.state.global_step,
            total_loss=float(losses.total.detach().cpu()),
            image_to_text_loss=float(losses.image_to_text.detach().cpu()),
            text_to_image_loss=float(losses.text_to_image.detach().cpu()),
            learning_rate=float(self.optimizer.param_groups[0]["lr"]),
        )

    def train_epoch(
        self,
        dataloader: Iterable[Tensor],
        *,
        prompt: str,
        max_steps: int | None = None,
        on_step: Callable[[TrainingStepMetrics], None] | None = None,
    ) -> list[TrainingStepMetrics]:
        """Train over one DataLoader pass, optionally stopping at a global step limit."""

        self.model.train()
        metrics: list[TrainingStepMetrics] = []
        for images in dataloader:
            if max_steps is not None and self.state.global_step >= max_steps:
                break
            step_metrics = self.train_step(images, prompt)
            metrics.append(step_metrics)
            if on_step is not None:
                on_step(step_metrics)
        return metrics

    def save_checkpoint(self, path: str | Path) -> Path:
        """Save Mapping Network and optimizer progress atomically."""

        checkpoint_path = Path(path).resolve()
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        payload = {
            "mapping_network": self.model.mapping_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "state": asdict(self.state),
            "trainer_config": asdict(self.config),
        }
        torch.save(payload, temporary_path)
        temporary_path.replace(checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore Mapping Network and optimizer progress from a local checkpoint."""

        checkpoint_path = Path(path).resolve()
        payload = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.mapping_network.load_state_dict(payload["mapping_network"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        self.scaler.load_state_dict(payload["scaler"])
        self.state = TrainerState(**payload["state"])
