"""Validated configuration objects used by the Pic2Word core."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MappingNetworkConfig:
    """Architecture settings for the image-to-token mapping network."""

    image_embedding_dim: int
    token_embedding_dim: int
    hidden_dim: int = 512
    hidden_layers: int = 2
    dropout: float = 0.1

    def __post_init__(self) -> None:
        positive_dimensions = {
            "image_embedding_dim": self.image_embedding_dim,
            "token_embedding_dim": self.token_embedding_dim,
            "hidden_dim": self.hidden_dim,
            "hidden_layers": self.hidden_layers,
        }
        for name, value in positive_dimensions.items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero; received {value}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1); received {self.dropout}")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Training defaults from the Pic2Word paper."""

    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    warmup_steps: int = 10_000
    epochs: int = 30
    prompt: str = "a photo of *"

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps cannot be negative")
        if self.epochs <= 0:
            raise ValueError("epochs must be greater than zero")
        if self.prompt.count("*") != 1:
            raise ValueError("training prompt must contain exactly one '*' placeholder")
