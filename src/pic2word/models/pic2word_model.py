"""End-to-end Pic2Word forward pass built from a frozen CLIP and a trainable MLP."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

from torch import Tensor, nn

from pic2word.models.clip_backbone import FrozenCLIPBackbone
from pic2word.models.mapping_network import MappingNetwork


class Pic2WordOutput(NamedTuple):
    """Intermediate representations returned for training and diagnostics."""

    image_features: Tensor
    pseudo_tokens: Tensor
    text_features: Tensor


class Pic2WordModel(nn.Module):
    """Connect frozen CLIP encoders to the trainable Pic2Word Mapping Network."""

    def __init__(
        self,
        backbone: FrozenCLIPBackbone,
        mapping_network: MappingNetwork,
    ) -> None:
        super().__init__()
        if mapping_network.config.image_embedding_dim != backbone.image_embedding_dim:
            raise ValueError("Mapping Network input dimension must match CLIP image output dimension")
        if mapping_network.config.token_embedding_dim != backbone.token_embedding_dim:
            raise ValueError("Mapping Network output dimension must match CLIP token dimension")
        self.backbone = backbone
        self.mapping_network = mapping_network

    def train(self, mode: bool = True) -> Pic2WordModel:
        """Train the Mapping Network while always keeping CLIP frozen and in eval mode."""

        super().train(mode)
        self.backbone.freeze()
        self.mapping_network.train(mode)
        return self

    def forward(self, images: Tensor, prompts: Sequence[str]) -> Pic2WordOutput:
        """Encode images, create pseudo-tokens, and reconstruct them through CLIP text."""

        image_features = self.backbone.encode_image(images, normalize=False)
        pseudo_tokens = self.mapping_network(image_features)
        text_features = self.backbone.encode_prompt_with_pseudo_token(
            prompts,
            pseudo_tokens,
            normalize=False,
        )
        return Pic2WordOutput(image_features, pseudo_tokens, text_features)
