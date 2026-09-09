"""Image embedding to pseudo-word token mapping network."""

from __future__ import annotations

from torch import Tensor, nn

from pic2word.config import MappingNetworkConfig


class MappingNetwork(nn.Module):
    """Map a CLIP image embedding into the CLIP token-embedding space.

    The backbone dimensions are supplied by the caller so the implementation works
    with more than one CLIP variant. The default hidden architecture follows Pic2Word:
    repeated Linear -> Dropout -> ReLU blocks followed by a linear output layer.
    """

    def __init__(self, config: MappingNetworkConfig) -> None:
        super().__init__()
        self.config = config

        blocks: list[nn.Module] = []
        input_dim = config.image_embedding_dim
        for _ in range(config.hidden_layers):
            blocks.extend(
                [
                    nn.Linear(input_dim, config.hidden_dim),
                    nn.Dropout(config.dropout),
                    nn.ReLU(),
                ]
            )
            input_dim = config.hidden_dim

        self.hidden = nn.Sequential(*blocks)
        self.output = nn.Linear(config.hidden_dim, config.token_embedding_dim)

    def forward(self, image_features: Tensor) -> Tensor:
        """Return pseudo-word tokens with shape ``[..., token_embedding_dim]``."""

        if image_features.shape[-1] != self.config.image_embedding_dim:
            raise ValueError(
                "Expected image feature dimension "
                f"{self.config.image_embedding_dim}, received {image_features.shape[-1]}"
            )
        return self.output(self.hidden(image_features))
