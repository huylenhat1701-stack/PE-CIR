"""Frozen OpenCLIP backbone with support for Pic2Word pseudo-token injection."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import open_clip
import torch
from open_clip.model import text_global_pool
from torch import Tensor, nn
from torch.nn import functional

from pic2word.models.token_injection import inject_pseudo_token


class FrozenCLIPBackbone(nn.Module):
    """Expose frozen CLIP image/text encoders needed by Pic2Word.

    CLIP parameters never receive gradients. The text forward pass itself still keeps
    autograd enabled so its output can send gradients back to the injected pseudo-token
    and, consequently, to the Mapping Network.
    """

    def __init__(
        self,
        clip_model: nn.Module,
        tokenizer: Callable[[Sequence[str]], Tensor],
        preprocess: Any = None,
    ) -> None:
        super().__init__()
        self.clip = clip_model
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self._validate_model_interface()
        self.freeze()

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_name: str = "ViT-L-14-quickgelu",
        pretrained: str = "openai",
        cache_dir: str | Path = "checkpoints/clip",
        device: str | torch.device = "cpu",
    ) -> FrozenCLIPBackbone:
        """Load an OpenCLIP model and freeze both encoders."""

        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained,
            device=device,
            cache_dir=str(Path(cache_dir).resolve()),
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        return cls(clip_model, tokenizer, preprocess)

    @property
    def device(self) -> torch.device:
        """Return the device holding the CLIP parameters."""

        return next(self.clip.parameters()).device

    @property
    def image_embedding_dim(self) -> int:
        """Dimension of the CLIP visual representation."""

        return int(self.clip.visual.output_dim)

    @property
    def token_embedding_dim(self) -> int:
        """Dimension required for an injected CLIP word token."""

        return int(self.clip.token_embedding.embedding_dim)

    @property
    def logit_scale(self) -> Tensor:
        """Return CLIP's positive contrastive scale."""

        return self.clip.logit_scale.exp()

    def _validate_model_interface(self) -> None:
        required = (
            "visual",
            "token_embedding",
            "positional_embedding",
            "transformer",
            "attn_mask",
            "ln_final",
            "text_pool_type",
            "text_projection",
            "logit_scale",
            "encode_image",
        )
        missing = [name for name in required if not hasattr(self.clip, name)]
        if missing:
            raise TypeError(f"CLIP model is missing required attributes: {', '.join(missing)}")

    def freeze(self) -> None:
        """Freeze CLIP while leaving the surrounding Pic2Word model trainable."""

        self.clip.eval()
        for parameter in self.clip.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> FrozenCLIPBackbone:
        """Keep CLIP in evaluation mode even when its parent model is training."""

        super().train(mode)
        self.clip.eval()
        return self

    def tokenize(self, prompts: Sequence[str]) -> Tensor:
        """Tokenize prompts and move their IDs to the CLIP device."""

        token_ids = self.tokenizer(list(prompts))
        if not isinstance(token_ids, Tensor) or token_ids.ndim != 2:
            raise TypeError("CLIP tokenizer must return a rank-2 torch.Tensor")
        return token_ids.to(self.device)

    def placeholder_token_id(self, placeholder: str = "*") -> int:
        """Return the CLIP token ID for a one-token placeholder such as ``*``."""

        token_ids = self.tokenize([placeholder])[0]
        non_padding_positions = token_ids.ne(0).nonzero(as_tuple=False).flatten()
        if non_padding_positions.numel() != 3:
            raise ValueError(
                f"Placeholder {placeholder!r} must map to exactly one token between SOT and EOT"
            )
        return int(token_ids[non_padding_positions[1]].item())

    def encode_image(self, images: Tensor, *, normalize: bool = False) -> Tensor:
        """Encode images without building gradients through the frozen vision encoder."""

        with torch.no_grad():
            return self.clip.encode_image(images.to(self.device), normalize=normalize)

    def encode_text_embeddings(
        self,
        token_ids: Tensor,
        token_embeddings: Tensor,
        *,
        normalize: bool = False,
    ) -> Tensor:
        """Run CLIP's text encoder from already-created token embeddings."""

        if token_ids.ndim != 2 or token_embeddings.ndim != 3:
            raise ValueError("Expected token IDs [batch, sequence] and embeddings [batch, sequence, dim]")
        if token_ids.shape != token_embeddings.shape[:2]:
            raise ValueError("Token ID and embedding batch/sequence shapes must match")
        if token_embeddings.shape[-1] != self.token_embedding_dim:
            raise ValueError(
                f"Expected token dimension {self.token_embedding_dim}, "
                f"received {token_embeddings.shape[-1]}"
            )

        cast_dtype = self.clip.transformer.get_cast_dtype()
        sequence_length = token_embeddings.shape[1]
        positional_embedding = self.clip.positional_embedding[:sequence_length].to(cast_dtype)
        x = token_embeddings.to(cast_dtype) + positional_embedding
        x = self.clip.transformer(x, attn_mask=self.clip.attn_mask)
        x = self.clip.ln_final(x)
        x = text_global_pool(
            x,
            token_ids,
            self.clip.text_pool_type,
            eos_token_id=getattr(self.clip, "text_eos_id", None),
        )
        if self.clip.text_projection is not None:
            if isinstance(self.clip.text_projection, nn.Linear):
                x = self.clip.text_projection(x)
            else:
                x = x @ self.clip.text_projection
        return functional.normalize(x, dim=-1) if normalize else x

    def encode_prompt_with_pseudo_token(
        self,
        prompts: Sequence[str],
        pseudo_tokens: Tensor,
        *,
        placeholder: str = "*",
        normalize: bool = False,
    ) -> Tensor:
        """Insert one image-derived token into every prompt and encode the result."""

        prompt_list = list(prompts)
        if len(prompt_list) != pseudo_tokens.shape[0]:
            raise ValueError("The number of prompts must equal the pseudo-token batch size")
        invalid = [prompt for prompt in prompt_list if prompt.count(placeholder) != 1]
        if invalid:
            raise ValueError(f"Every prompt must contain exactly one {placeholder!r} placeholder")

        # CLIP's BPE can merge adjacent punctuation, for example ``*,`` becomes one
        # token. Surrounding the placeholder with spaces keeps it independently
        # addressable without changing the semantic prompt shown to the user.
        tokenizable_prompts = [
            " ".join(prompt.replace(placeholder, f" {placeholder} ").split())
            for prompt in prompt_list
        ]
        token_ids = self.tokenize(tokenizable_prompts)
        token_embeddings = self.clip.token_embedding(token_ids)
        injected_embeddings = inject_pseudo_token(
            token_ids,
            token_embeddings,
            pseudo_tokens,
            placeholder_token_id=self.placeholder_token_id(placeholder),
        )
        return self.encode_text_embeddings(
            token_ids,
            injected_embeddings,
            normalize=normalize,
        )
