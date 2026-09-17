"""Preserve/Edit factorization and counterfactual verification for CF-PE-CIR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import Tensor, nn
from torch.nn import functional


@dataclass(frozen=True, slots=True)
class CFPEOutput:
    """Representations produced before gallery retrieval and reranking."""

    query: Tensor
    preserve: Tensor
    edit: Tensor
    composed_edit: Tensor
    gates: Tensor


@dataclass(frozen=True, slots=True)
class ConstraintScores:
    """Probabilities used by the Stage-2 constraint verifier."""

    preserve: Tensor
    edit: Tensor
    violation: Tensor


class PreserveEditFactorizer(nn.Module):
    """Split reference semantics into complementary Preserve and Edit components.

    The frozen CLIP global image embedding is expanded into learnable semantic slots.
    A text-conditioned gate implements the equations in the experiment runbook:
    ``g_i = sigmoid(w^T tanh(W_z z_i + W_t t))`` and complementary weighted pools.
    """

    def __init__(self, embedding_dim: int, *, slots: int = 8, hidden_dim: int = 512) -> None:
        super().__init__()
        if embedding_dim <= 0 or slots <= 1 or hidden_dim <= 0:
            raise ValueError("embedding_dim/hidden_dim must be positive and slots must exceed one")
        self.embedding_dim = embedding_dim
        self.slots = slots
        self.slot_projection = nn.Linear(embedding_dim, slots * embedding_dim)
        self.image_gate = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.text_gate = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.gate_score = nn.Linear(hidden_dim, 1, bias=False)
        self.slot_norm = nn.LayerNorm(embedding_dim)

    def forward(self, reference: Tensor, modification: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if reference.ndim != 2 or modification.ndim != 2 or reference.shape != modification.shape:
            raise ValueError("reference and modification must have identical [batch, dim] shapes")
        if reference.shape[-1] != self.embedding_dim:
            raise ValueError("reference embedding dimension does not match the factorizer")
        slots = self.slot_projection(reference).reshape(-1, self.slots, self.embedding_dim)
        slots = self.slot_norm(slots + reference.unsqueeze(1))
        gate_hidden = torch.tanh(
            self.image_gate(slots) + self.text_gate(modification).unsqueeze(1)
        )
        gates = torch.sigmoid(self.gate_score(gate_hidden)).squeeze(-1)
        eps = torch.finfo(slots.dtype).eps
        edit = (gates.unsqueeze(-1) * slots).sum(dim=1) / (gates.sum(dim=1, keepdim=True) + eps)
        inverse = 1.0 - gates
        preserve = (inverse.unsqueeze(-1) * slots).sum(dim=1) / (
            inverse.sum(dim=1, keepdim=True) + eps
        )
        return preserve, edit, gates


class EditComposer(nn.Module):
    """Compose the edit representation and produce the normalized Stage-1 query."""

    def __init__(self, embedding_dim: int, *, hidden_dim: int = 512) -> None:
        super().__init__()
        self.edit_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.preserve_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.edit_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.text_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, preserve: Tensor, edit: Tensor, modification: Tensor) -> tuple[Tensor, Tensor]:
        composed_edit = self.edit_mlp(torch.cat((edit, modification), dim=-1))
        query = (
            self.preserve_projection(preserve)
            + self.edit_projection(composed_edit)
            + self.text_projection(modification)
        )
        return functional.normalize(query, dim=-1), composed_edit


class ConstraintVerifier(nn.Module):
    """Predict Preserve satisfied, Edit satisfied, and violation probabilities."""

    def __init__(self, embedding_dim: int, *, hidden_dim: int = 512) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 3),
        )

    def forward(
        self,
        candidates: Tensor,
        preserve: Tensor,
        composed_edit: Tensor,
        modification: Tensor,
    ) -> ConstraintScores:
        if candidates.ndim == 3:
            count = candidates.shape[1]
            preserve = preserve.unsqueeze(1).expand(-1, count, -1)
            composed_edit = composed_edit.unsqueeze(1).expand(-1, count, -1)
            modification = modification.unsqueeze(1).expand(-1, count, -1)
        logits = self.network(torch.cat((candidates, preserve, composed_edit, modification), dim=-1))
        probabilities = torch.sigmoid(logits)
        return ConstraintScores(probabilities[..., 0], probabilities[..., 1], probabilities[..., 2])


class CFPECIRModel(nn.Module):
    """Trainable B1/B4/B5 head operating on frozen CLIP embeddings."""

    VALID_VARIANTS: ClassVar[set[str]] = {"B1", "B4", "B5"}

    def __init__(
        self,
        embedding_dim: int,
        *,
        variant: str,
        slots: int = 8,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        variant = variant.upper()
        if variant not in self.VALID_VARIANTS:
            raise ValueError(f"variant must be one of {sorted(self.VALID_VARIANTS)}")
        self.embedding_dim = embedding_dim
        self.variant = variant
        self.factorizer = PreserveEditFactorizer(
            embedding_dim, slots=slots, hidden_dim=hidden_dim
        )
        self.composer = EditComposer(embedding_dim, hidden_dim=hidden_dim)
        self.verifier = ConstraintVerifier(embedding_dim, hidden_dim=hidden_dim)

    @property
    def uses_counterfactuals(self) -> bool:
        return self.variant in {"B4", "B5"}

    @property
    def uses_verifier(self) -> bool:
        return self.variant == "B5"

    def forward(self, reference: Tensor, modification: Tensor) -> CFPEOutput:
        reference = functional.normalize(reference, dim=-1)
        modification = functional.normalize(modification, dim=-1)
        preserve, edit, gates = self.factorizer(reference, modification)
        query, composed_edit = self.composer(preserve, edit, modification)
        return CFPEOutput(query, preserve, edit, composed_edit, gates)

    def verify(
        self,
        candidates: Tensor,
        output: CFPEOutput,
        modification: Tensor,
    ) -> ConstraintScores:
        return self.verifier(
            functional.normalize(candidates, dim=-1),
            output.preserve,
            output.composed_edit,
            functional.normalize(modification, dim=-1),
        )

    def rerank_scores(
        self,
        stage1_scores: Tensor,
        constraints: ConstraintScores,
        *,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 0.5,
    ) -> Tensor:
        """Apply ``S_G + a log p_P + b log p_E - y log p_V`` to Top-K candidates."""

        eps = torch.finfo(stage1_scores.dtype).eps
        return (
            stage1_scores
            + alpha * constraints.preserve.clamp_min(eps).log()
            + beta * constraints.edit.clamp_min(eps).log()
            - gamma * constraints.violation.clamp_min(eps).log()
        )
