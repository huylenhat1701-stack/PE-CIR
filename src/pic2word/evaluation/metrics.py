"""Ranking metrics shared by retrieval benchmarks."""

from __future__ import annotations

from collections.abc import Sequence


def recall_at_k(
    rankings: Sequence[Sequence[str]],
    target_ids: Sequence[str],
    ks: Sequence[int],
) -> dict[int, float]:
    """Compute Recall@K percentages for one ground-truth target per query."""

    if len(rankings) != len(target_ids):
        raise ValueError("Ranking and target counts must match")
    if not rankings:
        raise ValueError("At least one ranking is required")
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("Recall cutoffs must be positive integers")

    query_count = len(target_ids)
    return {
        k: 100.0
        * sum(target_id in ranking[:k] for ranking, target_id in zip(rankings, target_ids))
        / query_count
        for k in ks
    }
