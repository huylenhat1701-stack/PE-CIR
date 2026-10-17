"""Ranking and statistical inference metrics for composed image retrieval."""

from __future__ import annotations

from collections.abc import Sequence
import numpy as np


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


def naive_bootstrap_ci(
    diff_array: Sequence[float] | np.ndarray,
    n_boot: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Calculate mean delta and naive 95% bootstrap confidence interval (tuple-level I.I.D.)."""
    diff_arr = np.asarray(diff_array, dtype=np.float64)
    n = len(diff_arr)
    mean_val = float(np.mean(diff_arr) * 100.0)
    rng = np.random.default_rng(seed=seed)
    boot_means = [float(np.mean(rng.choice(diff_arr, size=n, replace=True)) * 100.0) for _ in range(n_boot)]
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)
    return mean_val, float(np.percentile(boot_means, lower_pct)), float(np.percentile(boot_means, upper_pct))


def cluster_bootstrap_ci(
    diff_array: Sequence[float] | np.ndarray,
    cluster_labels: Sequence[str | int] | np.ndarray,
    n_boot: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Calculate cluster-robust bootstrap confidence interval by resampling reference image clusters.

    Properly reflects the data structure (e.g. 4,432 tuples originating from M=102 reference images).
    Preserves intra-cluster correlation to prevent artificially narrow confidence intervals.
    Implemented via vectorized Multinomial sampling (O(B * M) matrix operations), executing 10,000 iterations in <70ms.

    Args:
        diff_array: Sequence of per-sample difference values (e.g. B4_win - B1_win).
        cluster_labels: Cluster identifier for each sample (e.g. reference image path/id).
        n_boot: Number of bootstrap iterations (default 10,000).
        seed: Random seed for reproducibility.
        alpha: Significance level (default 0.05 for 95% CI).

    Returns:
        (mean_delta_percent, ci_lower_percent, ci_upper_percent)
    """
    diff_arr = np.asarray(diff_array, dtype=np.float64)
    clusters = np.asarray(cluster_labels)
    if len(diff_arr) != len(clusters):
        raise ValueError("diff_array and cluster_labels must have identical length")

    unique_c = np.unique(clusters)
    m = len(unique_c)
    if m == 0:
        raise ValueError("cluster_labels must contain at least one cluster")

    # Pre-aggregate sum and counts per cluster -> vectors of length M
    k_vec = np.zeros(m, dtype=np.float64)
    n_vec = np.zeros(m, dtype=np.float64)
    for i, c in enumerate(unique_c):
        mask = clusters == c
        k_vec[i] = np.sum(diff_arr[mask])
        n_vec[i] = np.sum(mask)

    overall_mean = float((np.sum(k_vec) / np.sum(n_vec)) * 100.0)

    # Vectorized multinomial cluster count matrix: shape (n_boot, M)
    rng = np.random.default_rng(seed=seed)
    w_mat = rng.multinomial(m, [1.0 / m] * m, size=n_boot)

    # Fast matrix-vector dot products
    k_boot = w_mat @ k_vec
    n_boot_total = w_mat @ n_vec
    boot_means = (k_boot / n_boot_total) * 100.0

    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)
    ci_lower = float(np.percentile(boot_means, lower_pct))
    ci_upper = float(np.percentile(boot_means, upper_pct))

    return overall_mean, ci_lower, ci_upper
