"""Diagnostic of verifier-only retrieval signal on eligible CIRR K=50 queries.

Uses saved probabilities only. Does not train, run model inference, or test a
new combined reranking formula.
"""

from __future__ import annotations

import csv
import bisect
import hashlib
import json
import math
import random
import statistics
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output/B5_VERIFIER_RETRIEVAL_VALIDATION_20260926.zip"
ENTRY = "results/cirr/cirr_B5_K50_per_query.json"
OUT = ROOT / "output/B5_CIRR_VERIFIER_SIGNAL_DIAGNOSTIC_20260927"
EPS = 2 ** -23


def verifier_score(probabilities: dict) -> float:
    return (math.log(max(probabilities["p_P"], EPS))
            + math.log(max(probabilities["p_E"], EPS))
            + math.log(max(1.0 - probabilities["p_V"], EPS)))


def main() -> None:
    with zipfile.ZipFile(SOURCE) as z:
        raw = z.read(ENTRY)
        manifest = json.loads(z.read("MANIFEST.json"))
        expected = manifest["entries"][ENTRY]
        if hashlib.sha256(raw).hexdigest() != expected["sha256"] or len(raw) != expected["bytes"]:
            raise AssertionError("Source per-query JSON differs from evidence manifest")
        all_rows = json.loads(raw)
    if len(all_rows) != 4181:
        raise AssertionError("Unexpected CIRR query count")
    records = []
    positive_scores = []
    negative_scores = []
    for source_row, row in enumerate(all_rows, 1):
        if not row["rerank_eligible"]:
            continue
        target = row["target_id"]
        values = row["verifier_top_k_scores"]
        if len(values) != 50 or target not in values or not 1 <= row["stage1_target_rank"] <= 50:
            raise AssertionError(f"Bad eligible query at source row {source_row}")
        candidates = [(name, verifier_score(probs)) for name, probs in values.items()]
        target_score = next(score for name, score in candidates if name == target)
        others = [(name, score) for name, score in candidates if name != target]
        if len(others) != 49:
            raise AssertionError("Expected 49 non-targets")
        wins = sum(target_score > score for _, score in others)
        ties = sum(target_score == score for _, score in others)
        losses = 49 - wins - ties
        # Stable Stage-1 order only resolves exact score ties for ordinal rank.
        target_index = next(i for i, (name, _) in enumerate(candidates) if name == target)
        target_rank = 1 + sum(score > target_score or
                              (score == target_score and i < target_index)
                              for i, (_, score) in enumerate(candidates) if i != target_index)
        if ties == 0 and target_rank != losses + 1:
            raise AssertionError("Rank and pairwise outcomes disagree")
        best_non_target = max(others, key=lambda item: item[1])
        positive_scores.append(target_score)
        negative_scores.extend(score for _, score in others)
        tp = values[target]
        records.append({
            "source_row": source_row, "pair_id": row["pair_id"],
            "reference_id": row["reference_id"], "target_id": target,
            "caption": row["caption"],
            "stage1_target_rank": row["stage1_target_rank"],
            "verifier_only_target_rank": target_rank,
            "target_stage1_score": tp["stage1_score"],
            "target_p_P": tp["p_P"], "target_p_E": tp["p_E"], "target_p_V": tp["p_V"],
            "target_S_V": target_score,
            "best_non_target_id": best_non_target[0],
            "best_non_target_S_V": best_non_target[1],
            "target_minus_best_non_target_S_V": target_score - best_non_target[1],
            "pairwise_strict_wins": wins, "pairwise_ties": ties,
            "pairwise_losses": losses,
            "pairwise_strict_win_rate": wins / 49,
            "per_query_AUROC": (wins + 0.5 * ties) / 49,
        })
    if len(records) != 491:
        raise AssertionError(f"Expected 491 eligible queries, got {len(records)}")
    n = len(records)
    total_pairs = 49 * n
    wins = sum(r["pairwise_strict_wins"] for r in records)
    ties = sum(r["pairwise_ties"] for r in records)
    losses = sum(r["pairwise_losses"] for r in records)
    verifier_ranks = [r["verifier_only_target_rank"] for r in records]
    stage1_ranks = [r["stage1_target_rank"] for r in records]
    sorted_negatives = sorted(negative_scores)
    global_wins = sum(bisect.bisect_left(sorted_negatives, positive)
                      for positive in positive_scores)
    global_ties = sum(bisect.bisect_right(sorted_negatives, positive) -
                      bisect.bisect_left(sorted_negatives, positive)
                      for positive in positive_scores)
    global_auc = (global_wins + 0.5 * global_ties) / (len(positive_scores) * len(negative_scores))
    per_query_auc = [r["per_query_AUROC"] for r in records]
    rng = random.Random(0)
    boot = sorted(statistics.fmean(per_query_auc[rng.randrange(n)] for _ in range(n))
                  for _ in range(10_000))
    summary = {
        "source_archive": str(SOURCE),
        "source_per_query": ENTRY,
        "source_per_query_sha256": expected["sha256"],
        "scope": "CIRR B5 K=50, only 491 rerank-eligible queries; saved probabilities; no training or inference",
        "score": "S_V=log(p_P)+log(p_E)+log(1-p_V)",
        "positive_definition": "one ground-truth target per query",
        "negative_definition": "other 49 Stage-1 Top-50 candidates in the same query",
        "eligible_queries": n,
        "candidates_per_query": 50,
        "pairwise_comparisons": total_pairs,
        "pairwise_strict_wins": wins,
        "pairwise_ties": ties,
        "pairwise_losses": losses,
        "pairwise_strict_win_rate": wins / total_pairs,
        "AUROC": (wins + 0.5 * ties) / total_pairs,
        "mean_per_query_AUROC": statistics.fmean(r["per_query_AUROC"] for r in records),
        "global_pooled_AUROC": global_auc,
        "query_bootstrap_95_percent_interval_for_within_query_AUROC": [boot[249], boot[9749]],
        "bootstrap_note": "Exploratory resampling of queries (seed 0, 10000 resamples); shared images may reduce independence.",
        "verifier_only": {
            "mean_target_rank": statistics.fmean(verifier_ranks),
            "median_target_rank": statistics.median(verifier_ranks),
            "target_top_counts": {str(k): sum(rank <= k for rank in verifier_ranks) for k in (1, 5, 10)},
            "target_top_percent": {str(k): 100 * sum(rank <= k for rank in verifier_ranks) / n
                                   for k in (1, 5, 10)},
        },
        "stage1_context": {
            "mean_target_rank": statistics.fmean(stage1_ranks),
            "median_target_rank": statistics.median(stage1_ranks),
            "target_top_counts": {str(k): sum(rank <= k for rank in stage1_ranks) for k in (1, 5, 10)},
            "target_top_percent": {str(k): 100 * sum(rank <= k for rank in stage1_ranks) / n
                                   for k in (1, 5, 10)},
        },
        "random_uniform_top50_reference": {
            "mean_rank": 25.5, "median_rank": 25.5,
            "target_top_percent": {"1": 2.0, "5": 10.0, "10": 20.0},
            "pairwise_strict_win_rate": 0.5,
            "AUROC": 0.5,
        },
    }
    if abs(summary["mean_per_query_AUROC"] - summary["AUROC"]) > 1e-12:
        raise AssertionError("Macro and pooled AUC should agree with 49 negatives/query")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "per_query.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("WROTE", OUT, flush=True)


if __name__ == "__main__":
    main()
