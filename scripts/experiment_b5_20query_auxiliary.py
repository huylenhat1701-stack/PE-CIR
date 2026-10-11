"""Evaluate a bounded offline verifier-weight experiment on 20 fixed CIRR queries.

No model inference, training, or full-dataset metric pass. Reads stored Top-50
candidate scores for the first 20 eligible queries selected previously.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import zipfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output/B5_VERIFIER_RETRIEVAL_VALIDATION_20260926.zip"
SELECTION = ROOT / "output/B5_VIOLATION_FORMULA_CHECK_20260927/cirr_20_eligible_K50_stage1_old_new.csv"
OUT = ROOT / "output/B5_CIRR_20QUERY_AUXILIARY_20260927"
LAMBDAS = (0.0, 0.01, 0.05, 0.10, 0.20)
EPS = 2 ** -23


def verifier_score(values: dict) -> float:
    return (math.log(max(values["p_P"], EPS))
            + math.log(max(values["p_E"], EPS))
            + math.log(max(1.0 - values["p_V"], EPS)))


def rank_order(candidates: list[dict], lam: float) -> list[dict]:
    # Python's stable sort keeps the Stage-1 order at exact score ties.
    return sorted(candidates, key=lambda c: c["stage1_score"] + lam * c["verifier_score"],
                  reverse=True)


def pearson(xs: list[float], ys: list[float]) -> float:
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    a = [x - mx for x in xs]
    b = [y - my for y in ys]
    denom = math.sqrt(sum(x*x for x in a) * sum(y*y for y in b))
    return sum(x*y for x, y in zip(a, b, strict=True)) / denom if denom else float("nan")


def main() -> None:
    chosen = list(csv.DictReader(SELECTION.open(encoding="utf-8-sig", newline="")))
    assert len(chosen) == 20 and len({int(r["pair_id"]) for r in chosen}) == 20
    with zipfile.ZipFile(SOURCE) as z:
        raw = json.loads(z.read("results/cirr/cirr_B5_K50_per_query.json"))
    rows = [raw[int(item["source_row"]) - 1] for item in chosen]
    assert all(int(row["pair_id"]) == int(item["pair_id"]) and
               row["rerank_eligible"] and row["stage1_target_rank"] <= 50
               for row, item in zip(rows, chosen, strict=True))
    report_rows = []
    all_candidates = []
    selected_candidates = []
    sg_spans, verifier_spans, correlations, target_verifier_percentiles = [], [], [], []
    for row, prior in zip(rows, chosen, strict=True):
        target = row["target_id"]
        candidates = []
        for stage1_rank, (name, values) in enumerate(row["verifier_top_k_scores"].items(), 1):
            candidates.append({
                "pair_id": row["pair_id"], "candidate_id": name,
                "is_target": name == target, "stage1_rank": stage1_rank,
                "stage1_score": values["stage1_score"],
                "p_P": values["p_P"], "p_E": values["p_E"], "p_V": values["p_V"],
                "verifier_score": verifier_score(values),
            })
        assert len(candidates) == 50
        target_item = next(c for c in candidates if c["is_target"])
        assert target_item["stage1_rank"] == row["stage1_target_rank"]
        ordered = {lam: rank_order(candidates, lam) for lam in LAMBDAS}
        ranks = {lam: next(i for i, c in enumerate(ordered[lam], 1) if c["is_target"])
                 for lam in LAMBDAS}
        assert ranks[0.0] == row["stage1_target_rank"]
        # lambda=0.5 here is exactly the previous B formula with 0.5 on each log.
        prior_b_rank = next(i for i, c in enumerate(rank_order(candidates, 0.5), 1)
                            if c["is_target"])
        assert prior_b_rank == int(prior["B_log1m_rank"])
        for lam in LAMBDAS:
            for rank, c in enumerate(ordered[lam], 1):
                c[f"rank_lambda_{lam:.2f}"] = rank
                c[f"score_lambda_{lam:.2f}"] = c["stage1_score"] + lam * c["verifier_score"]
        sg = [c["stage1_score"] for c in candidates]
        v = [c["verifier_score"] for c in candidates]
        sg_spans.append(max(sg) - min(sg))
        verifier_spans.append(max(v) - min(v))
        correlations.append(pearson(sg, v))
        target_verifier_percentiles.append(sum(x <= target_item["verifier_score"] for x in v) / 50)
        report_rows.append({
            "pair_id": row["pair_id"], "reference_id": row["reference_id"],
            "target_id": target, "caption": row["caption"],
            "stage1_rank": ranks[0.0],
            **{f"target_rank_lambda_{lam:.2f}": ranks[lam] for lam in LAMBDAS[1:]},
            "target_stage1_score": target_item["stage1_score"],
            "target_p_P": target_item["p_P"], "target_p_E": target_item["p_E"],
            "target_p_V": target_item["p_V"],
            "target_verifier_score": target_item["verifier_score"],
            "target_verifier_percentile": target_verifier_percentiles[-1],
            "stage1_verifier_pearson": correlations[-1],
        })
        # Every query includes target and representative non-targets:
        # Stage-1 leader, verifier-only leader, and lambda=0.05 leader.
        picks = [("target", target_item),
                 ("stage1_top_non_target", next(c for c in candidates if not c["is_target"])),
                 ("verifier_top_non_target", max((c for c in candidates if not c["is_target"]),
                                                  key=lambda c: c["verifier_score"])),
                 ("lambda_0.05_top_non_target", next(c for c in ordered[0.05]
                                                    if not c["is_target"]))]
        seen = set()
        for role, c in picks:
            if c["candidate_id"] not in seen:
                selected_candidates.append({"role": role, **c})
                seen.add(c["candidate_id"])
        all_candidates.extend(candidates)
    with SOURCE.open("rb") as source_stream:
        archive_sha256 = hashlib.file_digest(source_stream, "sha256").hexdigest()
    summary = {"source_archive_sha256": archive_sha256,
               "source_per_query": "results/cirr/cirr_B5_K50_per_query.json",
               "selection": "first 20 rerank-eligible queries in original file order; pair IDs match prior 20-query CSV",
               "query_count": len(report_rows), "top_k": 50, "lambdas": LAMBDAS,
               "formula": "S_G + lambda * (log(p_P) + log(p_E) + log(1-p_V))",
               "scale": {"median_stage1_top50_span": statistics.median(sg_spans),
                         "median_verifier_top50_span": statistics.median(verifier_spans),
                         "median_stage1_verifier_pearson": statistics.median(correlations),
                         "median_target_verifier_percentile": statistics.median(target_verifier_percentiles)},
               "metrics": {}}
    for lam in LAMBDAS:
        ranks = [r["stage1_rank"] if lam == 0 else r[f"target_rank_lambda_{lam:.2f}"]
                 for r in report_rows]
        baseline = [r["stage1_rank"] for r in report_rows]
        movement = Counter("improved" if new < old else "worse" if new > old else "same"
                           for old, new in zip(baseline, ranks, strict=True))
        summary["metrics"][f"{lam:.2f}"] = {
            "mean_target_rank": statistics.fmean(ranks),
            "median_target_rank": statistics.median(ranks),
            "recall_percent": {str(c): 100 * sum(rank <= c for rank in ranks) / len(ranks)
                               for c in (1, 5, 10)},
            "movement_vs_stage1": {name: movement[name] for name in ("improved", "same", "worse")},
        }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for filename, data in (("target_ranks_20.csv", report_rows),
                           ("selected_candidate_scores_20.csv", selected_candidates),
                           ("all_top50_candidate_scores_20.csv", all_candidates)):
        with (OUT / filename).open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
    print(json.dumps(summary["metrics"], indent=2), flush=True)
    print("SCALE", summary["scale"], flush=True)
    print("WROTE", OUT, flush=True)


if __name__ == "__main__":
    main()
