"""Explore a global verifier strength on stored B5 Top-K probabilities.

This is offline postprocessing, not model inference. The validation datasets
were previously inspected, so the deterministic split is exploratory and
cannot be claimed as a pristine independent holdout.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output/B5_VERIFIER_RETRIEVAL_VALIDATION_20260926.zip"
OUT = ROOT / "output/B5_LAMBDA_EXPERIMENT_20260927"
EPS = 2 ** -23
LAMBDAS = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)


def split_name(dataset: str, row: dict) -> str:
    key = (f"cirr:{row['pair_id']}" if dataset == "cirr"
           else f"fashioniq:{row['category']}:{row['reference_id']}")
    bucket = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % 10
    return "dev" if bucket < 6 else "check"


def annotate(dataset: str, k: int, rows: list[dict]) -> list[dict]:
    result = []
    for row in rows:
        s1 = (row["stage1_target_rank"] if dataset == "cirr"
              else row["rank_info_full_verifier"]["stage1_target_rank"])
        eligible = s1 <= k
        target = row["target_id"]
        candidates = list(row["verifier_top_k_scores"].items())
        if len(candidates) != k or eligible != (target in row["verifier_top_k_scores"]):
            raise AssertionError("Bad Top-K membership")
        record = {"split": split_name(dataset, row), "stage1_rank": s1,
                  "target": target, "category": row.get("category"), "ranks": {},
                  "eligible": eligible}
        if eligible:
            items = [(name, value["stage1_score"],
                      0.5 * math.log(max(value["p_P"], EPS)) +
                      0.5 * math.log(max(value["p_E"], EPS)) -
                      0.5 * math.log(max(value["p_V"], EPS)))
                     for name, value in candidates]
            position = next(i for i, item in enumerate(items) if item[0] == target)
            if position + 1 != s1:
                raise AssertionError("Stage-1 order mismatch")
            group = set(row["group_members"]) - {row["reference_id"]} if dataset == "cirr" else set()
            record["group_hit"] = {}
            for lam in LAMBDAS:
                target_score = items[position][1] + lam * items[position][2]
                before = [item for i, item in enumerate(items) if
                          item[1] + lam * item[2] > target_score or
                          (item[1] + lam * item[2] == target_score and i < position)]
                record["ranks"][lam] = len(before) + 1
                if dataset == "cirr":
                    group_rank = sum(item[0] in group for item in before) + 1
                    record["group_hit"][lam] = {c: group_rank <= c for c in (1, 2, 3)}
            saved = (row["final_target_rank_full_verifier"] if dataset == "cirr"
                     else row["rank_info_full_verifier"]["final_target_rank"])
            if record["ranks"][1.0] != saved:
                raise AssertionError("Original formula rank mismatch")
        else:
            record["ranks"] = {lam: s1 for lam in LAMBDAS}
        result.append(record)
    return result


def evaluate(dataset: str, rows: list[dict], subset: str, lam: float) -> dict:
    subset_rows = [r for r in rows if subset == "all" or r["split"] == subset]
    if not subset_rows:
        raise AssertionError("Empty subset")
    ranks = [r["ranks"][lam] for r in subset_rows]
    base = {"queries": len(subset_rows),
            "eligible": sum(r["eligible"] for r in subset_rows)}
    if dataset == "cirr":
        base["recall_percent"] = {str(c): 100 * sum(rank <= c for rank in ranks) / len(ranks)
                                  for c in (1, 5, 10, 50)}
    else:
        counts = Counter(r["category"] for r in subset_rows)
        per_cat = {cat: {str(c): 100 * sum(r["category"] == cat and r["ranks"][lam] <= c
                                          for r in subset_rows) / counts[cat]
                         for c in (10, 50)} for cat in sorted(counts)}
        mean = {str(c): sum(per_cat[cat][str(c)] for cat in per_cat) / len(per_cat)
                for c in (10, 50)}
        base.update(category_counts=dict(counts), per_category_percent=per_cat,
                    mean_recall_percent=mean,
                    mean_of_r10_r50_percent=(mean["10"] + mean["50"]) / 2)
    base["eligible_movement"] = dict(Counter(
        "up" if r["stage1_rank"] > r["ranks"][lam] else
        "down" if r["stage1_rank"] < r["ranks"][lam] else "same"
        for r in subset_rows if r["eligible"]
    ))
    return base


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_results = []
    group_recall = {}
    with zipfile.ZipFile(SOURCE) as source:
        for dataset in ("cirr", "fashioniq"):
            metrics = json.loads(source.read(f"results/{dataset}/{dataset}_metrics.json"))
            for k in (50, 100):
                raw = json.loads(source.read(f"results/{dataset}/{dataset}_B5_K{k}_per_query.json"))
                rows = annotate(dataset, k, raw)
                for subset in ("dev", "check", "all"):
                    for lam in LAMBDAS:
                        all_results.append({"dataset": dataset, "k": k, "subset": subset,
                                            "lambda": lam, **evaluate(dataset, rows, subset, lam)})
                # Original full-set metrics must be reproduced at lambda=0 and 1.
                mm = metrics[f"B5_K{k}"]["mode_metrics"]["full_verifier"]
                for lam, key in ((0.0, "stage1_recall"), (1.0, "final_recall")):
                    got = next(r for r in all_results if r["dataset"] == dataset and
                               r["k"] == k and r["subset"] == "all" and r["lambda"] == lam)
                    if dataset == "cirr":
                        for cutoff in (1, 5, 10, 50):
                            if abs(got["recall_percent"][str(cutoff)] - mm[key][str(cutoff)]) > 1e-8:
                                raise AssertionError("CIRR Recall mismatch")
                    else:
                        cat_key = "stage1_per_category" if lam == 0 else "final_per_category"
                        for cat in got["per_category_percent"]:
                            for cutoff in (10, 50):
                                if abs(got["per_category_percent"][cat][str(cutoff)] -
                                       mm[cat_key][cat][str(cutoff)]) > 1e-8:
                                    raise AssertionError("Fashion-IQ Recall mismatch")
                if dataset == "cirr":
                    baseline = mm["stage1_group_recall"]
                    group_recall[str(k)] = {}
                    for lam in LAMBDAS:
                        group_recall[str(k)][str(lam)] = {
                            str(c): baseline[str(c)] + 100 * sum(
                                int(r["group_hit"][lam][c]) - int(r["group_hit"][0.0][c])
                                for r in rows if r["eligible"]) / len(rows)
                            for c in (1, 2, 3)
                        }
                    for cutoff in (1, 2, 3):
                        if abs(group_recall[str(k)]["1.0"][str(cutoff)] -
                               mm["group_recall"][str(cutoff)]) > 1e-8:
                            raise AssertionError("CIRR Group Recall mismatch")
                print(dataset, k, "done", flush=True)
    # Selection is specified in advance: use only K=50 dev, choose the smallest
    # lambda among metric ties, then report both K=50 and K=100 check results.
    selected = {}
    for dataset in ("cirr", "fashioniq"):
        dev = [r for r in all_results if r["dataset"] == dataset and r["k"] == 50 and r["subset"] == "dev"]
        score = (lambda r: r["recall_percent"]["10"] if dataset == "cirr"
                 else r["mean_of_r10_r50_percent"])
        chosen = max(dev, key=lambda r: (score(r), -r["lambda"]))
        selected[dataset] = {"lambda": chosen["lambda"], "dev_primary_percent": score(chosen),
                             "objective": "R@10" if dataset == "cirr" else "Mean Recall"}
    summary = {"source": str(SOURCE), "lambdas": LAMBDAS,
               "split": "SHA256(dataset and pair_id/reference_id), first 8 bytes mod 10; <6 dev, >=6 check",
               "warning": "Previously inspected validation data; check subset is exploratory, not pristine independent holdout.",
               "selection": selected, "results": all_results, "cirr_group_recall_all": group_recall}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "grid.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ("dataset", "k", "subset", "lambda", "queries", "eligible", "R1", "R5", "R10", "R50", "MeanRecall")
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for r in all_results:
            rec = {key: r[key] for key in fields[:6]}
            if r["dataset"] == "cirr":
                rec.update({f"R{c}": r["recall_percent"][str(c)] for c in (1, 5, 10, 50)})
            else:
                rec.update(R10=r["mean_recall_percent"]["10"],
                           R50=r["mean_recall_percent"]["50"],
                           MeanRecall=r["mean_of_r10_r50_percent"])
            writer.writerow(rec)
    print("SELECTION", selected)
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
