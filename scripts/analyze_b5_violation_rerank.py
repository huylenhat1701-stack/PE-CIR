"""Re-rank stored B5 Top-K scores offline; no model inference or training.

Reads the verified evidence ZIP and writes a JSON summary plus a deterministic
20-query CIRR K=50 comparison. Candidate order in each JSON is Stage-1 order.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "output/B5_VERIFIER_RETRIEVAL_VALIDATION_20260926.zip"
OUT = ROOT / "output/B5_VIOLATION_FORMULA_CHECK_20260927"
EPS = 2 ** -23  # float32 epsilon used by the source evaluator
FORMULAS = ("stage1", "A_current", "B_log1m", "C_linear")


def scores(candidate: dict) -> dict[str, float]:
    sg = candidate["stage1_score"]
    pe = 0.5 * math.log(max(candidate["p_P"], EPS)) + 0.5 * math.log(max(candidate["p_E"], EPS))
    pv = candidate["p_V"]
    return {
        "stage1": sg,
        "A_current": sg + pe - 0.5 * math.log(max(pv, EPS)),
        "B_log1m": sg + pe + 0.5 * math.log(max(1.0 - pv, EPS)),
        "C_linear": sg + pe - 0.5 * pv,
    }


def rank_stats(deltas: list[int]) -> dict:
    count = Counter("up" if d > 0 else "down" if d < 0 else "same" for d in deltas)
    return {
        "up": count["up"], "same": count["same"], "down": count["down"],
        "mean_delta": statistics.fmean(deltas) if deltas else None,
        "median_delta": statistics.median(deltas) if deltas else None,
    }


def analyze_dataset(dataset: str, k: int, rows: list[dict], original_metrics: dict) -> tuple[dict, list[dict]]:
    ranks = {formula: [] for formula in FORMULAS}
    group_hits = {formula: Counter() for formula in FORMULAS}
    eligible_deltas = {formula: [] for formula in FORMULAS[1:]}
    selected: list[dict] = []
    verification_mismatches = []
    per_category = defaultdict(lambda: {formula: Counter() for formula in FORMULAS})
    category_counts = Counter()
    n_eligible = 0
    violation_terms = []
    sg_spans = []
    preserve_edit_spans = []
    violation_spans = []
    for row_number, row in enumerate(rows, 1):
        target = row["target_id"]
        original = (row["stage1_target_rank"] if dataset == "cirr"
                    else row["rank_info_full_verifier"]["stage1_target_rank"])
        candidates = list(row["verifier_top_k_scores"].items())
        if len(candidates) != k:
            raise AssertionError((dataset, k, row_number, "candidate count", len(candidates)))
        eligible = original <= k
        if eligible != (target in row["verifier_top_k_scores"]):
            raise AssertionError((dataset, k, row_number, "eligibility mismatch"))
        n_eligible += eligible
        all_scores = {candidate: scores(values) for candidate, values in candidates}
        current_terms = [-0.5 * math.log(max(v["p_V"], EPS)) for _, v in candidates]
        preserve_edit_terms = [0.5 * math.log(max(v["p_P"], EPS)) +
                               0.5 * math.log(max(v["p_E"], EPS)) for _, v in candidates]
        violation_terms.extend(current_terms)
        sg_spans.append(max(v["stage1_score"] for _, v in candidates) - min(v["stage1_score"] for _, v in candidates))
        preserve_edit_spans.append(max(preserve_edit_terms) - min(preserve_edit_terms))
        violation_spans.append(max(current_terms) - min(current_terms))
        rankings = {
            formula: sorted((candidate for candidate, _ in candidates),
                            key=lambda candidate: all_scores[candidate][formula], reverse=True)
            for formula in FORMULAS
        }
        if eligible:
            if rankings["stage1"].index(target) + 1 != original:
                raise AssertionError((dataset, k, row_number, "Stage-1 order mismatch"))
            saved_rank = (row["final_target_rank_full_verifier"] if dataset == "cirr"
                          else row["rank_info_full_verifier"]["final_target_rank"])
            if rankings["A_current"].index(target) + 1 != saved_rank:
                verification_mismatches.append((row_number, "rank", target))
        # Independently reproduce each saved final score, also for ineligible queries.
        for candidate, values in candidates:
            if abs(all_scores[candidate]["A_current"] - values["final_score_full_verifier"]) > 3e-6:
                verification_mismatches.append((row_number, "score", candidate))
        for formula in FORMULAS:
            rank = rankings[formula].index(target) + 1 if eligible else original
            ranks[formula].append(rank)
            if eligible and formula != "stage1":
                eligible_deltas[formula].append(original - rank)
            if dataset == "cirr" and eligible:
                group = set(row["group_members"])
                group.discard(row["reference_id"])
                group_rank = sum(candidate in group for candidate in rankings[formula][:rank - 1]) + 1
                for cutoff in (1, 2, 3):
                    group_hits[formula][cutoff] += group_rank <= cutoff
            if dataset == "fashioniq":
                category = row["category"]
                for cutoff in (10, 50):
                    per_category[category][formula][cutoff] += rank <= cutoff
        if dataset == "fashioniq":
            category_counts[row["category"]] += 1
        if dataset == "cirr" and k == 50 and eligible and len(selected) < 20:
            selected.append({
                "source_row": row_number, "pair_id": row["pair_id"],
                "reference_id": row["reference_id"], "target_id": target,
                "caption": row["caption"],
                "stage1_rank": original,
                "A_current_rank": ranks["A_current"][-1],
                "B_log1m_rank": ranks["B_log1m"][-1],
                "C_linear_rank": ranks["C_linear"][-1],
            })
    if verification_mismatches:
        raise AssertionError(f"Original formula not reproduced: {verification_mismatches[:5]}")
    summary = {"dataset": dataset, "k": k, "query_count": len(rows),
               "eligible_count": n_eligible, "original_formula_reproduced": True,
               "formulas": {},
               "score_scale": {
                   "median_stage1_topk_span": statistics.median(sg_spans),
                   "median_preserve_edit_term_span": statistics.median(preserve_edit_spans),
                   "median_current_violation_term_span": statistics.median(violation_spans),
                   "current_violation_term_min": min(violation_terms),
                   "current_violation_term_max": max(violation_terms),
               }}
    for formula in FORMULAS:
        result = {"rank_movement_eligible": rank_stats(eligible_deltas[formula]) if formula != "stage1" else None}
        if dataset == "cirr":
            result["recall_percent"] = {str(c): 100 * sum(rank <= c for rank in ranks[formula]) / len(rows)
                                        for c in (1, 5, 10, 50)}
            stage1_group = original_metrics["B5_K50"]["mode_metrics"]["full_verifier"]["stage1_group_recall"]
            # For an ineligible query, Top-K reranking does not change its group rank.
            # Replace eligible hits relative to Stage-1 in the published baseline.
            result["group_recall_percent"] = {
                str(c): stage1_group[str(c)] +
                        100 * (group_hits[formula][c] - group_hits["stage1"][c]) / len(rows)
                for c in (1, 2, 3)
            }
        else:
            result["per_category_recall_percent"] = {
                category: {str(c): 100 * per_category[category][formula][c] / category_counts[category]
                           for c in (10, 50)}
                for category in sorted(category_counts)
            }
            result["mean_recall_percent"] = {
                str(c): statistics.fmean(result["per_category_recall_percent"][category][str(c)]
                                         for category in category_counts)
                for c in (10, 50)
            }
            result["mean_of_r10_r50_percent"] = statistics.fmean(result["mean_recall_percent"].values())
        summary["formulas"][formula] = result
    # Independent check against archived metrics for both baseline and current.
    mm = original_metrics[f"B5_K{k}"]["mode_metrics"]["full_verifier"]
    for formula, key in (("stage1", "stage1_recall"), ("A_current", "final_recall")):
        if dataset == "cirr":
            for cutoff in (1, 5, 10, 50):
                if abs(summary["formulas"][formula]["recall_percent"][str(cutoff)] - mm[key][str(cutoff)]) > 1e-8:
                    raise AssertionError((dataset, k, formula, cutoff, "Recall mismatch"))
        else:
            key = "stage1_per_category" if formula == "stage1" else "final_per_category"
            for category in category_counts:
                for cutoff in (10, 50):
                    got = summary["formulas"][formula]["per_category_recall_percent"][category][str(cutoff)]
                    if abs(got - mm[key][category][str(cutoff)]) > 1e-8:
                        raise AssertionError((dataset, k, formula, category, cutoff, "Recall mismatch"))
    if dataset == "cirr":
        for cutoff in (1, 2, 3):
            got = summary["formulas"]["A_current"]["group_recall_percent"][str(cutoff)]
            expected = mm["group_recall"][str(cutoff)]
            if abs(got - expected) > 1e-8:
                raise AssertionError((dataset, k, "Group Recall mismatch", cutoff, got, expected))
    return summary, selected


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    examples = []
    with zipfile.ZipFile(ARCHIVE) as source:
        for dataset in ("cirr", "fashioniq"):
            original_metrics = json.loads(source.read(f"results/{dataset}/{dataset}_metrics.json"))
            for k in (50, 100):
                rows = json.loads(source.read(f"results/{dataset}/{dataset}_B5_K{k}_per_query.json"))
                result, selected = analyze_dataset(dataset, k, rows, original_metrics)
                all_summaries.append(result)
                examples.extend(selected)
                print(dataset, k, "done", flush=True)
    (OUT / "summary.json").write_text(json.dumps(all_summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "cirr_20_eligible_K50.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=examples[0].keys())
        writer.writeheader(); writer.writerows(examples)
    label = {"stage1": "Stage-1", "A_current": "A: −log(pV)",
             "B_log1m": "B: +log(1−pV)", "C_linear": "C: −pV"}
    lines = [
        "# Kiểm tra offline ba công thức reranking B5",
        "",
        "Ngày 27/09/2026. Dùng nguyên `stage1_score`, `p_P`, `p_E`, `p_V` của từng candidate trong Top-K đã lưu trong gói kết quả; **không train và không inference lại**. Giữ `α=β=γ=0,5`; chỉ đổi hạng tử violation. Query ngoài Stage-1 Top-K giữ nguyên rank.",
        "",
        "- **A, công thức đã chạy:** `S_G + 0,5log(p_P) + 0,5log(p_E) − 0,5log(p_V)`.",
        "- **B, joint constraint likelihood:** `S_G + 0,5log(p_P) + 0,5log(p_E) + 0,5log(1−p_V)`.",
        "- **C, bounded violation penalty:** `S_G + 0,5log(p_P) + 0,5log(p_E) − 0,5p_V`.",
        "",
        "Với `p_V` là xác suất violation, A không sai dấu theo nghĩa xếp ứng viên `p_V` thấp lên trên: `−log(p_V)` tăng khi `p_V` giảm. Nhưng A cộng **phần thưởng dương có thể rất lớn** cho `p_V` nhỏ; B và C cộng hình phạt không dương. Ba hàm khác nhau cả độ lớn lẫn dạng biến thiên, nên cần đối chiếu ranking thực tế.",
        "",
        "## CIRR validation (4.181 queries, %)",
        "",
        "| K | Công thức | R@1 | R@5 | R@10 | R@50 | Group R@1 | Group R@2 | Group R@3 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for x in all_summaries:
        if x["dataset"] != "cirr": continue
        for formula in FORMULAS:
            r = x["formulas"][formula]
            v = [r["recall_percent"][str(c)] for c in (1, 5, 10, 50)]
            g = [r["group_recall_percent"][str(c)] for c in (1, 2, 3)]
            lines.append(f"| {x['k']} | {label[formula]} | " + " | ".join(f"{n:.3f}" for n in v + g) + " |")
    lines += ["", "## Fashion-IQ validation (6.016 queries, %)", "",
              "`Mean R@K` là trung bình không trọng số của dress/shirt/toptee; `Mean Recall` là trung bình Mean R@10 và Mean R@50.", "",
              "| K | Công thức | Mean R@10 | Mean R@50 | Mean Recall |",
              "|---:|---|---:|---:|---:|"]
    for x in all_summaries:
        if x["dataset"] != "fashioniq": continue
        for formula in FORMULAS:
            r = x["formulas"][formula]
            lines.append(f"| {x['k']} | {label[formula]} | {r['mean_recall_percent']['10']:.3f} | {r['mean_recall_percent']['50']:.3f} | {r['mean_of_r10_r50_percent']:.3f} |")
    lines += ["", "### Fashion-IQ theo nhóm (R@10 / R@50, %)", "",
              "| K | Công thức | Dress | Shirt | Toptee |",
              "|---:|---|---:|---:|---:|"]
    for x in all_summaries:
        if x["dataset"] != "fashioniq": continue
        for formula in FORMULAS:
            p = x["formulas"][formula]["per_category_recall_percent"]
            cells = [f"{p[c]['10']:.3f} / {p[c]['50']:.3f}" for c in ("dress", "shirt", "toptee")]
            lines.append(f"| {x['k']} | {label[formula]} | " + " | ".join(cells) + " |")
    lines += ["", "## Target rank của query eligible", "",
              "Delta = Stage-1 rank − final rank; số dương là target được đẩy lên. Mean/median ở bảng này chỉ tính trên eligible.", "",
              "| Dataset | K | Công thức | Eligible | Lên | Không đổi | Xuống | Mean delta | Median delta |",
              "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for x in all_summaries:
        for formula in FORMULAS[1:]:
            m = x["formulas"][formula]["rank_movement_eligible"]
            lines.append(f"| {x['dataset']} | {x['k']} | {label[formula]} | {x['eligible_count']} | {m['up']} | {m['same']} | {m['down']} | {m['mean_delta']:.3f} | {m['median_delta']:.0f} |")
    lines += ["", "## 20 query CIRR eligible đầu tiên ở K=50", "",
              "Lấy **20 query eligible đầu tiên theo thứ tự gốc của file**, không chọn theo việc công thức nào thắng. Bảng đầy đủ ID/caption ở `cirr_20_eligible_K50.csv`.", "",
              "| Dòng gốc | Pair ID | Stage-1 | A cũ | B mới | C mới |",
              "|---:|---:|---:|---:|---:|---:|"]
    for row in examples:
        lines.append(f"| {row['source_row']} | {row['pair_id']} | {row['stage1_rank']} | {row['A_current_rank']} | {row['B_log1m_rank']} | {row['C_linear_rank']} |")
    lines += ["", "## Kiểm tra scale và kết luận", "",
              "Độ trải trung vị = max−min giữa candidate trong Top-K của mỗi query, rồi lấy trung vị qua các query.", "",
              "| Dataset | K | Stage-1 score | Preserve+Edit term | Hạng tử −0,5log(pV) |",
              "|---|---:|---:|---:|---:|"]
    for x in all_summaries:
        s = x["score_scale"]
        lines.append(f"| {x['dataset']} | {x['k']} | {s['median_stage1_topk_span']:.3f} | {s['median_preserve_edit_term_span']:.3f} | {s['median_current_violation_term_span']:.3f} |")
    lines += ["", "A đã được tính lại độc lập từ xác suất lưu trong JSON: score từng candidate, target rank, Recall và CIRR Group Recall khớp artifact gốc. Điều này xác nhận phép tính offline tái tạo được công thức đã chạy.",
              "", "**Kết luận:** nghi ngờ về scale có cơ sở: các hạng tử verifier biến thiên lớn hơn nhiều so với `S_G`. Tuy nhiên, B/C với cùng hệ số 0,5 **không cải thiện nhất quán** so với A và cả ba vẫn dưới Stage-1 ở các chỉ số tổng hợp chính. Vì vậy chưa thể quy toàn bộ việc giảm Recall cho riêng `−γlog(p_V)`, và **không có tín hiệu để tune λ theo điều kiện 'B/C tốt hơn rõ rệt'**. Không chọn công thức trên test ẩn; nếu làm thí nghiệm tiếp, cần định nghĩa development set độc lập và đánh giá lại trên tập giữ riêng.",
              "", "Giới hạn: đây là thay score trên Top-K và xác suất có sẵn, không chạy mô hình mới; một seed; chưa có kiểm định ý nghĩa thống kê. 20 query chỉ minh họa, kết luận dựa trên toàn bộ 4.181 và 6.016 queries.", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
