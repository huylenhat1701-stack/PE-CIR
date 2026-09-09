"""Build a Markdown acceptance report from Pic2Word training and CIRR metrics."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-summary",
        type=Path,
        default=Path("logs/local_final/pic2word/training_summary.json"),
    )
    parser.add_argument(
        "--training-metrics",
        type=Path,
        default=Path("logs/local_final/pic2word/training_metrics.csv"),
    )
    parser.add_argument(
        "--local-metrics",
        type=Path,
        default=Path("reports/cirr_val_metrics_local_final_partial.json"),
    )
    parser.add_argument(
        "--baseline-metrics",
        type=Path,
        default=Path("reports/cirr_val_metrics_partial.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/pic2word_local_acceptance.md"),
    )
    parser.add_argument("--loss-window", type=int, default=50)
    return parser


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def load_losses(path: Path) -> list[float]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        losses = [float(row["total_loss"]) for row in csv.DictReader(stream)]
    if not losses:
        raise ValueError(f"No training metrics found: {path}")
    return losses


def metric_rows(local: dict[str, Any], baseline: dict[str, Any]) -> str:
    rows: list[str] = []
    for title, key in (("Recall", "global_recall"), ("Group Recall", "group_recall")):
        for k, local_value in local[key].items():
            baseline_value = baseline[key][k]
            rows.append(
                f"| {title}@{k} | {local_value:.2f}% | {baseline_value:.2f}% |"
            )
    return "\n".join(rows)


def main() -> int:
    args = build_parser().parse_args()
    if args.loss_window <= 0:
        raise ValueError("loss-window must be greater than zero")

    summary = load_json(args.training_summary)
    local = load_json(args.local_metrics)
    baseline = load_json(args.baseline_metrics)
    losses = load_losses(args.training_metrics)
    window = min(args.loss_window, len(losses))
    first_loss = statistics.fmean(losses[:window])
    last_loss = statistics.fmean(losses[-window:])
    loss_reduction = 100.0 * (first_loss - last_loss) / first_loss
    comparable = bool(local.get("officially_comparable", not local.get("partial", False)))
    status = "Có thể so sánh chính thức" if comparable else "Chỉ nghiệm thu cục bộ (partial)"

    report = f"""# Báo cáo nghiệm thu cục bộ Pic2Word

## Kết luận

Pipeline `CC3M -> CLIP đóng băng -> Mapping Network -> checkpoint -> CIRR Recall` đã chạy
thành công. Trạng thái đánh giá: **{status}**.

## Huấn luyện

- Số ảnh huấn luyện: {summary['training_samples']}
- Batch size: {summary['batch_size']}
- Optimizer updates: {summary['global_step']}
- Thời gian: {summary['elapsed_seconds']:.2f} giây
- Loss trung bình {window} dòng đầu: {first_loss:.6f}
- Loss trung bình {window} dòng cuối: {last_loss:.6f}
- Mức giảm loss: {loss_reduction:.2f}%
- Checkpoint: `{summary['checkpoint']}`

## Đánh giá CIRR

- Truy vấn được chấm: {local['query_count']}/{local['total_query_count']}
  ({local['query_coverage_percent']:.2f}%)
- Ảnh ứng viên: {local['candidate_count']}/{local['total_candidate_count']}
  ({local['candidate_coverage_percent']:.2f}%)
- Truy vấn bị loại vì thiếu ảnh: {local['excluded_query_count']}

| Metric | Model local 500 bước | Checkpoint Pic2Word epoch 30 |
|---|---:|---:|
{metric_rows(local, baseline)}

## Giới hạn

Kết quả có `partial: true` và `officially_comparable: false`. Candidate pool nhỏ hơn bộ
CIRR đầy đủ và chỉ 36 truy vấn có đủ nhóm ảnh, vì vậy các số Recall này chứng minh code
chạy đúng từ đầu đến cuối nhưng không được dùng như kết quả tái lập chính thức của bài báo.
Muốn nghiệm thu chính thức phải bổ sung đủ 2.297 ảnh CIRR rồi đánh giá lại toàn bộ 4.181
truy vấn mà không dùng `--allow-partial`.
"""
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Saved acceptance report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
