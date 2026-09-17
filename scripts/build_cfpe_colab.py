"""Generate the guided CF-PE-CIR Colab notebook."""

from __future__ import annotations

import json
from pathlib import Path


def md(identifier: str, text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True), "id": identifier}


def code(identifier: str, text: str) -> dict:
    return {
        "cell_type": "code", "metadata": {}, "source": text.splitlines(True),
        "outputs": [], "execution_count": None, "id": identifier,
    }


cells = [
    md("cfpe-000", """# CF-PE-CIR theo runbook E2
Notebook này giữ CLIP ViT-L/14 đóng băng và tách rõ B0, B1, B4, B5. Chế độ `smoke` chỉ kiểm tra kỹ thuật; chế độ `screening` mới tạo 20K–50K pseudo-edits. Không dùng triplet CIRR/Fashion-IQ để train.
"""),
    code("cfpe-001", """MODE = "smoke"  # đổi thành "screening" sau khi smoke test đạt
AUDIT_APPROVED = False  # chỉ đặt True sau khi đã xem mining_audit.csv
RUN_B0_SCHEDULE = False  # bật nếu cần chạy lại B0 1K -> 10K -> 100K
RUN_B5_AFTER_GO = True
USE_GOOGLE_DRIVE = False  # smoke chạy ổn định trong /content; bật khi cần giữ checkpoint lâu dài
SETTINGS = {
    "smoke": {"num_shards": 2, "images_per_shard": 2000, "min_tuples": 20, "max_tuples": 1000, "steps": 20},
    "screening": {"num_shards": 20, "images_per_shard": 5000, "min_tuples": 20000, "max_tuples": 50000, "steps": 2000},
}[MODE]
print(MODE, SETTINGS)
"""),
    md("cfpe-002", """## 1. Nạp code và kiểm thử
Notebook tải trực tiếp phiên bản mới nhất từ nhánh `main`, sau đó cài dependency và chạy test.
"""),
    code("cfpe-003", """from pathlib import Path
import csv, json, os, shutil, subprocess, sys
repo = Path("/content/PE-CIR")
if repo.exists():
    shutil.rmtree(repo)
subprocess.run(["git", "clone", "--depth", "1",
    "https://github.com/huylenhat1701-stack/PE-CIR.git", str(repo)], check=True)
os.chdir(repo)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", ".[dev]", "huggingface_hub"], check=True)
subprocess.run([sys.executable, "-m", "pytest", "-q"], check=True)
"""),
    md("cfpe-004", """## 2. GPU và Google Drive
Mặc định smoke test lưu tạm trong `/content/CF-PE-CIR-output`. Đặt `USE_GOOGLE_DRIVE=True` ở ô đầu nếu muốn giữ dữ liệu và checkpoint sau khi phiên Colab kết thúc.
"""),
    code("cfpe-005", """import torch
assert torch.cuda.is_available(), "Hãy chọn Runtime > Change runtime type > GPU"
print(torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), "GB")
if USE_GOOGLE_DRIVE:
    from google.colab import drive
    if not Path("/content/drive/MyDrive").is_dir():
        drive.mount("/content/drive")
    drive_root = Path("/content/drive/MyDrive/CF-PE-CIR")
else:
    drive_root = Path("/content/CF-PE-CIR-output")
data_root = drive_root / "data"
runs_root = drive_root / "runs"
data_root.mkdir(parents=True, exist_ok=True)
runs_root.mkdir(parents=True, exist_ok=True)
print("Thư mục kết quả:", drive_root)
"""),
    md("cfpe-006", """## 3. Tải CC3M có caption
Mỗi shard khoảng 488 MB. `smoke` tải ít dữ liệu để kiểm tra code. `screening` cần nhiều shard vì yêu cầu 20K–50K pseudo-edits.
"""),
    code("cfpe-007", """from huggingface_hub import hf_hub_download, HfApi
revision = HfApi().dataset_info("pixparse/cc3m-wds").sha
image_root = data_root / "cc3m_images"
manifest_dir = data_root / "cc3m_manifests"
image_root.mkdir(exist_ok=True)
manifest_dir.mkdir(exist_ok=True)
for shard_id in range(SETTINGS["num_shards"]):
    filename = f"cc3m-train-{shard_id:04d}.tar"
    manifest = manifest_dir / f"{shard_id:04d}.csv"
    if manifest.is_file():
        continue
    archive = Path(hf_hub_download(repo_id="pixparse/cc3m-wds", repo_type="dataset",
        filename=filename, revision=revision, cache_dir="/content/hf-cache"))
    subprocess.run([sys.executable, "scripts/prepare_cc3m_shard.py", str(archive),
        "--output-dir", str(image_root), "--manifest", str(manifest),
        "--max-images", str(SETTINGS["images_per_shard"])], check=True)
combined = data_root / "cc3m_captioned.csv"
rows = []
for path in sorted(manifest_dir.glob("*.csv")):
    with path.open(encoding="utf-8") as stream:
        rows.extend(csv.DictReader(stream))
with combined.open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=["image", "caption"])
    writer.writeheader(); writer.writerows(rows)
captioned = sum(bool(row["caption"].strip()) for row in rows)
print("Ảnh:", len(rows), "có caption:", captioned, "revision:", revision)
"""),
    md("cfpe-008", """## 4. Mine pseudo-edits, CF-A và CF-B
Mở `mining_audit.csv` trên Drive và xem tối thiểu 500 dòng ở chế độ screening. Xác nhận Positive giữ Preserve và thỏa Edit; CF-A giữ Preserve nhưng sai Edit; CF-B thỏa Edit nhưng sai Preserve.
"""),
    code("cfpe-009", """pseudo_dir = data_root / f"pseudo_{MODE}"
pseudo_dir.mkdir(exist_ok=True)
all_tuples = pseudo_dir / "all.csv"
command = [sys.executable, "scripts/mine_pseudo_edits.py", "--source-manifest", str(combined),
    "--output", str(all_tuples), "--audit", str(pseudo_dir / "mining_audit.csv"),
    "--config-output", str(pseudo_dir / "mining_config.json"),
    "--min-tuples", str(SETTINGS["min_tuples"]), "--max-tuples", str(SETTINGS["max_tuples"]), "--seed", "0"]
if MODE == "smoke": command.append("--allow-small")
subprocess.run(command, check=True)
with all_tuples.open(encoding="utf-8") as stream:
    tuples = list(csv.DictReader(stream))
split = max(1, int(len(tuples) * 0.9))
for name, subset in (("train.csv", tuples[:split]), ("val.csv", tuples[split:])):
    with (pseudo_dir / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(tuples[0])); writer.writeheader(); writer.writerows(subset)
print("train:", split, "val:", len(tuples) - split, "audit:", pseudo_dir / "mining_audit.csv")
if MODE == "screening" and not AUDIT_APPROVED:
    raise RuntimeError("Dừng đúng thiết kế: xem mining_audit.csv rồi đặt AUDIT_APPROVED=True")
"""),
    md("cfpe-010", """## 5. Tùy chọn chạy lại B0 theo 1K → 10K → 100K
B0 dùng Pic2Word gốc. Mỗi quy mô tạo run riêng, không trộn checkpoint.
"""),
    code("cfpe-011", """import yaml
if RUN_B0_SCHEDULE:
    for sample_count in (1000, 10000, 100000):
        cfg = yaml.safe_load(Path("configs/train.yaml").read_text())
        cfg["training"].update(batch_size_per_device=4, max_steps=SETTINGS["steps"], seed=0, num_workers=0)
        cfg["data"].update(train_manifest=str(combined), image_root=str(image_root), max_samples=min(sample_count, len(rows)))
        root = runs_root / f"B0_{sample_count}_seed0"
        cfg["output"] = {"checkpoint_dir": str(root / "checkpoints"), "log_dir": str(root / "logs")}
        path = repo / f"configs/colab_b0_{sample_count}.yaml"; path.write_text(yaml.safe_dump(cfg))
        subprocess.run([sys.executable, "scripts/train.py", "--config", str(path), "--device", "cuda"], check=True)
else:
    print("Giữ checkpoint B0 hiện có. Bật RUN_B0_SCHEDULE nếu cần chạy lại.")
"""),
    md("cfpe-012", """## 6. Train B1 rồi B4
Loss khởi tạo đúng runbook: λfac=0.2, λCF=1.0. Mỗi model có thư mục artifact riêng.
"""),
    code("cfpe-013", """def run_variant(variant):
    cfg = yaml.safe_load(Path(f"configs/cfpe_{variant.lower()}.yaml").read_text())
    cfg["training"].update(batch_size_per_device=4, max_steps=SETTINGS["steps"], num_workers=0, seed=0)
    cfg["data"].update(train_manifest=str(pseudo_dir / "train.csv"), image_root=str(image_root))
    run_dir = runs_root / f"{variant}_{MODE}_seed0"
    cfg["output"]["run_dir"] = str(run_dir)
    cfg_path = repo / f"configs/colab_{variant.lower()}.yaml"; cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    subprocess.run([sys.executable, "-u", "scripts/train_cfpe.py", "--config", str(cfg_path), "--device", "cuda"], check=True)
    subprocess.run([sys.executable, "scripts/evaluate_counterfactual.py",
        "--manifest", str(pseudo_dir / "val.csv"), "--image-root", str(image_root),
        "--checkpoint", str(run_dir / "best_checkpoint.pt"), "--run-dir", str(run_dir),
        "--batch-size", "4", "--device", "cuda"], check=True)
    return run_dir
b1_dir = run_variant("B1")
b4_dir = run_variant("B4")
"""),
    md("cfpe-014", """## 7. GO gate và B5
B4 phải tăng cả CF-A/CF-B win hoặc giảm hard-negative error so với B1. Nếu không đạt, notebook dừng trước B5/full benchmark.
"""),
    code("cfpe-015", """def cf_metrics(path):
    return json.loads((path / "metrics.json").read_text())["counterfactual_evaluation"]
m1, m4 = cf_metrics(b1_dir), cf_metrics(b4_dir)
go = (m4["both_win_percent"] > m1["both_win_percent"] and
      m4["hard_negative_error_percent"] < m1["hard_negative_error_percent"])
print("B1:", m1); print("B4:", m4); print("GO B5:", go)
if go and RUN_B5_AFTER_GO:
    b5_dir = run_variant("B5")
else:
    b5_dir = None
    print("Dừng theo STOP RULE; chưa chạy B5/full package.")
"""),
    md("cfpe-016", """## 8. CIRR và Fashion-IQ validation
Chỉ chạy khi ảnh và annotation chính thức đầy đủ. Không dùng các benchmark này để train. CIRR thiếu ảnh sẽ dừng thay vì báo cáo partial.
"""),
    code("cfpe-017", """CIRR_ROOT = data_root / "cirr"
FASHIONIQ_ROOT = data_root / "fashioniq"
if b5_dir is not None:
    checkpoint = b5_dir / "best_checkpoint.pt"
    if CIRR_ROOT.is_dir():
        subprocess.run([sys.executable, "scripts/evaluate_cfpe_cirr.py",
            "--dataset-root", str(CIRR_ROOT), "--checkpoint", str(checkpoint),
            "--index", str(data_root / "indexes/cirr_val.pt"), "--run-dir", str(b5_dir / "cirr"),
            "--top-k", "50", "--device", "cuda"], check=True)
    else: print("Chưa có CIRR đầy đủ tại", CIRR_ROOT)
    if FASHIONIQ_ROOT.is_dir():
        subprocess.run([sys.executable, "scripts/evaluate_cfpe_fashioniq.py",
            "--dataset-root", str(FASHIONIQ_ROOT), "--checkpoint", str(checkpoint),
            "--index-dir", str(data_root / "indexes"), "--run-dir", str(b5_dir / "fashioniq"),
            "--top-k", "100", "--device", "cuda"], check=True)
    else: print("Chưa có Fashion-IQ đầy đủ tại", FASHIONIQ_ROOT)
"""),
    md("cfpe-018", """## 9. Kiểm tra artifact
Mỗi run phải có config, log, metrics, checkpoint, seed và git commit. Sau benchmark phải có thêm dự đoán từng query và constraint scores.
"""),
    code("cfpe-019", """for run_dir in [b1_dir, b4_dir] + ([b5_dir] if b5_dir else []):
    required = ["config.yaml", "train.log", "metrics.json", "per_query_predictions.json",
        "constraint_scores.json", "best_checkpoint.pt", "seed.txt", "git_commit.txt"]
    missing = [name for name in required if not (run_dir / name).is_file()]
    print(run_dir.name, "OK" if not missing else f"thiếu {missing}")
"""),
]

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "colab": {"name": "CF_PE_CIR_Colab.ipynb"},
        "accelerator": "GPU",
    },
    "cells": cells,
}
output = Path("notebooks/CF_PE_CIR_Colab.ipynb")
output.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
print(output.resolve())
