# PE-CIR

Triển khai CF-PE-CIR theo runbook E2. CLIP ViT-L/14 luôn đóng băng. Dự án giữ
baseline Pic2Word (B0) và bổ sung Preserve/Edit factorization (B1),
counterfactual ranking với CF-A/CF-B (B4), và constraint verifier rerank Top-K (B5).

## Chạy trên Colab

1. Mở Colab, chọn **File → Upload notebook**, tải `notebooks/CF_PE_CIR_Colab.ipynb`.
2. Chọn **Runtime → Change runtime type → GPU**.
3. Chạy từng ô; ở ô đầu tải `output/CF-PE-CIR-colab-code.zip` đi kèm.
4. Cho phép gắn Drive để lưu checkpoint và log.
5. Giữ `MODE = "smoke"` để kiểm tra pipeline trước; đổi sang `screening` để tạo
   20K–50K pseudo-edits từ nhiều CC3M shard có caption.
6. Kiểm tra `mining_audit.csv`, sau đó xác nhận `AUDIT_APPROVED = True`.
7. Notebook chạy B1, B4, kiểm tra GO gate rồi mới chạy B5 và benchmark đầy đủ.

Colab cấp GPU tùy thời điểm. Batch 4 chỉ là điểm khởi đầu thử nghiệm; nếu OOM,
giảm xuống 2 và chọn RUN_NAME mới. Batch nhỏ không đáp ứng mục tiêu batch 1024
của thí nghiệm lớn; cộng gradient đơn thuần không tăng số negative trong contrastive loss.

## Cài đặt và kiểm thử

Python >=3.11. Cài PyTorch phù hợp GPU trước nếu môi trường chưa có.

```sh
python -m pip install -e ".[dev]"
python -m pytest -q
python scripts/train.py --config configs/train_local.yaml --preflight-only
python scripts/train.py --config configs/train_local.yaml --device cuda
```

Cấu hình đường dẫn được tính từ thư mục chạy lệnh. Thư mục `data/` không nằm trong Git.
CSV huấn luyện cần cột `image` chứa đường dẫn ảnh tương đối với `image_root`.
Tạo thư mục đầu ra mới cho từng lần chạy. Code từ chối ghi đè run cũ; dùng
`--resume /path/to/last.pt` để tiếp tục. Resume giữ batch size, seed và cấu hình mô hình.
Checkpoint lưu số batch đã xử lý trong epoch và RNG Torch; với preprocessing xác định,
num_workers=0 và môi trường không đổi, phần dữ liệu đã chạy sẽ được bỏ qua khi resume.
Đổi môi trường GPU/thư viện có thể làm kết quả số học khác đi.

## Các model và STOP RULE

- B0: Pic2Word baseline.
- B1: P/E factorizer và Stage-1 query, loss `L_ret + 0.2 L_fac`.
- B4: B1 + margin ranking cho CF-A và CF-B, `lambda_CF=1.0`.
- B5: B4 + verifier BCE, `lambda_ver=0.5`, chỉ rerank Top-K 50/100.
- Chỉ chạy B5/full benchmark khi B4 cải thiện CF win và hard-negative error so với B1.
- Không fine-tune full CLIP, không dùng downstream triplet để train, không thêm
  MLLM/diffusion/multi-image trước khi B4/B5 chứng minh giá trị.

Các lệnh chính ngoài notebook:

```sh
python scripts/mine_pseudo_edits.py --source-manifest data/cc3m_captioned.csv \
  --output data/pseudo_edits/train.csv --audit runs/mining_audit.csv \
  --config-output runs/mining_config.json
python scripts/train_cfpe.py --config configs/cfpe_b1.yaml --device cuda
python scripts/train_cfpe.py --config configs/cfpe_b4.yaml --device cuda
python scripts/evaluate_counterfactual.py --manifest data/pseudo_edits/val.csv \
  --image-root data/cc_data/train --checkpoint runs/B4_seed0/best_checkpoint.pt \
  --run-dir runs/B4_seed0 --device cuda
```

CIRR và Fashion-IQ chỉ dùng validation để chọn cấu hình. Script đánh giá từ chối
CIRR thiếu ảnh thay vì tạo kết quả partial. Mỗi run lưu `config.yaml`, `train.log`,
`metrics.json`, `per_query_predictions.json`, `constraint_scores.json`,
`best_checkpoint.pt`, `seed.txt` và `git_commit.txt`.

Nguồn baseline: https://github.com/google-research/composed_image_retrieval
CC3M mẫu: https://huggingface.co/datasets/pixparse/cc3m-wds
Colab: https://research.google.com/colaboratory/faq.html
