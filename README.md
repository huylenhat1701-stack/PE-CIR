# PE-CIR

Triển khai baseline Pic2Word để chuẩn bị thí nghiệm Preserve–Edit theo
`PE-CIR_Experimental.docx`. CLIP ViT-L/14 đóng băng; chỉ train mapper MLP.

## Chạy trên Colab

1. Mở Colab, chọn **File → Upload notebook**, tải `notebooks/PE_CIR_Colab.ipynb`.
2. Chọn **Runtime → Change runtime type → GPU**.
3. Chạy từng ô; ở ô đầu tải `PE-CIR-colab-code.zip` đi kèm.
4. Cho phép gắn Drive để lưu checkpoint và log.
5. Notebook tải một shard CC3M, lấy tối đa 1.000 ảnh, train thử 20 bước với seed 0.
6. Xem log trước khi bật tiếp tục 200 bước. Đây vẫn là smoke test.

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

## Tiêu chí tiến tới PE-CIR

- Xác nhận Pic2Word trên CIRR val và Fashion-IQ val đầy đủ trước khi triển khai P/E gate.
- Báo cáo cũ chỉ có 36/4.181 truy vấn CIRR và không được dùng làm kết quả tái lập.
- Đánh giá CIRR lưu metrics cùng `.predictions.jsonl`. Thiếu ảnh hoặc giới hạn số query
  đều được gắn nhãn partial. Loader Fashion-IQ chưa được triển khai.
- Dùng validation để chọn hyperparameter; không tune test hoặc train bằng triplet benchmark.
- Khi baseline đạt yêu cầu: tạo pseudo-edit CC3M, kiểm tra chất lượng, rồi lần lượt
  A1–A5; log gate collapse, preserve/edit losses và đánh giá hard negative.
- Screening PE-CIR dùng 20–30% dữ liệu và seed 0; xác nhận full data; kết quả cuối
  dùng cố định seed 0, 1, 2 và báo cáo mean ± std.

Nguồn baseline: https://github.com/google-research/composed_image_retrieval
CC3M mẫu: https://huggingface.co/datasets/pixparse/cc3m-wds
Colab: https://research.google.com/colaboratory/faq.html
