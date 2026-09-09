# Báo cáo nghiệm thu cục bộ Pic2Word

## Kết luận

Pipeline `CC3M -> CLIP đóng băng -> Mapping Network -> checkpoint -> CIRR Recall` đã chạy
thành công. Trạng thái đánh giá: **Chỉ nghiệm thu cục bộ (partial)**.

## Huấn luyện

- Số ảnh huấn luyện: 1000
- Batch size: 2
- Optimizer updates: 500
- Thời gian: 200.46 giây
- Loss trung bình 50 dòng đầu: 1.280234
- Loss trung bình 50 dòng cuối: 0.549226
- Mức giảm loss: 57.10%
- Checkpoint: `C:\Pic2word\checkpoints\local_final\pic2word\last.pt`

## Đánh giá CIRR

- Truy vấn được chấm: 36/4181
  (0.86%)
- Ảnh ứng viên: 980/2297
  (42.66%)
- Truy vấn bị loại vì thiếu ảnh: 4145

| Metric | Model local 500 bước | Checkpoint Pic2Word epoch 30 |
|---|---:|---:|
| Recall@1 | 33.33% | 44.44% |
| Recall@5 | 69.44% | 77.78% |
| Recall@10 | 83.33% | 91.67% |
| Recall@50 | 86.11% | 97.22% |
| Group Recall@1 | 66.67% | 58.33% |
| Group Recall@2 | 77.78% | 69.44% |
| Group Recall@3 | 83.33% | 94.44% |

## Giới hạn

Kết quả có `partial: true` và `officially_comparable: false`. Candidate pool nhỏ hơn bộ
CIRR đầy đủ và chỉ 36 truy vấn có đủ nhóm ảnh, vì vậy các số Recall này chứng minh code
chạy đúng từ đầu đến cuối nhưng không được dùng như kết quả tái lập chính thức của bài báo.
Muốn nghiệm thu chính thức phải bổ sung đủ 2.297 ảnh CIRR rồi đánh giá lại toàn bộ 4.181
truy vấn mà không dùng `--allow-partial`.
