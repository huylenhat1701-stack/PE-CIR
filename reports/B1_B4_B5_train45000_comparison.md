# So sánh B1/B4/B5 trên 45.000 mẫu train

Kiểm tra ngày 25/09/2026. B5 đã hoàn tất inference ngày 23/09/2026 theo protocol; kết quả được tải về và xác minh trong lần kiểm tra này.

## Xác minh dữ liệu

Cả ba lượt có trạng thái `completed`, đủ 45.000 bản ghi. Đã tính lại các tỷ lệ từ điểm positive/CF-A/CF-B và đối chiếu với metrics: tất cả khớp. Positive chỉ thắng khi điểm lớn hơn nghiêm ngặt counterfactual. Hash file prediction B5 khớp protocol. Thứ tự modification/attribute type trùng nhau ở cả ba file.

Cả ba protocol ghi cùng manifest SHA-256 `7412d5b02ba5dda12f8d5f704a6d8be48237ddac06a405aedc1d7175675e5197`, 45.000 tuples và 13.582 đường dẫn ảnh duy nhất. Hash manifest lịch sử lúc train được xác minh cho B5; B1/B4 ghi `training_manifest_hash_verified: false`, không đồng nghĩa hash không khớp. Không có ID ảnh từng dòng trong prediction B1/B4 để đối chiếu trực tiếp với B5.

## Stage-1 trên tập train, trước rerank

| Chỉ số | B1 | B4 | B5 |
|---|---:|---:|---:|
| CF-A win (%) | 94,1067 | 95,8467 | 95,8378 |
| CF-B win (%) | 97,8889 | 98,9533 | 98,9533 |
| Both-win (%) | 92,1822 | 94,8667 | 94,8467 |
| Hard-negative error (%) | 7,8178 | 5,1333 | 5,1533 |
| Số mẫu thắng cả hai | 41.482 | 42.690 | 42.681 |

B4 cải thiện both-win 2,6844 điểm phần trăm so với B1; B5 cải thiện 2,6644 điểm so với B1. B5 thấp hơn B4 0,0200 điểm phần trăm, tương ứng tổng số mẫu thắng cả hai ít hơn 9 mẫu. Đây là chênh lệch mô tả; chưa có kiểm định để kết luận ý nghĩa thống kê hoặc tương đương.

## Verifier B5 trên tập train

Các số dưới đây được đọc từ báo cáo export B5; phần kiểm tra này không tính lại toàn bộ AUROC/verifier metrics.

| Đầu ra | Precision | Recall | F1 | AUROC | Accuracy |
|---|---:|---:|---:|---:|---:|
| Preserve | 0,920029 | 0,987733 | 0,952680 | 0,973329 | 0,934585 |
| Edit | 0,851234 | 0,863000 | 0,857077 | 0,891426 | 0,808119 |
| Violation | 0,834174 | 0,879933 | 0,856443 | 0,884455 | 0,803341 |
| Micro | 0,869012 | 0,910222 | 0,889140 | 0,924962 | 0,848681 |

45.000 tuples tạo ra 135.000 candidates và 405.000 quyết định nhị phân. Các nhãn dựa trên pseudo-edit protocol. Không có verifier đã huấn luyện tương ứng ở B1/B4 để so sánh các chỉ số này.

## Đối chiếu tập đánh giá 5.000 mẫu trước đó

| Chỉ số (%) | B1 | B4 | B5 |
|---|---:|---:|---:|
| CF-A win | 80,04 | 77,98 | 78,06 |
| CF-B win | 95,78 | 97,68 | 97,82 |
| Both-win | 77,54 | 76,58 | 76,84 |
| Hard-negative error | 22,46 | 23,42 | 23,16 |

Không gộp tập 5.000 với 45.000 để tuyên bố kết quả validation. Danh tính manifest đánh giá cũ và tính độc lập ảnh train/validation chưa được xác minh đầy đủ; protocol ghi nhận label noise và image overlap chưa được xử lý. Cải thiện trên train không chứng minh GO gate đã đạt. B5 được huấn luyện exploratory với manual override và khởi tạo mới, không warm-start từ B4.

## Đoạn dùng cho bản thảo

Trên 45.000 pseudo-edit tuples được khai báo là tập huấn luyện, B4 và B5 đạt tỷ lệ thắng đồng thời hai counterfactual lần lượt 94,87% và 94,85%, cao hơn mức 92,18% của B1. Hard-negative error tương ứng giảm từ 7,82% xuống 5,13% và 5,15%. B5 có điểm Stage-1 gần với B4, nhưng không cải thiện chỉ số này trong lượt đánh giá hiện tại. Verifier B5 đạt micro-F1 88,91% và micro-AUROC 0,9250 trên cùng tập train. Những kết quả này mô tả chất lượng trên dữ liệu huấn luyện, chưa chứng minh khả năng tổng quát hóa hoặc lợi ích của reranking trên gallery đầy đủ.

## Nguồn

- B1: `server_download_20260925_103850/B1_train45000/`.
- B4: `B4_train45000_20260925_104544/`.
- B5: `B5_train45000/inference/` và `B5_train45000/export/REPORT.md`.
- Bảng 5.000 mẫu: `B5_verifier_5000/b1_b4_b5_comparison.csv`.
