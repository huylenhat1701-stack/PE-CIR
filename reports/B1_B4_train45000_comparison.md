# So sánh B1 và B4 trên 45.000 mẫu train

Ngày kiểm tra: 25/09/2026.

## Phạm vi và kiểm tra

- Cả hai lượt đánh giá có trạng thái `completed`, đủ 45.000 bản ghi dự đoán.
- Đã tính lại CF-A win, CF-B win, both-win và hard-negative error từ các điểm similarity trong `constraint_scores.json`; các giá trị khớp `metrics.json`.
- Positive thắng khi điểm lớn hơn nghiêm ngặt điểm của counterfactual; điểm bằng nhau không tính thắng. Hard-negative error = 100% − both-win.
- Hai protocol ghi cùng manifest SHA-256: `7412d5b02ba5dda12f8d5f704a6d8be48237ddac06a405aedc1d7175675e5197`. Thứ tự modification và attribute type trong hai file prediction trùng nhau.
- Mỗi protocol ghi 13.582 đường dẫn ảnh duy nhất; 45.000 tuples không phải 45.000 ảnh độc lập. Batch inference = 16, GPU RTX 4090, đầu vào không đổi hash trong quá trình đánh giá.
- Đây là đánh giá Stage-1 trên tập được khai báo là train, trước rerank; không phải full-gallery retrieval hoặc validation độc lập.

## Kết quả

| Chỉ số | B1 (%) | B4 (%) | B4 − B1 (điểm phần trăm) |
|---|---:|---:|---:|
| CF-A win | 94,1067 | 95,8467 | +1,7400 |
| CF-B win | 97,8889 | 98,9533 | +1,0644 |
| Both-win | 92,1822 | 94,8667 | +2,6844 |
| Hard-negative error | 7,8178 | 5,1333 | −2,6844 |

## Đoạn mô tả dùng cho bản thảo

Trên tập 45.000 pseudo-edit tuples được khai báo là tập huấn luyện, B4 đạt tỷ lệ thắng CF-A và CF-B lần lượt là 95,85% và 98,95%, so với 94,11% và 97,89% của B1. Tỷ lệ thắng đồng thời cả hai counterfactual tăng từ 92,18% lên 94,87% (+2,68 điểm phần trăm), trong khi hard-negative error giảm từ 7,82% xuống 5,13%. Kết quả cho thấy B4 đạt hiệu quả phân biệt counterfactual cao hơn B1 trên tập train được đánh giá. Các chỉ số được đo ở Stage-1 trước rerank và không đủ để kết luận về khả năng tổng quát hóa trên dữ liệu chưa thấy.

## Giới hạn diễn giải

- `training_manifest_hash_verified` là `false` ở cả hai lượt: script không có hash lịch sử đủ để xác nhận danh tính manifest với dữ liệu tại thời điểm huấn luyện. Đây không phải thông báo hash không khớp. Hash hiện tại giống nhau hỗ trợ so sánh hai lượt đánh giá trên cùng manifest đã ghi nhận.
- Prediction không chứa ID/đường dẫn ảnh theo từng dòng; đối chiếu thứ tự dựa trên modification và attribute type cùng hash manifest trong protocol.
- Chỉ một checkpoint cho mỗi biến thể; không có kết quả nhiều seed hoặc khoảng tin cậy trong bảng này.
- Kết quả 5.000 mẫu trước đó phải trình bày riêng: both-win B1 = 77,54%, B4 = 76,58%. Cải thiện trên tập train này không đảo ngược kết luận B4 chưa vượt B1 trên tập đánh giá 5.000 mẫu đã báo cáo.
- B4 không có verifier được huấn luyện; `verifier_label_accuracy_percent: null` là phù hợp.

## Tệp nguồn

- B1: `server_download_20260925_103850/B1_train45000/`.
- B4: `B4_train45000_20260925_104544/`.
- Mỗi thư mục chứa `metrics.json`, `constraint_scores.json` và `evaluation_protocol.json`.
