# B5 FINAL AUDIT

Ngày lập: 23/09/2026. Run: `B5_override_20260922-085644`.

**Phạm vi:** tổng hợp bằng chứng hiện có và tính lại các chỉ số verifier từ prediction đã lưu của **5.000 samples**. Không huấn luyện lại, không sửa checkpoint, dataset hoặc training protocol. Đây là đánh giá counterfactual với ba candidate cho mỗi sample, không phải đánh giá retrieval trên toàn bộ gallery CIRR/Fashion-IQ.

**Trạng thái bằng chứng:** đã kiểm tra số lượng prediction, tính lại metrics/confusion matrix/Risk–Coverage và đối chiếu accuracy gốc. Chưa đủ bằng chứng để chứng nhận hoàn toàn danh tính split, tính độc lập train/validation hoặc liên kết checkpoint tại thời điểm inference.

## 1. Checkpoint

| Trường | Giá trị |
|---|---|
| Thư mục dự án trên server | `/home/iec/PE-CIR-20260921-134120` |
| Run | `runs/B5_override_20260922-085644` |
| Checkpoint được kiểm tra | `runs/B5_override_20260922-085644/best_checkpoint.pt` |
| Variant trong checkpoint | `B5` |
| Dung lượng | 120.485.982 bytes |
| `global_step` trong checkpoint | 12.605 |
| `best_loss` trong checkpoint | 0.3499754071235657 |
| Bước hoàn tất training trong log | 14.060 |
| Backbone theo config | `ViT-L-14-quickgelu`, pretrained `openai`; thiết kế giữ CLIP đóng băng |
| Head | `slots=8`, `hidden_dim=512` |

SHA-256 của checkpoint trong gói bằng chứng đã lưu:

```text
086eb8a18dff913fc933b55e3e6e784634311977066788d5cf6617062051ad1f
```

Theo mã `scripts/train_cfpe.py` hiện có, `best_checkpoint.pt` được chọn bằng **total training loss nhỏ nhất của một batch**, không phải validation loss hay validation F1 tốt nhất. Checkpoint tốt nhất và checkpoint cuối training vì vậy không nhất thiết cùng bước.

Hash trên được kiểm tra từ checkpoint trong gói `PE-CIR-training-results.zip`. File prediction cũ không ghi hash checkpoint lúc inference; việc nằm cùng thư mục run chưa đủ chứng minh bằng mật mã rằng prediction được tạo từ chính checkpoint này. Lượt export hiện tại chỉ đọc xác suất đã lưu, không chạy inference mới.

## 2. Seed

- Seed training của B1/B4/B5: **0**, theo `seed.txt` và `config.yaml` của từng run.
- Seed mining: **0**, theo `mining_config.original.json`.
- Seed/quy tắc chia split lịch sử: **chưa xác minh được từ gói bằng chứng**.
- Export metrics và Risk–Coverage không lấy mẫu ngẫu nhiên; dùng toàn bộ 5.000 record theo thứ tự JSON nguồn.
- Không có thí nghiệm nhiều seed trong bộ bằng chứng này; không báo cáo độ lệch chuẩn hay khoảng tin cậy giữa các seed.

## 3. Git commit

`git_commit.txt` của cả B1, B4 và B5 ghi **`unknown`**. Theo ghi chú đóng gói, mã nguồn được chuyển bằng ZIP nên không lưu được revision Git của lần chạy.

Không dùng commit của workspace hiện tại để thay thế commit lịch sử. Các hash file ở báo cáo này hỗ trợ truy vết artifact, không khôi phục được revision mã nguồn lúc training/inference.

## 4. Dataset + split

| Thành phần | Bằng chứng hiện có |
|---|---|
| Nguồn ảnh/caption theo mining config | `/home/iec/datasets/cc3m/subset_5000/cc3m_captioned.csv` |
| Image root theo training config | `/home/iec/datasets/cc3m/subset_5000/images` |
| Train manifest | `/home/iec/datasets/cc3m/pseudo_screening/train.csv` |
| Số train samples theo B5 run protocol | 45.000 pseudo-edit tuples |
| Tập đánh giá đã lưu | 5.000 records trong `constraint_scores.json` của từng B1/B4/B5 |
| Danh tính manifest đánh giá | Chưa xác minh: thiếu đường dẫn/hash manifest trong output đánh giá cũ |
| Image-disjoint train/validation | Không được xác nhận; run protocol ghi nhận vấn đề overlap đã biết |

Một sample là một tuple gồm **reference, positive, CF-A, CF-B và modification**, không đồng nghĩa với một ảnh duy nhất. Ảnh reference dùng để tạo query; ba ảnh còn lại là candidate được chấm điểm. Tên thư mục `subset_5000` không phải bằng chứng về số ảnh độc lập thực tế.

Phân bố `attribute_type` trong 5.000 prediction B5:

| Loại thuộc tính | Số samples | Tỷ lệ |
|---|---:|---:|
| Color | 4.660 | 93,20% |
| Size/shape | 286 | 5,72% |
| Material | 52 | 1,04% |
| Pattern | 2 | 0,04% |
| Tổng | 5.000 | 100% |

Số lượng 45.000 train và 5.000 evaluation phù hợp với cách chia 90/10 của 50.000 tuples đã mine, nhưng **chưa có manifest/quy trình split để xác nhận việc chia đó, tính bao phủ hoặc không trùng lặp**. `validation_manifest_identity_verified=false` trong run protocol.

## 5. Nguồn gốc 45.000 samples

`mining/mining_config.original.json` trong gói bằng chứng ghi:

| Trường | Giá trị |
|---|---|
| Source manifest | `/home/iec/datasets/cc3m/subset_5000/cc3m_captioned.csv` |
| `records_scanned` | 100.000 |
| `tuples_mined` | 50.000 |
| `max_tuples` | 50.000 |
| `min_tuples` | 20.000 |
| Seed | 0 |
| Thứ tự thuộc tính | color, pattern, material, size_shape |

Theo mã mining hiện có, các tuple được tạo bằng đối sánh từ khóa đối tượng và thuộc tính trong caption: positive cùng nhóm đối tượng nhưng đổi thuộc tính; CF-A giữ thuộc tính trước chỉnh sửa; CF-B có thuộc tính đích nhưng khác nhóm đối tượng. Đây là nhãn giả từ quy tắc caption, không phải nhãn được người đánh giá xác nhận cho toàn bộ ảnh.

B5 tái sử dụng train manifest của B4; `run_protocol.json` ghi **45.000 train samples** và hash của manifest:

```text
7412d5b02ba5dda12f8d5f704a6d8be48237ddac06a405aedc1d7175675e5197
```

Hash train manifest này được trích từ run protocol, chưa được tính lại từ CSV gốc trong lần lập báo cáo. Thiếu file split gốc nên chưa xác định được chính xác quy tắc chọn 45.000 tuples từ 50.000 tuples. `records_scanned=100000` cũng không chứng minh có 100.000 ảnh duy nhất.

**Phân biệt hai con số 45.000:** 45.000 ở trên là số tuples huấn luyện. Trong đánh giá, 5.000 samples × 3 candidates × 3 đầu ra verifier cũng bằng **45.000 quyết định nhị phân**, nhưng là đơn vị và tập dữ liệu khác.

## 6. Risk definition

Risk được định nghĩa là **tỷ lệ phân loại sai của verifier trong các quyết định được giữ lại**.

Với xác suất `p_i`, nhãn `y_i` và tập giữ lại `A(t)`:

```text
y_hat_i = 1 nếu p_i >= 0.5; ngược lại bằng 0
A(t) = {i: confidence_i >= t}
Risk(t) = tổng 1[y_hat_i != y_i] trên A(t) / |A(t)|
Coverage(t) = |A(t)| / N
```

Tính riêng cho Preserve, Edit, Violation với `N=15000` mỗi đầu ra; micro gộp cả ba với `N=45000`. Không tính risk tại coverage bằng 0 vì mẫu số bằng 0. Risk này **không phải** lỗi top-1 retrieval, hard-negative error hoặc xác suất violation trung bình.

## 7. Confidence definition

```text
confidence_i = max(p_i, 1 - p_i)
```

Confidence nằm trong [0.5, 1], biểu thị độ tin cậy của **nhãn nhị phân được dự đoán**. Chẳng hạn, `p_violation=0.95` có confidence 0.95 cho dự đoán “có violation”; không được diễn giải là candidate tốt với độ tin cậy 0.95.

Không dùng nhãn thật để xếp hạng confidence, không dùng retrieval similarity thay cho confidence, không thực hiện calibration hoặc tối ưu ngưỡng trên tập đánh giá. Confidence này không được khẳng định là xác suất đúng đã hiệu chuẩn.

## 8. Risk–Coverage protocol

1. Đọc đầy đủ 5.000 records từ `constraint_scores.json`; từ chối số lượng khác 5.000 hoặc xác suất không hợp lệ. Không cắt mẫu, bootstrap hoặc bỏ mẫu để cải thiện kết quả.
2. Với mỗi record, lấy cả positive, CF-A và CF-B, mỗi candidate có ba xác suất theo thứ tự Preserve/Edit/Violation.
3. Áp dụng ngưỡng phân loại cố định 0.5 và confidence tại mục 7.
4. Sắp xếp confidence giảm dần, quét mọi mức confidence khác nhau. Các quyết định bằng confidence được giữ lại **cùng một nhóm**, không phá tie bằng nhãn đúng/sai.
5. Xuất Coverage và Risk tại từng mức cho ba đầu ra và micro. Điểm cuối đạt coverage 1.0; risk tại điểm này bằng `1 - accuracy`.
6. Lưu toàn bộ điểm trong `risk_coverage.csv`. `risk_coverage.svg` chỉ giảm số điểm trùng cột pixel để vẽ gọn; CSV giữ nguyên toàn bộ mức ngưỡng. Không báo cáo AURC vì bản export hiện tại không tính chỉ số này.

| Đầu ra | Số quyết định | Risk tại coverage 100% |
|---|---:|---:|
| Preserve | 15.000 | 0.090600 |
| Edit | 15.000 | 0.278267 |
| Violation | 15.000 | 0.286267 |
| Micro | 45.000 | 0.218378 |

Artifact: [CSV đầy đủ](risk_coverage.csv), [biểu đồ SVG](risk_coverage.svg).

## 9. Verifier metrics

Nhãn theo protocol loss/evaluation hiện có:

| Candidate | Preserve | Edit | Violation |
|---|---:|---:|---:|
| Positive | 1 | 1 | 0 |
| CF-A | 1 | 0 | 1 |
| CF-B | 0 | 1 | 1 |

Ngưỡng nhị phân là `p >= 0.5`. Precision, Recall, F1 tính cho lớp 1. AUROC tính từ xác suất liên tục, cho nửa điểm với cặp có cùng score. Precision/Recall/F1 dùng 0 nếu mẫu số tương ứng bằng 0; trường hợp đó không xảy ra ở kết quả này.

| Verifier | Precision | Recall | F1 | AUROC | Accuracy |
|---|---:|---:|---:|---:|---:|
| Preserve | 0.916755 | 0.950400 | 0.933274 | 0.956352 | 0.909400 |
| Edit | 0.763859 | 0.843300 | 0.801616 | 0.791563 | 0.721733 |
| Violation | 0.731312 | 0.902000 | 0.807737 | 0.781804 | 0.713733 |
| Micro | 0.798939 | 0.898567 | 0.845829 | 0.863286 | 0.781622 |
| Macro | 0.803975 | 0.898567 | 0.847542 | 0.843240 | 0.781622 |

Micro gộp 45.000 quyết định; macro lấy trung bình ba chỉ số tương ứng của ba đầu ra. Micro AUROC gộp xác suất giữa các đầu ra, còn macro AUROC lấy trung bình AUROC từng đầu ra. Các quyết định có chung sample không được coi là 45.000 mẫu độc lập.

Micro accuracy tính lại là **78.16222222222222%**, khớp `verifier_label_accuracy_percent` gốc trong sai số số thực. Kết quả console trên server do người dùng cung cấp cũng khớp Precision/Recall/F1/AUROC đến sáu chữ số thập phân.

Artifact: [metrics JSON](verifier_metrics.json), [metrics CSV](verifier_metrics.csv), [prediction gốc 5.000 samples](predictions_5000.json), [15.000 candidate predictions](predictions_15000_candidates.csv).

## 10. Confusion matrix

Quy ước: **hàng là nhãn thật [0,1], cột là nhãn dự đoán [0,1]**, ma trận `[[TN, FP], [FN, TP]]`. “Nhãn thật” ở đây là nhãn giả quy định bởi loại candidate, không phải nhãn đã kiểm duyệt toàn bộ ảnh.

| Đầu ra | TN | FP | FN | TP | Tổng |
|---|---:|---:|---:|---:|---:|
| Preserve | 4.137 | 863 | 496 | 9.504 | 15.000 |
| Edit | 2.393 | 2.607 | 1.567 | 8.433 | 15.000 |
| Violation | 1.686 | 3.314 | 980 | 9.020 | 15.000 |
| Micro | 8.216 | 6.784 | 3.043 | 26.957 | 45.000 |

```text
Preserve                 Edit
          Pred 0 Pred 1             Pred 0 Pred 1
Actual 0    4137    863    Actual 0    2393   2607
Actual 1     496   9504    Actual 1    1567   8433

Violation                Micro
          Pred 0 Pred 1             Pred 0 Pred 1
Actual 0    1686   3314    Actual 0    8216   6784
Actual 1     980   9020    Actual 1    3043  26957
```

Các ma trận đều được tạo từ **cùng 5.000 samples**. Tổng 15.000/cột head hoặc 45.000/micro không có nghĩa lấy thêm samples ngoài tập đánh giá. Artifact: [confusion_matrix.csv](confusion_matrix.csv).

## 11. Evaluation protocol của B1/B4/B5

### Các run và cấu hình huấn luyện đã ghi nhận

| Variant | Run | Trọng số factorization / counterfactual / verifier |
|---|---|---|
| B1 | `B1_server_20260921-155218` | 0.2 / 0.0 / 0.0 |
| B4 | `B4_server_20260921-175002` | 0.2 / 1.0 / 0.0 |
| B5 | `B5_override_20260922-085644` | 0.2 / 1.0 / 0.5 |

Cấu hình chung: seed 0, 5 epochs, batch size 16/device, learning rate 0.0001, weight decay 0.1, AMP, 4 workers, max grad norm 1.0, cùng đường dẫn train manifest/image root. B4/B5 có CF margin 0.2. B5 run protocol ghi GPU NVIDIA GeForce RTX 4090 và khởi tạo B5 mới, **không warm-start checkpoint B4**.

### Protocol đánh giá counterfactual

Theo `scripts/evaluate_counterfactual.py` hiện có: model ở eval mode và `no_grad`; loader không shuffle, không bỏ batch cuối; tạo query từ reference/modification; chấm cosine similarity với positive, CF-A, CF-B. Định nghĩa:

- CF-A win: `score_positive > score_cf_a`.
- CF-B win: `score_positive > score_cf_b`.
- Both win: đồng thời thắng cả hai; trường hợp bằng điểm không tính là thắng.
- Hard-negative error: `1 - both_win_rate`.
- B5 bổ sung ba xác suất verifier cho cả ba candidates và nhãn tại mục 9; B1/B4 không báo cáo verifier metrics trong evaluator này.

Các điểm CF win được tính từ **Stage-1 query scores**, chưa áp dụng rerank B5. Vì vậy bảng dưới không đo riêng lợi ích của verifier reranking. Bộ đánh giá này không tìm kiếm gallery hay đo Recall@K của CIRR/Fashion-IQ.

| Variant | Samples | CF-A win (%) | CF-B win (%) | Both win (%) | Hard-negative error (%) |
|---|---:|---:|---:|---:|---:|
| B1 | 5.000 | 80.04 | 95.78 | 77.54 | 22.46 |
| B4 | 5.000 | 77.98 | 97.68 | 76.58 | 23.42 |
| B5 | 5.000 | 78.06 | 97.82 | 76.84 | 23.16 |

Đã đối chiếu cả ba file `constraint_scores.json`: đều có 5.000 records và cùng thứ tự `modification`/`attribute_type`. Tuy nhiên, do thiếu ID ảnh/hash manifest, đối chiếu này **không chứng minh các tuple ảnh hoàn toàn giống nhau**.

### Protocol export bổ sung hiện tại

`scripts/export_b5_verifier.py` chỉ đọc scores đã lưu, giữ threshold 0.5, xuất CSV/JSON/confusion matrix/Risk–Coverage và từ chối ghi đè thư mục output có sẵn. Không tải model, không thay dataset và không train. Đã kiểm tra các trường hợp AUROC có tie, threshold 0.5, confusion matrix, confidence tie, số lượng record và xác suất không hợp lệ; bốn kiểm thử đều đạt.

Đã đối chiếu hash nguồn với inventory trong gói kết quả, kiểm tra 15.000 dòng candidate và xác nhận risk tại coverage 1 bằng `1-accuracy` cho mọi đường cong.

## 12. Các deviation/issue

| Vấn đề | Bằng chứng và ảnh hưởng |
|---|---|
| B5 chạy dù GO gate chưa đạt | `manual_go_override=true`, `measured_metric_gate_passed=false`. B4 both-win thấp hơn B1 và hard-negative error cao hơn B1. B5 là run exploratory, không được mô tả là đã qua GO gate. |
| Khởi tạo B5 mới | `initialization` ghi fresh B5; không phải tiếp tục từ B4 checkpoint. Cần giữ mô tả này khi so sánh. |
| Thiếu Git revision lịch sử | Cả ba run ghi `unknown`; không tái lập chính xác source revision chỉ từ artifact hiện có. |
| Chưa xác minh split | Thiếu manifest đánh giá/hash và quy trình chia split. Run protocol đã ghi nhận overlap train/val; không có số đo overlap trong báo cáo này. |
| Nhãn giả và mất cân bằng | Nhãn suy ra từ caption/candidate type, có vấn đề label noise được ghi trong protocol; 93,2% mẫu đánh giá là color. Kết quả không chứng nhận độ đúng trên nhãn người hoặc tổng quát đều cho mọi thuộc tính. |
| Liên kết checkpoint–prediction chưa đầy đủ | Có hash checkpoint và scores hiện tại, nhưng thiếu checkpoint hash ở thời điểm inference và thiếu ID ảnh trong prediction cũ. |
| “Best” dựa trên training batch loss | Theo mã hiện có, không phải checkpoint được chọn bằng validation metric; thiếu Git commit nên không chứng nhận source lịch sử hoàn toàn giống source hiện tại. |
| File prediction lịch sử là placeholder | `per_query_predictions.json` trong gói cũ là `[]`. Dữ liệu thực của 5.000 samples nằm ở `constraint_scores.json`; file export mới không thay thế bằng chứng full-gallery retrieval. |
| Giới hạn của Risk–Coverage | Đo selective binary classification; chưa có query-level retrieval risk, calibration hoặc AURC. Không diễn giải là hiệu quả abstention của toàn bộ hệ thống retrieval. |
| Phạm vi 45.000 quyết định | Ba đầu ra × ba candidates × 5.000 samples; không phải 45.000 samples đánh giá độc lập. |
| Không dùng audit v3 thay cho audit run cũ | Các báo cáo caption/visual v3 thuộc phiên bản dữ liệu khác; không áp tỷ lệ lỗi v3 cho bộ 45.000/5.000 lịch sử. |

### Danh mục bằng chứng dùng cho báo cáo

- `output/PE-CIR-training-results.zip`: config, seed, Git commit, metrics, constraint scores, checkpoint, log, run protocol của các run; mining config và inventory.
- `reports/B5_verifier_5000/verifier_metrics.json`, `confusion_matrix.csv`, `risk_coverage.csv` và các file prediction đã export.
- Mã hiện có: `scripts/evaluate_counterfactual.py`, `scripts/export_b5_verifier.py`, `scripts/train_cfpe.py`, `scripts/mine_pseudo_edits.py`, `src/pic2word/training/losses.py`.
- Ảnh console server do người dùng cung cấp: checkpoint tồn tại, B5 hoàn tất 14.060 bước, đủ 5.000 samples và kết quả export khớp sáu chữ số thập phân.

SHA-256 của các file nguồn B5 trong gói đã kiểm tra:

| File | SHA-256 |
|---|---|
| `constraint_scores.json` | `d02c4d2e5b1aeb278f31638f2e9656094f7bb1c117c516ee74e748531f9c1657` |
| `config.yaml` | `8017960e3d5f4eca8edd9e2afdd1d5575ee29fc085afb78524469d312e0b4546` |
| `run_protocol.json` | `8e0607dc5faa0f8ed4e2b0da2ff94b8eecb37fa99f4175f7de79553e2877af72` |

**Kết luận audit:** đã hoàn tất tổng hợp prediction và các chỉ số verifier trên 5.000 samples theo dữ liệu đã lưu. Các thiếu hụt về Git revision, split/ID ảnh, checkpoint linkage và chất lượng nhãn được giữ nguyên và công khai; báo cáo này không chứng nhận các thiếu hụt đó đã được khắc phục.
