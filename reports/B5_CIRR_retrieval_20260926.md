# CIRR validation: B5 verifier → retrieval utility

Run: `b5_package_20260926_183042`; nguồn: `output/b5_cirr_results_20260926_183042.tar.gz`.

## Phạm vi và kiểm tra

- CIRR `rc2/val`: 4.181 queries, 2.297 candidate images, không thiếu ảnh theo preflight trên server.
- B1, B4, B5 dùng các checkpoint `best_checkpoint.pt` đã lưu; SHA-256 trong `metadata.json` khớp với ba bản checkpoint local. Training seed trong từng checkpoint là 0. Evaluation code ghi Git commit `unknown`; đó không phải Git commit lịch sử của training.
- `run.log` kết thúc bằng `Done`; gói có đủ `cirr_metrics.json`, `metadata.json`, sáu file per-query và log.
- Đã tính lại Recall@1/5/10/50 từ rank từng query cho bốn chế độ B5 tại K=50/100; tất cả khớp `cirr_metrics.json`. Mọi rank delta bằng `stage1_target_rank - final_target_rank`; số query đủ điều kiện bằng số target có Stage-1 rank ≤ K.
- Đây là validation CIRR, không phải hidden test. Gói chưa chứa kết quả Fashion-IQ.

## So sánh chính (đơn vị: %)

| Chỉ số | B4 Stage-1 | B5 Stage-1 | B5 Final K=50 | B5 Final K=100 |
|---|---:|---:|---:|---:|
| R@1 | 0,454 | 0,407 | 0,191 | 0,239 |
| R@5 | 2,129 | 2,153 | 1,315 | 1,028 |
| R@10 | 3,516 | 3,492 | 2,607 | 2,177 |
| R@50 | 11,720 | 11,744 | 11,744 | 9,950 |
| Group R@1 | 24,300 | 24,516 | 24,013 | 23,798 |
| Group R@2 | 44,941 | 44,750 | 44,391 | 44,463 |
| Group R@3 | 64,578 | 64,530 | 64,602 | 64,434 |

Tại K=50, R@50 không thể đổi vì verifier chỉ hoán vị 50 candidate đầu; không candidate nào đi vào hoặc rời Top-50. Tại K=100, B5 Final R@50 giảm 1,794 điểm phần trăm so với B5 Stage-1. Với cấu hình full verifier hiện tại, reranking làm giảm R@1/5/10 ở cả hai K; Group Recall cũng giảm nhẹ ở phần lớn ngưỡng.

## Rank từng query và rerank-eligible (full verifier)

Rank delta dương nghĩa là target được đẩy lên. Mean/median dưới đây tính trên toàn bộ 4.181 queries; query ngoài Top-K giữ nguyên rank.

| K | Target trong Stage-1 Top-K | Lên | Không đổi | Xuống | Mean rank delta | Median |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 491 (11,744%) | 207 | 10 | 274 | -0,276 | 0 |
| 100 | 773 (18,488%) | 315 | 9 | 449 | -1,318 | 0 |

Trong riêng nhóm đủ điều kiện, mean rank delta là -2,348 tại K=50 và -7,127 tại K=100. Số target bị đẩy xuống nhiều hơn số được đẩy lên ở cả hai K.

## Ablation B5 (Final Recall, đơn vị: %)

| K | Chế độ | R@1 | R@5 | R@10 | R@50 | Group R@1 | Group R@3 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 50 | Preserve-only | 0,407 | 1,507 | 2,727 | 11,744 | 24,013 | 64,530 |
| 50 | Edit-only | 0,120 | 0,741 | 1,483 | 11,744 | 24,348 | 64,602 |
| 50 | Preserve+Edit | 0,239 | 1,339 | 2,679 | 11,744 | 24,133 | 64,674 |
| 50 | Full verifier | 0,191 | 1,315 | 2,607 | 11,744 | 24,013 | 64,602 |
| 100 | Preserve-only | 0,431 | 1,411 | 2,583 | 10,930 | 23,846 | 64,458 |
| 100 | Edit-only | 0,120 | 0,502 | 1,172 | 7,773 | 23,726 | 64,482 |
| 100 | Preserve+Edit | 0,263 | 1,363 | 2,320 | 9,998 | 23,750 | 64,506 |
| 100 | Full verifier | 0,239 | 1,028 | 2,177 | 9,950 | 23,798 | 64,434 |

Preserve-only tốt nhất trong bốn ablation về R@1/5/10 tại K=50; tại K=100 nó tốt nhất về R@1/5/10/50. Tuy vậy, ngay cả Preserve-only cũng không cải thiện đồng thời các ngưỡng so với Stage-1. Không có cơ sở kết luận verifier hiện tại tăng retrieval utility trên CIRR.

## Dữ liệu truy vết

`cirr_B5_K50_per_query.json` và `cirr_B5_K100_per_query.json` lưu target rank trước/sau, rank delta và `rerank_eligible` cho từng query. Với mỗi candidate Stage-1 Top-K, file lưu `stage1_score`, `p_P`, `p_E`, `p_V` và `final_score` của cả bốn ablation. Full verifier dùng `S_G + 0,5 log(p_P) + 0,5 log(p_E) - 0,5 log(p_V)`; các chế độ khác và hệ số được ghi trong `metadata.json`.

**Kết luận phạm vi:** Hoàn tất phần CIRR của package. Fashion-IQ dress/shirt/toptee, Mean Recall và phân tích tương ứng vẫn chưa có vì chưa tìm thấy bộ dữ liệu trên server.
