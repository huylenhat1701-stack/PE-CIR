# Kiểm tra ảnh v3

Đã xem 40 tuple (160 vị trí ảnh) ở độ phân giải thu nhỏ tối đa 384 pixel. Đây là mẫu chẩn đoán gồm 20 tuple được ưu tiên theo cờ lỗi và 20 tuple ngẫu nhiên trong phần còn lại, không phải mẫu đại diện toàn bộ dữ liệu.

Kết luận: 21 tuple không nên sử dụng nguyên trạng; 19 tuple cần kiểm tra thêm; 0 tuple được phê duyệt để train. Chưa hoàn tất quy mô audit 500/200 trong tài liệu.

Các lỗi đã thấy: đa nghĩa iris/plate, gán màu chậu cho hoa, chọn loài hoa làm negative cho lớp flower, caption không phản ánh chính xác vật thể trong ảnh. Các vấn đề cần xem thêm gồm màu sắc, chất liệu, thay đổi cảnh và phong cách.

Giữ dữ liệu/checkpoint cũ. Không train bộ v3 nguyên trạng. Cần sửa lỗi ngữ nghĩa và xác minh ảnh trước khi mở rộng hoặc chạy thí nghiệm mới.

| Mẫu | Split / dòng CSV nguồn | Kết luận | Ghi chú kiểm tra |
|---|---|---|---|
| 1 | val / 145 | Loại tuple hiện tại | CF-B is orange lilies, still flowers with the requested color; violates the claimed negative label. |
| 2 | val / 403 | Loại tuple hiện tại | CF-B is orange lilies, still flowers. Reference focuses on a person/dog in a field, unlike the positive close-up. |
| 3 | train / 4528 | Loại tuple hiện tại | Positive is a human eye, while reference and CF-A are iris flowers: object sense is not preserved. |
| 4 | train / 2225 | Loại tuple hiện tại | Positive is a green cactus in a brown pot, not a brown flower. |
| 5 | val / 245 | Loại tuple hiện tại | CF-B is pink roses, which are flowers with the requested color. Positive is an illustration with small decorative flowers. |
| 6 | val / 225 | Loại tuple hiện tại | CF-B is orange lilies, still flowers. Reference is a flower tattoo, positive is a real flower. |
| 7 | train / 944 | Loại tuple hiện tại | CF-B depicts black roses and thus still satisfies generic black flower; reference/positive also change photo to illustration. |
| 8 | train / 385 | Loại tuple hiện tại | Positive is a green cactus in a brown pot, not a brown flower. |
| 9 | val / 144 | Loại tuple hiện tại | CF-B is orange lilies, still flowers. Red/orange distinctions are also ambiguous in these thumbnails. |
| 10 | train / 3753 | Loại tuple hiện tại | Positive is a green cactus in a brown pot, not a brown flower. |
| 11 | val / 517 | Loại tuple hiện tại | CF-B is pink roses, still flowers. Positive additionally contains presentation-template graphics. |
| 12 | train / 4656 | Loại tuple hiện tại | Positive is a cactus in a brown pot, not a brown flower; CF-B contains eggs, not the named chicken. |
| 13 | train / 1221 | Loại tuple hiện tại | Positive is a cactus in a brown pot. Reference/CF-A include wedding scenery and a dog as prominent additional content. |
| 14 | train / 1546 | Loại tuple hiện tại | CF-B is an orange rose, still a flower with the target color; styles also differ. |
| 15 | val / 394 | Loại tuple hiện tại | CF-B visibly contains yellow lilies, so wrong-object claim is not supported. |
| 16 | train / 895 | Loại tuple hiện tại | CF-B depicts black roses and thus still satisfies generic black flower; positive changes photographic scene to pattern. |
| 17 | val / 399 | Loại tuple hiện tại | CF-B contains white roses, still flowers; monochrome photo further complicates color verification. |
| 18 | train / 4154 | Loại tuple hiện tại | Positive is a cactus in a brown pot, not a brown flower. Orchid pattern in reference is also lost. |
| 19 | train / 2623 | Cần kiểm tra thêm | Reference is a cartoon, positive is a real ginger kitten; CF-B caption refers to tree lights and image focuses on ornaments. Requires relabeling at image level. |
| 20 | train / 2055 | Loại tuple hiện tại | Positive is a cactus in a brown pot, not a brown flower. |
| 21 | val / 532 | Loại tuple hiện tại | Plate conflates dish/cutting board, metal surface and directional signs. These do not preserve the same object sense. |
| 22 | val / 30 | Cần kiểm tra thêm | Jacket types appear caption-compatible, but wearer, composition and collage style change; leather cannot be verified confidently from thumbnail alone. |
| 23 | val / 207 | Cần kiểm tra thêm | Illustration, painting and photo are mixed; the claimed yellow house is not clearly established in the CF-B thumbnail. |
| 24 | val / 543 | Cần kiểm tra thêm | Door material task may be plausible, but actual material and preservation beyond object type need original-resolution inspection. |
| 25 | val / 550 | Cần kiểm tra thêm | Reference cat appears tan/orange tabby rather than clearly gray; positive contains multiple cats. Color labels require review. |
| 26 | val / 303 | Cần kiểm tra thêm | CF-A is a cartoon doctor rather than a comparable photographic coat scene; all other visual content also changes. |
| 27 | train / 3574 | Cần kiểm tra thêm | Positive purple cat is a cartoon while reference and CF-A are real cats. Object-type-only labels do not verify preservation of style. |
| 28 | train / 1020 | Cần kiểm tra thêm | Red means ginger fur here, not literal red; pose/background change and CF-B shirt is not the main visual mass. Requires an explicit color convention. |
| 29 | train / 676 | Cần kiểm tra thêm | Materials are caption-compatible but objects are repurposed planters versus industrial bottles; material and preservation need closer review. |
| 30 | val / 519 | Cần kiểm tra thêm | Brown/black bear labels may encode species as well as color; CF-B is an illustration. Requires scope-specific review. |
| 31 | train / 1508 | Cần kiểm tra thêm | Positive shirt appears dark burgundy/black under lighting; cut, wearer and context change. Do not approve color solely from caption. |
| 32 | val / 514 | Cần kiểm tra thêm | Worn adult dress, product child dress and cartoon fairy are mixed. Only a broad garment category is preserved. |
| 33 | val / 67 | Cần kiểm tra thêm | Broad flower/color labels appear compatible at thumbnail level, but species, arrangement and scene are not preserved; no general CIR approval. |
| 34 | val / 323 | Cần kiểm tra thêm | Jacket category is plausible; positive is an outfit collage and CF-B runway boots require closer material inspection. |
| 35 | train / 2376 | Cần kiểm tra thêm | Reference foreground pink tulip and purple background flowers make the edited subject ambiguous. |
| 36 | train / 4665 | Loại tuple hiện tại | Positive visibly resembles a shallow wooden bowl of seeds, not a vase. The caption object label is unreliable. |
| 37 | val / 137 | Cần kiểm tra thêm | Rendered white door, real black doors and conceptual door image are mixed. Object type alone does not preserve scene/style. |
| 38 | train / 946 | Cần kiểm tra thêm | Reference is monochrome; material is inferred largely from captions, with large scene and style changes. |
| 39 | train / 3757 | Cần kiểm tra thêm | Ball conflates Christmas ornament, toy ball and red-white illustrated football. CF-B building color is not clearly yellow in the crop. |
| 40 | train / 3370 | Cần kiểm tra thêm | CF-B includes pink rose-like decoration around the plate as well as cookies. Absence of the preserved object cannot be established; scenes also change. |
