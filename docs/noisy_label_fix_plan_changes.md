# Nhật ký sửa đổi phương án xử lý Noisy Labels

Ngày cập nhật: 2026-07-14

Tài liệu được sửa đổi: [`noisy_label_fix_plan.md`](./noisy_label_fix_plan.md)

## 1. Thay đổi phạm vi

- Thu hẹp phạm vi triển khai, chỉ sử dụng bộ dữ liệu **95-Cloud** để huấn luyện.
- Xác định `src/data/preprocess_38cloud.py` nằm ngoài phạm vi hiện tại.
- Loại bỏ các nội dung đề xuất triển khai đồng thời cho cả 38-Cloud và 95-Cloud.

## 2. Thay đổi ngưỡng gán nhãn

- Đặt ngưỡng tỷ lệ mây thành `cloud_ratio_threshold = 0.10` (10%).
- Crop được gán nhãn `cloud` khi tỷ lệ pixel mây lớn hơn hoặc bằng 10%.
- Crop được gán nhãn `clear` khi tỷ lệ pixel mây nhỏ hơn 10%.
- Phân biệt rõ hai loại ngưỡng:
  - `cloud_ratio_threshold = 0.10`: tạo ground-truth label từ mask.
  - `probability_threshold = 0.50`: chuyển xác suất dự đoán của mô hình thành nhãn cloud/clear.
- Đề xuất đổi tên tham số ngưỡng của `preprocess_95cloud.py` thành `--cloud_ratio_threshold` để tránh nhầm với ngưỡng xác suất của mô hình.

## 3. Bổ sung cấu trúc lưu mask

- Không lưu mask chung trong thư mục `cloud/` hoặc `clear/` để tránh `CloudDataset` đọc nhầm mask như ảnh đầu vào.
- Bổ sung thư mục `masks/` riêng trong dữ liệu đã tiền xử lý và trong từng split.
- Ảnh và mask dùng cùng tên file để ghép cặp, ví dụ:

```text
cloud/scene_a_p0.npy
masks/scene_a_p0.npy
```

- Mask được lưu dưới dạng `uint8` với giá trị `0` cho clear và `1` cho cloud.
- Bổ sung yêu cầu kiểm tra mask bị thiếu, dư, trùng tên hoặc sai kích thước.

## 4. Sửa đổi đề xuất cho pipeline

### `src/data/preprocess_95cloud.py`

- Lưu mask tương ứng với từng patch ảnh.
- Sử dụng ngưỡng tỷ lệ mây mặc định 10%.
- Tạo và dọn thư mục mask khi dùng `--force`.
- Kiểm tra image-mask pairing sau khi tiền xử lý.

### `src/data/split_dataset.py`

- Giữ nguyên nguyên tắc split theo scene để tránh data leakage.
- Copy hoặc move mask cùng với patch ảnh vào train, validation và test.
- Bổ sung số lượng ảnh, mask và kết quả kiểm tra pairing vào manifest.

### `src/data/cloud_dataset.py`

- Thay danh sách nhãn tĩnh bằng các record `(image_path, mask_path)`.
- Crop ảnh và mask bằng cùng một tọa độ.
- Tính lại nhãn từ crop mask với ngưỡng 10%.
- Training dùng random crop; validation và test dùng center crop cố định.
- Không dùng nhãn suy ra từ thư mục `cloud/clear` làm nhãn huấn luyện.

### `src/train.py` và `src/eval.py`

- Bổ sung `--cloud_ratio_threshold`, mặc định `0.10`.
- Giữ `--threshold` hiện tại cho ngưỡng xác suất dự đoán.
- Ghi lại dataset, crop size, seed và cả hai ngưỡng trong cấu hình kết quả.

## 5. Điều chỉnh `pos_weight`

- Không tính `pos_weight` từ số file trong thư mục `cloud/clear`, vì nhãn crop động có thể khác nhãn patch nguồn.
- Đề xuất ước lượng phân phối nhãn bằng một số crop cố định trên mỗi patch với seed xác định.
- Tính `pos_weight = clear_crop_count / cloud_crop_count` từ phân phối crop đã ước lượng.
- Bổ sung thí nghiệm khi tắt `pos_weight` để tách ảnh hưởng của dynamic labeling và class weighting.

## 6. Bổ sung kế hoạch kiểm thử

- Kiểm tra image-mask pairing và shape.
- Kiểm tra biên ngưỡng 10% bằng số pixel nguyên.
- Kiểm tra ảnh và mask luôn được crop cùng tọa độ.
- Kiểm tra random crop khi train và center crop xác định khi validation/test.
- Kiểm tra split không làm thất lạc mask hoặc gây data leakage giữa các scene.
- Kiểm tra `pos_weight` được tính từ nhãn crop.
- Bổ sung thí nghiệm đối chứng giữa nhãn patch nguồn và dynamic crop labeling trên cùng scene split và seed.

## 7. Điều chỉnh kết luận

- Thay tuyên bố "độ chính xác tuyệt đối" bằng kết luận thận trọng hơn: phương án giúp loại bỏ nguồn sai lệch do nhãn patch nguồn không khớp với random crop.
- Nêu rõ Ground Truth Mask vẫn có thể chứa lỗi annotation.
- Nêu rõ việc đọc thêm mask làm tăng I/O trong quá trình huấn luyện.
- Yêu cầu có kết quả đối chứng trước khi kết luận dynamic labeling cải thiện metric.

## 8. Trạng thái hiện tại

- Đã cập nhật tài liệu phương án.
- Đã triển khai lưu mask cho `preprocess_95cloud.py` với ngưỡng mặc định 10%.
- Đã triển khai scene-level split giữ nguyên image-mask pair.
- Đã triển khai dynamic crop labeling trong `CloudDataset`.
- Đã chuyển `pos_weight` sang ước lượng từ phân phối nhãn crop.
- Đã bổ sung cấu hình provenance cho train và eval.
- Đã thêm test tự động cho preprocess, pairing, split, crop và biên ngưỡng 10%.
- Chưa tạo lại dữ liệu 95-Cloud.
- Chưa chạy huấn luyện hoặc thí nghiệm đối chứng.
