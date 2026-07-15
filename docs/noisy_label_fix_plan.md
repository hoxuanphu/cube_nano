# Kế hoạch xử lý Noisy Labels do Random Crop trên 95-Cloud

## 1. Phạm vi và quy ước

Phương án này chỉ áp dụng cho bộ dữ liệu **95-Cloud**. `preprocess_38cloud.py` nằm ngoài phạm vi thay đổi hiện tại.

- Patch nguồn được lưu với kích thước mặc định `384x384`.
- Mô hình nhận crop kích thước mặc định `256x256`. `crop_size` vẫn có thể cấu hình, nhưng không được lớn hơn patch nguồn.
- Một pixel được xem là mây khi giá trị Ground Truth Mask lớn hơn `0`.
- Ngưỡng gán nhãn theo diện tích mây là `cloud_ratio_threshold = 0.10` (10%).
- Nhãn crop được xác định theo công thức:

```python
cloud_ratio = np.mean(crop_mask > 0)
label = 1.0 if cloud_ratio >= 0.10 else 0.0
```

Ngưỡng 10% ở đây là **ngưỡng tỷ lệ pixel mây trong Ground Truth Mask**, không phải ngưỡng xác suất dự đoán của mô hình. Ngưỡng xác suất dùng cho metric/inference vẫn là một tham số độc lập, mặc định `0.5`.

## 2. Vấn đề hiện tại

Pipeline hiện gán một nhãn duy nhất cho patch nguồn `384x384`, sau đó `CloudDataset` random crop xuống `256x256` nhưng vẫn giữ nhãn của patch nguồn.

Ví dụ, một patch nguồn có tỷ lệ mây từ 10% trở lên được gán nhãn `cloud`. Tuy nhiên, một crop ngẫu nhiên bên trong patch này có thể chứa dưới 10% mây, thậm chí không chứa mây, nhưng vẫn nhận nhãn `cloud`. Điều này tạo ra nhãn không khớp với nội dung thực tế mà mô hình nhìn thấy.

Mục tiêu của phương án là tính lại nhãn từ đúng vùng mask tương ứng với crop ảnh được đưa vào mô hình.

## 3. Cấu trúc dữ liệu đề xuất

Mask không được lưu chung trong thư mục `cloud/` hoặc `clear/`, vì `CloudDataset` hiện quét toàn bộ file `*.npy` trong hai thư mục này và có thể đọc nhầm mask 2D như một ảnh đầu vào.

Cấu trúc sau tiền xử lý:

```text
data/processed/all/
|-- cloud/
|   `-- scene_a_p0.npy
|-- clear/
|   `-- scene_a_p1.npy
`-- masks/
    |-- scene_a_p0.npy
    `-- scene_a_p1.npy
```

Cấu trúc sau khi chia tập dữ liệu:

```text
data/processed/
|-- train/
|   |-- cloud/
|   |-- clear/
|   `-- masks/
|-- val/
|   |-- cloud/
|   |-- clear/
|   `-- masks/
`-- test/
    |-- cloud/
    |-- clear/
    `-- masks/
```

File ảnh và mask dùng cùng tên file để ghép cặp. Ví dụ, ảnh `cloud/scene_a_p0.npy` phải có mask tương ứng tại `masks/scene_a_p0.npy`.

Hai thư mục `cloud/clear` tiếp tục được giữ để tương thích với pipeline hiện tại. Nhãn thư mục được tính từ toàn bộ patch nguồn với ngưỡng 10%, nhưng chỉ được xem là metadata của patch nguồn. Nhãn dùng để train phải được tính lại từ crop mask.

## 4. Các thay đổi cần thực hiện

### Bước 1: Lưu mask trong `preprocess_95cloud.py`

Sửa `src/data/preprocess_95cloud.py` như sau:

- Đổi tên tham số ngưỡng của script thành `--cloud_ratio_threshold` và đặt mặc định `0.10`. Có thể giữ `--threshold` làm alias tạm thời nếu cần tương thích với lệnh cũ.
- Tạo thêm thư mục `masks/` trong output directory.
- Với mỗi patch ảnh, lưu `patch_gt` nhị phân dưới dạng `uint8`, trong đó `0` là clear và `1` là cloud.
- Ảnh và mask phải dùng cùng stem, ví dụ `scene_a_p0.npy`.
- Kiểm tra tất cả các kênh ảnh và mask có cùng kích thước trước khi cắt patch.
- Khi dùng `--force`, xóa cả ảnh cũ và mask cũ để tránh ghép ảnh mới với mask tồn dư.
- Sau khi tiền xử lý, kiểm tra không có image-mask pair bị thiếu hoặc dư.

Mask nhị phân `0/1` và mask `0/255` có cùng dung lượng khi đều dùng `uint8`. Dùng `0/1` để thống nhất biểu diễn và làm rõ ý nghĩa dữ liệu; kết quả của điều kiện `mask > 0` không thay đổi.

### Bước 2: Giữ image-mask pair khi chia dữ liệu

Sửa `src/data/split_dataset.py` như sau:

- Tiếp tục split theo scene để tránh data leakage.
- Chỉ thu thập patch ảnh từ `cloud/*.npy` và `clear/*.npy`.
- Với mỗi patch ảnh, yêu cầu mask cùng tên phải tồn tại trong `masks/`.
- Copy hoặc move ảnh và mask vào cùng split.
- Tạo thư mục `train/masks`, `val/masks`, `test/masks`.
- `--force` phải dọn cả ba thư mục mask.
- Manifest bổ sung số lượng ảnh, số lượng mask và kết quả kiểm tra pairing cho từng split.
- Dừng với lỗi rõ ràng nếu thiếu mask, trùng stem hoặc kích thước tập ảnh và mask không khớp.

### Bước 3: Crop ảnh và mask bằng cùng tọa độ

Sửa `src/data/cloud_dataset.py` như sau:

- Thêm tham số `cloud_ratio_threshold`, mặc định `0.10`.
- Tạo danh sách record `(image_path, mask_path)` thay vì danh sách ảnh và nhãn tĩnh.
- Không đưa file trong `masks/` vào `self.files`.
- Trong `__getitem__`, nạp cả ảnh và mask rồi kiểm tra:
  - Ảnh có dạng `(H, W, C)` với `C` là `3` hoặc `4`.
  - Mask có dạng `(H, W)`.
  - Kích thước không gian của ảnh và mask giống nhau.
  - `crop_size` không lớn hơn `H` hoặc `W`.
- Chỉ sinh tọa độ crop một lần rồi dùng cho cả ảnh và mask.
- Training dùng random crop như hiện tại.
- Validation và test dùng center crop cố định để metric có thể tái lập.
- Tính `cloud_ratio` từ crop mask và gán nhãn `1.0` khi tỷ lệ lớn hơn hoặc bằng `0.10`.
- Không sử dụng `self.labels` được suy ra từ thư mục `cloud/clear` để train hoặc đánh giá.

Các phép flip và xoay 90 độ hiện tại không làm thay đổi tỷ lệ mây, nên có thể áp dụng sau khi nhãn đã được tính. Nếu sau này thêm spatial transform có crop, erase hoặc thay đổi vùng nhìn thấy, transform đó phải được áp dụng đồng bộ lên ảnh và mask trước khi tính nhãn.

### Bước 4: Truyền ngưỡng nhãn vào train và eval

Sửa `src/train.py` và `src/eval.py`:

- Thêm CLI argument `--cloud_ratio_threshold`, mặc định `0.10`.
- Truyền giá trị này vào `CloudDataset` cho train, validation và test.
- Giữ `--threshold` hiện tại cho ngưỡng xác suất dự đoán của mô hình, mặc định `0.5`.
- Lưu `cloud_ratio_threshold`, `crop_size`, dataset name và seed vào cấu hình kết quả để có thể tái lập thí nghiệm.

Hai ngưỡng phải được đặt tên và ghi log riêng:

```text
cloud_ratio_threshold = 0.10  # Tạo ground-truth label từ mask
probability_threshold = 0.50  # Chuyển model probability thành cloud/clear
```

### Bước 5: Tính lại `pos_weight`

Không tiếp tục tính `pos_weight` bằng số file trong thư mục `cloud/clear`, vì nhãn crop động có thể khác nhãn của patch nguồn.

Khi bật `--use_pos_weight`, cần ước lượng phân phối nhãn theo đúng crop policy:

- Dùng seed cố định.
- Lấy một số crop cố định trên mỗi patch nguồn, ví dụ 8 crop.
- Tính nhãn từng crop bằng `cloud_ratio_threshold = 0.10`.
- Tính `pos_weight = clear_crop_count / cloud_crop_count`.
- Ghi số crop cloud/clear và `pos_weight` vào log.
- Dừng với lỗi rõ ràng nếu không tìm thấy crop cloud nào.

Trong thí nghiệm đối chứng đầu tiên, cần ghi nhận cả kết quả khi tắt `pos_weight` để tách ảnh hưởng của dynamic labeling khỏi ảnh hưởng của class weighting.

## 5. Luồng xử lý sau khi sửa

```text
95-Cloud TIFF image + GT mask
        |
        v
preprocess_95cloud.py
        |
        +--> source image patch 384x384
        `--> paired binary mask 384x384
        |
        v
scene-level train/val/test split
        |
        v
CloudDataset
        |
        +--> chọn một tọa độ crop
        +--> crop ảnh và mask cùng tọa độ
        +--> cloud_ratio = mean(crop_mask > 0)
        `--> label = cloud_ratio >= 0.10
        |
        v
MobileNetV3 training/evaluation
```

Mask chỉ được dùng để tạo ground-truth label trong quá trình train/eval. Mask không phải input của MobileNetV3 và không cần thiết khi export ONNX hoặc inference trên Jetson Nano.

## 6. Kế hoạch kiểm thử

### Unit tests

- Mask có đúng tên và đúng kích thước với ảnh sau tiền xử lý.
- Với crop có `N` pixel, `ceil(0.10 * N) - 1` pixel mây tạo nhãn `0`.
- Với crop có `N` pixel, `ceil(0.10 * N)` pixel mây tạo nhãn `1`.
- Crop có tỷ lệ mây lớn hơn 10% tạo nhãn `1`.
- Ảnh và mask luôn được crop bằng cùng tọa độ, bao gồm các crop tại bốn cạnh.
- Training tạo được nhiều crop khác nhau với cùng source patch.
- Validation/test trả về center crop và nhãn giống nhau giữa nhiều lần chạy.
- Dataset báo lỗi khi thiếu mask, mask 3D, sai kích thước hoặc `crop_size` lớn hơn patch nguồn.
- Split giữ toàn bộ image-mask pair trong cùng scene và cùng split.
- Mask không bị `CloudDataset` đọc như ảnh đầu vào.
- `pos_weight` được tính từ nhãn crop ước lượng, không phải số file trong thư mục.

### Kiểm tra tích hợp

- Chạy preprocess trên một số scene 95-Cloud và xác nhận số ảnh bằng số mask.
- Chạy scene-level split và xác nhận không có orphan image hoặc orphan mask.
- Duyệt toàn bộ train/val/test dataset bằng `DataLoader` để kiểm tra shape và dtype.
- Ghi phân phối nhãn source patch và phân phối nhãn crop sau dynamic labeling.
- Ghi tỷ lệ crop bị đổi nhãn so với nhãn source patch để định lượng mức noisy label đã loại bỏ.

### Thí nghiệm đối chứng

So sánh hai lần train trên cùng scene split, seed, model và hyperparameters:

1. Baseline: crop ảnh nhưng giữ nhãn patch nguồn.
2. Dynamic labeling: crop ảnh-mask và tính nhãn với ngưỡng 10%.

Báo cáo tối thiểu loss, accuracy, precision, recall, F1, confusion matrix và phân phối nhãn crop. Không kết luận phương án làm tăng độ chính xác tuyệt đối nếu chưa có kết quả đối chứng.

## 7. Kết quả kỳ vọng và giới hạn

Phương án sẽ loại bỏ nguồn sai lệch do nhãn patch nguồn không khớp với nội dung của random crop. Nó vẫn giữ được lợi ích augmentation của random crop và không làm thay đổi kiến trúc MobileNetV3 hoặc pipeline TensorRT.

Phương án không đảm bảo loại bỏ mọi noisy label, vì Ground Truth Mask của 95-Cloud vẫn có thể chứa sai sót annotation. Việc nạp thêm mask cũng làm tăng I/O trong quá trình training, nhưng mask `uint8` kích thước `384x384` có chi phí nhỏ so với patch ảnh nhiều kênh.

Tiêu chí hoàn thành là pipeline không có image-mask pair bị thiếu, nhãn crop được tính đúng với ngưỡng 10%, validation/test có tính xác định, `pos_weight` phản ánh phân phối crop, và kết quả đối chứng được ghi lại trên cùng một scene split.
