# Nắn Ảnh Hình Học Trên Vệ Tinh (Onboard Geometric Correction)

Tài liệu này mô tả chi tiết quá trình nắn ảnh hình học (geometric preprocessing) chạy trực tiếp trên các thiết bị máy tính trên vệ tinh. Phần cứng triển khai có thể là các vi xử lý nhúng (Edge SoC) hoặc máy tính biên trên vệ tinh.

Đây là bước tiền xử lý nhằm chuẩn bị dữ liệu đầu vào sao cho khớp với yêu cầu của AI model (thường sử dụng phần cứng gia tốc AI như NPU, TPU hoặc GPU), đồng thời cung cấp thông tin ánh xạ không gian (georeferencing) cho phép các hệ thống dưới mặt đất tiếp tục xử lý.

## 1. Mục tiêu và Nguyên tắc chung

- **Mục tiêu:** Chuyển đổi dữ liệu ảnh thô (raw capture) từ camera sang một lưới ảnh mục tiêu (model-grid) có kích thước, tỷ lệ (GSD), và đặc tính hình học đồng nhất mà model AI (Cloud patch classifier) đã được huấn luyện (train).
- **Phạm vi xử lý:** Chỉ giải quyết các vấn đề hình học: hiệu chỉnh méo ống kính (lens distortion), đồng đăng ký các kênh màu R/G/B (co-registration), resample về lưới đích.
- **Không thực hiện:** Nắn ảnh bằng mô hình độ cao số (Full DEM Orthorectification), hiệu chỉnh màu sắc, cân bằng trắng, sửa độ sáng, histogram matching hay khử nhiễu. Quá trình xử lý không được làm thay đổi tính chất phân bố màu sắc (distribution) của dữ liệu.
- **Biên giới độc lập:** Phần nắn ảnh hình học không bị ràng buộc bởi các thông số chuẩn hóa đặc thù (normalization), kích thước patch, hoặc tensor layout (HWC sang NCHW) của một model AI cụ thể. Những bước này được dành riêng cho `InferenceAdapter`.

## 2. Đầu vào (Inputs)

Quá trình nắn ảnh cần nhiều thành phần để đảm bảo tính an toàn (fail-safe) và tái lập được (reproducible).

1. **Raw Capture (Ảnh thô):**
   - Ảnh được chụp từ camera, lưu dưới dạng strip hoặc block (thường có các dải màu R, G, B bị lệch nhau hoặc méo).
   - Đi kèm metadata: thứ tự kênh, bit depth, hệ số gamma, cờ nguyên vẹn (integrity flags).
2. **PreprocessingProfile (Hợp đồng tiền xử lý):**
   - Chứa cấu hình lưới đích (Target Grid): kích thước extent, độ phân giải GSD mục tiêu, loại nội suy (kernel resampling).
   - Quy định về kiểu dữ liệu xuất (output dtype), policy làm tròn (rounding), mask hợp lệ (validity encoding).
3. **CalibrationBundle (Bộ tham số hiệu chuẩn camera):**
   - Các thông số hình học (Intrinsic) như: focal length, principal point, các hệ số méo xuyên tâm và tiếp tuyến (radial/tangential distortion coefficients).
   - Tham số ngoại lai (Extrinsic): boresight matrix.
   - Các phép biến đổi đồng đăng ký (Co-registration mapping) giữa các kênh.
4. **ComputeProfile (Cấu hình năng lực thiết bị):**
   - Xác định thiết bị chạy và nền tảng phần cứng tương ứng (ví dụ: các dòng SoC ARM, kiến trúc có GPU/NPU tích hợp).
   - Các ngưỡng giới hạn cấp phát RAM, Disk, giới hạn hàng đợi in-flight. 
   - Thiết lập backend thực thi nắn ảnh (ví dụ: Multi-core CPU với SIMD, GPU, hoặc bộ xử lý ảnh ISP chuyên dụng).

## 3. Đầu ra (Outputs - PreprocessArtifact)

Kết thúc bước nắn ảnh hình học, hệ thống tạo ra một bundle (artifact) được ghi nguyên tử (atomic) trên ổ đĩa, bao gồm:

1. **Lưới ảnh đã nắn (`model-grid.tif` hoặc mảng dữ liệu tương đương):**
   - Ảnh có cấu trúc hình học đồng nhất, sẵn sàng để đọc bởi `InferenceAdapter` hoặc chia thành các patch cho phần cứng gia tốc AI (NPU/GPU/TPU).
2. **Validity Mask (`validity.tif`):**
   - Kênh đánh dấu mỗi pixel trên ảnh đích có hợp lệ hay không (dựa trên phép tính support từ resampling kernel). Không được nội suy một vùng pixel lỗi thành vùng hợp lệ giả.
3. **Validity-Reason Mask (`validity-reasons.tif`):**
   - Bản đồ các nguyên nhân gây ra sự không hợp lệ cho pixel đó (VD: Nằm ngoài biên, ảnh nguồn bị NoData, thiếu kênh, v.v.).
4. **Ánh xạ Pixel và Metadata (`preprocess.json`):**
   - Chứa thông tin hàm ánh xạ từ tâm pixel lưới nguồn sang lưới đích (`source_to_model` mapping) và ngược lại (hoặc source footprint).
   - Ghi nhận lại các fingerprint của bộ profile, calibration đã dùng để truy xuất độ tin cậy.
5. **Manifest:**
   - Tập hợp các mã hash (checksum) để chứng minh bộ dữ liệu đầu ra không bị gián đoạn hay sai lệch lúc ghi/đọc.

## 4. Nguyên lý hoạt động (Pipeline)

### Bước 1: Khởi động và Giải quyết Hợp đồng (Contract Resolution)
- Xác thực vân tay số (fingerprint), chữ ký của Capture, PreprocessingProfile, CalibrationBundle.
- **Resource Admission:** Đánh giá xem có đủ bộ nhớ (RAM hệ thống, bộ nhớ NPU/GPU chia sẻ) và ổ cứng (disk bounds) dựa trên tính toán lý thuyết từ `ComputeProfile`. Nếu ước lượng vượt trần, quá trình bị từ chối ngay từ đầu để tránh lỗi tràn bộ nhớ (Out-of-memory).

### Bước 2: Đọc dữ liệu (Source Reader)
- Dữ liệu ảnh thô được đọc theo block/strip để tiết kiệm RAM.
- Giữ nguyên các giá trị mẫu (sample values). Không thực hiện scale màu (không dùng min/max ảnh).

### Bước 3: Lập kế hoạch Nắn (Transform Planning)
- Tính toán vùng quan tâm nguồn (Source ROI) và lề (halo) cần thiết cho mỗi strip ảnh đích.
- Dựa vào `CalibrationBundle`, tính toán các hàm ánh xạ (mapping transform). Một phép biến đổi chuẩn thường trải qua: Khử méo ống kính -> Ánh xạ các màu về cùng một không gian hình học -> Resample về lưới ảnh đích `model-grid`.

### Bước 4: Chạy phép biến đổi (Warp Backend)
- **Hình học:** Sử dụng thuật toán (như Affine, Polynomial, hay mô hình biến dạng Brown-Conrady) tính ngược (inverse) từ tọa độ lưới đích về lưới nguồn.
- **Resampling:** Lấy giá trị màu bằng phương pháp nội suy đã chỉ định (Bilinear, Area, Nearest Neighbor...). Việc tính toán này diễn ra trên số thực độ chính xác cao (float32/float64).
- **Validity propagation:** Các vùng mask không hợp lệ cũng được resample. Nếu một pixel đích lấy thông tin chủ yếu từ pixel nguồn bị đánh dấu NoData, nó cũng bị đánh mask invalid thông qua ngưỡng tính toán (`validity_policy_id`). Không cho phép pixel lỗi "lây nhiễm" và che giấu mình dưới giá trị màu khả dĩ.
- Quá trình này chạy theo strip. Sau khi xong một strip, nó ép kiểu (cast) về Output dtype và lưu bộ đệm/ghi ra đĩa.

### Bước 5: Đóng gói và Ký số Artifact
- Sau khi toàn bộ các dải đã nắn xong, hệ thống kiểm tra lần 2 dung lượng (Admission tầng 2).
- Ghi checksum và publish `manifest.json`. Đến lúc này AI Model Adapter mới được phép đọc artifact để nạp vào hardware accelerator (NPU/GPU/TPU).

## 5. Fail-Safe và Quản trị Lỗi

Quá trình nắn ảnh hình học chạy với nguyên tắc fail-closed (thiên về tính an toàn):
- Nếu cấu hình tiền xử lý thiếu hoặc không khớp vân tay mã hóa.
- Nếu RAM hoặc I/O bị lỗi tràn.
- Nếu calibration bundle bị quá hạn so với sensor.
=> Pipeline sẽ trả về lỗi, ví dụ `RESOURCE_REJECTED` hoặc `INVALID_INPUT`. Khi đó ảnh gốc sẽ được dán nhãn `RETAIN_FOR_GROUND` (Lưu lại gửi về trạm mặt đất). AI/Vệ tinh không được phép tùy tiện xóa hay điền các giá trị bừa bãi vào vùng ảnh bị hỏng.
