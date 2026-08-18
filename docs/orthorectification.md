# Tài liệu về Phần Nắn Ảnh Trực Giao (Orthorectification)

Tài liệu này trình bày về phương pháp và chiến lược đối với việc nắn ảnh trực giao (orthorectification) trong dự án `cube_nano`.

## 1. Phân biệt Nắn ảnh hình học (Onboard) và Nắn ảnh trực giao (Ground)

Trong kiến trúc của hệ thống, chúng ta phân tách rõ ràng giữa hai quá trình:

1. **Nắn ảnh hình học trên vệ tinh (Onboard Geometric Correction):**
   - Chỉ thực hiện hiệu chỉnh méo ống kính (lens distortion), đồng đăng ký (co-registration) các kênh màu, và resample về lưới không gian (grid) yêu cầu của AI model.
   - Kết quả đầu ra là lưới ảnh (patch-result grid) phục vụ trực tiếp cho quá trình inference (TensorRT) để phân loại mây/không mây.
   - **Không sử dụng** mô hình độ cao số (DEM - Digital Elevation Model) để nắn ảnh địa hình chi tiết do giới hạn về tài nguyên phần cứng (Jetson Nano / Orin Nano).

2. **Nắn ảnh trực giao (Ground Orthorectification):**
   - Lồng ghép mô hình độ cao số (DEM) vào quá trình nắn ảnh để loại bỏ các biến dạng hình học do địa hình gồ ghề và độ nghiêng của camera gây ra.
   - Sản phẩm đầu ra thường là các bản đồ trực giao GeoTIFF hoặc Cloud Optimized GeoTIFF (COG) có tọa độ địa lý chính xác đến từng pixel.
   - **Được thực hiện hoàn toàn tại trạm mặt đất (Ground Station)** nơi có đầy đủ tài nguyên tính toán và cơ sở dữ liệu DEM.

## 2. Tại sao không nắn ảnh trực giao (Full DEM Orthorectification) trên vệ tinh?

Dựa theo thiết kế và giới hạn tài nguyên của hệ thống `cube_nano`, nắn ảnh trực giao bằng DEM nằm **ngoài phạm vi (out of scope)** xử lý onboard vì các lý do sau:

- **Giới hạn bộ nhớ và lưu trữ:** Jetson Nano/Orin Nano có giới hạn khắt khe về RAM và Disk. Một bộ dữ liệu DEM toàn cầu hoặc khu vực đủ độ phân giải chiếm dung lượng lưu trữ lớn và bộ nhớ RAM đáng kể khi truy xuất, vượt quá `ComputeProfile` cho phép.
- **Chi phí tính toán:** Quá trình giao hội tia (ray intersection) lặp với bề mặt DEM đòi hỏi năng lực tính toán cao. Onboard ưu tiên tài nguyên cho mạng Neural Network phân loại mây.
- **Yêu cầu của Model:** Khối AI (patch classifier) chỉ yêu cầu đưa các pixel vào đúng `InputSpec` (grid) đã train. Việc bù trừ quang sai do địa hình không ảnh hưởng đáng kể đến việc nhận diện mây.
- **Pipeline Fail-closed:** Nếu quá trình nắn ảnh onboard bị lỗi do thiếu dữ liệu DEM, sẽ gây lãng phí ảnh (RETAIN_FOR_GROUND). Thay vào đó, telemetry được truyền về nguyên vẹn để mặt đất tự xử lý.

## 3. Kiến trúc nắn ảnh trực giao tại Trạm mặt đất (Ground)

Khi nhiệm vụ yêu cầu sản phẩm bản đồ có độ chính xác vị trí cực cao, việc nắn ảnh trực giao sẽ được tiến hành dưới trạm mặt đất bằng cách kết hợp các dữ liệu đã được downlink.

### 3.1. Dữ liệu đầu vào từ Vệ tinh (Downlink Data)
- **Source Capture (Raw Image):** Ảnh thô chưa qua nắn trực giao.
- **Telemetry & Georef Sidecar (`<capture>.georef.json`):** Chứa thông tin về vị trí quỹ đạo (position, velocity), tư thế vệ tinh (attitude/quaternion), thời gian (exposure time, row timing) và mô hình trái đất.
- **Calibration Bundle:** Thông số camera (focal length, principal point, distortion coefficients), boresight matrix.

### 3.2. Dữ liệu bổ sung tại Trạm (Ground Data)
- **DEM / DSM (Digital Elevation/Surface Model):** Mô hình địa hình chi tiết khu vực chụp ảnh.
- **GCPs (Ground Control Points - Tùy chọn):** Điểm khống chế mặt đất để tăng độ chính xác nắn chỉnh.

### 3.3. Quy trình nắn trực giao
1. **Dựng mô hình hình học (Rigorous Sensor Model):** Trạm mặt đất sử dụng RPC (Rational Polynomial Coefficients) hoặc mô hình vật lý trực tiếp dựa trên dữ liệu quỹ đạo và tư thế vệ tinh.
2. **Ray Tracing với DEM:** Quét từng pixel từ không gian ảnh lên DEM (hoặc ngược lại) thông qua phương trình giao hội không gian (collinearity equations).
3. **Resampling:** Gán giá trị màu (Radiometry) từ ảnh thô sang lưới ảnh đích (Ground grid) bằng thuật toán thích hợp (Bilinear, Cubic, v.v.). Không giống onboard, ở ground có thể thực hiện cân bằng trắng, sửa màu hoặc pan-sharpening.
4. **Xuất bản đồ:** Tạo định dạng tiêu chuẩn như **GeoTIFF** / **COG** tích hợp sẵn hệ tọa độ (CRS).

## 4. Kết luận

Trong hệ sinh thái phần mềm hiện tại của vệ tinh `cube_nano`, **nắn ảnh trực giao (Orthorectification) không được thực hiện trên Jetson/OBC**. Nhiệm vụ của hệ thống onboard dừng lại ở bước nắn hình học chuẩn bị lưới TensorRT và tính toán tham chiếu địa lý (Georeferencing Metadata).

Mọi yêu cầu về ảnh nắn trực giao chất lượng cao được xử lý ở bước hậu kỳ tại trạm mặt đất, qua đó giải phóng tài nguyên quý giá trên vệ tinh, đảm bảo tính an toàn (fail-closed) và hiệu năng (power/thermal limit) của mission.
