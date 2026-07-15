# Phương án Huấn luyện Mô hình Hỗ trợ Ảnh Vệ tinh 3 Kênh và 4 Kênh

Để xây dựng một mô hình duy nhất (single model) có thể linh hoạt nhận và xử lý cả ảnh vệ tinh 3 kênh (RGB) và 4 kênh (RGB + NIR), chúng ta cần giải quyết vấn đề bất đồng nhất về số chiều (dimension) ở lớp đầu vào (Input Layer).

Dưới đây là 3 phương án kiến trúc và huấn luyện tối ưu nhất (xếp từ dễ triển khai đến tối ưu hóa hệ thống):

## 1. Phương án Zero-Padding + Channel Dropout (Khuyên dùng)
Đây là phương án phổ biến nhất, dễ triển khai nhất và đồ thị tính toán (ONNX/TensorRT) sau khi export sẽ cực kỳ ổn định.

*   **Về Kiến trúc (Architecture):** Thiết kế/Cấu hình mô hình sao cho lớp đầu vào **cố định luôn nhận 4 kênh**.
*   **Về Dữ liệu đầu vào (Inference):** Khi người dùng hoặc hệ thống đưa vào ảnh 3 kênh, ta tiến hành chèn thêm một kênh thứ 4 chứa toàn số `0` (Zero-Padding) hoặc giá trị hằng số (như trung bình của các kênh) để tạo thành tensor 4 kênh trước khi đưa vào mô hình.
*   **Chiến lược Huấn luyện (Training):** Để mô hình không bị suy giảm hiệu năng khi kênh 4 bị thiếu, trong lúc train (bằng tập dữ liệu 4 kênh gốc), ta áp dụng kỹ thuật **Channel Dropout**:
    *   Cấu hình một xác suất ngẫu nhiên (ví dụ `p = 0.3` hoặc `p = 0.5`).
    *   Với xác suất này, ta chủ động "tắt" (thay bằng tensor chứa toàn số 0) toàn bộ kênh thứ 4 của các ảnh trong một batch huấn luyện.
    *   **Tác dụng:** Ép mạng nơ-ron phải học cách trích xuất các đặc trưng chính từ 3 kênh RGB để đoán kết quả, và coi kênh NIR thứ 4 là một thông tin "bổ trợ" (có thì dự đoán chính xác hơn, không có thì vẫn hoạt động tốt).

## 2. Phương án Kiến trúc Phân nhánh Stem (Multi-branch Input Stem)
Phương án này can thiệp vào kiến trúc (Architecture-level) để tối ưu hóa khối lượng tính toán (FLOPs) nếu ảnh đưa vào phần lớn chỉ có 3 kênh. Ở phương án 1, ảnh 3 kênh bù số 0 vào thì mô hình vẫn tốn phép nhân/cộng với số 0 một cách vô ích.

*   **Về Kiến trúc:** Thay vì dùng 1 lớp `Conv2d(in_channels=4, out_channels=64)` ở đầu vào, ta tách thành 2 lớp chạy song bắt song:
    *   Nhánh 1 (`Stem_RGB`): Lớp `Conv2d(in_channels=3, out_channels=64)` chuyên xử lý 3 kênh RGB.
    *   Nhánh 2 (`Stem_NIR`): Lớp `Conv2d(in_channels=1, out_channels=64)` chuyên xử lý kênh thứ 4.
    *   Đầu ra của 2 nhánh này sau đó được cộng lại với nhau theo element-wise: `Output = Stem_RGB(x_rgb) + Stem_NIR(x_nir)`.
*   **Về Dữ liệu đầu vào:**
    *   Nếu là ảnh 4 kênh: Dữ liệu được chia ra, chạy qua cả 2 nhánh tương ứng rồi cộng kết quả lại.
    *   Nếu là ảnh 3 kênh: Dữ liệu chỉ chạy qua nhánh `Stem_RGB`. Nhánh `Stem_NIR` bị vô hiệu hóa (không tốn tài nguyên tính toán), ta coi đầu ra của nó là tensor `0`.
*   **Chiến lược Huấn luyện:** Giống Phương án 1, ta vẫn phải thỉnh thoảng tắt ngẫu nhiên (dropout) dữ liệu đi vào nhánh `Stem_NIR` trong lúc train để mô hình không dựa hoàn toàn vào nhánh này.

## 3. Phương án Bộ điều hợp Trọng số (Weight Adapter / Projection)
Phương án này cực kỳ phù hợp nếu bạn muốn sử dụng lại một mô hình **Pre-trained 3 kênh** siêu mạnh (như MobileNetV3, ResNet) đã được huấn luyện sẵn trên ImageNet mà không muốn phá vỡ lớp đầu vào (stem layer) của chúng.

*   **Về Kiến trúc:** Giữ nguyên mô hình lõi chỉ nhận 3 kênh. Tạo thêm một module xử lý nhỏ (gọi là Adapter, ví dụ như 1 lớp `Conv2d 1x1`).
*   **Về Dữ liệu đầu vào:**
    *   Nếu ảnh 3 kênh: Bỏ qua Adapter, đưa thẳng dữ liệu vào mô hình lõi.
    *   Nếu ảnh 4 kênh: Đưa dữ liệu qua lớp Adapter để "chiếu" (project) hoặc nén từ 4 kênh xuống 3 kênh không gian đặc trưng, sau đó kết quả mới được đưa vào mô hình lõi.
*   **Chiến lược Huấn luyện:** Quá trình huấn luyện sẽ được thực hiện từ đầu (hoặc fine-tune) trên cả khối Adapter và mô hình lõi. Adapter sẽ tự học được cách trộn lẫn thông tin của không gian RGB và kênh NIR thành một biểu diễn 3 kênh mới mà mô hình lõi có thể hiểu được tốt nhất.

---

> [!TIP]
> **Đánh giá & Lựa chọn:**
> *   Nếu ứng dụng AI của bạn dự kiến deploy trên vệ tinh nhỏ (CubeSat/NanoSat) qua **TensorRT** hay thiết bị Edge: Ưu tiên chọn **Phương án 1**. Định dạng ONNX rất hạn chế với các luồng rẽ nhánh (if/else dynamic). Việc đảm bảo kích thước tensor input luôn cố định ở dạng `[1, 4, H, W]` sẽ giúp trình biên dịch TensorRT tối ưu hóa đồ thị và engine mượt mà nhất, tránh các lỗi overhead không đáng có.
> *   Nếu vi xử lý trên vệ tinh cực kỳ yếu, cần chắt chiu từng phép toán tính toán (FLOPs) và framework AI có khả năng xử lý Dynamic Shape tốt: Nên cân nhắc **Phương án 2**.
