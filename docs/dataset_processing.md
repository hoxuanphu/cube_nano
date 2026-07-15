# Tài liệu Phân tích và Xử lý Dataset

Tài liệu này trình bày chi tiết về quy trình xử lý dữ liệu (Dataset Processing) cho mô hình nhận diện mây (Cloud Detection), bao gồm đặc điểm của bộ dữ liệu, các bài toán đặt ra, những vấn đề thực tế gặp phải trong quá trình thao tác với dữ liệu, và các phương án giải quyết đã được áp dụng trong codebase.

---

## 1. Đặc điểm Dataset

Dự án hiện đang sử dụng các bộ dữ liệu ảnh vệ tinh phổ biến cho bài toán nhận diện mây, chủ yếu là **38-Cloud** và **95-Cloud**. Các đặc điểm chính bao gồm:

*   **Định dạng file:** Dữ liệu gốc được cung cấp dưới dạng các file `.TIF` (TIFF) độc lập cho từng kênh màu.
*   **Các kênh phổ (Channels):** Dữ liệu là ảnh đa phổ (multi-spectral), thường bao gồm 4 kênh: Red (Đỏ), Green (Xanh lá), Blue (Xanh dương), và NIR (Cận hồng ngoại).
*   **Kích thước ảnh:** Các ảnh vệ tinh gốc (scene) có độ phân giải không gian cực kỳ lớn, không cố định và vượt quá khả năng xử lý trực tiếp của các phần cứng thông thường (GPU/TPU).
*   **Nhãn (Ground Truth):** Đi kèm với mỗi scene là một file mask nhị phân, trong đó các giá trị pixel lớn hơn 0 (thường là 1 hoặc 255) đại diện cho sự xuất hiện của mây (cloud), và 0 là nền (clear).

---

## 2. Các bài toán cần giải quyết

Để huấn luyện thành công một mô hình Deep Learning (ví dụ: Segmentation model) từ dữ liệu gốc, quy trình xử lý dữ liệu cần giải quyết các bài toán sau:

*   **Định dạng hóa đầu vào (Data Integration):** Gom nhóm và ghép (stack) các file `.TIF` riêng lẻ của từng kênh thành một tensor đa kênh (3 kênh RGB hoặc 4 kênh RGB+NIR) duy nhất tương ứng với một vùng địa lý.
*   **Chia nhỏ không gian (Spatial Partitioning):** Chuyển đổi các ảnh vệ tinh khổng lồ thành các khung hình (patches) nhỏ hơn với kích thước cố định để có thể đưa vào mạng Neural Network (ví dụ: đầu vào cần kích thước `256x256` hoặc `384x384`).
*   **Phân vùng dữ liệu (Dataset Splitting):** Chia dữ liệu thành các tập Huấn luyện (Train), Xác thực (Validation) và Kiểm thử (Test) phục vụ cho vòng đời phát triển mô hình.
*   **Quản lý bộ nhớ và tối ưu I/O:** Đảm bảo quá trình load dữ liệu trong lúc training diễn ra nhanh chóng, không gây ra hiện tượng thắt cổ chai (bottleneck) ở Disk I/O.

---

## 3. Vấn đề gặp phải

Trong quá trình triển khai, có một số thách thức và vấn đề đặc thù phát sinh:

*   **Rò rỉ dữ liệu (Data Leakage):** Nếu chia ngẫu nhiên các patches (khung hình nhỏ) vào tập Train và Validation, các patches nằm cạnh nhau thuộc cùng một ảnh vệ tinh gốc (scene) có đặc trưng cực kỳ giống nhau sẽ bị phân tán vào cả hai tập. Điều này khiến mô hình "học vẹt" và làm điểm số Validation cao ảo, không phản ánh đúng năng lực tổng quát hóa.
*   **Giới hạn phần cứng:** Không thể load toàn bộ ảnh gốc lên RAM hay VRAM do kích thước file `.TIF` quá lớn.
*   **Mất cân bằng dữ liệu (Data Imbalance):** Mật độ mây phân bố không đồng đều. Nhiều vùng hoàn toàn quang mây (clear), trong khi có vùng lại đặc mây. Nếu lấy ngẫu nhiên, mô hình có thể bị thiên lệch (bias).
*   **Sự thiếu nhất quán về định dạng:** Các file có thể có phần mở rộng viết hoa (`.TIF`) hoặc viết thường (`.tif`), cũng như cấu trúc tên file (tiền tố `red_`, `green_`) có thể sai khác đôi chút giữa 38-Cloud và 95-Cloud.
*   **Kiểu dữ liệu đa dạng:** File `.TIF` có thể lưu dưới dạng số nguyên (integer) hoặc số thực (float) với các giới hạn cực đại (max value) khác nhau, làm khó khăn cho quá trình chuẩn hóa (normalization).
*   **Tính linh hoạt của kênh (Channels Flexibility):** Đôi khi cần test mô hình chỉ với 3 kênh RGB, hoặc 4 kênh RGB+NIR, nhưng dữ liệu lại cố định.

---

## 4. Phương án giải quyết

Để giải quyết triệt để các vấn đề trên, codebase hiện tại đã triển khai các phương án tinh vi trong các file như `preprocess_38cloud.py`, `preprocess_95cloud.py`, `split_dataset.py`, và `cloud_dataset.py`:

### A. Trích xuất Patch (Patch Extraction) thay vì Resize
*   Thay vì thu nhỏ (resize) ảnh lớn sẽ làm hỏng cấu trúc không gian và chi tiết của đám mây, script tiền xử lý quét qua ảnh gốc bằng một cửa sổ trượt (sliding window) kích thước cố định (ví dụ: `384x384`).
*   Những patches ở rìa ảnh không đủ kích thước sẽ bị loại bỏ để đảm bảo tất cả input đều có shape đồng nhất (`384x384xC`).
*   Kết quả được lưu thành các file `.npy` nhỏ gọn, giúp tăng tốc độ đọc I/O đáng kể khi training.

### B. Phân loại Patch và Thresholding (Cloud Ratio)
*   Mỗi patch trích ra sẽ tính toán **tỷ lệ phần trăm pixel chứa mây** (Cloud Ratio) dựa vào Ground Truth.
*   Áp dụng một ngưỡng `cloud_ratio_threshold` (ví dụ: `0.05` cho 38-Cloud và `0.10` cho 95-Cloud). Nếu tỷ lệ mây vượt ngưỡng này, patch đó được phân loại và lưu vào thư mục `cloud`, ngược lại lưu vào thư mục `clear`.
*   Việc chia rõ hai nhóm `cloud` và `clear` giúp Dataset loader có thể lấy mẫu (sampling) cân bằng hơn khi cần thiết.

### C. Scene-level Splitting (Chống rò rỉ dữ liệu)
*   Script `split_dataset.py` không chia dữ liệu dựa trên các file `.npy` độc lập.
*   Nó sẽ trích xuất **Scene ID** từ tên file của từng patch (ví dụ: từ `patch_10_11_p5.npy` lấy ra `patch_10_11`).
*   Hệ thống gom tất cả các patches thuộc cùng một Scene ID lại với nhau, và thực hiện thao tác chia Train/Val/Test trên cấp độ Scene. Đảm bảo toàn bộ patches của một vùng địa lý chỉ nằm trọn ở một trong ba tập.

### D. Tiền xử lý động (Dynamic Augmentation & Normalization) trong Dataset
Lớp `CloudDataset` (`cloud_dataset.py`) xử lý dữ liệu realtime khi model gọi tới:
*   **Chuẩn hóa linh hoạt (Robust Normalization):** Tự động phát hiện kiểu dữ liệu (`float` hay `integer`). Nếu là `integer`, nó chia cho giá trị lớn nhất của kiểu đó. Nếu là `float`, nó tự đánh giá dải giá trị (ví dụ: `> 255.0` thì chia cho `65535.0`, ngược lại chia `255.0`), sau đó clip về khoảng `[0.0, 1.0]`.
*   **Xử lý số lượng kênh (Channel Padding/Slicing):** Nếu mô hình yêu cầu 4 kênh nhưng dữ liệu chỉ có 3 kênh, class sẽ tự động pad thêm 1 kênh ma trận `0`. Nếu ảnh có nhiều kênh hơn yêu cầu, nó sẽ cắt bớt đi.
*   **Random Cropping:** Ảnh patch lưu ổ cứng là `384x384`, nhưng khi training, DataLoader sẽ crop ngẫu nhiên một vùng `256x256` (giúp data augmentation tự nhiên). Khi Validation/Test sẽ lấy crop ở vị trí trung tâm (center crop).
*   **Data Augmentation:** Tích hợp lật ảnh ngẫu nhiên (Flip), xoay ảnh (Rotate 90, 180, 270) và đặc biệt là kỹ thuật **Channel Dropout** (xóa ngẫu nhiên kênh NIR bằng cách set kênh số 4 về 0 với xác suất `channel_dropout_p`) giúp mô hình tăng tính bền vững (robustness) ngay cả khi dữ liệu vệ tinh bị lỗi hoặc thiếu dải phổ hồng ngoại.

### E. Xử lý chuyên biệt cho 95-Cloud Dataset (`preprocess_95cloud.py`)
Do cấu trúc và yêu cầu của 95-Cloud có những đặc thù riêng, quy trình tiền xử lý bộ dữ liệu này có thêm các cơ chế kiểm soát lỗi nghiêm ngặt:
*   **Ngưỡng Cloud Ratio khắt khe hơn:** Ở 95-Cloud, mặc định một patch được gắn nhãn `cloud` khi có từ 10% (`0.10`) diện tích là mây (trong khi 38-Cloud có thể là 5%).
*   **Xử lý sai khác tên file (Tolerant File Search):** Hàm tìm kiếm file (`_find_file`) được thiết kế đặc biệt để quét thư mục và không phân biệt chữ hoa/chữ thường đối với phần mở rộng (ví dụ: `.TIF` và `.tif`), giúp tránh lỗi khi chạy trên các hệ điều hành khác nhau (Linux/Windows).
*   **Kiểm chứng cặp Image - Mask (Validation Pairs):** Trước khi hoàn tất, script bắt buộc chạy bước kiểm chứng toàn vẹn (`validate_output_pairs`). Nó đối chiếu chéo từng file patch sinh ra ở thư mục `cloud` và `clear` với các file trong thư mục `masks`, nhằm ngăn chặn triệt để tình trạng "thiếu mask" (missing mask) hoặc "thừa mask" (orphan mask) - một lỗi rất phổ biến khi xử lý dữ liệu lớn.
*   **Lưu trực tiếp mask nhị phân:** Quá trình trích xuất không chỉ cắt ảnh gốc thành patch (`.npy`) mà còn tính toán và lưu trực tiếp mask nhị phân (`0` hoặc `1` kiểu `uint8`) vào thư mục `masks` ngay trong lúc chạy, tạo sự đồng bộ 1-1 về mặt không gian (spatial) và định dạng.
