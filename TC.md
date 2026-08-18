Khác với **TM Transfer Frame** (Telemetry - đường truyền xuống) mà chúng ta đã phân tích ở bức ảnh trước, **TC Transfer Frame** (Telecommand - đường truyền lên để điều khiển vệ tinh) tuân theo một chuẩn riêng là **CCSDS 232.0-B**. 

Bởi vì việc gửi lệnh lên vệ tinh mang tính "sống còn", cấu trúc của TC Transfer Frame được thiết kế với cơ chế kiểm soát nghiêm ngặt hơn (thường đi kèm với giao thức bảo vệ COP-1) để đảm bảo vệ tinh không thực thi sai lệnh hoặc bỏ sót lệnh.

Dưới đây là cấu trúc chi tiết của một **TC Transfer Frame**:

### 1. Primary Header (Phần đầu - 5 Byte)
Phần đầu của khung TC nhỏ hơn khung TM một chút (5 byte thay vì 6 byte) và có các trường dữ liệu được tối ưu cho việc điều khiển:

*   **Version Number (2 bit):** Luôn là `00` đối với chuẩn TC hiện tại.
*   **Bypass Flag (1 bit):** Rất quan trọng. 
    *   Nếu = `0` (Normal): Lệnh này phải đi qua bộ lọc bảo vệ COP-1 (phải kiểm tra số thứ tự, nếu sót thì yêu cầu gửi lại).
    *   Nếu = `1` (Bypass): Lệnh khẩn cấp! Yêu cầu vệ tinh bỏ qua mọi quy trình kiểm tra rườm rà và thực thi ngay lập tức (thường dùng khi vệ tinh đang gặp nguy hiểm).
*   **Control Command Flag (1 bit):**
    *   `0`: Khung này chứa lệnh thực thi thông thường (Data).
    *   `1`: Khung này chứa lệnh điều khiển của chính giao thức COP-1 (ví dụ lệnh "Hãy reset lại bộ đếm số thứ tự").
*   **Reserved Spares (2 bit):** Dự phòng, thường mang giá trị `00`.
*   **Spacecraft ID - SCID (10 bit):** Mã định danh của vệ tinh (giống hệt TM Frame, để đảm bảo trạm mặt đất không gửi nhầm lệnh cho vệ tinh khác).
*   **Virtual Channel ID - VCID (6 bit):** Kênh ảo. (Lưu ý: TC có tới 6 bit cho VCID, cung cấp tối đa 64 kênh ảo, trong khi TM chỉ có 3 bit).
*   **Frame Length (10 bit):** Kích thước của toàn bộ khung truyền (chiều dài thực tế trừ đi 1).
*   **Frame Sequence Number (8 bit):** Số thứ tự của khung (`N(S)`). Vệ tinh sẽ dựa vào số này để biết lệnh có bị gửi sót, gửi trùng hay lộn xộn không.

### 2. Transfer Frame Data Field (Phần thân - Độ dài thay đổi)
Đây là phần "ruột" mang Payload. Tương tự như đường truyền xuống, phần ruột này chủ yếu sẽ chứa:
*   **Segment Header (Tùy chọn - 1 byte):** Nếu một gói lệnh quá dài và phải cắt làm nhiều khúc (phân mảnh), byte này sẽ cho biết đây là khúc đầu, khúc giữa hay khúc cuối.
*   **Space Packet (Chuẩn CCSDS 133.0-B):** Trái tim của lệnh. Nó chứa **APID = 0** (như bạn thấy trong file JSON là `FW_PACKET_COMMAND`), và chứa mã Opcode (như `12288`) cùng các tham số cấu hình mà kỹ sư mặt đất đã nhập vào.

### 3. Frame Error Control Field - FECF (Phần đuôi bảo vệ - 2 Byte)
*   Chứa mã **CRC-16** (Cyclic Redundancy Check).
*   Mặc dù ở TM Frame phần này là "tùy chọn", nhưng đối với TC Frame, FECF gần như là **bắt buộc**. 
*   **Cơ chế:** Trạm mặt đất trước khi phát lệnh lên anten sẽ dùng một công thức toán học tính toán toàn bộ khung dữ liệu và nhét kết quả vào 2 byte này. Vệ tinh nhận được sóng, sẽ dùng công thức y hệt tính toán lại. Nếu kết quả lệch đi dù chỉ 1 bit, vệ tinh sẽ vứt bỏ ngay lập tức (vì sóng vô tuyến bị nhiễu, nếu cố chạy lệnh bị lỗi bit có thể gây cháy nổ hoặc hỏng hệ thống).

**Tóm tắt sự khác biệt:** Nhìn chung, TC Transfer Frame có tính năng cốt lõi là **Bypass Flag** và **Sequence Number 8-bit** kết hợp chặt chẽ với cơ chế phản hồi tự động (COP-1) để biến đường truyền vô tuyến đầy nhiễu thành một kênh truyền lệnh cực kỳ đáng tin cậy.