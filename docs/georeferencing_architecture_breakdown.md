# Phân tích & Phân chia Góc độ Kiến trúc Georeferencing `cube_nano`

Tài liệu này tổng hợp các góc độ phân tích, nhận xét chuyên gia và phân chia cấu trúc của bản kiến trúc [georeferencing_architecture.md](file:///c:/Users/phuhx1/Documents/cube_nano/docs/georeferencing_architecture.md).

---

## 1. Đánh giá Chuyên gia Kỹ thuật (Expert Review)

### 1.1. Nhận xét Tổng quan
Bản kiến trúc đạt tiêu chuẩn cao về **kỹ thuật hệ thống không gian (Space-grade systems engineering)**:
- **Tách biệt Geometry & Radiometry**: Preprocessing onboard chỉ thực hiện hiệu chỉnh hình học (distortion, co-registration, resampling) mà không can thiệp radiometric (white balance, tone mapping, denoise) nhằm bảo toàn phân phối dữ liệu huấn luyện.
- **Tư duy Fail-Closed / Fail-Safe**: Mọi trạng thái lỗi (telemetry trễ, ray miss, thermal limit, OOD input) đều mặc định fallback về `RETAIN_FOR_GROUND`. Jetson đóng vai trò kiến nghị policy, còn OBC nắm thẩm quyền xóa file thực sự.
- **Tối ưu tài nguyên nhúng**: Xử lý theo dạng Strip/Patch có Halo buffer giúp khống chế RAM peak trên Jetson Nano (4GB) và Orin Nano.

### 1.2. Khuyến nghị Kỹ thuật Chuyên sâu cho Triển khai
1. **Trôi dạt nhiệt độ (Thermal Boresight Drift)**: Cần bổ sung bảng hiệu chỉnh theo dải nhiệt độ (Temperature-dependent Calibration) cho gá ống kính camera nếu thử nghiệm TVAC cho thấy lệch boresight giữa chu kỳ Eclipse/Sunlit.
2. **Đồng bộ thời gian Rolling Shutter**: Với vận tốc quỹ đạo $\sim 7.5\text{ km/s}$, lệch $1\text{ ms}$ gây sai số $7.5\text{ m}$ mặt đất. Tín hiệu ngắt PPS từ GNSS phải nối trực tiếp tới phần cứng camera trigger và ADCS.
3. **Mô hình độ cao khi Direct Georeferencing**: Nên ưu tiên `ellipsoid_direct` kết hợp độ cao bề mặt trung bình (Mean Surface Elevation) của ROI cho mục đích AI Downlink decision để tiết kiệm compute/RAM so với `dem_direct`.
4. **Kiểm thử Parity**: Xây dựng test-suite tự động so sánh kết quả warp C++/CUDA onboard với GDAL `gdalwarp` ở Ground để phát hiện lệch số thực hoặc quy ước pixel-center.

---

## 2. Góc độ 1: 4 Khối Kiến trúc Kỹ thuật Cốt lõi (Functional Pillars)

Dưới góc độ thiết kế hệ thống phần mềm vệ tinh hoàn chỉnh, bản kiến trúc được chia làm **4 khối chức năng chính**:

```text
 ┌────────────────────────────────────────────────────────────────────────┐
 │ 1. KHỐI NGUYÊN TẮC AN TOÀN & HỢP ĐỒNG MODEL (Safety & Model Contract)  │
 ├────────────────────────────────────────────────────────────────────────┤
 │ 2. KHỐI PIPELINE NẮN ẢNH & GEOREFERENCING (Geometric & Georef Core)    │
 ├────────────────────────────────────────────────────────────────────────┤
 │ 3. KHỐI QUẢN LÝ THỰC THI & TÀI NGUYÊN NHÚNG (Runtime & Compute Core)   │
 ├────────────────────────────────────────────────────────────────────────┤
 │ 4. KHỐI TÍCH HỢP HỆ THỐNG & KIỂM THỬ FLIGHT (Integration & Verification)│
 └────────────────────────────────────────────────────────────────────────┘
```

### Khối 1: Nguyên tắc An toàn & Hợp đồng Model (Safety & Model Contract Core)
- **Cơ sở tài liệu**: Mục 1, 3.
- **Nhiệm vụ**: 
  - Đảm bảo tính nhất quán *Train-Inference Parity*.
  - Thực thi nguyên tắc *Fail-Closed* (lỗi input/OOD $\rightarrow$ không suy luận).
  - Định nghĩa **An toàn quyết định Downlink**: Jetson chỉ tạo `DecisionPolicy` record, OBC/F' đối chiếu fingerprint và kiểm tra manifest trước khi thực thi xóa/gửi.

### Khối 2: Pipeline Nắn ảnh & Georeferencing (Geometric & Georef Core)
- **Cơ sở tài liệu**: Mục 4, 5, 6.
- **Nhiệm vụ**:
  - Nhận input RGB, kiểm tra NoData và tạo `geometric validity mask`.
  - Ánh xạ hình học: Hiệu chỉnh méo Brown-Conrady/LUT, co-registration R/G/B, resample về GSD model grid.
  - Tính Direct Georeferencing (Direct Ray-Tracing $R_{e\leftarrow b}R_{b\leftarrow c}K^{-1}\tilde{\mathbf{u}}$, ma trận sai số $\Sigma_g$).
  - Xuất sidecar metadata: `.preprocess.json`, `.georef.json` và log kết quả phân loại `.patch-results.jsonl`.

### Khối 3: Quản lý Thực thi & Tài nguyên Nhúng (Runtime & Compute Core)
- **Cơ sở tài liệu**: Mục 4.3, 8.
- **Nhiệm vụ**:
  - Phân định rõ `PreprocessingProfile` (dữ liệu) và `ComputeProfile` (phần cứng execution).
  - Quản lý ngân sách RAM/Disk nhúng ($W_{peak} \le W_{\text{RAM,max}}, D_{peak} \le D_{\text{disk,max}}$) cho Jetson Nano & Orin Nano.
  - Thiết lập chính sách điện năng/nhiệt độ (`nvpmodel`, thermal throttling) và tự khôi phục sau reset.

### Khối 4: Tích hợp Hệ thống & Tiêu chuẩn Kiểm thử (Integration & Verification Core)
- **Cơ sở tài liệu**: Mục 2, 7, 9, 10.
- **Nhiệm vụ**:
  - Định hình lộ trình tích hợp vào codebase `cube_nano` (các file `src/`).
  - Thiết lập chuẩn giao tiếp giữa Jetson và OBC qua F'.
  - Thiết lập 10 gate kiểm thử chấp nhận flight (Unit test, Parity test, HIL benchmark, TVAC campaign).

---

## 3. Góc độ 2: 10 Phần theo Cấu trúc Tài liệu (Document Layout)

Nếu tra cứu theo bố cục chi tiết của tài liệu:

1. **Phần 1: Quyết định kiến trúc** — Tóm tắt các quyết định cốt lõi và diagram luồng production.
2. **Phần 2: Trạng thái hiện tại của dự án** — Đánh giá hiện trạng các module `src/` và thành phần cần bổ sung.
3. **Phần 3: Hợp đồng model và nguyên tắc an toàn** — Các quy tắc Train-Inference Parity, Fail-Closed, An toàn xóa ảnh và Chữ ký số chống Rollback.
4. **Phần 4: Đầu vào cho preprocessing onboard** (Gồm 4.1 Metadata, 4.2 Dữ liệu hình học, 4.3 Schema `ComputeProfile`).
5. **Phần 5: Pipeline nắn ảnh và inference onboard** — Chi tiết 5 bước từ Resolve contract, Validate source, Warp hình học, Normalization cố định đến Patch Classifier & DecisionPolicy.
6. **Phần 6: Ánh xạ pixel và georeferencing** (Gồm 6.1 Schema `.preprocess.json` và 6.2 Phương trình Direct Georeferencing ray-tracing & error propagation).
7. **Phần 7: Tích hợp vào `cube_nano`** — Lộ trình 9 bước cập nhật code và thêm CLI/sidecar.
8. **Phần 8: Tài nguyên onboard và vận hành flight** — Bất đẳng thức ngân sách RAM/Disk, khác biệt 2 dòng Jetson, Power/Thermal containment.
9. **Phần 9: Kiểm thử và gate chấp nhận flight** — Tiêu chuẩn kiểm thử unit, parity, HIL soak test và flight campaign.
10. **Phần 10: Quyết định và gate phát hành** — Tổng kết các tiêu chuẩn bắt buộc phải đạt trước khi phát hành flight profile.
