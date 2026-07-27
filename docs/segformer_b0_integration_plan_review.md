# Nhận xét chuyên gia về kế hoạch tích hợp SegFormer-B0

## 1. Phạm vi đánh giá

Tài liệu được đánh giá: [segformer_b0_integration_plan.md](segformer_b0_integration_plan.md).

Việc đánh giá tập trung vào tính đúng đắn AI/ML, chất lượng dữ liệu, khả năng tái lập, contract giữa model và runtime, tính khả thi của ONNX/TensorRT, và khả năng tích hợp vào mission runtime hiện có.

## 2. Kết luận

Kế hoạch ban đầu đúng hướng về kiến trúc nhưng chưa đủ chặt để triển khai end-to-end. Đánh giá tổng quát:

- Mức định hướng AI/ML: `8/10`.
- Mức sẵn sàng triển khai trước khi sửa: `5/10`.
- Quyết định: chấp thuận có điều kiện sau khi bổ sung output/validity contract, decision threshold, model routing, radiometric parity và các acceptance gate tuyệt đối.

Các điểm mạnh chính là tách rõ classification và segmentation, chia dữ liệu theo scene, không thay đổi âm thầm MobileNetV3, dùng golden vectors cho các backend, version hóa artifact và duy trì khả năng rollback.

## 3. Findings

### 3.1. Nghiêm trọng - Chưa có đường chạy hai model end-to-end

Kế hoạch ban đầu chỉ đề xuất thêm manifest, runtime và inference file cho SegFormer. Tuy nhiên, runtime hiện tại vẫn ràng buộc với MobileNetV3:

- [sat_ai/manifest.py](../sat_ai/manifest.py) kiểm tra cứng contract của model release MobileNetV3.
- [protocol/schemas.py](../protocol/schemas.py) không có trường chọn `model_task` trong request phân tích.
- [sat_ai/worker_process.py](../sat_ai/worker_process.py) luôn gọi pipeline classification hiện tại.
- Deadline và benchmark được lấy từ một deployment/model profile duy nhất.

Nếu không chốt ownership và routing, SegFormer có thể chỉ hoạt động như một script độc lập mà không thực sự tích hợp vào worker, product và downlink.

Khuyến nghị cho MVP là mỗi deployment profile và worker chỉ kích hoạt một `model_task` cùng một model release tại thời điểm khởi động. Việc chọn model theo từng request phải để sang một schema version sau, kèm policy load/unload và memory admission riêng.

### 3.2. Nghiêm trọng - Contract output trộn logits với product mask

Model SegFormer trả tensor logits, trong khi sản phẩm cuối là cloud mask `uint8`. Kế hoạch ban đầu dùng cùng khái niệm output cho cả hai lớp này, làm cho việc kiểm tra PyTorch/ONNX/TensorRT thiếu xác định.

Cần tách ba contract:

```text
ModelOutputSpec: logits float [N, C, H/4, W/4]
PostprocessSpec: resize + probability + threshold + validity policy
ProductSpec: cloud_mask uint8 [H, W] và validity_mask uint8 [H, W]
```

Các tham số resize như mode, `align_corners`, class axis và thứ tự threshold phải được version hóa. Golden test phải kiểm tra cả logits, probability/mask sau postprocess và cloud coverage.

### 3.3. Nghiêm trọng - `ignore_index=255` và cloud value `255` bị nhập nhằng

Trong training target, `255` được dùng cho pixel bỏ qua; trong product mask, `255` lại có nghĩa là cloud. Ngoài ra, [preprocess_95cloud.py](../src/data/preprocess_95cloud.py) hiện chuyển mọi giá trị ground truth lớn hơn không thành cloud, nên có thể làm mất encoding NoData nếu nguồn có trạng thái này.

Cần lưu validity mask riêng. Padding, NoData và phần ngoài ROI phải bị loại khỏi loss, metric và mẫu số coverage. Cloud mask có thể vẫn dùng `0/255`, nhưng không được diễn giải khi validity mask bằng không.

### 3.4. Cao - Chưa tách pixel threshold khỏi coverage threshold

Postprocess bằng `argmax` ngầm khóa ngưỡng pixel quanh xác suất 0.5. Trong thực tế, pixel cloud threshold cần được chọn theo constraint false-clear, còn coverage threshold quyết định giữ hoặc loại ROI/scene.

Cần một `DecisionSpec` riêng, tối thiểu gồm:

- Pixel cloud probability threshold.
- ROI/scene cloud coverage limit.
- Cost hoặc constraint cho false-clear và false-cloud.
- Phương pháp chọn threshold trên validation và quy tắc khóa threshold trước test.

### 3.5. Cao - Tiêu chí nghiệm thu AI chỉ mang tính tương đối

Việc SegFormer tốt hơn coarse MobileNet baseline là cần thiết nhưng chưa đủ. Một segmenter trung bình vẫn có thể thắng mask theo ô mà chưa đáp ứng yêu cầu nhiệm vụ.

Acceptance gate phải có ngưỡng tuyệt đối đã được phê duyệt trước training cho cloud IoU/Dice, cloud recall hoặc false-clear, coverage error và quyết định giữ/loại. Metric cần được báo cáo theo cả micro-pixel và macro-scene, kèm confidence interval bootstrap theo scene.

Boundary F1 phải định nghĩa tolerance. T48PYS chỉ là quantitative holdout khi có ground truth độc lập; nếu không, nó chỉ là external smoke test.

### 3.6. Cao - Radiometric contract chưa phải là gate trước training

Chia `uint16` cho `65535` chỉ dựa trên dtype, không xác định được pixel đang biểu diễn DN, TOA reflectance hay surface reflectance. Việc thử nhiều normalization không thay thế cho việc xác định sensor, product, processing level, scale/offset và NoData.

Train, eval và runtime phải dùng cùng một preprocessor có version. Mean/std hoặc clip range chỉ được fit trên train scenes. Không được tuyên bố model hỗ trợ domain của T48PYS cho đến khi có tập đích có nhãn và vượt acceptance gate.

### 3.7. Cao - Đặc tả training và reproducibility còn thiếu

Cần khóa implementation SegFormer, pretrained artifact, checksum, license và dependency version. Loss phải chỉ rõ trọng số CE/Dice, cách áp dụng ignore mask, cách xử lý batch không có cloud và epsilon. Crop sampling cần quan sát class imbalance nhưng validation/test phải giữ phân phối tự nhiên.

Các candidate cuối nên được chạy nhiều seed hoặc ít nhất phải có scene-level uncertainty; không chọn model từ một run duy nhất nếu chênh lệch nhỏ.

### 3.8. Trung bình - TensorRT feasibility được kiểm tra quá muộn

Graph compatibility, peak memory và toolchain target có thể làm thay đổi kiến trúc hoặc output contract. Vì vậy cần một vertical-slice export/build bằng model cấu trúc trước khi đầu tư toàn bộ thời gian training.

Target phải được pin bằng SKU, JetPack/L4T, TensorRT, CUDA, precision, power mode và memory budget. Benchmark production phải đo cả đọc dữ liệu, preprocessing, inference, stitching và ghi product, không chỉ latency theo tile.

### 3.9. Thấp - Ước lượng và số pha chưa nhất quán

Kế hoạch ban đầu nhắc `P7-P8` dù không có P8. Tổng ngày của các pha cũng không khớp với tổng MVP. Cần dùng một phép cộng rõ ràng và tách engineering effort khỏi thời gian train/benchmark chờ thiết bị.

## 4. Các thay đổi bắt buộc đã đưa vào kế hoạch

| Finding | Điều chỉnh trong kế hoạch mới |
|---|---|
| Runtime chưa có routing | Chọn model task theo deployment/worker lúc khởi động; thêm task-aware manifest, job snapshot và telemetry |
| Output contract nhập nhằng | Tách `ModelOutputSpec`, `PostprocessSpec` và `ProductSpec` |
| NoData không rõ | Thêm `validity_mask` riêng và loại invalid khỏi loss/metric/coverage |
| Threshold chưa tách | Thêm `DecisionSpec` với pixel threshold và coverage limit riêng |
| Acceptance chỉ tương đối | Thêm absolute gate, macro-scene metric và bootstrap confidence interval |
| Radiometry chưa khóa | Biến raw-data/radiometric audit và shared preprocessor thành data gate |
| TensorRT quá muộn | Đưa feasibility spike lên trước full training |
| Ước lượng sai | Đánh số lại P0-P7 và tính tổng engineering effort nhất quán |

## 5. Quyết định chuyên gia

Kế hoạch sửa đổi có thể được dùng làm baseline triển khai sau khi P0 điền đầy đủ các ngưỡng AI, SLO và parity tolerance còn phụ thuộc mission/target. Không được promote model sang production nếu các giá trị này vẫn để trống, nếu T48PYS chưa có nhãn nhưng bị dùng như bằng chứng chất lượng, hoặc nếu runtime chỉ chạy được qua script ngoài worker chính thức.
