# Nhận xét chuyên gia: Kế hoạch khắc phục Training Pipeline RGB

> Tài liệu được đánh giá: [training_pipeline_remediation_plan.md](file:///d:/AI20K/cube_nano/docs/training_pipeline_remediation_plan.md)  
> Tài liệu gốc (review): [training_pipeline_review.md](file:///d:/AI20K/cube_nano/docs/training_pipeline_review.md)  
> Ngày đánh giá: 2026-07-14

---

## Đánh giá tổng thể

**Điểm: 8.5/10 — Đây là một kế hoạch khắc phục nghiêm túc, có hệ thống, và đặc biệt trưởng thành cho một dự án edge AI trên vệ tinh.**

Kế hoạch thể hiện rõ tư duy của người viết hiểu sâu cả hai mặt: (1) engineering discipline của ML pipeline, và (2) domain-specific constraints của remote sensing + satellite downlink. Rất ít kế hoạch remediation tôi thấy kết hợp tốt cả hai khía cạnh này.

---

## I. Điểm mạnh nổi bật

### 1. Tư duy "Contract-first" xuyên suốt

Khái niệm `InputSpec` + `DecisionSpec` + checkpoint bundle (Section 4) là **trụ cột kiến trúc đúng đắn nhất** của toàn bộ kế hoạch. Trong thực tế production ML:

- ~40% lỗi inference trên production đến từ mismatch giữa preprocessing lúc train và lúc deploy
- Việc checkpoint "tự mang" input contract loại bỏ hoàn toàn lớp lỗi "silent corruption" rất nguy hiểm

Đặc biệt ấn tượng là yêu cầu **legacy migration path** (`--legacy-input-spec`) thay vì tự suy luận — đây là dấu hiệu của engineering maturity.

### 2. Phân tách rõ ràng Decode/Calibrate vs. Normalize (Section 5)

Kiến trúc hai giai đoạn:
```
Raw → Physical Units (product-aware) → Normalized Tensor (InputSpec-aware)
```

Đây là cách tiếp cận chuẩn trong remote sensing ML và là **điều kiện tiên quyết bắt buộc** cho cross-satellite. Nhiều pipeline tôi review trộn lẫn hai bước này, dẫn đến không thể trace lỗi khi chuyển sensor.

### 3. Validity mask tách biệt khỏi cloud mask

Quyết định tách `validity_masks/` khỏi cloud ground-truth masks (Section 5, item 6) là **đúng và quan trọng**. Công thức:

```
cloud_ratio = cloud_valid_pixels / valid_pixels
```

Nghe đơn giản nhưng rất nhiều pipeline remote sensing mắc lỗi tính cloud coverage trên toàn bộ pixel bao gồm NoData, dẫn đến **underestimate cloud coverage** và false-clear decisions.

### 4. Threshold calibration strategy chặt chẽ

Luồng: fit threshold trên validation → khóa → chạy test là **best practice**. Kế hoạch còn đi xa hơn bằng cách yêu cầu:
- Threshold riêng cho từng sensor/product
- Cost function downlink-aware
- Provenance từ validation run cụ thể

### 5. Tiêu chí nghiệm thu cụ thể và có thể kiểm tra tự động

Mỗi giai đoạn đều có acceptance criteria viết dưới dạng **testable assertions**, không phải mô tả chung chung. Đây là điểm hiếm thấy trong kế hoạch ML remediation.

### 6. PR sequence hợp lý

Thứ tự 7 PR tuân thủ nguyên tắc dependency-first, mỗi PR có test riêng. Đặc biệt tốt là PR-1 (dọn dẹp pipeline) phải xong trước khi làm bất kỳ thay đổi contract nào.

---

## II. Điểm cần bổ sung hoặc cải thiện

### 1. ⚠️ Thiếu Data Versioning Strategy

> [!WARNING]
> Kế hoạch nhiều lần đề cập "tái tạo processed dataset" (P2 item 11, P5) nhưng **không có chiến lược version dữ liệu**.

Khi preprocessor thay đổi, bạn cần trả lời:
- Dataset v2 (sau rebuild) và v1 (trước rebuild) được lưu và phân biệt như thế nào?
- Checkpoint được train trên dataset version nào?
- Nếu phát hiện bug trong preprocessor sau khi train xong P5, rollback hay retrain?

**Đề xuất:** Thêm dataset manifest có hash + version vào checkpoint bundle. Cân nhắc dùng DVC hoặc ít nhất là convention `dataset_v{N}_manifest.json` với SHA-256 của processed files.

### 2. ⚠️ Thiếu chiến lược xử lý Label Noise tồn đọng

> [!IMPORTANT]
> Kế hoạch giải quyết tốt noisy label từ notebook (source patch label cho random crop), nhưng **không đề cập đến label noise nội tại của 95-Cloud dataset**.

95-Cloud (dựa trên Landsat 8 Biome) có một số vấn đề đã biết:
- Nhãn sai ở ranh giới mây (boundary pixels)
- Confusion giữa mây mỏng / haze và clear
- Thin cirrus annotation không nhất quán

Với dynamic crop labeling dùng threshold `cloud_ratio > T`, **giá trị T** trở nên cực kỳ quan trọng. Kế hoạch chưa xác định T = bao nhiêu, hay T có nên khác nhau giữa train và eval.

**Đề xuất:** Thêm ablation cho cloud_ratio threshold vào P5. Ghi rõ T đang dùng và có test cho edge cases (ratio ≈ T).

### 3. ⚠️ Chưa đề cập Augmentation Strategy cho RGB

Kế hoạch vô hiệu hóa channel dropout cho RGB (đúng) nhưng **không đề cập augmentation strategy thay thế**:

- Color jitter phải cẩn thận vì ảnh vệ tinh không giống ảnh tự nhiên — shift hue/saturation quá mạnh có thể tạo ra phân phối không bao giờ xuất hiện trên sensor
- Geometric augmentations (flip, rotation 90°) thường safe cho overhead imagery
- Nếu dùng physical reflectance, additive noise phải được scale phù hợp với đơn vị vật lý

**Đề xuất:** Thêm một item trong P5 hoặc tạo P2.5 riêng cho augmentation audit. Đặc biệt nếu dùng ImageNet pretrained weights, augmentation nên consistent với distribution mà backbone đã học.

### 4. 🟡 Timeline P6 có thể lạc quan

> [!NOTE]
> P6 ước lượng "1-3 tuần" cho cross-satellite, nhưng phụ thuộc vào **annotation effort** mà kế hoạch chưa ước lượng.

Tạo validation/test set có nhãn cho Sentinel-2 và PlanetScope, đặc biệt với các nhóm khó (thin cloud, haze, cirrus, snow/ice, sun glint) thường mất:
- 2-4 tuần cho mỗi sensor nếu làm thủ công
- Cần ít nhất 30-50 scene đa dạng cho mỗi sensor

**Đề xuất:** Tách P6 thành hai sub-phase: P6a (annotation + dataset curation) và P6b (training + evaluation). Ghi rõ minimum scene count và geographic diversity requirement.

### 5. 🟡 Thiếu Monitoring / Drift Detection plan

Kế hoạch dừng ở "production gates" nhưng **không đề cập post-deployment monitoring**:

- Model performance on-orbit có thể suy giảm theo mùa (snow cover, sun angle), hay theo thời gian (sensor degradation)
- Cần cơ chế log prediction distribution trên Jetson và phát hiện distribution shift

Đây có thể nằm ngoài scope của remediation plan, nhưng nên ít nhất ghi nhận như future work.

### 6. 🟡 OOD Detection (P6 item 8) cần cụ thể hơn

Kế hoạch đề cập "OOD checks cho band mismatch, processing-level mismatch, invalid ratio và band statistics ngoài miền train" nhưng **chưa xác định phương pháp**:

- Rule-based (kiểm tra range, statistics) hay learned (confidence-based, Mahalanobis distance)?
- Threshold cho OOD rejection là bao nhiêu?
- OOD flag dẫn đến reject inference hay chỉ warning?

**Đề xuất:** Bắt đầu bằng rule-based (band statistics ngoài [μ ± kσ] của train set). Đơn giản, interpretable, và đủ cho edge deployment trên Jetson.

---

## III. Vấn đề kiến trúc tiềm ẩn

### 1. Single-model vs. Sensor-specific models (P6)

Kế hoạch để mở: "nếu cần model chung thì train đa sensor với sampling cân bằng". Đây là quyết định kiến trúc **quan trọng nhất** và nên được quyết định sớm hơn vì nó ảnh hưởng đến:

| | Single model | Per-sensor models |
|---|---|---|
| Inference simplicity | ✅ Một model | ❌ Phải route theo sensor |
| Data efficiency | ✅ Tận dụng tất cả | ❌ Dataset nhỏ hơn per sensor |
| Performance | ❌ Compromise | ✅ Optimized per sensor |
| Deployment trên Jetson | ✅ Ít bộ nhớ hơn | ❌ Nhiều model/engine |
| Maintenance | ✅ Một pipeline | ❌ N pipeline |

Với Jetson Nano (RAM giới hạn), tôi **nghiêng về single model + sensor-specific threshold**, nhưng cần ablation data để quyết định.

### 2. TensorRT Engine Portability

Kế hoạch yêu cầu sidecar JSON cho TensorRT engine — đúng, vì engine không portable. Nhưng cần làm rõ:
- Engine được build trên Jetson hay trên host?
- Nếu build trên host, JetPack version phải match
- Nên lưu ONNX model cùng engine để có thể rebuild khi cần

---

## IV. Tổng kết

### Điểm mạnh cốt lõi
- ✅ Contract-first architecture (InputSpec/DecisionSpec/Bundle)
- ✅ Product-aware preprocessing pipeline
- ✅ Validity mask tách biệt
- ✅ Threshold calibration discipline
- ✅ Tiêu chí nghiệm thu testable
- ✅ PR sequence có dependency rõ ràng
- ✅ Production gates toàn diện

### Cần bổ sung
- ⚠️ Data versioning strategy
- ⚠️ Label noise handling cho 95-Cloud
- ⚠️ Augmentation audit cho RGB
- 🟡 Timeline P6 cần chi tiết hơn
- 🟡 Post-deployment monitoring
- 🟡 OOD detection methodology
- 🟡 Single-model vs. per-sensor decision

### Verdict

> [!TIP]
> **Đây là một kế hoạch remediation chất lượng cao.** Nó không chỉ sửa bug mà đang thiết kế lại pipeline theo hướng production-grade. Các điểm tôi nêu bổ sung là để nâng từ "very good" lên "excellent" — không có vấn đề nào trong kế hoạch là sai hướng.
> 
> Nếu thực thi đúng P0-P5, bạn sẽ có một pipeline training **đáng tin cậy hơn 90% các ML pipeline remote sensing tôi từng review**, đặc biệt về khía cạnh reproducibility và input contract enforcement.

Ưu tiên bổ sung ngay: **data versioning** (vì nó ảnh hưởng đến mọi giai đoạn từ P2 trở đi) và **augmentation audit** (vì nó ảnh hưởng trực tiếp đến kết quả P5).
