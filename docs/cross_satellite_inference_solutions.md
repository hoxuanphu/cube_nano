# Phương án suy luận chéo vệ tinh (Cross-Satellite Inference)

## 1. Phạm vi và bối cảnh

Tài liệu này đánh giá và đề xuất pipeline đưa mô hình phát hiện mây được huấn luyện trên dữ liệu Landsat 8 / 95-Cloud sang ảnh từ cảm biến khác như Sentinel-2 hoặc PlanetScope.

Phạm vi được giới hạn ở **ảnh quang học ba kênh RGB**. Model đích phải nhận tensor theo thứ tự `[Red, Green, Blue]`; các band NIR, SWIR, thermal, hyperspectral và dữ liệu SAR nằm ngoài phạm vi. Một checkpoint đã train với bốn kênh không được coi là checkpoint RGB: phải retrain/fine-tune model ba kênh hoặc chứng minh từ training rằng model hỗ trợ thiếu kênh một cách có chủ đích.

“Ảnh RGB” cũng phải được phân loại theo product. Ba band reflectance của một analytic product không tương đương ảnh true-color 8-bit đã gamma correction, tone mapping hoặc nén JPEG. Không trộn hai loại này trong cùng input contract nếu model chưa được train cho cả hai miền.

Trong repository hiện tại, MobileNetV3-Small là mô hình **phân loại nhị phân theo patch** `256 x 256`, không phải mô hình segmentation pixel-level. Mỗi patch chỉ trả về một xác suất `cloud/clear`; mask của ảnh lớn được tạo bằng cách gán kết quả đó cho toàn bộ patch. Vì vậy:

- Đầu ra là mask thô theo ô, không biểu diễn chính xác biên mây.
- Mục tiêu phù hợp là sàng lọc ảnh hoặc vùng ảnh trước khi downlink.
- Các chỉ số đánh giá chính phải ở mức patch và mức quyết định giữ/loại ảnh.

Cross-satellite inference là bài toán **domain shift**. Calibration hoặc chuẩn hóa ảnh chỉ làm giảm một phần sai lệch; không có phép tiền xử lý đơn lẻ nào bảo đảm một checkpoint Landsat sẽ tổng quát tốt sang mọi cảm biến khác.

## 2. Đánh giá chuyên gia

### 2.1. Kết luận tổng quát

Phiên bản ban đầu nhận diện đúng phần lớn nhóm rủi ro: sai band, sai radiometry, NoData, độ phân giải không gian, spectral shift và khác biệt processing level. Đây là một checklist chẩn đoán tốt.

Tuy nhiên, tài liệu chưa đủ an toàn để dùng làm đặc tả triển khai. Vấn đề lớn nhất là chưa yêu cầu preprocessing của training và inference phải giống nhau. Một số biện pháp như percentile stretch theo từng ảnh, histogram matching với một ảnh mẫu, clipping reflectance tại `1.0` hoặc Z-score theo từng ảnh có thể làm mất tín hiệu nhận diện mây và tạo thêm domain shift.

Đánh giá tổng thể: **6/10 ở mức định hướng; chưa production-ready**.

### 2.2. Điểm tốt

- Không tin thứ tự kênh chỉ dựa trên số lượng band hoặc kiểu file.
- Yêu cầu đọc metadata và không coi kênh Alpha của RGBA là một band quang phổ.
- Nhận diện đúng sự khác biệt giữa DN, TOA reflectance và surface reflectance.
- Nhận diện đúng tác động của GSD lên vùng mặt đất mà receptive field quan sát.
- Đề cập NoData, saturation, QA band, spectral response và processing level.
- Ưu tiên từ chối input mơ hồ thay vì âm thầm suy luận trên dữ liệu sai.

### 2.3. Các vấn đề phải khắc phục

| Mức độ | Vấn đề | Hậu quả | Khắc phục |
|---|---|---|---|
| Nghiêm trọng | Gọi model là segmentation | Chọn sai chỉ số và kỳ vọng sai về mask | Mô tả đúng là patch-level classification |
| Nghiêm trọng | Chỉ calibration dữ liệu inference | Phân phối inference khác dữ liệu train | Dùng cùng một preprocessor cho train và inference, sau đó retrain/fine-tune |
| Nghiêm trọng | Dùng `valid_range` hoặc dtype làm radiometric scale | Giá trị vật lý sai dù kết quả nằm trong `[0, 1]` | Decode theo sensor, product, processing level, scale và offset trong metadata |
| Cao | Percentile stretch `2-98%` theo từng ảnh | Mất độ sáng tuyệt đối; kết quả phụ thuộc thành phần cảnh | Không dùng trong baseline; chỉ dùng mapping cố định đã fit và kiểm chứng |
| Cao | Histogram matching với một ảnh Landsat mẫu | Ép cảnh đích sang phân phối không đại diện | Bỏ khỏi baseline; nếu cần, học mapping từ tập cặp ảnh đại diện |
| Cao | Z-score theo từng ảnh/tile | Mất tín hiệu brightness và spectral contrast | Chỉ dùng thống kê cố định của tập train và dùng giống nhau ở inference |
| Cao | Gán NoData thành clear | Làm giảm giả tạo cloud coverage | Xuất trạng thái invalid và loại khỏi mẫu số |
| Cao | Coi reflectance `> 1.0` là saturation | Có thể xóa tín hiệu mây sáng hợp lệ | Dùng QA/metadata; clipping theo input contract của model |
| Cao | Trộn analytic RGB với ảnh RGB đã render | Cùng giá trị pixel nhưng khác ý nghĩa vật lý và phân phối | Tách product contract hoặc train có chủ đích trên cả hai miền |
| Trung bình | Chỉ resize theo kích thước pixel | Band có thể lệch CRS, grid và point-spread function | Reproject, co-register và resample theo loại dữ liệu |
| Nghiêm trọng | Không có benchmark có nhãn trên sensor đích | Không biết giải pháp có cải thiện thật hay không | Lập test set độc lập theo sensor, scene, vùng địa lý và mùa |

## 3. Nguyên tắc thiết kế bắt buộc

### 3.1. Train/inference parity

Một checkpoint chỉ hợp lệ với đúng input contract đã dùng khi train. Mọi phép biến đổi sau phải giống nhau giữa training và inference:

- Ý nghĩa và thứ tự band.
- Processing level và đơn vị vật lý.
- Công thức scale/offset.
- Quy tắc xử lý NoData và saturation.
- GSD, CRS, pixel grid và resampling kernel.
- Miền clipping và normalization.

Nếu chuyển từ pipeline hiện tại, vốn chuẩn hóa chủ yếu theo dtype, sang reflectance vật lý thì phải tạo lại dữ liệu train và retrain hoặc fine-tune model. Không được thay riêng pipeline inference rồi tiếp tục dùng checkpoint cũ như thể hai miền đầu vào tương đương.

### 3.2. Product-aware thay vì dtype-aware

`uint16` chỉ mô tả cách lưu trữ, không mô tả ý nghĩa radiometric. Hai file `uint16` có thể lần lượt chứa DN thô, TOA reflectance đã lượng tử hóa hoặc surface reflectance với scale/offset khác nhau.

Preprocessor phải được chọn bằng tổ hợp:

```text
sensor + platform + product + processing_level + processing_baseline
```

Không được tự suy ra calibration chỉ từ `dtype`, `max()` hoặc `valid_range`.

### 3.3. Không suy luận khi metadata mơ hồ

Nếu hệ thống không xác định chắc chắn band, scale, offset, processing level hoặc NoData, kết quả phải là lỗi có thông tin. Có thể hỗ trợ cấu hình thủ công, nhưng cấu hình này phải được lưu cùng kết quả để truy vết.

## 4. Input contract có phiên bản

Mỗi checkpoint cần đi kèm một `InputSpec`, tối thiểu gồm:

```yaml
schema_version: 1
model_task: patch_classification
patch_size: 256
target_gsd_m: 30
channels: 3
band_order: [red, green, blue]
expected_units: reflectance
expected_processing_level: TOA
normalization:
  type: fixed_per_band
  clip_min: [...]       # Học/chốt từ train set
  clip_max: [...]
  mean: [...]           # Chỉ có nếu model được train với Z-score cố định
  std: [...]
nodata_policy: mask
invalid_patch_ratio: 0.5
```

Các giá trị cụ thể phải được tính từ dữ liệu train và lưu cùng checkpoint; không dùng số minh họa trong tài liệu làm mặc định ngầm.

## 5. Pipeline tiền xử lý đã hiệu chỉnh

### Bước 1: Nhận diện sensor và ánh xạ band

1. Đọc metadata từ product manifest, sidecar XML/MTL hoặc GeoTIFF tags.
2. Xác định sensor, product, processing level, processing baseline và thời điểm chụp.
3. Ánh xạ band theo vai trò phổ, không chỉ theo chỉ số kênh trong file.
4. Reorder về đúng `band_order` của `InputSpec`.
5. Với RGBA, chỉ bỏ Alpha khi metadata xác nhận rõ ba kênh còn lại là RGB đúng với product contract; nếu không thì từ chối input.

Ví dụ Sentinel-2 thường dùng B4/B3/B2 cho RGB. Ba band này đều có GSD danh nghĩa 10 m nhưng vẫn phải được đưa về target grid của model. Không đọc B8/B8A hoặc band thứ tư vào tensor trong pipeline RGB.

PlanetScope có nhiều thế hệ sensor và product khác nhau; thứ tự band phải lấy từ metadata hoặc cấu hình product cụ thể.

### Bước 2: Tạo validity mask trước mọi thống kê

Validity mask phải tổng hợp riêng các trạng thái:

- NoData hoặc fill value.
- Pixel ngoài footprint.
- Detector/sensor defect.
- Saturation theo QA hoặc metadata.
- Band bị thiếu hoặc không đồng đăng ký.

Không dùng NoData khi tính min/max, percentile, mean hoặc std. Không gán NoData thành clear. Với patch có tỷ lệ invalid vượt `invalid_patch_ratio`, bỏ qua inference và đánh dấu `invalid`. Với patch còn lại, giá trị fill phải cố định và giống quy tắc lúc train.

QA có chứa nhãn cloud, cirrus hoặc cloud shadow phải được xử lý có chủ đích:

- Nếu mục tiêu là đánh giá model độc lập, không dùng các bit này làm đầu vào hoặc bộ lọc vì có thể gây target leakage.
- Nếu mục tiêu là hệ thống production kết hợp model và QA, phải mô tả đây là sensor fusion/rule fusion và đánh giá toàn bộ hệ thống, không chỉ model.

### Bước 3: Radiometric calibration theo product

Áp dụng đúng công thức được khai báo bởi product. Dạng tổng quát chỉ có thể viết là:

```text
physical_value = decode(DN, scale_factor, add_offset, product_metadata)
```

Với product thực sự khai báo phép biến đổi tuyến tính, công thức có thể là:

```text
reflectance = DN * scale_factor + add_offset
```

Tuy nhiên, không được coi đây là công thức phổ quát. Một số product Level-1 còn cần hệ số band-specific và hiệu chỉnh góc mặt trời; một số product đã là reflectance lượng tử hóa và có offset thay đổi theo processing baseline.

`valid_range`, `10000` hoặc `65535` không tự động là mẫu số chuẩn hóa. Chúng chỉ được dùng khi đặc tả chính thức của đúng product quy định như vậy.

Processing level của train và inference phải thống nhất. TOA và surface reflectance không được trộn trong cùng checkpoint nếu training chưa được thiết kế để chịu được sự khác biệt đó. Đối với cloud detection, surface reflectance cũng không mặc nhiên tốt hơn TOA vì atmospheric correction có thể không đáng tin cậy trên mây.

### Bước 4: Đồng đăng ký và resampling không gian

1. Reproject tất cả band về cùng CRS và pixel grid.
2. Chọn target GSD đúng với `InputSpec`, ví dụ 30 m nếu train trên Landsat 8 30 m.
3. Khi giảm độ phân giải reflectance liên tục, ưu tiên area/average resampling.
4. Với QA, class mask và validity mask, dùng nearest-neighbor hoặc majority phù hợp.
5. Không trộn các band chưa co-register vào cùng tensor.

Việc resample Sentinel-2 từ 10 m xuống 30 m giúp patch `256 x 256` bao phủ diện tích mặt đất gần với training. Tuy nhiên, nó không loại bỏ hoàn toàn khác biệt point-spread function, modulation transfer function và spectral response giữa các sensor.

### Bước 5: Normalization cố định

Baseline được khuyến nghị:

1. Decode về cùng đơn vị vật lý.
2. Áp dụng miền clipping cố định theo từng band, được chốt từ train set và product contract.
3. Scale hoặc Z-score bằng tham số cố định lưu trong `InputSpec`.
4. Dùng đúng các tham số đó ở inference.

Không dùng mặc định:

- Chia theo cực đại của dtype.
- Chia theo `max()` của ảnh hoặc patch.
- Percentile stretch riêng từng ảnh/tile.
- Z-score riêng từng ảnh/tile.
- Histogram matching với một ảnh tham chiếu.

Reflectance lớn hơn `1.0` không đủ để kết luận pixel bão hòa. Chỉ clipping nếu miền clipping đó đã được định nghĩa trong input contract và model đã được train với cùng quy tắc.

### Bước 6: Suy luận và tạo output mask

Mỗi patch phải trả về:

```text
cloud_probability, cloud_label, valid_fraction, status
```

`status` tối thiểu gồm `valid`, `invalid_input` và `out_of_distribution`. Khi ghép mask ảnh lớn:

- Không tính vùng invalid là clear.
- Tính cloud coverage trên tổng diện tích valid.
- Lưu validity mask riêng hoặc ghi NoData vào output.
- Ghi lại `InputSpec`, sensor, product và phiên bản preprocessor trong metadata kết quả.

Vì model là patch classifier, có thể dùng sliding window có overlap và trung bình xác suất để giảm block artifact. Cách này chỉ làm mask mượt hơn, không biến model thành pixel-level segmentation.

## 6. Spectral harmonization và domain adaptation

Phạm vi RGB làm pipeline đơn giản hơn nhưng không loại bỏ spectral shift. Red, Green và Blue của hai sensor vẫn có spectral response function, độ rộng band và độ nhạy khác nhau.

### 6.1. SBAF

SBAF có thể là một baseline, nhưng không phải hệ số nhân cố định cho mọi cảnh. Hệ số phụ thuộc cặp spectral response function, band, khí quyển và loại bề mặt. Chỉ dùng bộ hệ số phù hợp đúng sensor/product, có nguồn gốc rõ ràng và được validation trên dữ liệu đích.

### 6.2. Histogram hoặc quantile mapping

Không dùng histogram matching với một ảnh Landsat mẫu. Nếu cần statistical harmonization:

1. Thu thập nhiều cặp ảnh Landsat và sensor đích gần thời điểm, đã co-register.
2. Fit mapping theo từng band trên tập train riêng.
3. Cố định mapping sau khi fit.
4. Đánh giá trên scene chưa xuất hiện khi fit.

Mapping này vẫn là xấp xỉ thống kê và không thay thế radiometric calibration.

### 6.3. Giải pháp AI được ưu tiên

Khi có dữ liệu sensor đích, thứ tự ưu tiên là:

1. Fine-tune checkpoint Landsat trên dữ liệu có nhãn của sensor đích.
2. Train đa sensor với sampling cân bằng và augmentation radiometric hợp lý.
3. Dùng sensor-specific normalization hoặc adapter nhỏ nếu một model chung không đủ ổn định.
4. Dùng pseudo-label/domain adaptation chỉ sau khi có tập validation có nhãn để kiểm soát confirmation bias.

Preprocessing giúp dữ liệu có ý nghĩa nhất quán; khả năng tổng quát chéo sensor cuối cùng vẫn phải được học và kiểm chứng bằng dữ liệu.

## 7. Calibration ngưỡng và phát hiện OOD

Ngưỡng xác suất `0.5` không mặc nhiên tối ưu cho sensor mới. Trên validation set của từng sensor cần:

- Chọn threshold theo chi phí false clear và false cloud của nhiệm vụ downlink.
- Báo cáo precision, recall, F1, AUROC và confusion matrix.
- Kiểm tra probability calibration bằng Brier score hoặc ECE.
- Lưu threshold theo sensor/product cùng model configuration.

Trước inference, kiểm tra OOD ở mức dữ liệu:

- Band hoặc processing level không đúng contract.
- Tỷ lệ invalid quá cao.
- Thống kê từng band nằm quá xa phân phối train.
- Sensor/product chưa được validation.

Trong các trường hợp này, hệ thống nên từ chối hoặc gắn cờ kết quả thay vì đưa ra quyết định downlink với độ tin cậy giả tạo.

## 8. Kế hoạch đánh giá thực nghiệm

### 8.1. Dữ liệu

Tạo test set có nhãn riêng cho Landsat 8, Sentinel-2 và từng product PlanetScope cần hỗ trợ. Chia dữ liệu theo scene, đồng thời tách vùng địa lý, mùa và điều kiện bề mặt để tránh leakage.

Test set phải có các nhóm khó:

- Mây mỏng, haze và cirrus.
- Mây nhỏ hoặc patch chỉ có ít mây.
- Tuyết/băng, cát sáng, đô thị sáng và sun glint.
- Biển tối, bóng mây và vùng NoData.
- Góc mặt trời và tỷ lệ mây khác nhau.

Vì chỉ dùng RGB, cần đặc biệt theo dõi lỗi giữa mây với tuyết/băng, cát sáng, haze và bề mặt có độ phản xạ cao. Thiếu các band ngoài vùng khả kiến có thể làm các trường hợp này khó phân biệt hơn; đây là giới hạn thông tin của đầu vào, không thể khắc phục hoàn toàn bằng normalization.

### 8.2. Chỉ số

Ở mức patch:

- Recall cloud, precision, F1, specificity và AUROC.
- False-clear rate vì đây thường là lỗi tốn băng thông downlink nhất.
- Calibration error và độ ổn định threshold.

Ở mức ảnh:

- Sai số cloud coverage trên vùng valid.
- Tỷ lệ quyết định giữ/loại ảnh đúng.
- Chi phí downlink tiết kiệm được và tỷ lệ ảnh hữu ích bị loại nhầm.

Không báo cáo IoU pixel-level như chỉ số chính cho checkpoint hiện tại vì đầu ra gốc chỉ là nhãn patch.

### 8.3. Ablation bắt buộc

So sánh tuần tự:

1. Checkpoint hiện tại và normalization hiện tại.
2. Band mapping + validity mask.
3. Radiometric calibration nhất quán và retrain.
4. Thêm spatial/spectral harmonization.
5. Fine-tune hoặc train đa sensor.

Chỉ giữ một bước nếu nó cải thiện trên test scene độc lập và không gây suy giảm không chấp nhận được trên Landsat.

## 9. Luồng xử lý production đề xuất

```text
Input product
    |
    v
Metadata validation ---- không rõ ----> Reject / yêu cầu cấu hình rõ ràng
    |
    v
Band mapping + validity mask
    |
    v
Product-specific radiometric decoding
    |
    v
Co-registration + resampling về target grid
    |
    v
Fixed normalization từ InputSpec
    |
    v
OOD checks ---- ngoài miền ----> Invalid/OOD result
    |
    v
Patch inference + calibrated sensor threshold
    |
    v
Cloud mask + validity mask + provenance metadata
```

## 10. Thứ tự triển khai

### P0 - Bắt buộc trước khi sửa inference

- Xác định chính xác product, processing level, dtype và phân phối giá trị của dữ liệu 95-Cloud đã dùng để train.
- Chốt kiến trúc `num_channels=3` và retrain/fine-tune checkpoint RGB; không dùng trực tiếp checkpoint bốn kênh.
- Tạo `InputSpec` và lưu nó cùng checkpoint/ONNX/TensorRT engine.
- Định nghĩa rõ model hiện tại là patch classifier.

### P1 - Preprocessor dùng chung

- Tạo module preprocessing dùng chung cho train, eval và inference.
- Thêm adapter theo từng sensor/product.
- Trả về cả tensor và validity mask.
- Thêm unit test cho band order RGB, xử lý Alpha, scale/offset, NoData và resampling.

### P2 - Dữ liệu và đánh giá sensor đích

- Tạo test set Sentinel-2/PlanetScope có nhãn và scene-level split.
- Chạy baseline trước khi harmonization.
- Calibration threshold riêng theo sensor/product.

### P3 - Retrain/fine-tune

- Tái tạo dữ liệu Landsat bằng preprocessor mới.
- Retrain hoặc fine-tune với calibrated reflectance.
- Fine-tune/train đa sensor nếu preprocessing đơn thuần chưa đạt yêu cầu.

### P4 - Tối ưu triển khai

- Export model mới sang ONNX/TensorRT.
- Benchmark latency, RAM, năng lượng và độ chính xác trên thiết bị đích.
- Thêm logging provenance, OOD status và monitoring drift.

## 11. Tiêu chí hoàn thành

Giải pháp chỉ được coi là sẵn sàng production khi:

- Không có đường suy luận nào tự đoán band hoặc calibration từ dtype/max.
- Train, eval và inference sử dụng cùng preprocessor có phiên bản.
- NoData không bị tính thành clear.
- Mọi sensor/product được hỗ trợ đều có validation set và threshold đã hiệu chỉnh.
- Kết quả lưu đủ metadata để tái lập preprocessing.
- Model đạt tiêu chí false-clear, false-cloud và chi phí downlink do dự án quy định trên scene độc lập.
