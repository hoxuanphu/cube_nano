# Kế hoạch tích hợp SegFormer-B0

## 1. Mục tiêu và phạm vi

Bổ sung semantic cloud segmentation trên tập dữ liệu 95-Cloud, đồng thời giữ nguyên MobileNetV3 cho task `patch_classification` hiện tại.

Model mới phải:

- Tạo cloud mask ở mức pixel.
- Tính cloud coverage trên pixel hợp lệ.
- Chạy được qua worker, product và downlink chính thức, không chỉ qua script độc lập.
- Có contract và model release riêng, không thay đổi âm thầm release MobileNetV3.
- Có thể rollback về deployment profile MobileNetV3 đã phát hành.

Phạm vi MVP là model RGB ba kênh, batch `1`, CPU/reference runtime và một TensorRT engine cho target đã được pin. Huấn luyện và validation dùng đầu vào native `[1, 3, H, W]` để giữ nguyên kích thước 95-Cloud; contract inference/ONNX/TensorRT vẫn cố định `256 x 256`. Model RGB+NIR, chọn model theo từng request, multi-model concurrent loading và cascade không thuộc MVP.

T48PYS chỉ được dùng như external smoke test nếu chưa có ground truth độc lập. Nó chỉ trở thành quantitative holdout sau khi có nhãn được kiểm tra chất lượng.

## 2. Các quyết định kiến trúc

### 2.1. Model và label

- Model family: SegFormer-B0.
- Task: `semantic_cloud_segmentation`.
- Class mapping: `clear=0`, `cloud=1`.
- Training target dtype: integer mask `[H, W]`.
- Training values: `0`, `1`; `ignore_index=255` chỉ dùng trong training/evaluation target.
- Input: RGB theo thứ tự `[red, green, blue]`, source dtype `uint16`, tensor `float32`, layout `NCHW`.
- Input shape huấn luyện/validation: `[1, 3, H, W]` với H×W native của source và batch `1`; không resize label.
- Input shape runtime/export MVP: `[1, 3, 256, 256]`.
- Segmentation head: hai output classes, không gọi nhầm là head có hai layer.

Implementation SegFormer, pretrained artifact, checksum, license và dependency version phải được pin ở P0. Không được suy ra preprocessing chỉ từ image processor mặc định của thư viện.

### 2.2. Tách ba output contract

Model release phải tách rõ output của graph, postprocess và product.

`ModelOutputSpec` dự kiến cho graph MVP:

```yaml
kind: semantic_logits
name: logits
shape: [1, 2, 64, 64]
logical_dtype: float32
physical_dtype: artifact-manifest-defined
class_axis: 1
classes: [clear, cloud]
output_stride: 4
```

Shape này phải được xác nhận bằng vertical slice P3 và sau đó đóng băng. Reference/PyTorch và ONNX MVP dùng output `float32`; physical output dtype của TensorRT phải được pin trong engine manifest và được chuyển về comparison dtype trước parity check. FP16 không làm thay đổi class/shape semantics.

`PostprocessSpec` MVP:

```yaml
postprocess_id: segformer-softmax-bilinear-v1
resize:
  mode: bilinear
  target: input_spatial_shape
  align_corners: false
probability:
  kind: softmax
  cloud_class_index: 1
threshold_source: decision_spec.pixel_cloud_probability_threshold_bp
invalid_policy: exclude-from-mask-metrics-and-coverage
```

`ProductSpec` MVP:

```yaml
cloud_mask:
  dtype: uint8
  clear_value: 0
  cloud_value: 255
validity_mask:
  dtype: uint8
  invalid_value: 0
  valid_value: 1
```

Cloud mask và validity mask là hai artifact khác nhau. Byte `0` tại pixel invalid trong cloud mask không được diễn giải là clear; validity mask mới là nguồn quyết định pixel có tham gia coverage hay không.

### 2.3. DecisionSpec

Không dùng `argmax` như một quyết định ngầm cố định ở 0.5. Mỗi model release phải có `DecisionSpec` riêng:

```yaml
decision_spec_id: cloud-segmentation-decision-v1
pixel_cloud_probability_threshold_bp: <set-in-P0>
coverage_limit_bp: <set-in-P0>
threshold_selection_metric: <set-in-P0>
false_clear_constraint: <set-in-P0>
calibration_id: none-or-versioned-calibrator
```

Pixel threshold được chọn trên validation set rồi khóa trước khi chạy test. Coverage limit là quyết định ở mức ROI/scene và không được dùng thay cho pixel threshold.

### 2.4. Cách vận hành hai model

MVP chọn đúng một model task khi worker khởi động:

```text
deployment profile A -> active_model_task=patch_classification
                         model_release=cloud-mobilenetv3-small-rgb-r1

deployment profile B -> active_model_task=semantic_cloud_segmentation
                         model_release=cloud-segformer-b0-rgb-r1
```

Command phân tích hiện tại không chọn model theo từng request. Job snapshot, health telemetry và product manifest phải ghi `model_task`, `model_release_id` và toàn bộ contract ID. Worker không load đồng thời hai model. Rollback được thực hiện bằng cách kích hoạt lại deployment profile MobileNetV3 đã xác minh.

Nếu sau MVP cần chọn model theo request, phải tạo protocol/schema version mới cùng memory admission, lazy-load và deadline policy riêng.

## 3. Hiện trạng repository và phạm vi tích hợp

- [preprocess_95cloud.py](../src/data/preprocess_95cloud.py) đã lưu image-mask pair nhưng đang nhị phân hóa bằng `ground_truth > 0`; cần tránh làm mất encoding invalid.
- [cloud_dataset.py](../src/data/cloud_dataset.py) chỉ tạo nhãn crop-level cho classifier; chưa trả segmentation mask và validity mask.
- [manifest.py](../sat_ai/manifest.py) và [model_manifest.yaml](../sat_ai/model_manifest.yaml) đang ràng buộc với MobileNetV3 và `binary_cloud_logit`.
- [sat_ai/inference.py](../sat_ai/inference.py) nhận một logit cho mỗi tile và tô toàn bộ tile.
- [worker_process.py](../sat_ai/worker_process.py) gọi trực tiếp classifier runtime hiện tại.
- [products.py](../sat_ai/products.py) ghi `cloud_positive_tile_area_ratio_bp`; chưa có `pixel_cloud_ratio_bp` và validity artifact.
- [protocol/schemas.py](../protocol/schemas.py) không mang model task theo request. MVP giữ wire command hiện tại và lấy task từ deployment profile.
- [inference_tensorrt.py](../src/inference_tensorrt.py) chưa xử lý segmentation tensor.
- [jetson-l4t-profile.yaml](../deploy/jetson-l4t-profile.yaml) chưa deployable vì thiếu engine, optimization profile và target benchmark.

Không tạo một manifest parser song song chỉ dùng cho SegFormer. Cần nâng manifest schema thành task-aware, đồng thời giữ backward compatibility hoặc migration test cho release MobileNetV3.

## 4. Kế hoạch theo gate và giai đoạn

### P0 - Đóng băng đặc tả, target và acceptance profile

Chốt trước khi sửa training pipeline:

- Exact SegFormer implementation và pretrained artifact, gồm source, version, checksum và license.
- `InputSpec`, `ModelOutputSpec`, `PostprocessSpec`, `ProductSpec` và `DecisionSpec`.
- Radiometric fields bắt buộc: sensor, platform, product, processing level, units, scale/offset, NoData, saturation, GSD và band order.
- Exact target: hardware SKU, OS/L4T, CUDA, TensorRT, precision, power mode, clock policy và memory budget.
- Absolute quality gates, parity tolerance và runtime SLO trong `acceptance_profile`.
- Model manifest schema version và migration policy cho MobileNetV3.

Không được để trống các ngưỡng acceptance khi bắt đầu final training. Giá trị có thể được thay đổi sau pilot nhưng phải tạo profile version mới và không được tối ưu trên test set.

Deliverable:

```text
InputSpec
ModelOutputSpec
PostprocessSpec
ProductSpec
DecisionSpec
AcceptanceProfile
TargetDeploymentSpec
```

Gate G0: các contract parse/validate được, không còn trường bắt buộc chưa xác định, và release MobileNetV3 cũ vẫn load được qua compatibility test.

### P1 - Audit dữ liệu, radiometry và split

- Xác định chính xác nguồn, sensor/product, processing level, units, scale/offset, NoData và GSD của dữ liệu 95-Cloud được dùng.
- Kiểm tra image-mask pairing, shape, dtype, mask values, file lỗi và duplicate.
- Xác định encoding ground truth trước khi nhị phân hóa; không dùng chung `255` cho cloud và invalid trong cùng training artifact.
- Tạo validity mask riêng từ metadata/QA/ground truth khi có. Nếu nguồn không có thông tin invalid, phải ghi rõ giả định toàn bộ source pixel là valid.
- Tính cloud ratio và normalization statistics chỉ trên pixel hợp lệ.
- Chia train/validation/test theo source scene; không chia ngẫu nhiên từng patch.
- Kiểm tra near-duplicate hoặc các scene có cùng nguồn không bị tách sang nhiều split.
- Cân bằng split theo cloud coverage strata khi có thể nhưng không dùng test để điều chỉnh model.
- Fit clip/mean/std chỉ trên train scenes rồi đóng băng trong `InputSpec`.
- Tạo lineage ID cho raw manifest, processed dataset, split và preprocessing config.

Không dùng thư mục `cloud/clear` làm ground truth chính cho segmentation. Đây chỉ là nhãn phụ phục vụ classifier.

Deliverable: processed dataset bất biến, validity masks, frozen split manifest, label-quality report và unit test chống scene leakage.

Gate G1: toàn bộ pair hợp lệ, split không leakage, invalid policy có kiểm thử và cùng raw manifest/config tạo cùng processed dataset ID.

### P2 - Dataset, preprocessor và augmentation

Bổ sung `src/data/segmentation_dataset.py` trả về tối thiểu:

```text
image: Tensor[C, H, W]
mask: Tensor[H, W]
validity_mask: Tensor[H, W]
scene_id: str
tile_coordinates: tuple[int, int, int, int]
```

Yêu cầu:

- Train, eval và runtime gọi chung một implementation preprocessing có version.
- Crop image, mask và validity mask tại cùng tọa độ.
- Flip, rotate và crop đồng bộ.
- Mask và validity mask chỉ dùng nearest-neighbor khi resize.
- Pixel invalid được gán `ignore_index=255` khi tạo target cho loss.
- Huấn luyện/validation native-size giữ nguyên H×W của source; decoder logits được bilinear-upsample về H×W target cho loss và metric, không resize target/validity. Runtime vẫn dùng tiling `256` độc lập với luồng training.
- Edge padding được đánh dấu invalid; training có sample padding phù hợp với inference edge policy.
- Có sampling theo cloud-ratio bins cho training nếu imbalance lớn; validation/test giữ phân phối tự nhiên.
- Validation/test dùng deterministic tiling phủ toàn bộ source patch/scene, không chỉ center crop.

Gate G2-data: cùng raw tile và `InputSpec` tạo tensor giống nhau trong train, eval và reference runtime.

### P3 - Vertical slice ONNX/TensorRT trước full training

Tạo một model cấu trúc hoặc checkpoint pilot để kiểm tra sớm toàn bộ đường kỹ thuật:

- Bổ sung `src/models/segformer_b0.py` với wrapper có output contract cố định.
- Export fixed batch `1`, input `[1, 3, 256, 256]`, output logits đã chốt.
- Chọn opset từ compatibility matrix của target, không mặc định dùng opset 11.
- Chạy PyTorch và ONNX Runtime trên golden tiles.
- Build thử TensorRT FP16 trên đúng target và kiểm tra LayerNorm, GELU, attention, reshape, matmul và Resize.
- Đo model load, workspace, peak memory và latency pilot.
- Kiểm tra raw logits và postprocessed output theo parity tolerance trong AcceptanceProfile.
- Xác nhận runtime có thể stream tile output mà không giữ probability map full-scene.

Nếu TensorRT graph hoặc memory budget không đạt, phải quyết định graph rewrite, precision khác, tile size khác hoặc model khác trước full training. Plugin tùy chỉnh chỉ dùng sau khi các phương án export tương thích thất bại và plugin được version/hash hóa.

Deliverable: pilot ONNX, pilot engine, golden vectors, feasibility report và output contract đã đóng băng.

Gate G2-target: graph build được trên target, memory/latency pilot nằm trong ngân sách sơ bộ và không còn operator blocker chưa có owner.

### P4 - Huấn luyện và ablation

Bổ sung:

```text
src/train_segmentation.py
src/eval_segmentation.py
```

Baseline training:

- `num_labels=2`, AdamW và AMP khi backend hỗ trợ.
- Learning rate candidates `6e-5` và `1e-4`.
- Weight decay ban đầu `1e-4`.
- 50-100 epochs, warmup, cosine decay và early stopping.
- Loss baseline:

```text
L = 1.0 * CrossEntropy(logits, target, ignore_index=255)
  + 1.0 * SoftDiceLoss(cloud_probability, cloud_target, valid_pixels_only)
```

- Dice tính cho cloud class trên toàn mini-batch với epsilon được pin trong training config; CE xử lý cả clear và cloud. Phải có unit test cho batch toàn clear, toàn cloud và không có pixel hợp lệ.
- Sample không có pixel hợp lệ bị loại trước loss. Nếu một mini-batch vẫn trở thành all-invalid, training loop phải bỏ qua optimizer step, tăng counter có giám sát và không đưa loss đó vào metric trung bình.
- Chọn checkpoint trên validation theo metric đã chốt ở P0; không chọn từ test.
- Theo dõi gradient, non-finite loss, class distribution và valid-pixel ratio.

Ablation tối thiểu:

- Physical/radiometric scaling theo InputSpec với fixed per-band statistics từ train split.
- ImageNet mean/std chỉ sau cùng một radiometric scaling hợp lệ.
- CE-only so với CE + Dice.
- Sampling tự nhiên so với cloud-ratio-bin sampling nếu imbalance đáng kể.

Không coi `uint16 / 65535` là physical normalization nếu raw-data audit chưa chứng minh contract đó. Candidate cuối nên chạy ít nhất ba seed; nếu tài nguyên không đủ, phải dùng scene-level bootstrap và ghi rõ giới hạn của single-seed run.

Checkpoint bundle phải lưu model/optimizer/scheduler/scaler state, epoch, global step, best metric, pretrained artifact ID, Git commit, dependency versions, dataset/split/InputSpec ID, class mapping, loss config, seed và validation metrics.

### P5 - Evaluation, calibration và model selection

Đánh giá SegFormer-B0 và MobileNetV3 trên cùng frozen test scenes 95-Cloud. MobileNetV3 được chuyển thành coarse mask trên đúng tile grid và valid area; không coi mask block là segmentation ground truth.

Quy trình bắt buộc:

1. Chọn pixel probability threshold và calibration trên validation set.
2. Khóa checkpoint, preprocessing, postprocessing và threshold.
3. Chạy test set một lần cho báo cáo release.
4. Báo cáo cả aggregate metric, macro-scene metric và confidence interval bootstrap theo scene.
5. Báo cáo theo cloud-coverage strata và valid-pixel strata.

Metric bắt buộc:

- Cloud IoU, Dice/F1, precision, recall và false-clear rate.
- Confusion matrix trên pixel hợp lệ.
- Boundary F1 với tolerance được định nghĩa trong AcceptanceProfile.
- Cloud coverage bias, MAE, RMSE và p95 absolute error theo scene/ROI.
- False-accept ảnh nhiều mây và false-reject ảnh hữu ích tại coverage limit.
- Calibration metric nếu sử dụng xác suất cho threshold/cascade.
- Model-only latency và end-to-end latency/RSS được báo cáo riêng.

T48PYS không được trộn vào split 95-Cloud. Không dùng nó để báo cáo IoU/Dice nếu chưa có independent ground truth. Khi có nhãn, báo cáo riêng như target-domain evaluation và theo dõi domain shift về sensor, GSD, radiometry và spectral response.

Gate G3: model vượt toàn bộ absolute AI gates và baseline-improvement gate; test không tham gia bất kỳ quyết định tuning nào.

### P6 - Reference runtime, worker và product integration

- Nâng model manifest thành task-aware; giữ compatibility test cho MobileNetV3 release hiện tại.
- Thêm model/runtime factory theo `active_model_task` của deployment profile.
- Ghi task, release, contract IDs và DecisionSpec vào immutable job snapshot.
- Dùng benchmark artifact của đúng model release để tính deadline.
- Bổ sung segmentation inference chạy theo tile, upsample logits, threshold probability và ghép mask theo ROI.
- Coverage chỉ tính trên valid pixels:

```text
pixel_cloud_ratio_bp = (cloud_and_valid_pixels * 10000) // valid_pixels
```

Phép chia nguyên lấy sàn phải dùng cùng canonical helper với classifier để giữ semantics basis-point nhất quán giữa runtime và protocol.

- Reject với `INSUFFICIENT_VALID_DATA` khi valid fraction dưới ngưỡng contract.
- Product gồm `cloud_mask.tif`, `validity_mask.tif`, crop/quicklook theo policy và manifest có provenance đầy đủ.
- Product manifest phân biệt `pixel_cloud_ratio_bp` với `cloud_positive_tile_area_ratio_bp`.
- Health telemetry công bố active model task/release và domain status.
- Giữ wire command hiện tại trong MVP; task không được thay đổi giữa lúc admit và hoàn tất job.

Tiling v1 dùng tile `256`, stride `256`. Edge padding là invalid. Không blend binary masks. Nếu overlap được thêm ở release sau, phải kết hợp logits/probabilities bằng weighting window đã version hóa rồi threshold một lần; mỗi valid pixel chỉ được đếm một lần trong coverage.

Gate G4: scene/ROI đi qua command -> worker -> segmentation -> product -> downlink, restart/timeout/cancel không để lại product dở dang, và toàn bộ test MobileNetV3 hiện có không regression.

### P7 - Trained export, TensorRT release và target benchmark

- Export checkpoint được chọn sang ONNX bằng output contract đã đóng băng.
- So sánh PyTorch, ONNX và TensorRT trên golden tiles, scene samples và threshold-near samples.
- Build TensorRT engine trên đúng target với builder flags, optimization profile và plugin hashes được lưu.
- Đo cold start, warm p50/p95/p99, throughput, peak RSS, CUDA/TensorRT workspace và deadline miss rate.
- Benchmark end-to-end gồm input reader, preprocessing, inference, stitching, product writing và cleanup.
- Chạy representative small/medium/large ROI cùng edge/NoData-heavy cases.
- Hash checkpoint, ONNX, engine, manifest, DecisionSpec, AcceptanceProfile, split và evaluation report.
- Chạy rollback drill về MobileNetV3 deployment profile.

Chỉ chuyển target profile sang `deployable: true` sau khi engine hash, target benchmark artifact, parity report và G0-G4 đều hợp lệ. Không suy ra GPU SLO từ CPU benchmark.

## 5. AcceptanceProfile và tiêu chí nghiệm thu

P0 phải tạo một file version hóa có ít nhất các trường sau và giá trị cụ thể được project owner phê duyệt:

```yaml
quality:
  min_cloud_iou: <required>
  min_cloud_dice: <required>
  min_cloud_recall: <required>
  max_false_clear_rate: <required>
  max_coverage_mae_bp: <required>
  max_coverage_p95_abs_error_bp: <required>
  boundary_f1_tolerance_pixels: <required>
  min_boundary_f1: <required>
decision:
  max_false_accept_cloudy_scene_rate: <required>
  max_false_reject_useful_scene_rate: <required>
parity:
  pytorch_onnx: <required>
  pytorch_tensorrt_fp16: <required>
runtime:
  max_cold_start_ms: <required>
  max_warm_p95_ms_per_tile: <required>
  max_end_to_end_deadline_miss_rate: <required>
  max_peak_rss_bytes: <required>
  min_valid_pixel_ratio: <required>
```

Release chỉ được nghiệm thu khi:

1. Dataset không scene leakage, pair/validity hợp lệ và lineage tái lập được.
2. SegFormer vượt các ngưỡng tuyệt đối và cải thiện cloud IoU/Dice cùng coverage MAE so với coarse MobileNet baseline.
3. Pixel threshold được chọn trên validation và khóa trước test.
4. PyTorch, ONNX và TensorRT đạt parity contract trên raw output lẫn postprocessed output.
5. Target đạt end-to-end latency, memory và deadline SLO.
6. Artifact được hash/version hóa và manifest tham chiếu đúng dependency.
7. Worker/product/downlink integration pass, không regression MobileNetV3.
8. Rollback drill về MobileNetV3 thành công.
9. Không dùng T48PYS không nhãn làm bằng chứng quantitative quality.

## 6. Test strategy

### Unit tests

- Mask mapping `0/1/255` và validity-mask semantics.
- Synchronized image/mask/validity augmentation.
- Loss với all-clear, all-cloud, mixed và no-valid-pixel batches.
- Softmax, resize, threshold và coverage basis-point integer conversion.
- Edge padding không tham gia coverage.
- Manifest validation theo task và backward compatibility MobileNetV3.

### Integration tests

- Scene-level split không leakage và deterministic lineage IDs.
- Train/eval/runtime preprocessing parity.
- PyTorch/ONNX/TensorRT golden vectors.
- Full ROI và partial edge ROI stitching correctness.
- Worker restart, cancellation, deadline và cleanup.
- Product bundle có cloud mask, validity mask và provenance đúng.
- Downlink round trip giữ nguyên artifact hashes.

### Evaluation tests

- Threshold selection chỉ đọc validation predictions.
- Test runner từ chối config chưa khóa hoặc split hash sai.
- Metric macro/micro và bootstrap chạy deterministic với seed cố định.
- Baseline và SegFormer dùng cùng test scenes, ROI và valid-pixel denominator.

## 7. Rủi ro và phụ thuộc

| Rủi ro | Mức độ | Biện pháp |
|---|---|---|
| Ground-truth/NoData encoding không rõ | Nghiêm trọng | Raw-data audit, validity mask riêng, không nhị phân hóa trước khi xác định encoding |
| Runtime vẫn hard-code MobileNet | Nghiêm trọng | Task-aware manifest/runtime factory, active task cố định theo worker profile, end-to-end gate |
| Logits và mask contract không đồng nhất | Nghiêm trọng | Ba contract riêng, golden vectors cho raw và postprocessed output |
| Domain shift sang sensor khác | Cao | Supported-domain allow-list, target set có nhãn, không dùng smoke test làm quality evidence |
| Radiometry train/inference lệch nhau | Cao | Shared versioned preprocessor, physical metadata contract, parity tests |
| SegFormer chậm hoặc tốn RAM | Cao | P3 feasibility spike, batch 1, FP16, stream output, full-pipeline benchmark |
| TensorRT không hỗ trợ graph | Cao | Export/build sớm, graph rewrite trước plugin, exact target fingerprint |
| False-clear cao tại threshold 0.5 | Cao | Validation threshold sweep, false-clear constraint và DecisionSpec |
| Seam ở biên tile | Trung bình | Padding parity; overlap release sau phải blend logits/probability, không blend mask |
| Probability map làm tăng RAM/disk | Trung bình | Chỉ giữ tile buffer và stream uint8 product |
| Single-seed variance | Trung bình | Ba seed cho candidate cuối hoặc scene bootstrap kèm giới hạn được ghi rõ |

## 8. Ước lượng

Ước lượng là engineering effort cho một kỹ sư, không bao gồm thời gian chờ train dài, cấp quyền dataset, mua/đặt thiết bị hoặc sửa lỗi toolchain ngoài kiểm soát:

- P0: 2-3 ngày.
- P1-P2: 4-7 ngày.
- P3: 2-4 ngày nếu target và toolchain có sẵn.
- P4-P5: 6-11 ngày, không tính thời gian compute chờ.
- P6: 4-7 ngày.
- P7: 5-10 ngày nếu có target.

Tổng CPU/reference MVP đến hết P6: **18-32 ngày công**. TensorRT release và target benchmark P7: **5-10 ngày công**. Tổng toàn bộ P0-P7: **23-42 ngày công**, cộng thời gian compute và external dependency.

Các pha data audit, dependency setup và target feasibility có thể chạy xen kẽ, nhưng không được bỏ qua gate để rút ngắn lịch. Fine-tune cho target-domain shift và cascade nằm ngoài ước lượng này.
