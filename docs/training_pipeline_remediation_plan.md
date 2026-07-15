# Kế hoạch khắc phục Training Pipeline RGB

## 1. Mục tiêu và quyết định kiến trúc

Kế hoạch này xử lý các vấn đề được phát hiện trong
[`training_pipeline_review.md`](./training_pipeline_review.md), với mục tiêu đưa pipeline
huấn luyện mô hình phân loại mây RGB từ mức baseline thử nghiệm lên mức có thể đánh giá
và triển khai cross-satellite một cách kiểm soát được.

> Cập nhật ngày 2026-07-14 sau vòng
> [`expert review`](./expert_review_remediation_plan.md),
> [`rebuttal`](./expert_review_remediation_plan_rebuttal.md) và
> [`response to rebuttal`](./expert_review_response_to_rebuttal.md).

Các quyết định nền tảng:

- Dùng `src/` làm pipeline chuẩn duy nhất.
- Notebook Kaggle chỉ dùng để cấu hình và gọi các entrypoint trong `src/`; không duy trì
  một bản sao riêng của dataset, training loop hoặc export logic.
- Chốt model chính là MobileNetV3-Small patch classifier với đầu vào ba kênh RGB theo
  thứ tự `[red, green, blue]`.
- Một checkpoint chỉ hợp lệ với đúng input contract, preprocessor và decision threshold
  đã được lưu cùng checkpoint.
- Mọi thay đổi radiometric phải được áp dụng đồng thời cho dữ liệu train, validation,
  test và inference, sau đó retrain hoặc fine-tune model.
- Dữ liệu, split, normalization và experiment phải có ID riêng, có lineage và có thể tái tạo.
- Contract violation, invalid input, distribution warning và semantic OOD là các trạng thái
  khác nhau; mỗi trạng thái phải có action rõ ràng.

Pipeline đích:

```text
Raw product
    -> Product adapter
    -> Calibrated/versioned representation + validity mask
    -> ProcessedDatasetID
    -> Scene-level split + SplitID
    -> Fit normalization trên train split + InputSpecID
    -> Train
    -> Checkpoint bundle + ExperimentID
    -> Scene evaluation + threshold calibration
    -> ONNX/TensorRT + metadata
    -> Input validation/domain guardrails
    -> Inference + monitoring
```

## 2. Trạng thái hiện tại

Các phần đã được triển khai trong `src/` và cần được giữ lại:

- Image và ground-truth mask được ghép theo filename.
- Nhãn được tính lại từ mask tương ứng với đúng random crop.
- `pos_weight` được ước lượng từ phân phối nhãn crop.
- Dataset được chia theo scene để hạn chế leakage.
- `src/train.py` đã tắt `cudnn.benchmark` và bật chế độ deterministic của cuDNN.
- Đã có unit test cho noisy label, image-mask pairing và scene-level split.

Các vấn đề còn phải xử lý:

- Notebook Kaggle vẫn gán nhãn source patch cho random crop và chưa deterministic.
- RGB chưa là cấu hình mặc định xuyên suốt dataset, model, train, eval, export và inference.
- Normalization còn dựa trên dtype hoặc `max()` thay vì sensor/product contract.
- Validation/test chỉ dùng một center crop cho mỗi source patch.
- Checkpoint chỉ chứa `state_dict`, không tự mang input contract và training state.
- Metric và threshold chưa phản ánh chi phí false-clear/false-cloud của bài toán downlink.
- Chưa có benchmark có nhãn trên sensor và product đích.

## 3. P0 - Khóa pipeline chuẩn và baseline

**Mức ưu tiên:** Bắt buộc  
**Ước lượng:** 1-2 ngày công

### Công việc

1. Chọn `src/train.py` làm entrypoint training chính thức.
2. Tạm ngừng sử dụng checkpoint được sinh từ notebook hiện tại cho đánh giá cuối cùng.
3. Xóa các implementation trùng lặp khỏi notebook, gồm `CloudDataset`, model factory,
   training loop, metric và ONNX export.
4. Chuyển notebook thành lớp điều phối thực hiện các bước:
   - Cài dependency.
   - Thiết lập đường dẫn và cấu hình thí nghiệm.
   - Gọi preprocessing và scene split.
   - Gọi `src/train.py`, `src/eval.py` và `src/export_onnx.py`.
   - Hiển thị log và artifact đầu ra.
5. Audit dữ liệu 95-Cloud để xác định chính xác sensor, product, processing level,
   đơn vị, NoData, scale/offset và GSD.
6. Lưu baseline provenance gồm Git commit, split manifest hash, seed, dependency version,
   Torch/CUDA/cuDNN version và cấu hình training.
7. Chạy lại toàn bộ test noisy-label hiện có trước khi bắt đầu thay đổi contract.

### Tiêu chí nghiệm thu

- Notebook không còn định nghĩa pipeline training riêng.
- Notebook và CLI tạo cùng loại nhãn cho cùng image-mask crop.
- Checkpoint notebook cũ được đánh dấu `legacy` và không được dùng để export production.
- Radiometric contract của 95-Cloud được xác định; nếu chưa xác định được thì phải ghi rõ
  cross-satellite đang bị chặn và không được tự suy luận từ dtype.

## 4. P1 - Chuẩn hóa RGB và checkpoint contract

**Mức ưu tiên:** Cao  
**Ước lượng:** 2-3 ngày công

### Công việc

1. Tạo `src/input_spec.py` với một schema có version, tối thiểu gồm:
   - `schema_version` và `preprocessor_version`.
   - `model_task=patch_classification`.
   - `channels=3` và `band_order=[red, green, blue]`.
   - `patch_size`.
   - Sensor, platform, product và processing level.
   - Expected units và scale/offset theo band.
   - Target GSD, CRS/grid policy và resampling method.
   - NoData policy và maximum invalid-pixel ratio.
   - Fixed clip range, mean và std theo band.
2. Tạo `DecisionSpec` riêng để lưu:
   - Probability threshold theo sensor/product.
   - Cloud-coverage threshold ở mức scene.
   - Cost hoặc constraint cho false-clear và false-cloud.
   - Phương pháp probability calibration nếu có.
3. Tạo `src/checkpoint.py` để lưu checkpoint bundle gồm:
   - Format version, model name và model state.
   - Optimizer, scheduler và AMP scaler state.
   - Epoch, global step và best validation metric.
   - `InputSpec`, `DecisionSpec` và training config.
   - `processed_dataset_id`, `split_id`, `input_spec_id`, `experiment_id` và runtime provenance.
4. Đổi cấu hình mặc định thành RGB trong:
   - `src/data/preprocess_95cloud.py`.
   - `src/data/cloud_dataset.py`.
   - `src/models/mobilenetv3.py`.
   - `src/train.py`.
   - `src/eval.py`.
   - `src/export_onnx.py`.
   - `src/inference_tensorrt.py`.
   - `src/inference_large_image_trt.py`.
5. Đặt `channel_dropout_p=0` và báo lỗi nếu channel dropout được bật cho model RGB.
6. Để eval, export và inference tự đọc channels, patch size, preprocessing và threshold
   từ checkpoint bundle. CLI override chỉ dùng cho debug và phải báo lỗi khi không khớp.
7. Hỗ trợ checkpoint `state_dict` cũ bằng một migration path tường minh như
   `--legacy-input-spec`; không tự suy luận normalization từ trọng số.
8. Thêm resume training từ bundle, bao gồm optimizer, scheduler, scaler và epoch.

### Tiêu chí nghiệm thu

- Lệnh train/eval/export chuẩn không cần truyền `--channels 3`.
- Checkpoint RGB không thể bị load âm thầm bằng model bốn kênh.
- Mismatch giữa checkpoint, ONNX, TensorRT engine và input bị chặn trước inference.
- Resume training khôi phục đúng learning rate, epoch và optimizer state.

## 5. P2 - Preprocessor product-aware dùng chung

**Mức ưu tiên:** Cao  
**Ước lượng:** 5-8 ngày công

### Công việc

1. Tạo `src/data/preprocessor.py` với API dùng chung cho train, eval và inference.
2. Preprocessor phải trả về ít nhất:

```text
tensor, validity_mask, preprocessing_metadata
```

3. Tạo product adapter riêng cho từng tổ hợp sensor/product/processing level được hỗ trợ.
   Adapter chịu trách nhiệm:
   - Xác nhận band identity và sắp xếp về RGB.
   - Decode DN bằng scale/offset đúng theo metadata product.
   - Tạo validity mask từ NoData, fill, footprint, saturation và band availability.
   - Co-register/reproject/resample về target grid khi cần.
4. Tách pipeline dữ liệu thành hai giai đoạn:
   - Với analytic product, decode/calibrate raw product về đơn vị vật lý nhất quán.
   - Với rendered RGB, khai báo rõ gamma/tone mapping/compression contract; không giả định
     có thể khôi phục reflectance vật lý.
   - Áp dụng fixed normalization được lưu trong `InputSpec`.
5. Chỉ fit clip/mean/std trên scene thuộc train split. Validation và test không được tham
   gia tính statistics.
6. Thêm thư mục hoặc artifact `validity_masks/`, tách biệt với cloud ground-truth masks.
7. Tính cloud ratio trên số pixel hợp lệ:

```text
cloud_ratio = cloud_valid_pixels / valid_pixels
```

8. Crop vượt `invalid_patch_ratio` phải bị loại khỏi training hoặc mang trạng thái
   `invalid`; không được gán nhãn clear.
9. Loại bỏ normalization dựa trên `img.max()`, `np.iinfo(dtype).max` hoặc heuristic
   `255/65535` trong dataset và large-image inference.
10. Đánh dấu edge padding của ảnh lớn là invalid để không làm giảm giả tạo cloud coverage.
11. Tái tạo processed dataset và retrain sau khi preprocessor mới hoàn thành.
12. Tạo hệ thống định danh phân tầng, không gộp mọi artifact thành một dataset ID:

```text
processed_dataset_id = hash(raw_manifest + preprocessor_version + preprocessing_parameters)
split_id             = hash(processed_dataset_id + scene_assignment)
input_spec_id        = hash(split_id + normalization_statistics + input_contract)
experiment_id        = hash(input_spec_id + model_config + training_config)
```

13. Mỗi manifest phải lưu parent ID để tạo lineage graph từ raw data đến release artifact.
14. Định nghĩa lifecycle policy:
   - Raw data và release dataset là immutable.
   - Intermediate data có retention period và có thể rebuild deterministic.
   - Khi phát hiện lỗi preprocessor, tạo version mới; không ghi đè artifact cũ.
   - Checkpoint bị ảnh hưởng phải bị revoke và retrain, không chỉ đổi metadata.
15. Chọn artifact registry phù hợp với hạ tầng. DVC có thể được dùng nhưng không thay thế
    manifest schema, validation gate và deterministic rebuild.

### Tiêu chí nghiệm thu

- Cùng một input và `InputSpec` tạo tensor giống nhau trong train, eval và inference.
- Không còn code path production suy luận radiometry từ dtype hoặc giá trị lớn nhất.
- NoData không tham gia normalization statistics và không bị tính là clear.
- Input thiếu hoặc mơ hồ metadata phải fail-fast với thông báo có thể xử lý được.
- Cùng raw manifest và preprocessing config phải tạo cùng `processed_dataset_id`.
- Đổi scene assignment chỉ làm đổi `split_id`, không làm đổi `processed_dataset_id`.
- Từ checkpoint phải truy được đầy đủ lineage đến raw manifest và preprocessor version.

## 6. P3 - Validation theo tiling và metric theo scene

**Mức ưu tiên:** Cao  
**Ước lượng:** 3-4 ngày công

### Công việc

1. Thêm deterministic tiling cho validation/test, với stride và edge policy cố định.
2. Bảo đảm toàn bộ source patch hoặc scene được phủ; không chỉ dùng một center crop.
3. Trả thêm metadata từ evaluation dataset:
   - `scene_id` và patch ID.
   - Tọa độ tile.
   - Valid fraction.
   - Ground-truth cloud ratio.
4. Báo cáo metric ở mức crop:
   - Loss, confusion matrix, precision, recall, F1 và specificity.
   - False-clear rate và false-cloud rate.
   - AUROC và PR-AUC/Average Precision.
   - Brier score và Expected Calibration Error.
5. Báo cáo metric ở mức scene:
   - Ground-truth và predicted cloud coverage trên vùng valid.
   - Mean absolute error của cloud coverage.
   - Quyết định giữ/loại ảnh.
   - False-accept ảnh nhiều mây và false-reject ảnh hữu ích.
6. Chọn best training checkpoint bằng PR-AUC hoặc validation loss thay vì F1 tại một
   threshold cố định.
7. Thực hiện probability-threshold sweep trên validation set bằng cost function downlink
   hoặc một constraint false-clear đã được thống nhất.
8. Khóa probability threshold trước khi chạy test; test set không được tham gia chọn
   normalization, calibration, model hoặc threshold.
9. Xuất metric tổng và metric phân nhóm theo sensor/product, địa lý, mùa, cloud coverage
   và nhóm cảnh khó.

### Tiêu chí nghiệm thu

- Cùng checkpoint và split luôn tạo cùng tập tile và metric.
- Mỗi scene trong validation/test được đánh giá trên toàn bộ vùng hợp lệ.
- Threshold trong release bundle có provenance từ validation run.
- Báo cáo test không chứa thao tác fit normalization, calibration hoặc threshold.

## 7. P4 - Reproducibility, notebook và export

**Mức ưu tiên:** Trung bình  
**Ước lượng:** 2-3 ngày công

### Công việc

1. Bổ sung deterministic `torch.Generator`, DataLoader worker seed và chế độ
   `deterministic={strict,warn,off}`.
2. Trong strict mode, bật deterministic algorithms và cấu hình CUDA cần thiết; dừng với
   lỗi rõ ràng nếu gặp operator không deterministic.
3. Notebook sử dụng cùng seed helper và luôn đặt `cudnn.benchmark=False`.
4. Log seed, package versions, CUDA, cuDNN, device và split hash trong checkpoint.
5. Export ONNX trực tiếp từ checkpoint bundle, không yêu cầu nhập lại channels/patch size.
6. Nhúng `InputSpec` và `DecisionSpec` vào ONNX metadata.
7. Tạo sidecar JSON có checksum cho TensorRT engine vì engine không tự mang đầy đủ
   training contract. Sidecar tối thiểu phải lưu:
   - Target device và GPU compute capability.
   - OS/L4T, JetPack, TensorRT, CUDA và driver version.
   - Plugin names, versions và hashes.
   - Builder flags, optimization profile, precision và calibration cache hash.
   - ONNX checksum, `InputSpecID`, `DecisionSpec` version và build timestamp.
8. So sánh numerical output giữa PyTorch và ONNX trên cùng batch kiểm thử.
9. Xác nhận ONNX input có shape `N x 3 x H x W` và output có shape `N x 1`.
10. Lưu ONNX làm artifact portable chính; build TensorRT engine trên thiết bị đích hoặc
    môi trường target-equivalent, sau đó smoke test trên chính target device.

### Tiêu chí nghiệm thu

- Hai smoke training run cùng seed tạo cùng label sequence và metric trong tolerance.
- Notebook và CLI tạo cùng artifact khi dùng cùng config.
- ONNX/TensorRT artifact không có metadata hoặc metadata sai checksum bị từ chối.
- PyTorch và ONNX probability sai khác không vượt tolerance đã định nghĩa.
- TensorRT engine chỉ được phát hành sau khi target-device smoke test và parity test đạt.

## 8. P5 - Retrain baseline RGB và normalization ablation

**Mức ưu tiên:** Bắt buộc trước cross-satellite  
**Ước lượng:** 5-8 ngày công/GPU, không tính thời gian chuẩn bị dữ liệu

### Normalization và class-weight ablation

Giữ nguyên architecture, scene split và seed set để so sánh:

1. Physical reflectance với fixed clipping/scaling.
2. Physical reflectance với fixed mean/std tính từ train split.
3. Physical reflectance với ImageNet mean/std.
4. Dynamic crop labeling không dùng `pos_weight`.
5. Dynamic crop labeling có `pos_weight` ước lượng từ crop.

Mỗi cấu hình nên chạy ít nhất ba seed. Không lựa chọn phương án chỉ dựa trên một run.

### Label-quality audit và cloud-ratio contract

1. Giữ baseline label contract hiện tại là:

```text
cloud_ratio_threshold = 0.10
label = 1.0 if cloud_ratio >= 0.10 else 0.0
```

2. Định nghĩa cloud-ratio threshold từ mục tiêu nghiệp vụ: tỷ lệ mây tối thiểu khiến patch
   không còn hữu ích cho quyết định downlink. Không chọn `T` chỉ vì một giá trị làm F1 hoặc
   PR-AUC cao hơn.
3. Nếu cần sensitivity analysis, chạy nhiều candidate `T`; trong mỗi experiment, dùng cùng
   một `T` cho train, validation và test. Final test chỉ chạy sau khi đã khóa `T`.
4. Không so sánh metric giữa các `T` như các task hoàn toàn tương đương vì mỗi `T` thay đổi
   label definition và class prevalence. Báo cáo cả utility/cost theo nghiệp vụ.
5. Audit annotation bằng stratified sampling theo:
   - Cloud-ratio bins, gồm cả vùng gần và xa threshold.
   - Scene, mùa, địa lý và loại bề mặt.
   - Thin cloud, haze, cirrus, vùng biên và NoData.
6. Dùng ít nhất hai annotator hoặc một reference độc lập cùng annotation guideline. Báo cáo
   agreement/disagreement rate và adjudication result trước khi kết luận label-noise level.
7. Không trình bày một loại lỗi annotation là lỗi đã biết của 95-Cloud nếu chưa có citation
   hoặc kết quả audit trực tiếp.

### Augmentation audit

1. Version hóa augmentation policy và lưu policy/version trong checkpoint bundle.
2. Giữ flip/rotation 90 độ làm geometric baseline, sau đó kiểm chứng label invariance.
3. Chỉ thử band-wise gain, additive noise, blur hoặc resampling khi parameter range được suy
   ra từ sensor noise, radiometry, PSF/GSD hoặc dữ liệu train; không dùng range tùy ý.
4. Ghi rõ augmentation được áp dụng trước hay sau calibration/normalization.
5. Chạy ablation augmentation trên cùng split, seed set và input contract.
6. Không yêu cầu augmentation mô phỏng phân phối ImageNet; ImageNet weights chỉ là
   initialization, còn augmentation phải phù hợp product đích.

### Tiêu chí lựa chọn

- PR-AUC, false-clear và downlink utility trên validation scene để chọn cấu hình.
- Final test scene chỉ dùng một lần để báo cáo cấu hình đã khóa.
- Brier/ECE và độ ổn định của threshold.
- Độ lệch metric giữa các seed.
- Không suy giảm không chấp nhận được trên các cảnh khó.
- Latency và memory vẫn phù hợp với Jetson Nano.

### Artifact đầu ra

- `release_model.pt` chứa checkpoint bundle đầy đủ.
- ONNX model và metadata.
- TensorRT build specification và metadata sidecar.
- Báo cáo ablation, threshold calibration và final test metrics.
- Label-quality/augmentation audit report.
- Dataset/split manifest, normalization statistics và toàn bộ lineage IDs.

## 9. P6 - Cross-satellite data, training và validation

**Mức ưu tiên:** Sau khi P0-P5 hoàn thành  
**Ước lượng:** Chốt sau annotation pilot và statistical power analysis

### P6a - Target definition, data curation và annotation

1. Chốt sensor, product, processing level và downlink decision target cần hỗ trợ.
2. Viết annotation guideline, quality-control process và adjudication rule trước khi gán nhãn.
3. Chạy annotation pilot để đo thời gian, disagreement, prevalence và within-scene correlation.
4. Tính số scene/sample cần thiết từ target false-clear/false-cloud, confidence interval,
   prevalence, design effect và số strata. Không dùng một scene count cứng không có power analysis.
5. Tạo train/validation/test set có nhãn riêng cho Sentinel-2 và từng PlanetScope product.
6. Chia holdout theo scene, vùng địa lý, mùa và thời điểm để hạn chế domain leakage.
7. Bổ sung các nhóm khó: mây mỏng, haze, cirrus, tuyết/băng, cát sáng, đô thị sáng,
   sun glint, biển tối, bóng mây và NoData.
8. Khóa test set và annotation version trước khi bắt đầu P6b.

### P6b - Baseline, model ablation và calibration

1. Chạy checkpoint Landsat RGB làm baseline trước harmonization hoặc fine-tune.
2. So sánh ba phương án trên cùng data split và evaluation contract:
   - Một model chung cho mọi sensor/product.
   - Shared backbone với sensor-specific normalization hoặc head.
   - Model riêng cho từng sensor/product.
3. Đánh giá negative transfer, data efficiency, storage, engine switching latency và peak RAM;
   không chốt kiến trúc chỉ từ giả định về Jetson RAM.
4. Chỉ dùng pseudo-label/domain adaptation sau khi có validation set có nhãn.
5. Calibrate probability threshold riêng theo sensor/product khi có đủ validation data.
6. Benchmark accuracy, latency, peak RAM, storage, switching latency và điện năng trên
   Jetson Nano.

### Input Validation and Domain Guardrails

Không gọi mọi kiểm tra đầu vào là semantic OOD. Triển khai theo tầng và action rõ ràng:

| Tầng | Kiểm tra | Action |
|---|---|---|
| Contract | Sensor, product, band identity/order, units, processing level | Reject |
| Validity | NoData ratio, footprint, saturation, physical bounds | `invalid_input` |
| Distribution | Robust quantile/MAD và multiband consistency đã calibrate | Warning/review |
| Semantic OOD | Learned/embedding method đã validation, nếu thực sự cần | Policy riêng |

Distribution guardrail phải được calibrate trên in-domain holdout để kiểm soát false rejection.
Không mặc định dùng `[mean +/- k * std]` như một production-ready OOD detector.

### Tiêu chí nghiệm thu

- Mỗi sensor/product production có adapter, validation set và threshold riêng.
- Model đạt giới hạn false-clear/false-cloud do nghiệp vụ downlink quy định.
- Kết quả được báo cáo theo scene độc lập, không chỉ theo crop.
- Kiến trúc single/shared/per-sensor được chọn bằng ablation trên sensor đích.
- Contract violation bị reject; invalid input và distribution warning không bị nhập nhằng
  với model prediction.
- Semantic OOD chỉ được dùng làm gate sau khi có validation và action policy riêng.

## 10. P7 - Post-deployment monitoring và feedback

**Mức ưu tiên:** Bắt buộc trước khi vận hành production dài hạn  
**Ước lượng:** 3-5 ngày cho telemetry ban đầu; feedback/labeling là hoạt động liên tục

### Công việc

1. Log theo sensor/product và model version:
   - Input contract violation rate.
   - Invalid-input và distribution-warning rate.
   - Robust band statistics, valid fraction và prediction-score distribution.
   - Tỷ lệ quyết định giữ/loại và predicted cloud coverage.
2. Thu thập telemetry thiết bị: latency, throughput, peak RAM, storage, temperature,
   throttling và inference errors.
3. Đặt alert threshold từ production baseline và operational SLO; không dùng threshold tùy ý.
4. Lưu `experiment_id`, engine fingerprint, `input_spec_id` và `DecisionSpec` version với mỗi
   inference batch để truy vết.
5. Thiết kế sampling/feedback loop để thu ground truth định kỳ, ưu tiên distribution warning,
   semantic OOD, score gần threshold và các strata thiếu dữ liệu.
6. Chỉ kết luận performance drift khi có ground truth/reference phù hợp. Không đồng nhất
   input/prediction drift với accuracy degradation.
7. Định nghĩa trigger cho recalibration, retraining, rollback hoặc revoke artifact.

### Tiêu chí nghiệm thu

- Monitoring phân biệt contract, validity, distribution, prediction và device telemetry.
- Alert được kiểm tra bằng dữ liệu replay hoặc fault injection trước production.
- Có thể truy từ một inference result về engine, checkpoint, input contract và dataset lineage.
- Có runbook cho warning spike, invalid spike, latency regression và performance regression.

## 11. Kế hoạch kiểm thử

### Unit tests

- Serialize/deserialize và schema migration của `InputSpec`/`DecisionSpec`.
- Checkpoint RGB, legacy checkpoint và channel mismatch.
- Deterministic generation và parent linkage của `processed_dataset_id`, `split_id`,
  `input_spec_id` và `experiment_id`.
- Band mapping, band order và xử lý RGBA.
- Scale/offset bằng các mảng có kết quả biết trước.
- NoData, validity mask, saturation và invalid-patch policy.
- Fixed normalization và kiểm tra không dùng statistics từ validation/test.
- Deterministic tiling, edge coverage và scene aggregation.
- Metric, ECE, Brier score và probability-threshold sweep trên dữ liệu tổng hợp.
- Cloud-ratio threshold dùng nhất quán cho train/validation/test trong một experiment.
- Augmentation policy được version hóa, serialize và lưu cùng checkpoint.
- Contract violation, invalid input và distribution warning tạo đúng status/action.
- Resume optimizer/scheduler/scaler state.

### Integration tests

- `preprocess -> split -> fit stats -> train 1 epoch -> resume -> eval -> export`.
- Duyệt toàn bộ train/val/test bằng DataLoader và kiểm tra tensor shape/dtype.
- Xác minh image, cloud mask và validity mask cùng scene/split.
- Notebook/CLI parity với cùng sample, crop coordinates và seed.
- PyTorch/ONNX output parity.
- TensorRT metadata/binding validation trước inference.
- TensorRT sidecar fingerprint và target-device smoke test.
- Cloud coverage chỉ sử dụng valid area làm mẫu số.
- Rebuild cùng raw/config tạo cùng processed ID; đổi split chỉ làm đổi split ID.
- Invalid input không được chuyển tiếp vào model như một patch clear hợp lệ.
- Monitoring event schema truy được về engine, checkpoint và dataset lineage.

### Reproducibility tests

- Hai smoke run cùng seed tạo cùng split, crop sequence và metric trong tolerance.
- Thay seed phải được phản ánh trong provenance và tạo sequence khác.
- Test set không bị thay đổi khi train configuration hoặc probability-threshold sweep thay đổi.

## 12. Thứ tự PR đề xuất

1. **PR-1:** Chọn `src` làm pipeline chuẩn, chuyển notebook thành orchestrator.
2. **PR-2:** `InputSpec`, checkpoint bundle, RGB defaults và legacy migration.
3. **PR-3:** Product-aware preprocessor, validity mask, lineage IDs và data lifecycle.
4. **PR-4:** Deterministic tiling, scene metrics và threshold calibration.
5. **PR-5:** Determinism, ONNX metadata, TensorRT fingerprint và parity tests.
6. **PR-6:** Label/augmentation audit, RGB retraining, ablation và release artifact.
7. **PR-7:** P6a target definition, annotation pilot, power analysis và data curation.
8. **PR-8:** P6b single/shared/per-sensor ablation và sensor-specific calibration.
9. **PR-9:** Input validation/domain guardrails và target-device benchmark.
10. **PR-10:** Monitoring telemetry, feedback sampling và operational runbooks.

Mỗi PR phải có test riêng và không được phụ thuộc vào artifact thủ công không có manifest.

## 13. Timeline và dependency

| Giai đoạn | Thời lượng ước tính | Phụ thuộc |
|---|---:|---|
| P0 - Pipeline chuẩn và baseline | 1-2 ngày | Môi trường test |
| P1 - RGB và checkpoint contract | 2-3 ngày | P0 |
| P2 - Preprocessing, lineage và lifecycle | 5-8 ngày | P0, xác định product metadata |
| P3 - Scene evaluation và metric | 3-4 ngày | P1, P2 |
| P4 - Determinism và export | 2-3 ngày | P1, P2 |
| P5 - Audit, retrain và ablation | 5-8 ngày công/GPU | P1-P4, dữ liệu đã rebuild |
| P6a - Data curation/annotation | Chốt sau pilot và power analysis | P5, target definition |
| P6b - Model ablation/calibration | 1-3 tuần thực nghiệm | P6a hoàn thành |
| P7 - Monitoring ban đầu | 3-5 ngày, sau đó liên tục | Release candidate từ P6b |

Tổng effort kỹ thuật cho P0-P5 khoảng 18-28 ngày công, chưa tính thời gian thu thập
nhãn, chờ GPU và benchmark trên phần cứng đích.

## 14. Production gates

Pipeline chỉ được coi là production-ready khi tất cả điều kiện sau được đáp ứng:

- Notebook và CLI sử dụng cùng implementation và cùng label contract.
- Mọi artifact tự nhận diện và sử dụng đúng RGB ba kênh.
- Checkpoint tự mang `InputSpec`, `DecisionSpec`, training state và provenance.
- Dataset, split, input spec và experiment có ID riêng cùng lineage có thể tái tạo.
- Có retention, rollback/rebuild và artifact-revocation policy.
- Không có đường production nào chuẩn hóa theo dtype hoặc `max()`.
- Train, eval và inference sử dụng cùng preprocessor có version.
- NoData/invalid pixels không bị tính thành clear hoặc cloud coverage denominator.
- Validation/test phủ scene bằng deterministic tiling.
- Cloud-ratio threshold xuất phát từ task/downlink policy và dùng nhất quán giữa các split.
- Probability threshold được fit trên validation theo mục tiêu downlink và khóa trước test.
- Label-quality và augmentation policy đã được audit, version hóa và lưu cùng artifact.
- Mỗi sensor/product được hỗ trợ có test scene, adapter và probability threshold độc lập.
- Single/shared/per-sensor architecture được chọn bằng ablation thay vì giả định.
- Contract violation, invalid input, distribution warning và semantic OOD có status/action riêng.
- PyTorch, ONNX và TensorRT đạt parity trong tolerance.
- TensorRT engine có full build fingerprint và đã smoke test trên target device.
- Model đạt giới hạn false-clear, false-cloud, latency, RAM và điện năng đã được dự án chốt.
- Monitoring, feedback sampling và operational runbook sẵn sàng trước vận hành dài hạn.
