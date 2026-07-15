# Phản biện nhận xét chuyên gia về kế hoạch khắc phục Training Pipeline RGB

> Tài liệu được phản biện: [`expert_review_remediation_plan.md`](./expert_review_remediation_plan.md)  
> Kế hoạch gốc: [`training_pipeline_remediation_plan.md`](./training_pipeline_remediation_plan.md)  
> Ngày phản biện: 2026-07-14

## 1. Kết luận tổng quát

Bản nhận xét chuyên gia nhìn chung hữu ích và đồng thuận với các hướng kiến trúc chính của
kế hoạch. Tuy nhiên, một số nhận định đang được trình bày quá tuyệt đối, thiếu bằng chứng
định lượng hoặc chưa chính xác về mặt kỹ thuật.

Các điểm cần điều chỉnh quan trọng nhất liên quan đến:

- Phương pháp OOD dựa trên `mean +/- k * std`.
- Lựa chọn single-model hay sensor-specific model.
- Nguồn gốc và label noise được cho là đã biết của 95-Cloud.
- Việc sử dụng cloud-ratio threshold khác nhau giữa train và eval.
- Mức độ data versioning đã có trong kế hoạch.
- Vai trò của augmentation đối với channel dropout và ImageNet pretrained weights.
- Điều kiện tương thích của TensorRT engine.
- Các con số định lượng không có nguồn hoặc phương pháp tính.

## 2. Các điểm cần phản biện

### 2.1. Cao: OOD bằng `[mu +/- k * sigma]` không đủ an toàn

Bản review đề xuất bắt đầu bằng rule-based OOD detection, cụ thể là coi band statistics
nằm ngoài `[mu +/- k * sigma]` của tập train là OOD, đồng thời nhận định cách này đủ cho
edge deployment trên Jetson.

Nhận định này chưa đủ chặt chẽ vì:

- Reflectance theo band thường không có phân phối Gaussian.
- Phân phối có thể đa mode và phụ thuộc mạnh vào loại bề mặt, mùa, góc mặt trời và tỷ lệ mây.
- Scene tuyết, băng, sa mạc hoặc mây dày hợp lệ có thể nằm ngoài khoảng này.
- Lỗi semantic như đảo band hoặc sai processing baseline vẫn có thể tạo mean/std nằm trong
  khoảng cho phép.
- Kiểm tra từng band độc lập không phát hiện đầy đủ shift trong tương quan giữa các band.

`[mu +/- k * sigma]` chỉ phù hợp như một gross sanity check, không nên được gọi là OOD
detector đủ dùng cho production.

Phương án phù hợp hơn cho baseline edge deployment:

1. Kiểm tra cứng metadata và `InputSpec`: sensor, product, processing level, band identity,
   units, scale/offset và GSD.
2. Kiểm tra NoData/invalid ratio và các giới hạn vật lý bắt buộc.
3. Dùng robust quantiles hoặc median/MAD thay cho giả định Gaussian đơn giản.
4. Hiệu chỉnh OOD threshold trên một in-domain holdout set để kiểm soát false rejection.
5. Chỉ thêm embedding-distance hoặc learned OOD khi rule-based validation không đủ.

OOD flag cũng phải có action rõ ràng:

- Contract mismatch: từ chối inference.
- Invalid ratio vượt ngưỡng: trả trạng thái `invalid_input`.
- Statistical warning nhẹ: gắn cờ và không tự động coi kết quả là đáng tin cậy.

### 2.2. Cao: Bảng single-model/per-sensor dùng các kết luận tuyệt đối

Bản review mô tả:

- Single model luôn phải compromise về performance.
- Per-sensor model luôn được tối ưu tốt hơn.
- Per-sensor model dẫn đến nhiều pipeline độc lập.
- Single model luôn phù hợp hơn với RAM giới hạn của Jetson Nano.

Các kết luận này không phải quy luật chung.

Multi-sensor training có thể cải thiện regularization và tổng quát hóa khi các domain chia sẻ
đặc trưng. Ngược lại, per-sensor model có thể overfit nếu mỗi sensor chỉ có ít dữ liệu. Negative
transfer của single model cũng chỉ có thể được xác định bằng thực nghiệm.

Nhiều checkpoint không đồng nghĩa với nhiều pipeline. Các model sensor-specific vẫn có thể
dùng chung:

- Product adapter framework.
- `InputSpec`/`DecisionSpec` schema.
- Training, evaluation và export code.
- Artifact registry và deployment process.

Jetson cũng không nhất thiết phải giữ mọi TensorRT engine trong RAM cùng lúc. Hệ thống có
thể route theo product metadata rồi chỉ load engine tương ứng. Khi đó, chi phí chính là storage
và model switching latency, không phải tổng RAM của tất cả model.

Vì vậy, quyết định trong kế hoạch gốc là hợp lý: hoàn thành pipeline Landsat RGB, thu thập dữ
liệu sensor đích, rồi chạy ablation giữa:

1. Một model chung.
2. Một backbone chung với sensor-specific normalization/head.
3. Model riêng theo sensor/product.

Không nên chốt single model sớm chỉ dựa trên giả định RAM.

### 2.3. Cao: Nguồn và label noise của 95-Cloud chưa được chứng minh

Bản review gọi 95-Cloud là dataset dựa trên Landsat 8 Biome và khẳng định dataset có các vấn
đề đã biết như thin cirrus annotation không nhất quán. Tài liệu không cung cấp nguồn cho các
khẳng định này.

[Repository chính thức của 95-Cloud](https://github.com/SorourMo/95-Cloud-An-Extension-to-38-Cloud-Dataset)
mô tả đây là phần mở rộng của 38-Cloud, sử dụng các scene Landsat 8 Collection 1 Level-1,
và quy định thin cloud/haze là cloud.

[Repository chính thức của 38-Cloud](https://github.com/SorourMo/38-Cloud-A-Cloud-Segmentation-Dataset)
cho biết ground truth được tạo thủ công. Điều này cho thấy annotation noise có thể tồn tại,
nhưng không tự chứng minh các lỗi cụ thể được liệt kê trong bản review.

Cách diễn đạt chính xác hơn:

- Label noise nội tại là một rủi ro cần đo bằng audit.
- Không khẳng định một loại annotation error là lỗi đã biết nếu chưa có nguồn hoặc kết quả audit.
- Lấy mẫu các crop gần decision boundary, thin cloud, haze và vùng biên để review thủ công.
- Báo cáo disagreement rate giữa annotator hoặc giữa ground truth với một QA/reference độc lập.

Đối với patch classifier, lỗi một số pixel ở biên chỉ làm đổi nhãn patch khi cloud ratio nằm
gần threshold. Vì vậy cần đo tỷ lệ crop nằm trong một uncertainty band quanh threshold, thay
vì mặc định xem mọi boundary error là nghiêm trọng như trong bài toán pixel segmentation.

### 2.4. Cao: Không nên dùng cloud-ratio threshold khác nhau giữa train và eval

Bản review dùng công thức `cloud_ratio > T` và đặt vấn đề liệu `T` có nên khác nhau giữa
train và eval.

Pipeline hiện tại định nghĩa rõ:

```text
cloud_ratio_threshold = 0.10
label = 1.0 if cloud_ratio >= 0.10 else 0.0
```

Định nghĩa này được ghi tại [`noisy_label_fix_plan.md`](./noisy_label_fix_plan.md) và được
triển khai trong `src/data/cloud_dataset.py`.

Ablation nhiều giá trị `T` là hợp lý, nhưng mỗi experiment phải dùng cùng một label contract
cho train, validation và test. Nếu train và eval dùng `T` khác nhau, model sẽ học một target
nhưng được đánh giá bằng target khác; metric khi đó không còn đo đúng khả năng thực hiện
nhiệm vụ đã train.

Quy trình đúng:

1. Chọn một tập candidate threshold trên development data.
2. Với mỗi candidate, tạo label nhất quán cho train/validation/test theo cùng rule.
3. Train và chọn cấu hình chỉ bằng train/validation.
4. Khóa label threshold và probability threshold trước khi chạy final test.

Cloud-ratio threshold và probability threshold phải tiếp tục là hai tham số độc lập.

### 2.5. Trung bình: Nhận định "không có data versioning strategy" là quá mức

Kế hoạch gốc đã có các thành phần data provenance cơ bản:

- Lưu dataset/split manifest hash trong checkpoint bundle.
- Lưu dataset manifest và normalization statistics trong release artifacts.
- Yêu cầu mỗi PR không phụ thuộc vào artifact thủ công không có manifest.
- Lưu preprocessor version và runtime provenance.

Do đó, nói rằng kế hoạch không có data versioning là không chính xác. Cách đánh giá phù hợp
hơn là: kế hoạch đã có traceability cơ bản nhưng chưa mô tả đầy đủ lifecycle management.

Các phần nên bổ sung gồm:

- Dataset ID có tính content-addressed.
- Artifact storage/registry.
- Retention policy cho raw, intermediate và release dataset.
- Quy trình rollback/rebuild khi phát hiện lỗi preprocessor.
- Quan hệ cha-con giữa dataset version, split version, `InputSpec` và checkpoint.

Dataset ID nên được tạo từ:

```text
raw_manifest_hash
+ preprocessor_version
+ preprocessing_parameters
+ split_manifest_hash
+ normalization_statistics_hash
```

DVC có thể được sử dụng, nhưng không phải điều kiện bắt buộc. DVC cũng không thay thế cho
manifest schema, deterministic rebuild và validation gates.

### 2.6. Trung bình: Augmentation không phải thứ thay thế channel dropout

Channel dropout và augmentation giải quyết các mục tiêu khác nhau:

- Channel dropout mô phỏng tình huống thiếu kênh thứ tư trong model RGB+NIR.
- Geometric/radiometric augmentation tạo invariance và regularization cho dữ liệu train.

Vì vậy, vô hiệu hóa channel dropout cho model RGB không tạo ra yêu cầu phải thêm một
augmentation khác để thay thế.

Code hiện tại cũng đã có horizontal flip, vertical flip và rotation 90 độ. Điểm còn thiếu
trong kế hoạch là audit và version hóa augmentation policy, không phải hoàn toàn không có
augmentation.

Nhận định augmentation phải consistent với distribution mà ImageNet backbone đã học cũng
chưa chính xác. Khi fine-tune, augmentation phải:

- Phù hợp với vật lý sensor và product đích.
- Bảo toàn label contract của bài toán cloud classification.
- Không tạo spectrum hoặc brightness không thể xuất hiện trong product.
- Được áp dụng sau hoặc trước normalization đúng theo định nghĩa của phép biến đổi.

ImageNet pretraining là initialization, không phải input distribution mà fine-tuning bắt buộc
phải duy trì.

Augmentation audit vẫn nên được thêm vào P5, gồm geometric transforms, band-wise gain/noise,
blur/resampling và các mức độ đã được hiệu chỉnh theo product metadata.

### 2.7. Trung bình: Physical units không bắt buộc cho mọi cross-satellite pipeline

Product-aware contract và train/inference parity là bắt buộc. Chuyển analytic products về
physical reflectance là baseline phù hợp nhất để giảm sai lệch radiometric giữa sensor.

Tuy nhiên, physical units không phải điều kiện bắt buộc cho mọi dạng RGB product. Một số
rendered RGB product đã trải qua gamma correction, tone mapping hoặc nén và không thể khôi
phục reflectance vật lý chính xác. Model vẫn có thể hoạt động cross-satellite nếu:

- Product representation được định nghĩa rõ.
- Train và inference dùng cùng representation.
- Không trộn analytic RGB với rendered RGB ngoài ý muốn.
- Model được train và validation trên đúng sensor/product contract.

Cách diễn đạt phù hợp hơn:

> Physical reflectance là representation ưu tiên cho analytic products, còn versioned,
> product-aware input contract mới là điều kiện bắt buộc chung.

### 2.8. Trung bình: TensorRT compatibility không chỉ phụ thuộc JetPack version

Bản review nêu rằng nếu build engine trên host thì JetPack version phải match. Đây là một
heuristic chưa đầy đủ.

Khả năng load TensorRT engine còn phụ thuộc vào:

- CPU/platform architecture.
- TensorRT và CUDA version.
- GPU compute capability và hardware-compatibility mode.
- OS/driver ABI.
- Plugin libraries và plugin version.
- Precision mode, calibration cache và builder flags.
- Version-compatibility/cross-platform flags nếu được hỗ trợ.

Exact JetPack match không tự bảo đảm engine tương thích; ngược lại, một số compatibility mode
có thể cho phép phạm vi tương thích rộng hơn. Tham khảo
[NVIDIA TensorRT Engine Compatibility](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/advanced.html#engine-compatibility).

Với Jetson Nano, phương án an toàn là:

1. Lưu ONNX làm artifact portable chính.
2. Build engine trên thiết bị đích hoặc môi trường target-equivalent.
3. Lưu đầy đủ build fingerprint trong sidecar.
4. Chạy smoke inference và numerical parity test trên chính thiết bị đích.

Kế hoạch gốc đã yêu cầu lưu ONNX model, TensorRT build specification và metadata sidecar,
nên việc lưu ONNX không phải nội dung đang bị thiếu.

### 2.9. Thấp: Các con số định lượng không có nguồn hoặc cách tính

Bản review sử dụng các con số:

- Khoảng 40% lỗi production inference đến từ preprocessing mismatch.
- Annotation mất 2-4 tuần cho mỗi sensor.
- Cần tối thiểu 30-50 scene cho mỗi sensor.
- Pipeline sau P0-P5 đáng tin cậy hơn 90% các pipeline remote sensing khác.

Các con số này không có citation, định nghĩa population hoặc phương pháp tính. Chúng không
nên được sử dụng làm engineering requirement hay căn cứ timeline.

Số lượng scene cần thiết phải xuất phát từ:

- Target false-clear/false-cloud rate.
- Desired confidence interval.
- Cloud prevalence và số positive/negative decisions.
- Correlation giữa các patch trong cùng scene.
- Số strata cần bao phủ theo sensor, product, địa lý, mùa và nhóm cảnh khó.
- Granularity của annotation: scene-level, patch-level hay pixel-level.

`30-50 scene` có thể đủ cho pilot nhưng vẫn thiếu để chứng minh một false-clear rate rất thấp.
Timeline cũng thay đổi lớn giữa label patch classification và vẽ cloud mask pixel-level.

## 3. Các góp ý nên giữ và bổ sung vào kế hoạch

Các nội dung sau trong bản review là hợp lý:

### 3.1. Hoàn thiện data lifecycle

Bổ sung artifact registry, dataset ID, retention và rollback/rebuild policy ngoài manifest/hash
đã có.

### 3.2. Label-quality audit

Thêm audit cho các crop gần cloud-ratio threshold, thin cloud, haze và vùng biên. Kết quả audit
phải được đo và lưu, không chỉ dựa trên nhận định định tính về dataset.

### 3.3. Augmentation audit

Version hóa augmentation policy và chạy ablation với các phép biến đổi phù hợp vật lý sensor.

### 3.4. Tách P6 theo dependency dữ liệu

Tách thành:

- P6a: target definition, data sourcing, annotation guideline, curation và quality control.
- P6b: baseline, fine-tune/multi-sensor ablation, calibration và final evaluation.

Không đặt một scene count cứng trước khi xác định target error rate và statistical power.

### 3.5. Post-deployment monitoring

Monitoring nên được thêm như một phase sau production release, nhưng chỉ theo dõi prediction
distribution là chưa đủ. Cần tối thiểu:

- Input contract violation rate.
- Invalid/OOD rate.
- Band statistics và score distribution theo sensor/product.
- Tỷ lệ quyết định giữ/loại.
- Telemetry về latency, RAM, nhiệt độ và lỗi inference.
- Cơ chế lấy mẫu dữ liệu và thu ground truth để đo performance drift thực sự.

Không có ground truth feedback thì chỉ có thể phát hiện data/prediction drift, không thể kết
luận model accuracy đã suy giảm.

### 3.6. TensorRT build provenance

Sidecar cần lưu target device, OS/L4T, JetPack, TensorRT, CUDA, GPU capability, plugin hashes,
builder flags, precision và ONNX checksum.

## 4. Kết luận

Bản nhận xét chuyên gia đúng về hướng tổng thể và đưa ra một số bổ sung hữu ích. Tuy nhiên,
các nội dung sau không nên đưa nguyên trạng vào remediation plan:

- Xem `[mu +/- k * sigma]` là OOD detector đủ dùng cho production.
- Chốt single model vì Jetson RAM trước khi có ablation sensor đích.
- Khẳng định các lỗi annotation cụ thể của 95-Cloud khi chưa có nguồn/audit.
- Dùng cloud-ratio threshold khác nhau giữa train và eval.
- Khẳng định kế hoạch hoàn toàn không có data versioning.
- Xem augmentation là phần thay thế channel dropout hoặc phải bám distribution ImageNet.
- Đồng nhất TensorRT compatibility với việc JetPack version match.
- Dùng các con số 40%, 90%, 2-4 tuần hoặc 30-50 scene như fact không cần kiểm chứng.

Phiên bản điều chỉnh nên giữ các đề xuất về data lifecycle, label audit, augmentation audit,
P6 data curation và monitoring, nhưng phải chuyển chúng thành yêu cầu có metric, provenance
và tiêu chí nghiệm thu rõ ràng.
