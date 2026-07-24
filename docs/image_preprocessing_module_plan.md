# Kế hoạch module nắn ảnh độc lập dataset và model

## 1. Mục tiêu

Tạo một boundary nắn ảnh hình học có thể dùng lại cho bất kỳ nguồn ảnh, dataset
hoặc model nào:

~~~text
Source image + CalibrationBundle + PreprocessingProfile
  -> geometric transform
  -> model-grid image + validity mask + validity-reason mask + mapping
~~~

"model-grid" là tên lưới đích do profile xác định, không phải giả định về một
model, dataset, số kênh, patch size, tensor layout hay normalization cụ thể.
Core preprocessing chỉ xử lý hình học, metadata không gian và tính hợp lệ pixel.
Nó không thực hiện radiometric enhancement, color correction, tone mapping,
denoise, normalization, chọn patch, chuyển HWC sang NCHW hoặc gọi TensorRT.

Target grid, output layout, output dtype, kernel, rounding, border và validity
phải đến từ "PreprocessingProfile". Warp tính nội bộ bằng "float32" hoặc
"float64" theo numeric contract của profile/backend, rồi mới cast sang
"output_dtype" theo rounding, clipping và non-finite policy đã version hóa.
Không có quy tắc core nào buộc raster phải là một dtype cụ thể hoặc tự đổi dtype
theo thống kê ảnh.

Normalization, sắp xếp band cho tensor, padding theo cửa sổ model và chuyển HWC
sang NCHW là trách nhiệm duy nhất của "InferenceAdapter", sau khi artifact nắn
ảnh đã được xác thực.

## 2. Hiện trạng và ràng buộc

- "process_large_image()" hiện nhận path ảnh, khởi tạo TensorRT, đọc strip,
  normalize, pad và transpose trong cùng một hàm. Luồng này phải được tách thành
  preprocessing và inference adapter.
- "ImageBlockReader.read_rows()" hiện chỉ trả ndarray, chưa trả validity hoặc
  validity-reason mask. Không đổi ngầm semantics của API này.
- "TiffReader" đã kiểm tra axes, shape, dtype, band order, fingerprint và
  "input_spec_id", nhưng chưa hiểu preprocessing mapping.
- "_MaskCache" hiện tạo dense mask; chưa có patch-result artifact có recovery.
- "NormalizationSpec" và "EngineInputSpec" là contract của model/engine; chúng
  không được trở thành input bắt buộc của "WarpBackend".
- Repository hiện chưa có package "src/preprocessing" hoặc packaging metadata.
  Plan phải tạo một public import path ổn định để reference implementation gọi
  module, không chỉ mô tả các class nội bộ.
- TIFF nén có thể kích hoạt decode/cache toàn ảnh. Flight profile phải từ chối
  path đó nếu hai tầng resource admission không chứng minh đủ RAM và disk.

Các điểm tham chiếu:

- "src/inference_large_image_trt.py"
- "src/tiff_reader.py"
- "src/input_contract.py"
- "src/resource_guards.py"
- "docs/georeferencing_architecture.md", mục 5-9

## 3. Boundary kiến trúc

~~~text
RawCaptureReader
  -> ContractResolver / ArtifactTrust
  -> ValidityBuilder
  -> TransformPlanner
  -> WarpBackend
  -> BlockProcessor
  -> ArtifactWriter
  -> PreprocessedArtifactReader
  -> InferenceAdapter (tùy chọn, ngoài core)
~~~

| Thành phần | Trách nhiệm | Không được làm |
|---|---|---|
| "RawCaptureReader" | Đọc sample nguồn và metadata đã khai báo | Warp, normalize, chọn band cho model, gọi TensorRT |
| "ContractResolver" | Xác thực bundle, resolve profile/calibration/compute profile | Đoán profile từ dtype hoặc thống kê ảnh |
| "ValidityBuilder" | Tạo validity và reason mask từ NoData, thiếu kênh, biên, mapping và support | Biến invalid thành pixel hợp lệ hoặc class suy diễn |
| "TransformPlanner" | Tính target grid, source ROI, halo, transform và mapping | Thay đổi radiometry hoặc chọn kernel mặc định |
| "WarpBackend" | Áp dụng distortion, co-registration và resampling theo plan | Normalize, chọn patch size, chọn tensor layout hoặc tự chọn profile |
| "ArtifactWriter" | Ghi output, masks, mapping, checksum và manifest nguyên tử | Publish artifact dở dang |
| "InferenceAdapter" | Kiểm tra tương thích model, chọn/reorder band, patch, normalization, HWC-to-NCHW và TensorRT | Đọc raw source hoặc tự warp |
| "ReferencePipeline" | Gọi public preprocessing facade rồi chuyển artifact sang reference consumer | Import private backend hoặc tự lặp lại read/warp/normalize |
| "ReferencePreprocessedInference" | Chỉ nhận "PreprocessArtifact" đã xác thực và chạy adapter | Nhận raw image/path/ndarray, mở ảnh phụ hoặc gọi source reader |
| "DecisionPolicy/OBC" | Quyết định giữ/xóa và thực thi xóa | Cho Jetson quyền xóa source |

Core có thể chạy và publish "PreprocessArtifact" mà không có engine. Một pipeline
có inference chỉ nối artifact đó với "InferenceAdapter" sau một compatibility
gate riêng.

## 4. Hai contract độc lập

### 4.1. PreprocessingProfile

"PreprocessingProfile" là contract duy nhất điều khiển biến đổi hình học. Nó
phải bất biến, có ID/version/digest và quy định ít nhất:

- source representation được chấp nhận, schema kênh và NoData semantics;
- calibration selector/compatibility, pixel convention, transform direction,
  loại transform và giới hạn chất lượng calibration;
- target grid: extent, resolution/GSD, origin, axes, pixel convention và
  spatial semantics;
- source ROI, halo và border policy;
- kernel resampling ảnh, policy resampling validity/reason, ngưỡng support;
- internal numeric precision, output layout, output dtype, rounding,
  clipping/cast và non-finite policy;
- encoding của validity mask và validity-reason mask, gồm enum/bit layout;
- implementation version và tolerance CPU/GPU.

"CalibrationBundle" là artifact tin cậy riêng chứa tham số hoặc LUT cụ thể.
Profile chỉ xác định calibration nào được chấp nhận và cách dùng nó; không nhúng
hay suy diễn calibration từ ảnh.

"output_layout" là layout raster, ví dụ YXC hoặc CYX, không phải tensor layout
của engine. Profile không chứa normalization, patch size, TensorRT input dtype
hay band order mà một model yêu cầu.

### 4.2. ModelCompatibilityProfile / EngineInputSpec

"ModelCompatibilityProfile" là contract của model. Trong code hiện có, nó có thể
được biểu diễn bởi "EngineInputSpec" gắn với engine manifest. Contract này quy
định:

- engine/model fingerprint và target runtime tương thích;
- required band order và channel count;
- input tensor layout, dtype, shape/patch size, batch và padding policy;
- normalization, clip hoặc standardization;
- các điều kiện train-inference parity.

Adapter chỉ chấp nhận artifact khi output schema của "PreprocessingProfile" có
thể đáp ứng "EngineInputSpec": các band yêu cầu có mặt, semantics/grid/profile
được chấp nhận và conversion layout/dtype là tường minh. Không được dùng
"EngineInputSpec" để sửa target grid, kernel, rounding, mapping hay validity
policy của artifact đã nắn.

Ví dụ cấu hình cho deployment hiện tại, nhưng không phải giới hạn của thuật
toán:

~~~text
PreprocessingProfile.output_dtype = uint16
EngineInputSpec.normalization = dtype-range
EngineInputSpec.patch_size = 256
~~~

Các giá trị trên nằm ở hai contract khác nhau. Một profile nắn ảnh khác có thể
chọn dtype/layout/kernel/grid khác, và một engine khác có thể chọn normalization
hoặc patch size khác mà không thay đổi "WarpBackend".

### 4.3. Processing bundle và trust gate

Tạo bundle bất biến, liên kết bằng SHA-256 và sau này có chữ ký, issuer/key ID
và generation:

- "CaptureManifest": capture ID, completion marker, source fingerprint,
  sensor/product, dimensions, source layout/dtype, channel schema và NoData
  semantics.
- "PreprocessingProfile": contract ở mục 4.1.
- "CalibrationBundle": camera serial/version, distortion hoặc LUT,
  co-registration, frame/pixel convention, transform direction và digest.
- "ComputeProfile": backend, max in-flight strips, queue/batch limit, RAM/disk
  budgets, temporary-storage policy và thermal policy.
- "EngineInputSpec" cùng engine manifest: chỉ bắt buộc khi caller yêu cầu
  inference.

Preprocessing-only run xác thực bốn artifact đầu. Inference pipeline xác thực
thêm engine/model contract tại adapter boundary. Thiếu, hết hạn, sai fingerprint,
sai generation hoặc sai issuer của bất cứ artifact bắt buộc nào phải bị từ chối
trước khi đọc source, nạp engine hoặc publish artifact.

### 4.4. Trạng thái run

~~~text
NEW
  -> VALIDATING
  -> ADMITTED
  -> PROCESSING
  -> VERIFYING
  -> COMPLETE
~~~

Các trạng thái terminal lỗi gồm "INVALID_INPUT", "UNTRUSTED_ARTIFACT",
"RESOURCE_REJECTED", "CALIBRATION_ERROR", "IO_FAULT" và "RUNTIME_FAULT".
Mọi trạng thái lỗi phải có reason code và dẫn đến "RETAIN_FOR_GROUND".
"COMPLETE" chỉ có thể xảy ra sau khi manifest đã được publish nguyên tử và
checksum đã xác minh.

### 4.5. Public module API bắt buộc

Module phải được import bằng một facade ổn định:

~~~python
from preprocessing import (
    ArtifactOpenRequest,
    CalibrationBundle,
    CaptureManifest,
    ComputeProfile,
    PreprocessArtifact,
    PreprocessFailure,
    PreprocessRequest,
    PreprocessingProfile,
    open_preprocessed_artifact,
    preprocess_capture,
)
~~~

"src/preprocessing/__init__.py" là public boundary và khai báo "__all__". Caller,
bao gồm reference implementation, không được import trực tiếp "warp_backend",
"artifact_writer" hoặc module private khác. Thay đổi public symbol hoặc semantics
phải tăng public API version; thay đổi artifact schema phải tăng schema version
riêng.

"PreprocessRequest" là immutable request và chứa tối thiểu:

- source descriptor/path và "CaptureManifest";
- "CalibrationBundle";
- "PreprocessingProfile";
- "ComputeProfile";
- output artifact target và run/correlation ID.

"ArtifactOpenRequest" cũng immutable và chứa artifact path, expected
source/profile/calibration fingerprints, trust policy, "ComputeProfile" và
run/correlation ID. Artifact mode không nhận calibration/profile thay thế để
ghi đè metadata đã ký trong artifact.

Không truyền "EngineInputSpec", TensorRT engine hoặc normalization vào request
của core. Public entry point không dùng positional bag hoặc dictionary mơ hồ:

~~~text
preprocess_capture(request: PreprocessRequest)
    -> PreprocessArtifact | PreprocessFailure

open_preprocessed_artifact(request: ArtifactOpenRequest)
    -> PreprocessArtifact | PreprocessFailure

PreprocessArtifact.open() -> ModelGridReader
ModelGridReader.read_block(row_start, row_end)
    -> {
         image: ndarray[profile.output_layout, profile.output_dtype],
         validity_yx: ndarray[profile.validity_encoding],
         validity_reason_yx: ndarray[profile.reason_encoding],
         mapping_ref,
         provenance
       }
~~~

"PreprocessFailure" phải chứa terminal state, reason code, safe action và
provenance đủ để reference trả kết quả fail-closed. Lỗi input/runtime dự kiến
được biểu diễn bằng "PreprocessFailure"; exception chỉ dành cho programmer error
hoặc invariant bị phá vỡ.

"ModelGridReader" là protocol mới. "validity_yx" quyết định pixel có được dùng
hay không; "validity_reason_yx" giữ nguyên nhân invalid theo policy version hóa,
ví dụ source NoData, missing channel, outside mapping, border hoặc insufficient
resampling support. Reader phải hỗ trợ context manager để ownership/cleanup rõ
ràng. Reason mask không được suy ra lại từ giá trị màu.

### 4.6. Import và packaging contract

V1 dùng import path duy nhất là "preprocessing". Development có thể chạy với
"src" trên "PYTHONPATH", nhưng không được sửa "sys.path" trong code production.
Repository phải thêm "pyproject.toml" theo src-layout để:

- test cài editable package trong development;
- build wheel reproducible cho image deployment;
- reference và CLI dùng cùng package đã cài;
- kiểm tra import trong clean process, không phụ thuộc working directory.

Flight/deployment artifact phải pin package version và wheel digest. Không copy
file rời hoặc vendor một bản "WarpBackend" khác vào reference.

## 5. Cấu trúc package đề xuất

~~~text
src/preprocessing/
  __init__.py           # public facade + __all__ + API version
  api.py                # PreprocessRequest, preprocess/open entry points
  contracts.py          # PreprocessingProfile, schema, enums, validation
  resolver.py           # resolve và kiểm tra preprocessing trust bundle
  source_reader.py      # raw capture -> source blocks
  validity.py           # validity + validity-reason policy
  transform_plan.py     # target grid, mapping, ROI, halo
  warp_backend.py       # CPU/GPU numeric contract, không biết model
  block_processor.py    # producer/consumer bounded
  artifact_writer.py    # raster/masks/sidecars/checksum/publish
  artifact_reader.py    # verify và đọc model-grid artifact
  errors.py             # error classes và reason codes
src/inference_adapter.py         # model-specific, ngoài preprocessing core
src/reference_preprocessed_inference.py
                                 # reference consumer chỉ nhận PreprocessArtifact
src/preprocess_for_inference.py  # CLI/orchestration dùng public facade
pyproject.toml                   # src-layout package/install contract
~~~

"create_tiff_sidecar.py" vẫn giữ vai trò tạo AI sidecar cho một raster cụ thể.
Preprocessing tạo fingerprint và sidecar mới cho output; không gán lại sidecar
của raw source. Metadata thuộc "EngineInputSpec" phải ở engine/inference
sidecar, không trở thành yêu cầu chung của artifact preprocessing.

## 6. Các pha triển khai

### Pha 0 - Chốt hai acceptance contract

Chốt trước khi viết warp:

1. "PreprocessingProfile": calibration selector, transform direction, target
   grid, kernel, halo, border, validity/reason semantics, output layout/dtype,
   internal float precision và cast/rounding.
2. "ModelCompatibilityProfile"/"EngineInputSpec": band order, tensor shape,
   patch/window policy, tensor dtype và normalization cho từng model.
3. Contract nối hai lớp: artifact schema nào một adapter được phép nhận, và
   conversion nào là hợp lệ/tường minh.
4. Public Python API: import path, request/result types, error semantics,
   ownership/context manager và API/schema versioning.
5. RAM, disk, latency và thermal budget riêng của "ComputeProfile".

Nếu chưa có calibration thực tế, chỉ cho phép identity/affine profile để kiểm
chứng boundary; không giả lập Brown-Conrady cho flight.

**Gate:** mỗi profile mẫu validate deterministically, không có default mơ hồ,
không có thông số model trong "PreprocessingProfile", và public API contract
được freeze trước khi reference tích hợp.

### Pha 0.5 - Public package scaffold

- Tạo "src/preprocessing/__init__.py" và "api.py" với public exports đã chốt.
- Tạo "pyproject.toml" cho src-layout và một version source duy nhất.
- Không import TensorRT, model code hoặc "EngineInputSpec" khi import package
  preprocessing.
- Thêm clean-process import smoke test và test "__all__"; mọi public symbol phải
  import được sau editable install và wheel install.
- Cho phép implementation chưa hoàn chỉnh trả failure có reason
  "NOT_IMPLEMENTED" trong development, nhưng không được publish profile
  production hoặc artifact complete từ stub.

**Gate:** "from preprocessing import preprocess_capture" hoạt động độc lập
working directory; package import không khởi tạo decoder/GPU/TensorRT và không
có side effect filesystem.

### Pha 1 - Contract resolver, trust gate và admission hai tầng

- Validate "CaptureManifest", "PreprocessingProfile", "CalibrationBundle" và
  "ComputeProfile"; validate engine bundle chỉ khi pipeline có inference.
- Kiểm tra fingerprint, schema version, generation, thời gian hiệu lực, issuer
  và compatibility giữa capture, profile và calibration.
- Thực hiện admission tầng 1 trước decode/cache/engine allocation bằng upper
  bound bảo thủ theo profile và compute profile.
- Thực hiện admission tầng 2 sau khi decoder/backend (và TensorRT nếu có) đã
  cấp phát thực tế, trước khi materialize output và tại các điểm reserve disk
  quan trọng. Actual usage hoặc free space lệch khỏi bound phải abort fail-closed.
- Chuẩn hóa state transition, run reason code và pixel validity reason code.

**Gate:** input/profile/calibration không tin cậy bị từ chối trước khi đọc dữ
liệu; thiếu resource ở bất kỳ tầng nào không được tạo artifact complete.

### Pha 2 - Source reader, validity và reason mask

- Đọc theo block/strip, giữ nguyên sample values; chỉ đổi axes sang working
  layout nội bộ đã tài liệu hóa, không scale theo min/max.
- Chấp nhận source dtype/representation được "PreprocessingProfile" khai báo,
  thay vì giả định một dtype hoặc bộ band cố định.
- Tạo "validity_yx" độc lập với sample values và "validity_reason_yx" theo
  encoding profile. Khi nhiều nguyên nhân cùng xảy ra, lưu đầy đủ bit/enum theo
  policy thay vì chỉ giữ một heuristic reason.
- Trong flight profile, từ chối compressed/full-decode path nếu admission không
  chứng minh được bound.

**Gate:** source layout, byte order, shape, channel schema, NoData và reason mask
được giữ đúng theo profile; sample hợp lệ không bị đổi giá trị tại reader.

### Pha 3 - Transform planner và warp backend

- Tính source ROI, halo, target grid và mapping cho từng output strip hoàn toàn
  từ "PreprocessingProfile" cùng "CalibrationBundle".
- Hỗ trợ trước identity/affine; sau đó thêm LUT hoặc Brown-Conrady theo
  calibration bundle thật.
- Warp image, validity và reason support theo cùng mapping nhưng với kernel/policy
  riêng được profile chỉ định. Validity/reason không được nội suy như radiometry
  để tạo pixel hợp lệ giả.
- Thực hiện resampling và accumulation trong "float32" hoặc "float64". Sau khi
  kiểm tra support, non-finite và range, cast output theo output dtype, rounding
  và clipping của profile.
- Backend CPU là baseline; GPU chỉ được bật sau golden-test parity theo cùng
  profile, tolerance và mapping convention.

**Gate:** output blockwise khớp oracle trên lưới tổng hợp trong tolerance; thay
đổi strip size hoặc halo không làm thay đổi image, validity, reason mask hay
mapping ngoài tolerance đã công bố.

### Pha 4 - Materialized PreprocessArtifact

V1 dùng artifact trên disk để tách core khỏi inference path hiện tại:

~~~text
<capture>.artifact/
  model-grid.tif        # layout, dtype, codec theo PreprocessingProfile
  validity.tif          # encoding theo validity policy
  validity-reasons.tif  # encoding/bit layout theo reason policy
  output.json           # output schema, axes, dtype, channel schema
  preprocess.json       # source/model grids, mapping, calibration, profile IDs
  manifest.json         # checksums, status, profile IDs, generation
~~~

- Ghi staging files trên cùng filesystem.
- Reserve disk cho output tạm, image/masks/reason sidecars, checksum, recovery
  headroom và các consumer đã được admission.
- Tính checksum sau khi đóng từng file.
- Verify staging bundle rồi publish "manifest.json" cuối cùng bằng atomic rename.
- Artifact thiếu file, checksum sai hoặc "status != complete" không được mở.
- Public "preprocess_capture()" với identity fixture phải chạy end-to-end qua
  resolver, reader, warp, writer và trả "PreprocessArtifact"; không gọi private
  backend trực tiếp từ test consumer.

"preprocess.json" phải liên kết source fingerprint, output fingerprint,
"PreprocessingProfile", "CalibrationBundle", source/model grid, cả hai chiều
mapping hoặc source footprint, validity/reason policy và error bound. Nó không
được bắt buộc chứa patch size, normalization hay TensorRT input dtype.

**Gate:** crash hoặc lỗi I/O không tạo artifact mà consumer có thể nhầm là
complete; raw capture luôn còn nguyên; public API identity call tạo artifact có
thể verify/open/read bằng facade.

### Pha 5 - Tích hợp reference qua public module và inference adapter

Reference implementation hiện tại là "src/inference_large_image_trt.py". Giữ
CLI legacy như compatibility wrapper trong development, nhưng production path
phải gọi public facade. Tạo thêm
"src/reference_preprocessed_inference.py" làm reference consumer chuẩn. Có hai
mode orchestration tường minh:

1. raw mode: gọi "preprocess_capture()" đúng một lần rồi đưa artifact vào
   "InferenceAdapter";
2. artifact mode: gọi "open_preprocessed_artifact()" để verify artifact có sẵn,
   không re-warp.

~~~text
reference.process_capture(request)
  -> preprocessing.preprocess_capture(request.preprocessing)
     hoặc preprocessing.open_preprocessed_artifact(request.artifact)
  -> PreprocessFailure => RETAIN_FOR_GROUND
  -> resolve EngineInputSpec và compatibility gate
  -> initialize TensorRT
  -> ReferencePreprocessedInference(configured_adapter).run(artifact)
~~~

"inference_large_image_trt.py" chỉ được import từ package root
"preprocessing". Sau migration, production path không được trực tiếp tạo
"TiffReader" cho raw source, gọi normalization, padding hoặc transpose; các
thao tác đó lần lượt thuộc core reader hoặc "InferenceAdapter". Không duy trì
hai implementation preprocessing cùng có thể chạy production.

#### Reference consumer chỉ nhận output của module nắn ảnh

"src/reference_preprocessed_inference.py" không phải raw-image entry point. API
reference bắt buộc có dạng:

~~~python
from preprocessing import PreprocessArtifact
from inference_adapter import InferenceAdapter, InferenceResult


class PreprocessedInferenceReference:
    def __init__(self, adapter: InferenceAdapter) -> None:
        ...

    def run(self, artifact: PreprocessArtifact) -> InferenceResult:
        ...
~~~

Đầu vào ảnh duy nhất của "run()" là "PreprocessArtifact" được trả về từ
"preprocess_capture()" hoặc "open_preprocessed_artifact()". Engine,
"EngineInputSpec" và normalization được bind vào "InferenceAdapter" khi khởi
tạo reference object; chúng là dependency thực thi, không phải image input.

File reference này:

- chỉ mở dữ liệu qua "artifact.open()" và "ModelGridReader";
- không nhận source path, artifact path, ndarray hoặc file handle;
- không import "TiffReader", PIL, tifffile, rasterio hay source-reader backend;
- không tìm hoặc mở raw image, calibration image, mask image hay ảnh phụ khác;
- không tự verify path hoặc tự warp; orchestration phải chuyển vào một
  "PreprocessArtifact" đã xác thực;
- đọc image block, validity, validity-reason và mapping từ cùng artifact.

Nếu caller chỉ có artifact path, caller phải gọi
"open_preprocessed_artifact(ArtifactOpenRequest(...))" trước, xử lý
"PreprocessFailure", rồi mới truyền object thành công vào "run()". Reference
consumer không được thêm overload nhận path để rút ngắn bước này.

"InferenceAdapter" phải:

- đọc image theo output layout của artifact, đồng thời đọc validity và reason
  mask;
- xác thực band schema/output dtype/profile compatibility với "EngineInputSpec";
- chọn/reorder bands theo engine contract, chia/pad cửa sổ theo patch policy và
  giữ pad là invalid;
- tính valid fraction, bỏ qua patch dưới ngưỡng và ghi summary của validity
  reasons; không suy diễn invalid thành clear/cloud;
- áp dụng normalization đúng một lần, cast sang tensor dtype và chuyển HWC sang
  NCHW theo engine contract;
- không sửa raster artifact, mapping, kernel, rounding hoặc mask do warp tạo ra.

Reference integration test phải chứng minh:

- clean import và call public facade từ reference;
- raw mode gọi preprocessing đúng một lần, artifact mode không gọi warp;
- cả hai mode đều hội tụ tại
  "PreprocessedInferenceReference.run(PreprocessArtifact)";
- "PreprocessFailure" được truyền thành safe result, không nạp TensorRT;
- reference không import private preprocessing modules;
- signature của consumer không nhận path/ndarray và static import test chặn mọi
  raw/auxiliary image reader;
- normalization và HWC-to-NCHW chỉ chạy một lần trong adapter.

Train-inference parity là test của cặp "PreprocessingProfile" +
"ModelCompatibilityProfile", không phải ràng buộc của "WarpBackend" đơn lẻ. Test
phải chứng minh cùng source samples tạo ra cùng tensor sau band order,
crop/window, padding, normalization và tensor layout.

**Gate:** reference import/call được package đã cài; file
"reference_preprocessed_inference.py" chỉ nhận "PreprocessArtifact", không đọc
raw/auxiliary image và không còn duplicate preprocessing trong production path;
tensor gửi TensorRT khớp engine manifest và pipeline training tương ứng;
artifact hình học vẫn có thể được dùng bởi engine khác qua adapter tương thích.

### Pha 6 - Patch-result và recovery

- Thay dense "_MaskCache" trong production path bằng "PatchResultWriter".
- Record tối thiểu: patch coordinate, model window, valid fraction,
  validity-reason summary, probability, inference status, source mapping
  reference, preprocessing profile ID, engine fingerprint, timestamp và reason
  code.
- Ghi tuần tự theo patch row, checksum và manifest cuối run.
- Hỗ trợ restart: phát hiện record thiếu/duplicate, resume hoặc fail closed.
- Dense TIFF chỉ là export ground/debug hoặc compute profile đã phê duyệt.

**Gate:** manifest thiếu, record duplicate/missing, checksum sai, validity
không đủ hoặc inference dở dang đều không thể trở thành decision loại ảnh.

### Pha 7 - Validation và rollout

- Golden tests cho identity/affine/LUT, nhiều source/output dtype, layout, target
  grid, kernel, rounding, clipping và border policy.
- Test validity/reason propagation: NoData, missing channel, outside mapping,
  border, insufficient support và nhiều lý do đồng thời.
- CPU/GPU parity, strip/halo invariance và mapping provenance.
- Test compatibility adapter cho từng engine: band order, tensor dtype,
  patch/window, padding, normalization và train-inference parity.
- Test wheel/editable install, clean-process public import và API version.
- Reference seam test cho raw/artifact mode, typed failure propagation, call
  count và cấm private imports.
- Contract test cho "reference_preprocessed_inference.py": "run()" chỉ nhận
  "PreprocessArtifact", dùng fake "ModelGridReader", không import/call image
  reader và không mở path khác.
- Integration test source -> preprocess artifact -> fake TensorRT.
- Fault injection: trust failure, calibration lỗi, thiếu disk/RAM ở từng tầng
  admission, codec lỗi, timeout, reset, checksum mismatch và manifest partial.
- HIL benchmark riêng cho mỗi "ComputeProfile" trên Jetson Nano và Orin Nano.

Rollout theo thứ tự: offline -> ground inference -> onboard shadow mode -> chỉ
sau khi georeferencing và OBC contract hoàn tất mới cho phép policy mission.

## 7. Resource admission hai tầng

Tầng 1 là preflight admission sau trust gate và trước mọi cấp phát lớn. Nó dùng
upper bound bảo thủ; core preprocessing phải tính:

~~~text
OS/OBC reserve
  + decoder/source strip + halo
  + float warp working buffers
  + validity + validity-reason buffers
  + mapping/control-grid buffers
  + bounded queue
  + artifact staging/output + checksum/recovery headroom
~~~

Khi pipeline có inference, estimate tầng 1 cộng thêm allocation engine/CUDA,
normalized batch và patch-result buffer. Core không tự nạp TensorRT chỉ để hoàn
thành preprocessing.

Tầng 2 là runtime/commit admission: đo allocation thực tế sau khi backend đã
khởi tạo, reserve output trước khi writer ghi, và kiểm tra lại RAM/disk tại các
điểm chuyển strip hoặc publish. Cả hai bất đẳng thức phải giữ với safety margin:

~~~text
W_peak <= W_RAM,max
D_temporary + D_result + D_headroom <= D_disk,max
~~~

Không dùng swap, full-raster LUT, full-raster float working image, dense mask,
full decode cache hoặc temporary raster ngoài budget. Nếu bất kỳ tầng nào thất
bại, dừng run, không publish manifest complete và giữ source cho ground.

## 8. Fail-safe / fail-closed matrix

| Tình huống | Kết quả | Hành động |
|---|---|---|
| Thiếu capture marker hoặc source fingerprint sai | "INVALID_INPUT" | Giữ source |
| Profile, calibration, compute hoặc engine bundle bắt buộc không tin cậy | "UNTRUSTED_ARTIFACT" | Giữ source |
| Artifact preprocessing không tương thích với engine contract | "INVALID_INPUT" tại adapter | Giữ source; không re-warp ngầm |
| Không đủ RAM/disk ở admission tầng 1 hoặc 2 | "RESOURCE_REJECTED" | Hủy staging, giữ source |
| Ray/mapping/warp hoặc cast policy lỗi | "CALIBRATION_ERROR" hoặc "RUNTIME_FAULT" | Hủy staging, giữ source |
| Pixel không có support hợp lệ | validity=0 kèm reason mask | Không suy luận pixel/patch đó |
| Patch valid fraction thấp | patch invalid kèm reason summary | Không suy luận |
| TensorRT/CUDA/timeout/reset lỗi | "RUNTIME_FAULT" | Hủy kết quả dở dang, giữ source |
| Artifact manifest/checksum lỗi | Không publish/mở artifact | Giữ source |
| Georef quality không đạt | georef_valid=false | "RETAIN_FOR_GROUND" |

Module preprocessing không được phát lệnh xóa capture. Quyền xóa thuộc OBC/F'
sau khi decision record bền vững và source fingerprint khớp.

## 9. Acceptance criteria

Module chỉ sẵn sàng khi:

- Core chạy được với CaptureManifest + CalibrationBundle + PreprocessingProfile +
  ComputeProfile, không cần dataset/model/engine.
- Package cài/import được bằng "from preprocessing import preprocess_capture"
  trong clean process và public API không có import/initialization side effect.
- "inference_large_image_trt.py" production path gọi public facade; không đọc
  raw source, warp, normalize hoặc transpose bằng implementation riêng.
- Raw mode gọi warp đúng một lần; artifact mode verify/open mà không re-warp.
- "reference_preprocessed_inference.py" tồn tại như consumer mẫu; image input
  duy nhất của "run()" là "PreprocessArtifact", và mọi block/mask/mapping đều
  đến từ artifact đó.
- Reference consumer không nhận path/ndarray/file handle, không import raw image
  reader và không mở ảnh phụ.
- Target grid, output layout/dtype, kernel, rounding, cast và validity/reason
  policy đều được lấy từ PreprocessingProfile; không có hard-code model.
- Warp nội bộ dùng float và chỉ cast ở output boundary theo profile.
- Output image, validity mask, validity-reason mask và mapping deterministic,
  đồng thời không phụ thuộc strip size trong tolerance.
- Không có heuristic theo min/max ảnh hoặc patch.
- Normalization, band reordering, patch size, tensor dtype và HWC-to-NCHW chỉ
  nằm trong InferenceAdapter/EngineInputSpec.
- Resource admission tầng 1 từ chối run trước resource không kiểm soát; tầng 2
  chặn publish khi allocation thực tế hoặc disk reserve lệch budget.
- Artifact chỉ được consumer khi manifest complete và checksum hợp lệ.
- Inference không xử lý patch invalid như clear; patch result giữ reason summary.
- Mọi lỗi có reason code và dẫn đến RETAIN_FOR_GROUND.
- Nano/Orin dùng cùng semantics profile nhưng có compute profile, budget, backend
  và engine evidence riêng.
- Chưa bật DELETE_CAPTURE trước khi georeferencing, patch-result,
  DecisionPolicy và OBC/F' đạt kiểm thử hệ thống.

## 10. Ngoài phạm vi của phiên bản đầu

- Orthorectification DEM đầy đủ.
- GeoTIFF/COG ground product.
- Radiometric calibration, color enhancement hoặc denoise.
- Tự động scale/ép dtype không được khai báo trong PreprocessingProfile.
- Model normalization, tensor layout conversion hoặc engine execution trong
  WarpBackend.
- Quyền xóa capture từ Jetson.
- GPU warp production trước khi có benchmark và parity evidence.
