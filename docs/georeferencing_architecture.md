# Kiến trúc nắn ảnh onboard và georeferencing cho `cube_nano`

Tài liệu này mô tả pipeline xử lý ảnh RGB **trên vệ tinh** trước khi đưa vào
model phát hiện mây TensorRT của dự án. Mục tiêu là tạo tensor RGB có pixel grid
phù hợp với `InputSpec`; georeferencing là metadata bổ trợ để truy vết kết quả
và xử lý sản phẩm tại trạm mặt đất.

`Nắn ảnh` trong tài liệu này **chỉ** là nắn hình học: hiệu chỉnh méo ống kính,
đồng đăng ký các kênh R/G/B, crop/rotate/resample về grid đích và ánh xạ pixel.
Pipeline không làm dark/flat-field, white balance, gamma, tone mapping, color
correction, denoise hay radiometric decode onboard. Camera phải cung cấp RGB ở
representation cố định đúng với dữ liệu train; đây không đồng nghĩa với việc bắt
buộc tạo GeoTIFF hoặc orthorectify đầy đủ bằng DEM cho mọi ảnh.

## 1. Quyết định kiến trúc

- Preprocessing/nắn pixel chạy onboard, trước TensorRT inference.
- Flight profile chỉ nhận input ba kênh theo thứ tự `[R, G, B]`. RGB product,
  bit depth, gamma/tone mapping và compression phải đúng contract đã train;
  mismatch bị từ chối, không có bước tự hiệu chỉnh màu/sáng.
- Xử lý theo strip/patch, giới hạn RAM và không mặc định đọc-ghi lại toàn bộ
  raster đã nắn.
- Tất cả phép biến đổi phải là xác định, được version hóa, và tương đương với
  pipeline train/evaluation của checkpoint.
- Runtime phải chọn tường minh một `ComputeProfile` cho `jetson-nano` hoặc
  `jetson-orin-nano`. Profile này chỉ điều khiển engine, backend thực thi,
  resource/power/thermal policy; nó không được thay đổi RGB representation,
  geometric transform, normalization hay bất cứ phần nào của model input contract.
- AI/`DecisionPolicy` tạo quyết định tường minh `KEEP_FOR_DOWNLINK` hoặc
  `DELETE_CAPTURE` từ cloud result và georeferencing. OBC/F' xác thực record rồi
  thực thi gửi hoặc xóa đúng source capture; khi AI hoặc telemetry có lỗi, ảnh
  luôn được giữ.
- Artifact onboard mặc định là patch-result grid nhỏ, không phải dense cloud
  raster. Dense TIFF chỉ là sản phẩm dẫn xuất, bật khi một `ComputeProfile`
  đã có disk/RAM budget và mission thật sự cần nó.
- Georeferencing onboard sinh metadata và ánh xạ giữa các pixel grid; nó không
  thay thế preprocessing pixel cho model.
- Orthorectification đầy đủ bằng DEM và xuất GeoTIFF/COG có thể thực hiện tại
  ground khi sản phẩm bản đồ cần độ chính xác cao.

Luồng production mục tiêu:

```text
boot -> resolve signed ComputeProfile + validate target runtime
    -> capture + calibration + telemetry
    -> validate RGB product/input contract
    -> source integrity + geometric validity mask
    -> geometric correction, co-registration, resampling to model grid
    -> fixed normalization from InputSpec
    -> TensorRT patch classification
    -> patch-result grid + georef sidecar + DecisionPolicy
    -> KEEP_FOR_DOWNLINK | DELETE_CAPTURE | RETAIN_FOR_GROUND
    -> OBC validates record, then sends or deletes the matching capture
```

Model hiện tại là patch classifier, không phải pixel-level segmentation. Mỗi
patch được phân loại `cloud` hoặc `clear`; dense mask, nếu được dẫn xuất, chỉ là
việc gán cùng nhãn cho toàn bộ pixel trong patch. Nắn ảnh không làm output này
trở thành segmentation chính xác đến từng pixel.

Trong tài liệu này, "phân đoạn" nghĩa là phân vùng/phân đoạn thô trên lưới patch
có georeferencing. Nếu mission cần pixel-level segmentation, phải thay model,
dataset, `InputSpec`, TensorRT engine và toàn bộ validation; không được đổi tên
output patch classifier thành segmentation pixel-level.

## 2. Trạng thái hiện tại của dự án

- `src/create_tiff_sidecar.py` tạo input sidecar cho AI: fingerprint, axes,
  shape, dtype, thứ tự band, `input_spec_id` và normalization. Nó chưa mô tả
  RGB product representation, geometric calibration, target grid hay georeferencing.
- `src/tiff_reader.py` xác thực sidecar, sắp xếp band, đọc TIFF theo strip và
  áp dụng normalization của engine manifest. Reader chưa có bước nắn ảnh.
- `src/inference_large_image_trt.py` chia ảnh thành patch, chạy TensorRT và ghi
  mask TIFF thường. Mask có cùng width/height với ảnh reader đưa vào, không tự
  mang thông tin geospatial.
- Engine manifest hiện chỉ khóa `band_order`, `patch_size` và normalization.
  Flight profile phải bổ sung RGB product representation, GSD đích, resampling
  và geometric calibration qua preprocessing contract riêng hoặc tham chiếu của nó.

Vì vậy, đây là kiến trúc mục tiêu cần bổ sung vào pipeline; các module hiện tại
chưa tự động thực hiện calibration hay warp pixel.

## 3. Hợp đồng model và nguyên tắc an toàn

### 3.1. Train-inference parity

Checkpoint chỉ hợp lệ với đúng biểu diễn input đã dùng lúc train. Các phép sau
phải giống nhau giữa train, evaluation và inference onboard:

- ý nghĩa, thứ tự và số lượng kênh RGB;
- RGB product/processing level, bit depth, gamma/tone mapping và compression
  được hỗ trợ;
- quy tắc source integrity, NoData và geometric validity mask;
- GSD, pixel grid, co-registration và resampling kernel;
- clipping và normalization cố định.

Không được chỉ thay preprocessing lúc inference rồi dùng checkpoint cũ như thể
phân phối input không đổi. Nếu đổi RGB product, bit depth/gamma/tone mapping,
GSD hoặc phép resample có ý nghĩa, phải tạo lại dữ liệu train và retrain/fine-tune,
sau đó validation trên scene độc lập.

### 3.2. Fail closed

Không chạy model khi không xác định được RGB product, channel order, geometric
calibration hoặc pixel grid cần thiết. Kết quả phải mang `status=invalid_input` hoặc
`status=out_of_distribution`; không điền giá trị tùy ý và không coi pixel invalid
là `clear`.

Không áp dụng percentile stretch, histogram matching, white balance, gamma/tone
mapping, color correction, denoise hoặc z-score theo từng ảnh/tile. Chúng không
nằm trong scope nắn hình học và có thể làm RGB distribution khác với dữ liệu train.

### 3.3. An toàn quyết định downlink

AI/`DecisionPolicy` được phép quyết định `KEEP_FOR_DOWNLINK` hoặc
`DELETE_CAPTURE`, nhưng Jetson không tự xóa source capture. OBC/F' chỉ thực thi
decision sau khi xác nhận `capture_id`, source fingerprint, complete manifest,
`inference_status=valid`, `georef_valid=true`, `decision_policy_id` và checksum
của decision record. Với dữ liệu mission, mọi trạng thái không thỏa các điều kiện
này có action an toàn là `RETAIN_FOR_GROUND`.

| Điều kiện | Decision từ AI | Hành động OBC/F' |
|---|---|---|
| Contract, model, georef và health hợp lệ; policy kết luận giữ | `KEEP_FOR_DOWNLINK` | Queue/gửi source capture và decision artifact |
| Contract, model, georef và health hợp lệ; policy kết luận xóa | `DELETE_CAPTURE` | Persist decision record, đối chiếu fingerprint rồi xóa đúng source capture |
| Input/profile/calibration không hợp lệ hoặc OOD | `RETAIN_FOR_GROUND` | Giữ source capture |
| Thiếu resource, timeout, CUDA/TensorRT lỗi, reset hoặc thermal limit | `RETAIN_FOR_GROUND` | Giữ source capture, không dùng kết quả dở dang |
| Georef không đạt quality gate | `RETAIN_FOR_GROUND` | Giữ source capture; không phát tọa độ hoặc lệnh xóa |

Mỗi run phải kết thúc bằng một record nguyên tử có `capture_id`, source fingerprint,
inference/georef status, decision, reason code, policy version và artifact
fingerprint. OBC phải coi record thiếu, checksum sai hoặc trạng thái không xác
định là `RETAIN_FOR_GROUND`; lệnh xóa chỉ hợp lệ sau khi record bền vững và đúng
với source capture đang lưu.

### 3.4. Điều kiện kích hoạt profile cho sensor bay

Một `PreprocessingProfile` không được bật flight chỉ vì band/dtype khớp. Trước
khi phát hành, sensor/product/processing level đó phải có tập validation độc lập
đại diện cho mùa, bề mặt, góc chiếu và các negative khó; threshold phải được fit
theo chi phí false-clear/false-cloud của policy downlink. Domain statistic chỉ là
guardrail, không thay thế validation có nhãn.

Calibration phải gắn với sensor serial, version, khoảng hiệu lực và điều kiện
vận hành đã kiểm chứng. Calibration hết hạn, không đúng serial, hoặc không đạt
quality gate sau launch phải làm profile trở thành `invalid_input`, không được
ngoại suy bằng hệ số mặc định.

### 3.5. Artifact tin cậy và tương thích runtime

Fingerprint SHA-256 phát hiện thay đổi nhưng không tự xác thực nguồn phát hành.
Engine manifest, preprocessing profile, compute profile và geometric calibration
bundle từ
ground phải có chữ ký, `key_id`, generation chống rollback, thời điểm hiệu lực
và chain-of-trust được OBC/Jetson kiểm tra trước khi dùng. Profile cũ, bị replay
hoặc không xác thực phải bị từ chối.

TensorRT engine phải được build và smoke-test cho đúng GPU capability, L4T,
JetPack, CUDA, TensorRT, plugin hash, precision và builder flags. ONNX là
artifact portable; TensorRT engine là artifact theo target runtime, không chia
sẻ mặc định giữa Jetson Nano và Jetson Orin Nano.

## 4. Đầu vào cho preprocessing onboard

### 4.1. Capture và sensor/product metadata

Mỗi capture cần có `capture_id` ổn định và liên kết với:

- source image fingerprint, shape, dtype và band layout;
- sensor, platform, RGB product, processing level và processing baseline;
- channel order, bit depth, gamma/tone mapping, compression và RGB representation;
- NoData, fill value, source integrity flags và geometric calibration version;
- sensor serial, geometric calibration validity interval và quality state;
- exposure start/end, row timing nếu applicable và clock synchronization status;
- capture-complete marker, byte count/checksum và storage integrity status;
- input sidecar AI và engine manifest được chọn.

`uint8`/`uint16`, ba kênh hoặc tên file không đủ để suy ra RGB representation.
Nếu metadata không đủ, pipeline phải từ chối hoặc nhận cấu hình RGB product đã
được version hóa từ mission control. Pipeline không được tự áp dụng gain, scale,
white balance, gamma hoặc color matrix để "đoán" representation.

### 4.2. Dữ liệu hình học

Phép nắn hình học cần dùng calibration phù hợp với loại sensor:

- intrinsics: focal length, pixel pitch, principal point và distortion;
- extrinsics/boresight từ camera frame sang body frame;
- mapping đồng đăng ký giữa các kênh R/G/B, kể cả chromatic shift nếu calibration
  hình học của camera mô tả nó;
- target grid: CRS/local frame nếu cần, GSD, origin, extent và resampling kernel.

Nếu cần gán tọa độ mặt đất, capture cần bổ sung timestamp có time scale, vị trí,
vận tốc, attitude quaternion, frame, đơn vị và uncertainty. Rolling shutter và
pushbroom cần state theo hàng hoặc theo nhóm hàng, không dùng một attitude duy
nhất cho cả ảnh.

### 4.3. Compute mode cho Jetson Nano và Jetson Orin Nano

`ComputeProfile` là contract triển khai riêng với `PreprocessingProfile`.
`PreprocessingProfile` trả lời "đưa pixel nào vào model"; `ComputeProfile` trả
lời "thiết bị nào và trong giới hạn nào được phép tạo tensor đó". Cùng source,
preprocessing profile và engine input contract phải tạo tensor tương đương trong
tolerance đã quy định trên cả hai board.

CLI/runtime nhận `--compute-profile` từ cấu hình deployment đáng tin cậy, ví dụ
`jetson-nano-4gb` hoặc `jetson-orin-nano-<ram-sku>`. Không tự chọn mode chỉ từ
tên máy, RAM khả dụng hoặc một phép benchmark lúc boot. Trước mỗi run, runtime
đối chiếu profile với capability thực tế; mismatch phải từ chối inference và
trả ảnh về policy `retain_for_ground`.

Schema tối thiểu của profile cần có:

```json
{
  "schema_version": 1,
  "compute_profile_id": "jetson-nano-4gb-l4t-<version>",
  "device_family": "jetson-nano",
  "target_runtime": {
    "l4t": "<version>",
    "jetpack": "<version>",
    "cuda": "<version>",
    "tensorrt": "<version>",
    "gpu_compute_capability": "<value>"
  },
  "execution": {
    "allowed_engine_fingerprints": ["<sha256>"],
    "preprocess_backend": "cpu-blockwise",
    "max_batch_size": 1,
    "max_inflight_strips": 1,
    "dense_mask_allowed": false
  },
  "resource_limits": {
    "runtime_reserve_bytes": "<positive integer>",
    "max_working_set_bytes": "<positive integer>",
    "max_temporary_disk_bytes": "<positive integer>"
  },
  "power_thermal_policy_id": "<board-and-carrier-specific-policy>",
  "health_policy_id": "<watchdog-timeout-and-safe-action>",
  "security": {
    "key_id": "<issuer-key-id>",
    "generation": "<monotonic integer>",
    "not_before": "<mission time>",
    "expires_at": "<mission time>",
    "signature": "<detached signature>"
  }
}
```

Hai family dùng chung API và artifact schema, nhưng không dùng chung TensorRT
engine nếu target runtime/GPU capability không khớp. `jetson-nano` là mode bảo
thủ: batch, queue, strip và temporary storage chỉ được tăng sau benchmark HIL
trên Nano. `jetson-orin-nano` có thể cho phép batch lớn hơn hoặc backend GPU cho
warp sau benchmark riêng, nhưng không được tự thay đổi kernel, GSD hay
normalization đã train. Cả hai profile phải đặt giới hạn dương, cụ thể và được
đo trên đúng board/carrier; các giá trị placeholder không phải default runtime.

## 5. Pipeline nắn ảnh và inference onboard

### Bước 1 - Resolve contract trước khi đọc raster

Load engine manifest và input sidecar, kiểm tra fingerprint, axes, dtype,
`band_order=[R,G,B]` và RGB representation. Resolve preprocessing profile theo:

```text
sensor + platform + rgb_product + processing_level + geometric_calibration_id + input_spec_id
```

Profile phải quy định rõ các phép hình học được phép áp dụng, GSD/grid đích,
kernel, validity policy và phiên bản implementation. Không đoán profile từ dtype
hoặc từ thống kê của ảnh.

Trước khi nạp TensorRT hoặc cấp phát cache, control plane phải xác nhận capture
đã hoàn tất và checksum hợp lệ; xác thực chữ ký/generation của engine,
preprocessing profile, compute profile và geometric calibration bundle; sau đó mới chạy
resource admission. Một capture thiếu marker hoàn tất, artifact không tin cậy
hoặc estimate vượt budget phải tạo record `resource_rejected`/`invalid_input`
và được giữ cho ground.

### Bước 2 - Xác thực RGB source và geometric validity mask

Đọc strip RGB nguồn mà không thay đổi giá trị R/G/B. Kiểm tra channel order, bit
depth, gamma/tone mapping, compression và source integrity đúng profile; bất kỳ
mismatch nào đều là `invalid_input`. Không decode DN, không scale/offset, không
color correction và không dùng thống kê ảnh để thay đổi RGB.

Tạo geometric validity mask cho NoData/fill do source cung cấp, pixel thiếu kênh,
biên crop/padding và vùng không có ánh xạ hình học hợp lệ. Mask này được warp
cùng ảnh và không biến pixel invalid thành `clear`.

### Bước 3 - Nắn hình học trên vệ tinh

Áp dụng các phép cần thiết để mọi pixel của tensor đầu vào nằm trên target grid
của model:

1. hiệu chỉnh distortion đã được calibration;
2. đồng đăng ký các kênh R/G/B về cùng pixel grid;
3. resample về GSD/extent đích nếu profile yêu cầu;
4. propagate validity mask cùng phép biến đổi.

Với quy ước tọa độ pixel-center, một phép ánh xạ affine có thể được biểu diễn
bằng:

$$
\tilde{\mathbf{p}}_m =
\begin{bmatrix} a & b & c \\ d & e & f \\ 0 & 0 & 1 \end{bmatrix}
\tilde{\mathbf{p}}_s,
\qquad
\tilde{\mathbf{p}}_s=(x_s+0.5,\,y_s+0.5,\,1)^T.
$$

Trong đó $\tilde{\mathbf{p}}_m=(x_m+0.5,\,y_m+0.5,\,1)^T$ và chỉ số ảnh
$(x_s,y_s)$, $(x_m,y_m)$ đều bắt đầu từ 0.

Với LUT hoặc polynomial, ma trận trên được thay bằng hàm
`T_source_to_model` đã hash trong preprocessing sidecar. Mọi implementation
phải giữ cùng quy ước pixel-center, hướng trục và border policy.

Nếu calibration bundle dùng mô hình Brown--Conrady, với
$x=(u-c_x)/f_x$, $y=(v-c_y)/f_y$ và $r^2=x^2+y^2$, distortion có thể được mô tả
bởi:

$$
\begin{aligned}
x_d &= x(1+k_1r^2+k_2r^4+k_3r^6)+2p_1xy+p_2(r^2+2x^2),\\
y_d &= y(1+k_1r^2+k_2r^4+k_3r^6)+p_1(r^2+2y^2)+2p_2xy.
\end{aligned}
$$

Đây chỉ là công thức tham chiếu; calibration có thể dùng LUT hoặc mô hình khác.
Các hệ số và chiều ánh xạ (distorted-to-undistorted hay ngược lại) phải được
ghi rõ trong calibration bundle.

Đồng đăng ký kênh có thể biểu diễn bằng một transform riêng cho mỗi kênh:

$$
\mathbf{p}_{s,\mathrm{ref}}^{(c)}=
A_c\mathbf{p}_s^{(c)}+\mathbf{b}_c,
\qquad c\in\{R,G,B\},
$$

trong đó kênh tham chiếu thường có $A_c=I$ và $\mathbf{b}_c=\mathbf{0}$.

Với RGB, kernel mặc định phải là kernel đã được train và validation; khi giảm độ
phân giải, area/average là ứng viên nếu đó là đúng representation lúc train.
Với geometric validity mask, dùng nearest-neighbour hoặc majority đã được chỉ
định. Không dùng interpolation làm biến mất NoData hoặc tạo pixel hợp lệ giả.

Với một band RGB, resampling có thể viết tổng quát là:

$$
I_m(\mathbf{p}) =
\frac{\sum_{\substack{\mathbf{q}\in\mathcal{N}(\mathbf{p})\\V_s(\mathbf{q})=1}}
w(\mathbf{p},\mathbf{q})\,I_s(\mathbf{q})}
{\sum_{\substack{\mathbf{q}\in\mathcal{N}(\mathbf{p})\\V_s(\mathbf{q})=1}}
w(\mathbf{p},\mathbf{q})},
$$

trong đó mẫu có mẫu số bằng 0 là invalid. Mask đầu ra không được suy ra từ
giá trị màu mà từ support hợp lệ:

$$
V_m(\mathbf{p}) = \mathbf{1}\left[
\frac{\sum_{\mathbf{q}} w(\mathbf{p},\mathbf{q})V_s(\mathbf{q})}
{\sum_{\mathbf{q}} w(\mathbf{p},\mathbf{q})}
\ge \tau_{\mathrm{valid}}\right].
$$

Đối với nearest-neighbour hoặc majority, `w` và ngưỡng phải được cố định trong
`validity_policy_id`; không được thay đổi theo từng ảnh.

Phép nắn phải chạy theo strip có halo đủ cho kernel, sau đó chỉ đưa các patch
hoàn chỉnh vào batch. Không mặc định ghi một TIFF trung gian đã nắn. Nếu mission
bắt buộc lưu raster đã nắn, phải đặt trước disk budget và ghi chung fingerprint,
profile và mapping pixel grid.

Triển khai dùng pipeline producer/consumer có số strip in-flight bị chặn bởi
`ComputeProfile`. Working set phải tính đồng thời source strip + halo, decoded
block, warp output, validity mask, normalized batch, CUDA/TensorRT allocation,
patch-result buffer và filesystem buffer. Không tạo full-raster LUT hoặc dense
mask trong hot path; LUT lớn phải là artifact đã budget/hashed hoặc thay bằng
control grid có thể nội suy theo block.

### Bước 4 - Fixed normalization và TensorRT

Chỉ sau khi geometry đã đúng contract mới áp dụng `NormalizationSpec` cố định từ
engine manifest. Normalization là chuyển đổi tensor bắt buộc của model, không
phải hiệu chỉnh radiometric hay color enhancement. Tensor đầu vào phải có thứ tự
`[R, G, B]`,
shape `[N, C, H, W]`, dtype và `patch_size` đúng với engine.

Nếu `NormalizationSpec` dùng các tham số cố định theo kênh, giá trị đưa vào model
được mô tả bởi:

$$
z_{c,p}=\operatorname{clip}\left(N_c(x_{c,p}),\,l,\,u\right),
$$

với các dạng được hỗ trợ:

$$
N_c(x)=
\begin{cases}
x, & \text{identity},\\
\dfrac{x-d_{\min}}{d_{\max}-d_{\min}}, & \text{dtype-range với input integer},\\
s_cx+o_c, & \text{scale-offset},\\
\dfrac{s_cx+o_c-\mu_c}{\sigma_c}, & \text{standardize}.
\end{cases}
$$

Với input float ở dạng `dtype-range`, miền `[0,1]` là tiền điều kiện; `d_min` và
`d_max` lấy từ dtype, không lấy từ min/max của ảnh hoặc tile. `l`, `u` chỉ
xuất hiện khi contract có khai báo clipping.

Patch có tỷ lệ invalid vượt ngưỡng của profile không được suy luận; ghi nhận
`invalid` trong output. Vùng invalid không tham gia mẫu số khi tính cloud
coverage.

Với patch $P_i$, tỷ lệ hợp lệ được tính là:

$$
v_i=\frac{\sum_{p\in P_i} w_pV_p}{\sum_{p\in P_i}w_p},
\qquad
\text{chỉ suy luận khi }v_i\ge v_{\min}.
$$

`w_p` thường bằng nhau trên model grid đều; nếu dùng trọng số diện tích hoặc
footprint địa lý, quy tắc đó phải được version hóa cùng validity policy.

### Bước 5 - Patch segmentation, georeferencing và quyết định giữ/xóa

Model là patch classifier, vì vậy artifact chuẩn là `<capture>.patch-results.jsonl`
hoặc format binary stream tương đương, không phải mask pixel-level. Header và
mỗi record theo patch phải mang tối thiểu:

```text
capture_id, patch_row, patch_col, model_window, valid_fraction,
cloud_probability, cloud_label, inference_status, source_mapping_ref,
input_spec_id, preprocessing_profile_id, compute_profile_id,
engine_fingerprint, threshold, georef_valid, decision_policy_id,
decision, decision_reason, timestamp
```

Writer append record theo hàng patch và chỉ publish manifest cuối cùng bằng
atomic rename sau khi đủ record. Record thiếu, duplicate, checksum sai hoặc
`inference_status != valid` làm run không đủ điều kiện cho policy loại ảnh.
Dense coarse mask là artifact dẫn xuất từ patch-result grid, phải gắn
`spatial_semantics=patch_classification`, và chỉ được tạo khi
`dense_mask_allowed=true` trong `ComputeProfile`.

Chỉ số dùng cho downlink phải được gọi đúng nghĩa là **tỷ lệ diện tích valid
thuộc các patch được gán cloud**, không phải cloud fraction pixel-level:

$$
A_i=\sum_{p\in P_i}w_pV_p,
\qquad
C_{\mathrm{cloud}}=
\frac{\sum_{i=1}^{M}A_i\,\mathbf{1}[\ell_i=\mathrm{cloud}]}
{\sum_{i=1}^{M}A_i}.
$$

Nếu mẫu số bằng 0, coverage là `invalid`, không phải 0. `A_i` là diện tích trên
model grid hoặc diện tích ground tùy contract; hai nghĩa này không được trộn lẫn.

`A_i` tính từ validity mask và edge policy. Sau khi đầy đủ patch-result
grid và georef sidecar, `DecisionPolicy` version hóa dùng threshold, minimum valid
area, `georef_valid` và model/decision version để tạo đúng một decision:
`KEEP_FOR_DOWNLINK`, `DELETE_CAPTURE` hoặc `RETAIN_FOR_GROUND`. Không điền patch
invalid thành `clear`, không dùng kết quả dở dang, và không cho inference process
tự xóa source capture. OBC thực thi `KEEP_FOR_DOWNLINK` bằng gửi/queue capture;
với `DELETE_CAPTURE`, OBC lưu decision record rồi chỉ xóa file có fingerprint
trùng `source_fingerprint` trong record.

## 6. Ánh xạ pixel và georeferencing

Georeferencing không còn là bước thay thế nắn pixel; nó là metadata kèm theo để
biết patch-result grid hoặc dense mask dẫn xuất đang ở lưới nào và để đưa kết quả
về tọa độ mặt đất khi quality gate cho phép.

### 6.1. Pixel-grid mapping bắt buộc

Mỗi preprocessing run phải sinh `<capture>.preprocess.json`, liên kết bằng
fingerprint với source image và output. Schema version hóa cần có:

```json
{
  "schema_version": 1,
  "source_fingerprint": {"algorithm": "sha256", "digest": "<sha256>"},
  "input_spec_id": "<engine-input-spec>",
  "preprocessing_profile_id": "<sensor-product-profile>",
  "source_grid": {
    "width": 10000,
    "height": 10000,
    "pixel_coordinates": "pixel_center",
    "axes": "x_right_y_down"
  },
  "model_grid": {"width": 3334, "height": 3334, "gsd_m": 30.0},
  "mapping": {
    "source_to_model": {
      "kind": "affine|camera_lut|polynomial",
      "parameters_ref": "<content-addressed-transform-artifact>",
      "parameters_digest": "<sha256>",
      "calibration_id": "<calibration-id>",
      "interpolation": "area",
      "border_policy": "invalid",
      "estimated_error_px_1sigma": "<value>"
    },
    "model_to_source": {
      "kind": "inverse|source_footprint",
      "parameters_ref": "<content-addressed-transform-artifact>",
      "parameters_digest": "<sha256>"
    }
  },
  "validity_policy_id": "<policy-id>",
  "output": {
    "spatial_semantics": "patch_classification",
    "patch_size": 256,
    "edge_policy": "invalid"
  }
}
```

`source_to_model` phải tham chiếu một transform bất biến, có payload/checksum và
quy ước tọa độ rõ ràng; version string hoặc mô tả tự do không đủ. Resampling
không luôn khả nghịch, nên `model_to_source` có thể là source footprint của một
pixel/patch thay vì "inverse" giả. Mọi patch-result phải truy được model window
về source footprint với sai số mapping đã công bố.

Với transform affine, công thức trên là ánh xạ tâm pixel; với transform phi tuyến,
`T_source_to_model` phải được áp dụng tại từng tâm pixel hoặc theo control grid có
sai số nội suy đã công bố. Không được coi `T^{-1}` là nghịch đảo chính xác nếu
resampling đã làm mất thông tin.

### 6.2. Georeferencing onboard cho quyết định giữ/xóa

Với mission này, AI sinh `<capture>.georef.json` cho patch-result grid trước khi
`DecisionPolicy` cho phép `DELETE_CAPTURE`. Sidecar khai báo
`georef_mode=ellipsoid_direct|dem_direct`; không mặc định coi footprint bốn góc
là affine GeoTransform. `georef_mode=none` chỉ hợp lệ cho policy mission khác
không cho phép xóa dựa trên AI.

Với direct georeferencing, sidecar bắt buộc có:

- time system, exposure start/end, row/group-row timing, clock-sync quality và
  telemetry freshness;
- position, velocity, frame của state vector, nguồn ephemeris và rule nội suy;
- quaternion cùng chiều biến đổi/frame nguồn-đích, body/camera boresight và
  camera calibration serial/version;
- ellipsoid/DEM, Earth-orientation model, ray-intersection rule và failure mode;
- footprint và sparse **reference geolocations** của source/model grid; chỉ gọi
  là GCP khi có ground control thực, độc lập với camera model;
- error budget thành phần và tổng hợp, ví dụ attitude/time/orbit/boresight/camera/
  surface-intersection, cùng horizontal/vertical error bound và quality flags.

Với một pixel ảnh thu được tại thời điểm $t$, một ray giao mặt đất có thể
được mô tả bởi:

$$
\mathbf{r}_e(\mathbf{u},t)=
R_{e\leftarrow b}(t)R_{b\leftarrow c}K^{-1}\tilde{\mathbf{u}},
\qquad
\mathbf{g}=\mathbf{s}_e(t)+\lambda^*\mathbf{r}_e,
$$

trong đó $\tilde{\mathbf{u}}=(u,v,1)^T$, $K$ là ma trận intrinsics, $R$ là các
phép quay đã khai báo frame, $\mathbf{s}_e(t)$ là vị trí sensor trong frame $e$,
và $\lambda^*>0$ được chọn sao cho $\mathbf{g}$ nằm trên ellipsoid hoặc DEM đã
khai báo. Ray miss, nghiệm không dương hoặc không hội tụ phải tạo
`georef_valid=false`.

Nếu $\mathbf{z}$ là vector trạng thái và hiệu chỉnh đầu vào, sai số có thể được
lan truyền bằng:

$$
\Sigma_g=J\Sigma_zJ^T+\Sigma_{\mathrm{model}},
\qquad
E_h(k)=k\sqrt{\lambda_{\max}(P_h\Sigma_gP_h^T)}.
$$

`k`, phép chiếu ngang $P_h$ và cách suy ra giới hạn đứng phải nằm trong quality
policy; không gọi một giá trị $1\sigma$ là error bound nếu chưa công bố mức tin
cậy.

`georef_valid=true` chỉ khi telemetry, frame, calibration, ray intersection và
error bound đều đạt ngưỡng mission đã version hóa. Telemetry cũ, time discontinuity,
frame không resolve, calibration không khớp hoặc ground error vượt ngưỡng phải
cho `georef_valid=false` với reason code và buộc `DecisionPolicy` trả
`RETAIN_FOR_GROUND`; không phát sinh tọa độ giả hoặc lệnh xóa. Full DEM
orthorectification và GeoTIFF/COG vẫn mặc định ở ground.

## 7. Tích hợp vào `cube_nano`

1. Thêm `src/runtime/compute_profile.py` và `src/runtime/artifact_trust.py` với
   profile registry, signature/generation validator, capability probe, resource
   admission, health state machine và safe-action policy. Không nạp TensorRT trước
   khi các kiểm tra này hoàn tất.
2. Thêm `src/preprocessing/` với `PreprocessingProfile`, validator metadata và
   block/strip transformer. Module này là nguồn chung cho train, evaluation và
   inference, hoặc có test parity chống lại implementation train.
3. Mở rộng `ImageBlockReader`/`TiffReader` bằng reader đã nắn để
   `process_large_image()` nhận patch trên model grid mà vẫn giữ memory bounded.
   Bỏ fallback full decode trong flight profile khi nó không qua resource admission;
   không chèn preprocessing vào `create_tiff_sidecar.py`.
4. Thay writer dense `_MaskCache` trong hot path bằng `PatchResultWriter` có
   append log, manifest nguyên tử, checksum và recovery sau reset. Dense TIFF là
   export tùy chọn cho ground hoặc profile đã phê duyệt.
5. Tạo CLI riêng, ví dụ `src/preprocess_for_inference.py`, để resolve profile,
   viết preprocess sidecar và stream trực tiếp sang inference. CLI nhận
   `--compute-profile`, xác thực artifact bundle và từ chối input/target runtime
   mơ hồ.
6. Giữ `create_tiff_sidecar.py` cho contract AI của file nguồn/đã xử lý. Nếu
   preprocessing tạo raster trung gian, raster này cần input sidecar mới và
   fingerprint riêng; không gán lại sidecar của source.
7. Thêm `src/georeferencing/` và `create_georef_sidecar.py` để tạo mapping và
   georef sidecar cho patch-result grid. `DecisionPolicy` của mission này không
   được tạo `DELETE_CAPTURE` khi module này không đạt `georef_valid=true`.
8. Định nghĩa interface F' giữa OBC và Jetson: request/cancel/status, heartbeat,
   timeout, reset recovery, artifact-complete acknowledgement, `decision` và
   source fingerprint. OBC thực thi `KEEP_FOR_DOWNLINK` bằng gửi/queue và
   `DELETE_CAPTURE` bằng xóa đúng capture sau khi persist record; Jetson không có
   command xóa source capture.
9. Đóng gói source image, capture manifest, input sidecar, preprocessing sidecar,
   compute profile, geometric calibration bundle, engine manifest, patch-result manifest và
   georef sidecar (nếu có) trong product manifest. Manifest phải lưu fingerprint,
   issuer/key ID và generation của mọi artifact.

## 8. Tài nguyên onboard và vận hành flight

### 8.1. Resource admission và output bounded

Trước capture/inference, `ComputeProfile` phải admission một upper bound gồm:

```text
OS/OBC reserve + TensorRT/CUDA + decoder + source strip/halo + warp output
+ validity mask + normalized batch + queue + patch-result buffer + filesystem headroom
```

Có thể kiểm tra bound bằng hai bất đẳng thức độc lập:

$$
W_{\mathrm{peak}}=\sum_j W_j\le W_{\mathrm{RAM,max}},
\qquad
D_{\mathrm{peak}}=D_{\mathrm{temporary}}+D_{\mathrm{result}}+D_{\mathrm{headroom}}
\le D_{\mathrm{disk,max}}.
$$

Các $W_j$ phải tính theo số strip/patch đang in-flight thực tế; không được nhân
ngầm theo toàn bộ kích thước raster. `filesystem headroom` là phần dành cho
manifest/checksum/recovery, không được dùng lại cho tensor hoặc cache.

Tổng phải nhỏ hơn đồng thời RAM và disk limit của profile với safety margin đã
đo. Kiểm tra lại sau khi TensorRT đã cấp phát thực tế. Không dùng swap, full
decode cache, full-raster LUT, dense mask hoặc temporary TIFF khi chúng không
nằm trong budget đã phê duyệt. TIFF không có true-stream backend bị từ chối trong
flight profile nếu fallback decode/cache vượt bound.

Patch-result grid được ghi tuần tự và có kích thước tỉ lệ với số patch, không
với số pixel. Source capture được giữ cho đến khi OBC nhận manifest complete và
thực thi `KEEP_FOR_DOWNLINK` hoặc `DELETE_CAPTURE`; decision artifact vẫn được
giữ sau khi xóa để audit. Khi storage thiếu hoặc writer không thể hoàn tất
manifest, run là `runtime_fault` và source được giữ cho ground; OBC không suy
diễn rằng ảnh là `clear`.

### 8.2. Hai board, hai profile thực thi

`jetson-nano` và `jetson-orin-nano` dùng cùng API, input/output schema và
numeric preprocessing contract, nhưng có engine, batch limit, queue depth,
backend, RAM/disk budget, power/thermal envelope và benchmark evidence riêng.
Nano bắt đầu với mode bảo thủ; Orin chỉ được tăng batch hoặc chuyển warp sang GPU
sau khi chứng minh latency, memory bandwidth, power và thermal margin trên đúng
board/carrier. Không giả định CPU và GPU độc lập về DRAM hoặc dùng engine Nano
trên Orin/ngược lại.

### 8.3. Power, thermal và containment

`nvpmodel` chỉ nhận mode được xác nhận trên chính board, carrier và JetPack/L4T
của `ComputeProfile`; không hard-code `ULP`, `15W` hay bất kỳ tên mode chung nào.
F' phải chuyển state có timeout, heartbeat và acknowledgment; OBC hoặc hardware
supervisor phải có khả năng watchdog/reset Jetson độc lập. Thermal throttle,
brownout, timeout, reset, filesystem error hoặc CUDA error chuyển run sang
`retain_for_ground` và phải để lại reason code có thể truy vết.

`ComputeProfile` không phải bằng chứng board đã space-qualify. Trước flight cần
phân tích/kiểm thử ở cấp hệ thống cho thermal-vacuum, power rail, boot/reboot,
storage integrity và các rủi ro radiation/SEU/SEL theo quỹ đạo mission. Jetson
chỉ là payload compute; OBC và phần cứng an toàn vẫn giữ authority cho data
retention và spacecraft safety.

### 8.4. Telemetry và promote profile

Ghi `compute_profile_id`, engine/calibration/profile fingerprints, power mode
thực tế, latency, throughput, peak RAM/disk, queue depth, temperature, thermal
throttle, reset/error counter, status/action và georef quality. Chỉ promote một
profile sau benchmark cold/warm cache, ảnh lớn, burst capture và soak test đạt
các limit mission đã chốt.

## 9. Kiểm thử và gate chấp nhận flight

- Unit test RGB input contract: channel order, bit depth, gamma/tone mapping,
  compression, source integrity, NoData/fill, geometric calibration serial/expiry
  và capture completeness. Xác nhận pipeline không đổi RGB ngoài phép resample
  hình học và fixed normalization đã train.
- Unit test distortion, co-registration, resampling, validity propagation,
  edge policy và deterministic block/halo behavior trên lưới có đáp án.
- Test parity: cùng source/preprocessing profile phải tạo tensor tương đương trong
  tolerance ở train, evaluation, Jetson Nano và Jetson Orin Nano. Mỗi board dùng
  engine riêng và có engine-specific probability tolerance.
- Test sensor/product target với scene có nhãn độc lập; khóa threshold, minimum
  valid area và false-clear/false-cloud limit trước test cuối cùng.
- Test `PatchResultWriter`/`DecisionPolicy`: restart giữa lúc ghi, thiếu/duplicate
  record, checksum sai, manifest chưa complete, validity-weighted coverage,
  semantics patch-level, `KEEP_FOR_DOWNLINK`, `DELETE_CAPTURE` và fallback
  `RETAIN_FOR_GROUND`.
- Test mapping theo forward transform, source footprint và sai số không gian công
  bố; không yêu cầu `source -> model -> source` là inverse chính xác sau resample.
- Test georef bằng state/attitude/timing tổng hợp có đáp án, frame mismatch, stale
  telemetry, time discontinuity, ray miss và error-bound vượt ngưỡng. `GCP` chỉ
  xuất hiện trong fixture có ground control độc lập. Mọi `georef_valid=false`
  phải chặn `DELETE_CAPTURE`.
- Test artifact trust: signature sai, key không tin cậy, generation cũ/replay,
  engine/JetPack/TensorRT/GPU mismatch đều bị từ chối và dẫn đến `retain_for_ground`.
- HIL benchmark/soak test riêng cho từng `ComputeProfile`: cold/warm cache, ảnh
  lớn, burst capture, low RAM/disk, thermal throttle, timeout TensorRT, CUDA lỗi,
  brownout/reset và recovery giữa lúc ghi artifact.
- Chạy campaign hệ thống với OBC/F' và phần cứng target để xác nhận watchdog,
  telemetry, safe action, data retention và power/thermal behavior trước flight.
  Xác nhận OBC chỉ gửi capture có `KEEP_FOR_DOWNLINK`, chỉ xóa capture có
  fingerprint khớp `DELETE_CAPTURE`, và không xóa khi record invalid/incomplete.

Một profile chỉ được phát hành khi toàn bộ test trên đạt, engine smoke-test đúng
target, resource/power/thermal limit còn margin, và runbook rollback/revoke có
chủ sở hữu rõ ràng.

## 10. Quyết định và gate phát hành

- Nắn ảnh phục vụ cloud model chạy onboard, trước TensorRT, chỉ giới hạn ở phép
  hình học đã train, validation và phát hành. Không có radiometric/color
  correction onboard ngoài fixed normalization của model.
- Pipeline flight stream theo strip/patch, thực hiện georeferencing cho patch grid
  và phát `PatchResultWriter` + `DecisionPolicy` output; không mặc định tạo raster
  lớn, dense mask, full decode cache hoặc GeoTIFF.
- Hỗ trợ cả Jetson Nano và Jetson Orin Nano bằng `ComputeProfile` tường minh.
  Mode phần cứng chỉ thay execution policy; mỗi mode có engine, artifact bundle,
  resource/power/thermal budget và benchmark admission riêng.
- Input sidecar AI, preprocessing sidecar, compute profile và georeferencing
  sidecar là các contract riêng. Chúng, geometric calibration bundle và engine manifest
  phải liên kết bằng fingerprint, chữ ký, issuer/key ID và generation.
- Kết quả patch classification là phân đoạn thô theo patch có georeferencing.
  Tỷ lệ dùng cho quyết định là validity-weighted cloud-labelled area, không phải
  segmentation hoặc cloud fraction pixel-level.
- Georeferencing có error bound và là gate cho quyết định xóa. `georef_valid`
  không đạt buộc `RETAIN_FOR_GROUND`; orthorectification DEM và GeoTIFF/COG vẫn
  mặc định ở ground.
- Khi inference, georef và artifact đều hợp lệ, AI/`DecisionPolicy` tạo
  `KEEP_FOR_DOWNLINK` hoặc `DELETE_CAPTURE`; OBC/F' xác thực record rồi gửi hoặc
  xóa đúng capture. AI lỗi, input/OOD lỗi, resource/thermal fault hoặc artifact
  thiếu buộc `RETAIN_FOR_GROUND`.
- Không phát hành flight profile trước khi toàn bộ gate ở mục 9 đạt trên board,
  carrier, JetPack/L4T và mission operating envelope tương ứng.

## Tài liệu tham khảo

- [Cross-satellite inference và input contract](cross_satellite_inference_solutions.md)
- [GDAL Geotransform Tutorial](https://gdal.org/en/stable/tutorials/geotransforms_tut.html)
- [GDAL Raster Data Model: affine, GCP và RPC](https://gdal.org/en/stable/user/raster_data_model.html)
- [GDAL `gdalwarp`](https://gdal.org/en/stable/programs/gdalwarp.html)
- [OGC GeoTIFF Standard 1.1](https://docs.ogc.org/is/19-008r4/19-008r4.html)
