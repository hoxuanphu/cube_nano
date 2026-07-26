# Task tracker: module nắn ảnh

Tracker này theo dõi việc triển khai theo
[image_preprocessing_module_plan.md](image_preprocessing_module_plan.md).
Mục tiêu là có một nơi cập nhật trạng thái, bằng chứng hoàn thành và blocker;
chi tiết thiết kế chuẩn vẫn nằm trong plan.

## Quy ước

Trạng thái dùng các giá trị:

- "TODO": chưa bắt đầu.
- "IN_PROGRESS": đang thực hiện, chưa đạt gate.
- "BLOCKED": có blocker đã ghi ở mục Blockers.
- "DONE": đã có bằng chứng và đạt gate.
- "DEFERRED": cố ý để sau phiên bản hiện tại.

Mỗi task chỉ chuyển sang "DONE" khi cột Evidence chứa link tới code, test,
manifest mẫu hoặc benchmark tương ứng. Không đánh dấu hoàn thành chỉ vì đã
viết code nhưng chưa chạy gate.

## Snapshot

| Trường | Giá trị |
|---|---|
| Scope | Core preprocessing độc lập dataset/model và inference adapter |
| Overall status | "IN_PROGRESS" |
| Current phase | P7-04/P7-05 - HIL và shadow rollout (P7-01..P7-03 tự động đã hoàn tất; hardware/OBC còn blocker) |
| Release target | Chưa chốt |
| Last reviewed | 2026-07-24 |
| Tracker owner | "Codex" |
| Source plan | "docs/image_preprocessing_module_plan.md" |

Tình trạng hiện tại: P0/P0.5/P1 đã hoàn thành contract, public facade,
packaging, resolver/trust gate, state machine và admission hai tầng. P2/P3 đã
hoàn thành generic source reader, validity/reason masks, identity/affine
transform planner và CPU warp baseline; G2 đã DONE. P4 đã hoàn thành staging
writer, checksum/manifest atomic, verified artifact reader và identity call
end-to-end qua public facade; G3 đã DONE. P5-01..P5-05, P5-07..P5-08 và toàn
bộ P6 đã có implementation/test. Tên integration đã chuẩn hóa theo chức năng:
`preprocessed_inference.py`, `PreprocessedInference` và
`InferenceProcessRequest`. P5-06 đã chạy end-to-end với fake TensorRT;
TensorRT smoke thật còn IN_PROGRESS vì chưa có engine manifest/input spec và
board đích (B-03/B-04). P7-01..P7-03 đã chạy campaign tự động với 54 test pass
và full suite 141 passed, 9 subtests; P7-04 đã có harness guard nhưng bị BLOCKED
do chưa có Jetson Nano/Orin và GPU parity. P7-05 đã review policy gate nhưng
chưa thể shadow rollout do thiếu georeferencing flight bundle và interface OBC/F'.
GPU P3-05 vẫn DEFERRED cho tới khi có numeric parity và benchmark trên compute
profile đích.

## Work items

| ID | Phase | Work item / exit gate | Owner | Status | Depends on | Evidence |
|---|---|---|---|---|---|---|
| P0-01 | P0 | Chốt schema "PreprocessingProfile": calibration selector, transform, target grid, kernel, validity, output layout/dtype, rounding/cast và numeric precision | "Codex" | "DONE" | - | [contracts.py](../src/preprocessing/contracts.py), [test_preprocessing_contracts.py](../tests/test_preprocessing_contracts.py) |
| P0-02 | P0 | Chốt schema "ModelCompatibilityProfile" canonical: band order, tensor shape, patch/window, tensor dtype, normalization và runtime fingerprint; legacy "EngineInputSpec" dùng adapter tường minh | "Codex" | "DONE" | P0-01 | [contracts.py](../src/preprocessing/contracts.py), [input_contract.py](../src/input_contract.py), [test_preprocessing_contracts.py](../tests/test_preprocessing_contracts.py) |
| P0-03 | P0 | Chốt artifact schema nối hai contract; chứng minh core không cần model/engine để tạo artifact | "Codex" | "DONE" | P0-01, P0-02 | [api.py](../src/preprocessing/api.py), [test_preprocessing_contracts.py](../tests/test_preprocessing_contracts.py) |
| P0-04 | P0 | Chốt fixture profile tối thiểu identity/affine và calibration thật; validation deterministic, không có default mơ hồ | "Codex" | "DONE" | P0-01 | [contracts.py](../src/preprocessing/contracts.py), [test_preprocessing_contracts.py](../tests/test_preprocessing_contracts.py) |
| P0-05 | P0 | Freeze public API: import path, PreprocessRequest, artifact/failure result, reader lifecycle và API/schema versioning | "Codex" | "DONE" | P0-01, P0-03 | [__init__.py](../src/preprocessing/__init__.py), [api.py](../src/preprocessing/api.py), [errors.py](../src/preprocessing/errors.py) |
| P0-06 | P0.5 | Tạo package facade, pyproject src-layout, __all__ và clean-process editable/wheel import smoke test | "Codex" | "DONE" | P0-05 | [pyproject.toml](../pyproject.toml), [__init__.py](../src/preprocessing/__init__.py), [test_preprocessing_contracts.py](../tests/test_preprocessing_contracts.py); editable và wheel smoke pass 2026-07-24 |
| P1-01 | P1 | Implement contract resolver và schema validation cho capture, preprocessing, calibration, compute profile | "Codex" | "DONE" | P0-01, P0-04 | [resolver.py](../src/preprocessing/resolver.py), [contracts.py](../src/preprocessing/contracts.py), [test_preprocessing_trust_admission.py](../tests/test_preprocessing_trust_admission.py) |
| P1-02 | P1 | Implement artifact trust: SHA-256, signature/issuer/key ID, generation, expiry và fingerprint linkage | "Codex" | "DONE" | P1-01 | [resolver.py](../src/preprocessing/resolver.py), [contracts.py](../src/preprocessing/contracts.py), [test_preprocessing_trust_admission.py](../tests/test_preprocessing_trust_admission.py) |
| P1-03 | P1 | Implement state machine NEW -> VALIDATING -> ADMITTED -> PROCESSING -> VERIFYING -> COMPLETE và terminal reason codes | "Codex" | "DONE" | P1-01, P1-02 | [errors.py](../src/preprocessing/errors.py), [api.py](../src/preprocessing/api.py), [test_preprocessing_trust_admission.py](../tests/test_preprocessing_trust_admission.py) |
| P1-04 | P1 | Implement resource admission tầng 1 trước decode/cache/engine allocation | "Codex" | "DONE" | P0-01, P1-01 | [admission.py](../src/preprocessing/admission.py), [test_preprocessing_trust_admission.py](../tests/test_preprocessing_trust_admission.py) |
| P1-05 | P1 | Implement resource admission tầng 2 sau allocation thực tế và trước commit/publish | "Codex" | "DONE" | P1-04 | [admission.py](../src/preprocessing/admission.py), [test_preprocessing_trust_admission.py](../tests/test_preprocessing_trust_admission.py) |
| P2-01 | P2 | Refactor source reader thành block/strip reader generic, giữ sample values và source schema | "Codex" | "DONE" | P0-01, P1-01 | [source_reader.py](../src/preprocessing/source_reader.py), [test_preprocessing_geometry.py](../tests/test_preprocessing_geometry.py) |
| P2-02 | P2 | Implement validity_yx độc lập với giá trị ảnh | "Codex" | "DONE" | P2-01, P0-01 | [validity.py](../src/preprocessing/validity.py), [test_preprocessing_geometry.py](../tests/test_preprocessing_geometry.py) |
| P2-03 | P2 | Implement validity_reason_yx versioned enum/bit mask và propagation của nhiều nguyên nhân | "Codex" | "DONE" | P2-02, P0-01 | [validity.py](../src/preprocessing/validity.py), [warp_backend.py](../src/preprocessing/warp_backend.py), [test_preprocessing_geometry.py](../tests/test_preprocessing_geometry.py) |
| P2-04 | P2 | Đạt gate NoData, missing channel, outside mapping, border và compressed/full-decode budget | "Codex" | "DONE" | P1-04, P2-01, P2-03 | [source_reader.py](../src/preprocessing/source_reader.py), [test_preprocessing_geometry.py](../tests/test_preprocessing_geometry.py) |
| P3-01 | P3 | Implement transform planner: target grid, source ROI, halo, pixel convention và mapping hai chiều/footprint | "Codex" | "DONE" | P0-01, P0-04 | [transform_plan.py](../src/preprocessing/transform_plan.py), [test_preprocessing_geometry.py](../tests/test_preprocessing_geometry.py) |
| P3-02 | P3 | Implement CPU warp baseline với internal float32/float64 numeric contract | "Codex" | "DONE" | P3-01, P2-02 | [warp_backend.py](../src/preprocessing/warp_backend.py), [test_preprocessing_geometry.py](../tests/test_preprocessing_geometry.py) |
| P3-03 | P3 | Implement profile-driven kernel, border, rounding, clipping, non-finite handling và cast output dtype | "Codex" | "DONE" | P3-02, P0-01 | [warp_backend.py](../src/preprocessing/warp_backend.py), [test_preprocessing_geometry.py](../tests/test_preprocessing_geometry.py) |
| P3-04 | P3 | Đạt strip/halo invariance và CPU golden-test tolerance cho image, masks, reason và mapping | "Codex" | "DONE" | P3-03, P2-04 | [transform_plan.py](../src/preprocessing/transform_plan.py), [warp_backend.py](../src/preprocessing/warp_backend.py), [test_preprocessing_geometry.py](../tests/test_preprocessing_geometry.py) |
| P3-05 | P3 | GPU backend chỉ được bật sau numeric parity và resource benchmark trên từng compute profile | "Codex" | "DEFERRED" | P3-04, P1-05 | [warp_backend.py](../src/preprocessing/warp_backend.py), [test_preprocessing_geometry.py](../tests/test_preprocessing_geometry.py); GPU bị fail-closed cho tới khi có parity/HIL evidence |
| P4-01 | P4 | Implement staging writer cho image, validity, reason mask, output metadata và preprocess mapping | "Codex" | "DONE" | P2-04, P3-04 | [artifact_writer.py](../src/preprocessing/artifact_writer.py), [test_preprocessing_artifacts.py](../tests/test_preprocessing_artifacts.py) |
| P4-02 | P4 | Implement checksum, manifest complete và atomic rename; không publish artifact dở dang | "Codex" | "DONE" | P4-01, P1-02 | [artifact_writer.py](../src/preprocessing/artifact_writer.py), [api.py](../src/preprocessing/api.py), [test_preprocessing_artifacts.py](../tests/test_preprocessing_artifacts.py) |
| P4-03 | P4 | Implement artifact reader verify đầy đủ file, checksum, profile/calibration digest và schema | "Codex" | "DONE" | P4-02 | [artifact_reader.py](../src/preprocessing/artifact_reader.py), [api.py](../src/preprocessing/api.py), [test_preprocessing_artifacts.py](../tests/test_preprocessing_artifacts.py) |
| P4-04 | P4 | Fault injection crash/I/O/disk-full chứng minh source được giữ và artifact incomplete không mở được | "Codex" | "DONE" | P1-05, P4-02, P4-03 | [test_preprocessing_artifacts.py](../tests/test_preprocessing_artifacts.py); rename failure, simulated disk-full, checksum và incomplete-manifest tests pass |
| P4-05 | P4 | Public preprocess_capture identity call chạy end-to-end và artifact verify/open/read chỉ qua facade | "Codex" | "DONE" | P0-06, P4-03 | [api.py](../src/preprocessing/api.py), [test_preprocessing_trust_admission.py](../tests/test_preprocessing_trust_admission.py); identity artifact verify/open/read pass |
| P5-01 | P5 | Tạo InferenceAdapter độc lập với WarpBackend và đọc được artifact theo output layout profile | "Codex" | "DONE" | P4-03, P0-02 | [inference_adapter.py](../src/inference_adapter.py), [artifact_reader.py](../src/preprocessing/artifact_reader.py), [test_preprocessing_p5_p6.py](../tests/test_preprocessing_p5_p6.py) |
| P5-02 | P5 | Compatibility gate: band availability/order, grid semantics, output dtype/layout và engine fingerprint | "Codex" | "DONE" | P5-01, P0-02 | [inference_adapter.py](../src/inference_adapter.py), [test_preprocessing_p5_p6.py](../tests/test_preprocessing_p5_p6.py) |
| P5-03 | P5 | Adapter thực hiện patch/window, padding validity, normalization, tensor dtype và HWC-to-NCHW | "Codex" | "DONE" | P5-02 | [inference_adapter.py](../src/inference_adapter.py), [test_preprocessing_p5_p6.py](../tests/test_preprocessing_p5_p6.py) |
| P5-04 | P5 | Refactor inference_large_image_trt raw mode gọi public preprocessing facade đúng một lần; failure chặn TensorRT | "Codex" | "DONE" | P4-05, P5-03 | [preprocessed_inference.py](../src/preprocessed_inference.py), [inference_large_image_trt.py](../src/inference_large_image_trt.py), [test_preprocessing_p5_p6.py](../tests/test_preprocessing_p5_p6.py); production_contract fail-closed nếu chưa có typed request |
| P5-05 | P5 | Train-inference parity cho từng cặp PreprocessingProfile + EngineInputSpec | "Codex" | "DONE" | P5-03, P5-04 | [input_contract.py](../src/input_contract.py), [inference_adapter.py](../src/inference_adapter.py), [test_preprocessing_p5_p6.py](../tests/test_preprocessing_p5_p6.py) |
| P5-06 | P5 | End-to-end source -> public module -> artifact -> fake TensorRT và TensorRT smoke test | "Codex" | "IN_PROGRESS" | P4-04, P5-05 | [preprocessed_inference.py](../src/preprocessed_inference.py), [inference_adapter.py](../src/inference_adapter.py), [test_preprocessing_p5_p6.py](../tests/test_preprocessing_p5_p6.py); fake TensorRT pass, real smoke chờ B-03/B-04 |
| P5-07 | P5 | Artifact mode verify/open không re-warp; cấm private imports và duplicate production preprocessing | "Codex" | "DONE" | P0-06, P5-04, P5-06 | [preprocessed_inference.py](../src/preprocessed_inference.py), [preprocessing/patch_result_writer.py](../src/preprocessing/patch_result_writer.py), [test_preprocessing_p5_p6.py](../tests/test_preprocessing_p5_p6.py) |
| P5-08 | P5 | Tạo preprocessed_inference.py; run() chỉ nhận PreprocessArtifact, không nhận path/ndarray và không đọc raw/ảnh phụ | "Codex" | "DONE" | P5-03, P5-07 | [preprocessed_inference.py](../src/preprocessed_inference.py), [test_preprocessing_p5_p6.py](../tests/test_preprocessing_p5_p6.py) |
| P6-01 | P6 | Implement PatchResultWriter append theo patch row, checksum và manifest | "Codex" | "DONE" | P5-08 | [patch_result_writer.py](../src/patch_result_writer.py), [test_preprocessing_p5_p6.py](../tests/test_preprocessing_p5_p6.py) |
| P6-02 | P6 | Ghi mapping reference, valid fraction, validity-reason summary, status và engine/profile fingerprint | "Codex" | "DONE" | P6-01, P5-02 | [patch_result_writer.py](../src/patch_result_writer.py), [inference_adapter.py](../src/inference_adapter.py), [test_preprocessing_p5_p6.py](../tests/test_preprocessing_p5_p6.py) |
| P6-03 | P6 | Recovery test missing/duplicate record, resume hoặc fail closed | "Codex" | "DONE" | P6-01 | [patch_result_writer.py](../src/patch_result_writer.py), [test_preprocessing_p5_p6.py](../tests/test_preprocessing_p5_p6.py) |
| P6-04 | P6 | Chặn decision khi artifact/patch result thiếu, invalid hoặc inference dở dang | "Codex" | "DONE" | P6-02, P6-03 | [decision_policy.py](../src/decision_policy.py), [patch_result_writer.py](../src/patch_result_writer.py), [test_preprocessing_p5_p6.py](../tests/test_preprocessing_p5_p6.py) |
| P7-01 | P7 | Golden campaign nhiều input/output dtype, layout, target grid, kernel, rounding và border | "Codex" | "DONE" | P3-04 | [test_preprocessing_p7.py](../tests/test_preprocessing_p7.py), [P7 validation report](image_preprocessing_module_p7_validation_report.md); identity/affine campaign pass, LUT/Brown-Conrady vẫn fail-closed theo B-01 |
| P7-02 | P7 | Validity/reason propagation campaign và non-finite/cast edge cases | "Codex" | "DONE" | P2-03, P3-03 | [test_preprocessing_p7.py](../tests/test_preprocessing_p7.py), [validity.py](../src/preprocessing/validity.py); 54-test P7 campaign pass |
| P7-03 | P7 | Fault-injection campaign cho trust, calibration, RAM/disk hai tầng, codec, timeout, reset và checksum | "Codex" | "DONE" | P1-05, P4-04, P6-03 | [test_preprocessing_p7.py](../tests/test_preprocessing_p7.py), [inference_adapter.py](../src/inference_adapter.py), [P7 validation report](image_preprocessing_module_p7_validation_report.md) |
| P7-04 | P7 | HIL benchmark/soak riêng cho Jetson Nano và Orin Nano theo ComputeProfile | "Codex" | "BLOCKED" | P3-05, P5-08, P7-03 | [preprocessing_p7_hil_benchmark.py](../scripts/preprocessing_p7_hil_benchmark.py), [P7 validation report](image_preprocessing_module_p7_validation_report.md); host smoke 5/5 pass, Nano/Orin run blocked by B-04 |
| P7-05 | P7 | Shadow-mode rollout review với georeferencing, DecisionPolicy và OBC/F' | "Codex" | "BLOCKED" | P6-04, P7-04 | [P7 validation report](image_preprocessing_module_p7_validation_report.md), [test_preprocessing_p7.py](../tests/test_preprocessing_p7.py); georef/OBC/F' system contract còn thiếu (B-06) |

## Core invariant checklist

Các invariant này phải được kiểm tra trong code review và test gate:

- [x] Core chạy được với source image, calibration và PreprocessingProfile,
      không cần dataset/model/engine.
- [x] Package cài/import được bằng public path preprocessing trong clean process;
      import không khởi tạo decoder, GPU, TensorRT hoặc ghi filesystem.
- [x] Reference chỉ import package root và production path không gọi private
      preprocessing backend.
- [x] Reference raw mode gọi preprocessing đúng một lần; artifact mode không
      re-warp; typed failure chặn khởi tạo TensorRT.
- [x] preprocessed_inference.run() chỉ nhận PreprocessArtifact; không
      nhận path/ndarray/file handle và không đọc raw hoặc ảnh phụ.
- [x] Reference consumer chỉ dùng artifact.open()/ModelGridReader; static import
      test chặn TiffReader, PIL, tifffile, rasterio và source-reader backend.
- [x] Không có hard-code tên model, output dtype, normalization, /65535,
      patch size hoặc tensor layout trong WarpBackend.
- [x] Target grid, kernel, rounding, border, output layout/dtype và validity đều
      được resolve từ PreprocessingProfile.
- [x] Warp nội bộ dùng float; cast chỉ xảy ra ở output boundary theo profile.
- [x] validity_yx và validity_reason_yx độc lập với pixel color và giữ đủ
      nguyên nhân khi có nhiều lỗi.
- [x] Normalization, band reorder, patching và HWC-to-NCHW chỉ nằm ở adapter.
- [x] Artifact chỉ publish sau checksum và atomic manifest commit.
- [x] Resource admission có cả preflight bound và runtime/commit check.
- [x] Mọi lỗi đều có reason code và safe action RETAIN_FOR_GROUND.

## Gate dashboard

| Gate | Điều kiện mở gate | Status | Evidence / link |
|---|---|---|---|
| G0 - Contract/public API freeze | P0-01..P0-06 DONE; schema, facade, packaging và clean import đã review | "DONE" | [contracts.py](../src/preprocessing/contracts.py), [__init__.py](../src/preprocessing/__init__.py), [pyproject.toml](../pyproject.toml), [test_preprocessing_contracts.py](../tests/test_preprocessing_contracts.py) |
| G1 - Trust/admission | P1-01..P1-05 DONE; reject trước allocation và fail closed sau allocation | "DONE" | [resolver.py](../src/preprocessing/resolver.py), [admission.py](../src/preprocessing/admission.py), [test_preprocessing_trust_admission.py](../tests/test_preprocessing_trust_admission.py) |
| G2 - Geometric transform | P2-01..P3-04 DONE; golden parity và strip invariance | "DONE" | [source_reader.py](../src/preprocessing/source_reader.py), [validity.py](../src/preprocessing/validity.py), [transform_plan.py](../src/preprocessing/transform_plan.py), [warp_backend.py](../src/preprocessing/warp_backend.py), [test_preprocessing_geometry.py](../tests/test_preprocessing_geometry.py) |
| G3 - Atomic artifact/public call | P4-01..P4-05 DONE; crash/I/O an toàn và facade identity call hoạt động | "DONE" | [artifact_writer.py](../src/preprocessing/artifact_writer.py), [artifact_reader.py](../src/preprocessing/artifact_reader.py), [test_preprocessing_artifacts.py](../tests/test_preprocessing_artifacts.py), [test_preprocessing_trust_admission.py](../tests/test_preprocessing_trust_admission.py) |
| G4 - Inference integration | P5-01..P5-08 implementation complete; artifact-only consumer, no image reader/private imports và parity | "IN_PROGRESS" | [inference_adapter.py](../src/inference_adapter.py), [preprocessed_inference.py](../src/preprocessed_inference.py), [test_preprocessing_p5_p6.py](../tests/test_preprocessing_p5_p6.py); P5-06 real TensorRT smoke còn thiếu |
| G5 - Recovery/decision | P6-01..P6-04 DONE; invalid/incomplete không thành decision | "DONE" | [patch_result_writer.py](../src/patch_result_writer.py), [decision_policy.py](../src/decision_policy.py), [test_preprocessing_p5_p6.py](../tests/test_preprocessing_p5_p6.py) |
| G6 - Hardware rollout | P7-01..P7-05 DONE; HIL, shadow mode và OBC review | "BLOCKED" | [P7 validation report](image_preprocessing_module_p7_validation_report.md), [preprocessing_p7_hil_benchmark.py](../scripts/preprocessing_p7_hil_benchmark.py); P7-04/P7-05 chưa đạt do B-01/B-03/B-04/B-06 |

Không promote gate tiếp theo khi gate trước còn BLOCKED, trừ khi có decision
log ghi rõ phạm vi thử nghiệm bị giới hạn và safe action.

## Blockers và risks

| ID | Blocker/risk | Impact | Mitigation / next action | Owner | Status |
|---|---|---|---|---|---|
| B-01 | Chưa chốt calibration flight và chiều mapping | Không thể chứng minh geometric correctness | Cung cấp calibration bundle thật; chỉ dùng identity/affine cho boundary test | "TBD" | "OPEN" |
| B-02 | Chưa chốt output storage/codec cho từng compute profile | P4 chỉ cung cấp baseline TIFF memmap không nén; codec/benchmark deployment chưa finalize | Giữ baseline uncompressed fail-closed; chốt codec, compression policy và benchmark decode path theo từng ComputeProfile ở P7 | "TBD" | "OPEN" |
| B-03 | Chưa có engine manifest/input spec cho deployment | Chưa thể chạy TensorRT smoke thật và khóa fingerprint deployment | Fake engine/profile fixture đã chạy; cung cấp engine manifest/input spec thật trước G4 promotion | "Codex" | "OPEN" |
| B-04 | Chưa có RAM/disk/thermal benchmark trên board đích và chưa có GPU numeric parity | Không được enable flight profile hoặc bật GPU P3-05 | CPU baseline đã fail-closed GPU; chạy HIL, đo peak usage/latency/power/thermal margin và golden parity trên từng ComputeProfile | "Codex" | "OPEN" |
| B-05 | Legacy CLI còn giữ raw reader/normalization/pad/transpose cho compatibility development | Không được dùng legacy path trong production | `production_contract=True` fail-closed nếu thiếu typed request; production typed path đã qua facade. Xóa compatibility branch sau rollout | "Codex" | "MITIGATED" |
| B-06 | Chưa có georeferencing flight sidecar và interface OBC/F' cho shadow mode | Không được promote policy mission hoặc thực thi DELETE_CAPTURE | Chốt georef error-bound/state contract và F' request/cancel/status/heartbeat/timeout/reset/ack; chạy system campaign với OBC trước khi mở G6 | "TBD" | "OPEN" |

## Decision log

| Date | ID | Decision | Rationale / impact | Owner |
|---|---|---|---|---|
| 2026-07-24 | D-01 | Tách PreprocessingProfile khỏi EngineInputSpec | Core nắn ảnh độc lập model/dataset; model-specific behavior ở adapter | "TBD" |
| 2026-07-24 | D-02 | Cho phép output dtype/layout/kernel/grid theo profile | Không biến cấu hình deployment hiện tại thành giới hạn thuật toán | "TBD" |
| 2026-07-24 | D-03 | Giữ validity mask và thêm validity-reason mask | Phân biệt pixel invalid và nguyên nhân để recovery/decision/audit | "TBD" |
| 2026-07-24 | D-04 | Giữ atomic artifact, trust bundle, admission hai tầng và state machine | Bảo đảm fail-closed, provenance và giữ source khi run không hoàn tất | "TBD" |
| 2026-07-24 | D-05 | Public import path là preprocessing; reference chỉ gọi facade | Biến core thành module dùng lại được và ngăn duplicate/private integration | "TBD" |
| 2026-07-24 | D-06 | Thêm preprocessed_inference.py chỉ nhận PreprocessArtifact | Consumer mẫu không đọc raw/ảnh phụ và chỉ dùng output đã xác thực của module nắn ảnh | "TBD" |
| 2026-07-24 | D-07 | Fingerprint ký trên canonical contract content; trust envelope bị loại khỏi content digest | Tránh vòng lặp digest/signature và giữ digest ổn định khi gắn trust metadata | "Codex" |
| 2026-07-24 | D-08 | TrustPolicy secure-by-default; unsigned fixture chỉ chạy khi caller chọn TrustPolicy.development() | Không để mode thử nghiệm vô tình trở thành flight trust policy | "Codex" |
| 2026-07-24 | D-09 | Runtime admission dùng cùng `artifact_staging_multiplier` với preflight; facade chỉ chuyển `PreprocessError`, không bắt lỗi lập trình; `ModelCompatibilityProfile` là tên canonical và legacy `EngineInputSpec` có adapter tường minh | Không để tầng runtime bypass disk bound, không che giấu invariant bug, và không nhập nhằng contract model cũ/mới | "Codex" |
| 2026-07-24 | D-10 | Compatibility gate chạy trước `engine_factory`; raw/artifact mode hội tụ tại `PreprocessedInference.run(PreprocessArtifact)` | Không khởi tạo TensorRT khi artifact/trust/profile không tương thích và không re-warp artifact | "Codex" |
| 2026-07-24 | D-11 | Patch result append-only theo patch row, giữ staging để resume; mọi record thiếu/duplicate/checksum/inference dở dang dẫn tới `RETAIN_FOR_GROUND` | Bảo đảm decision không suy diễn từ run chưa hoàn chỉnh và OBC là bên duy nhất thực thi xóa | "Codex" |

## Update log

| Date | Change | Updated by |
|---|---|---|
| 2026-07-24 | Tạo tracker và đồng bộ backlog với plan module nắn ảnh | "TBD" |
| 2026-07-24 | Bổ sung public module, packaging và reference-integration tasks/gates | "TBD" |
| 2026-07-24 | Bổ sung artifact-only reference consumer và contract test | "TBD" |
| 2026-07-24 | Hoàn thành P0/P0.5/P1: contracts, facade, packaging, resolver, trust và two-tier admission; full suite 60 passed, 9 subtests | "Codex" |
| 2026-07-24 | Remediate review findings: đồng bộ runtime/preflight staging multiplier, thu hẹp exception boundary, thêm legacy EngineInputSpec adapter và thu gọn `__all__`; full suite 64 passed, 9 subtests | "Codex" |
| 2026-07-24 | Hoàn thành P2/P3-01..P3-04: generic block reader, validity/reason masks, identity/affine planner, CPU warp, profile-driven cast và strip/halo invariance; geometry suite 11 passed | "Codex" |
| 2026-07-24 | G2 DONE; full suite sau P2/P3: 75 passed, 9 subtests. P3-05 GPU DEFERRED và giữ fail-closed do thiếu parity/HIL benchmark | "Codex" |
| 2026-07-24 | Hoàn thành P4: TIFF/memmap staging writer, mapping/masks/metadata, checksum + complete manifest + atomic publish, verified reader, facade identity end-to-end và fault injection I/O/disk-full; full suite 82 passed, 9 subtests | "Codex" |
| 2026-07-24 | Triển khai P5/P6: InferenceAdapter + compatibility gate/padding/normalization/layout, reference raw/artifact facade integration, append-only PatchResultWriter với checksum/resume và DecisionPolicy validity-weighted fail-closed; fake TensorRT integration pass; full suite 87 passed, 9 subtests | "Codex" |
| 2026-07-24 | Hoàn thành P7-01..P7-03: golden dtype/layout/grid/kernel/rounding/border, validity/reason/non-finite/cast và fault-injection trust/calibration/admission/codec/checksum/timeout/reset; P7 campaign 54 passed, full suite 141 passed, 9 subtests | "Codex" |
| 2026-07-24 | Thêm guarded HIL/soak harness và shadow-mode review evidence; host smoke 5/5 pass nhưng Nano/Orin và OBC/F' chưa có nên P7-04/P7-05, G6 BLOCKED theo B-01/B-03/B-04/B-06 | "Codex" |
| 2026-07-24 | Chuẩn hóa naming integration: `reference_preprocessed_inference.py` -> `preprocessed_inference.py`, `PreprocessedInferenceReference` -> `PreprocessedInference`; đồng bộ import, test, packaging và docs; full suite 141 passed, 9 subtests | "Codex" |

Khi cập nhật tracker, sửa đồng thời Snapshot, Work items, Gate dashboard,
Blockers và Update log; không xóa lịch sử decision hoặc evidence cũ.
