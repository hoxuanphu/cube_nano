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
| Current phase | P0 - chốt contract |
| Release target | Chưa chốt |
| Last reviewed | 2026-07-24 |
| Tracker owner | "TBD" |
| Source plan | "docs/image_preprocessing_module_plan.md" |

Tình trạng hiện tại: plan đã được tách thành "PreprocessingProfile" và
"ModelCompatibilityProfile"/"EngineInputSpec"; phần code implementation chưa
được đánh dấu hoàn thành.

## Work items

| ID | Phase | Work item / exit gate | Owner | Status | Depends on | Evidence |
|---|---|---|---|---|---|---|
| P0-01 | P0 | Chốt schema "PreprocessingProfile": calibration selector, transform, target grid, kernel, validity, output layout/dtype, rounding/cast và numeric precision | "TBD" | "TODO" | - | - |
| P0-02 | P0 | Chốt schema "ModelCompatibilityProfile"/"EngineInputSpec": band order, tensor shape, patch/window, tensor dtype, normalization và runtime fingerprint | "TBD" | "TODO" | P0-01 | - |
| P0-03 | P0 | Chốt artifact schema nối hai contract; chứng minh core không cần model/engine để tạo artifact | "TBD" | "TODO" | P0-01, P0-02 | - |
| P0-04 | P0 | Chốt fixture profile tối thiểu identity/affine và calibration thật; validation deterministic, không có default mơ hồ | "TBD" | "TODO" | P0-01 | - |
| P0-05 | P0 | Freeze public API: import path, PreprocessRequest, artifact/failure result, reader lifecycle và API/schema versioning | "TBD" | "TODO" | P0-01, P0-03 | - |
| P0-06 | P0.5 | Tạo package facade, pyproject src-layout, __all__ và clean-process editable/wheel import smoke test | "TBD" | "TODO" | P0-05 | - |
| P1-01 | P1 | Implement contract resolver và schema validation cho capture, preprocessing, calibration, compute profile | "TBD" | "TODO" | P0-01, P0-04 | - |
| P1-02 | P1 | Implement artifact trust: SHA-256, signature/issuer/key ID, generation, expiry và fingerprint linkage | "TBD" | "TODO" | P1-01 | - |
| P1-03 | P1 | Implement state machine NEW -> VALIDATING -> ADMITTED -> PROCESSING -> VERIFYING -> COMPLETE và terminal reason codes | "TBD" | "TODO" | P1-01, P1-02 | - |
| P1-04 | P1 | Implement resource admission tầng 1 trước decode/cache/engine allocation | "TBD" | "TODO" | P0-01, P1-01 | - |
| P1-05 | P1 | Implement resource admission tầng 2 sau allocation thực tế và trước commit/publish | "TBD" | "TODO" | P1-04 | - |
| P2-01 | P2 | Refactor source reader thành block/strip reader generic, giữ sample values và source schema | "TBD" | "TODO" | P0-01, P1-01 | - |
| P2-02 | P2 | Implement validity_yx độc lập với giá trị ảnh | "TBD" | "TODO" | P2-01, P0-01 | - |
| P2-03 | P2 | Implement validity_reason_yx versioned enum/bit mask và propagation của nhiều nguyên nhân | "TBD" | "TODO" | P2-02, P0-01 | - |
| P2-04 | P2 | Đạt gate NoData, missing channel, outside mapping, border và compressed/full-decode budget | "TBD" | "TODO" | P1-04, P2-01, P2-03 | - |
| P3-01 | P3 | Implement transform planner: target grid, source ROI, halo, pixel convention và mapping hai chiều/footprint | "TBD" | "TODO" | P0-01, P0-04 | - |
| P3-02 | P3 | Implement CPU warp baseline với internal float32/float64 numeric contract | "TBD" | "TODO" | P3-01, P2-02 | - |
| P3-03 | P3 | Implement profile-driven kernel, border, rounding, clipping, non-finite handling và cast output dtype | "TBD" | "TODO" | P3-02, P0-01 | - |
| P3-04 | P3 | Đạt strip/halo invariance và CPU golden-test tolerance cho image, masks, reason và mapping | "TBD" | "TODO" | P3-03, P2-04 | - |
| P3-05 | P3 | GPU backend chỉ được bật sau numeric parity và resource benchmark trên từng compute profile | "TBD" | "TODO" | P3-04, P1-05 | - |
| P4-01 | P4 | Implement staging writer cho image, validity, reason mask, output metadata và preprocess mapping | "TBD" | "TODO" | P2-04, P3-04 | - |
| P4-02 | P4 | Implement checksum, manifest complete và atomic rename; không publish artifact dở dang | "TBD" | "TODO" | P4-01, P1-02 | - |
| P4-03 | P4 | Implement artifact reader verify đầy đủ file, checksum, profile/calibration digest và schema | "TBD" | "TODO" | P4-02 | - |
| P4-04 | P4 | Fault injection crash/I/O/disk-full chứng minh source được giữ và artifact incomplete không mở được | "TBD" | "TODO" | P1-05, P4-02, P4-03 | - |
| P4-05 | P4 | Public preprocess_capture identity call chạy end-to-end và artifact verify/open/read chỉ qua facade | "TBD" | "TODO" | P0-06, P4-03 | - |
| P5-01 | P5 | Tạo InferenceAdapter độc lập với WarpBackend và đọc được artifact theo output layout profile | "TBD" | "TODO" | P4-03, P0-02 | - |
| P5-02 | P5 | Compatibility gate: band availability/order, grid semantics, output dtype/layout và engine fingerprint | "TBD" | "TODO" | P5-01, P0-02 | - |
| P5-03 | P5 | Adapter thực hiện patch/window, padding validity, normalization, tensor dtype và HWC-to-NCHW | "TBD" | "TODO" | P5-02 | - |
| P5-04 | P5 | Refactor inference_large_image_trt raw mode gọi public preprocessing facade đúng một lần; failure chặn TensorRT | "TBD" | "TODO" | P4-05, P5-03 | - |
| P5-05 | P5 | Train-inference parity cho từng cặp PreprocessingProfile + EngineInputSpec | "TBD" | "TODO" | P5-03, P5-04 | - |
| P5-06 | P5 | End-to-end source -> public module -> artifact -> fake TensorRT và TensorRT smoke test | "TBD" | "TODO" | P4-04, P5-05 | - |
| P5-07 | P5 | Reference artifact mode verify/open không re-warp; cấm private imports và duplicate production preprocessing | "TBD" | "TODO" | P0-06, P5-04, P5-06 | - |
| P5-08 | P5 | Tạo reference_preprocessed_inference.py; run() chỉ nhận PreprocessArtifact, không nhận path/ndarray và không đọc raw/ảnh phụ | "TBD" | "TODO" | P5-03, P5-07 | - |
| P6-01 | P6 | Implement PatchResultWriter append theo patch row, checksum và manifest | "TBD" | "TODO" | P5-08 | - |
| P6-02 | P6 | Ghi mapping reference, valid fraction, validity-reason summary, status và engine/profile fingerprint | "TBD" | "TODO" | P6-01, P5-02 | - |
| P6-03 | P6 | Recovery test missing/duplicate record, resume hoặc fail closed | "TBD" | "TODO" | P6-01 | - |
| P6-04 | P6 | Chặn decision khi artifact/patch result thiếu, invalid hoặc inference dở dang | "TBD" | "TODO" | P6-02, P6-03 | - |
| P7-01 | P7 | Golden campaign nhiều input/output dtype, layout, target grid, kernel, rounding và border | "TBD" | "TODO" | P3-04 | - |
| P7-02 | P7 | Validity/reason propagation campaign và non-finite/cast edge cases | "TBD" | "TODO" | P2-03, P3-03 | - |
| P7-03 | P7 | Fault-injection campaign cho trust, calibration, RAM/disk hai tầng, codec, timeout, reset và checksum | "TBD" | "TODO" | P1-05, P4-04, P6-03 | - |
| P7-04 | P7 | HIL benchmark/soak riêng cho Jetson Nano và Orin Nano theo ComputeProfile | "TBD" | "TODO" | P3-05, P5-08, P7-03 | - |
| P7-05 | P7 | Shadow-mode rollout review với georeferencing, DecisionPolicy và OBC/F' | "TBD" | "TODO" | P6-04, P7-04 | - |

## Core invariant checklist

Các invariant này phải được kiểm tra trong code review và test gate:

- [ ] Core chạy được với source image, calibration và PreprocessingProfile,
      không cần dataset/model/engine.
- [ ] Package cài/import được bằng public path preprocessing trong clean process;
      import không khởi tạo decoder, GPU, TensorRT hoặc ghi filesystem.
- [ ] Reference chỉ import package root và production path không gọi private
      preprocessing backend.
- [ ] Reference raw mode gọi preprocessing đúng một lần; artifact mode không
      re-warp; typed failure chặn khởi tạo TensorRT.
- [ ] reference_preprocessed_inference.run() chỉ nhận PreprocessArtifact; không
      nhận path/ndarray/file handle và không đọc raw hoặc ảnh phụ.
- [ ] Reference consumer chỉ dùng artifact.open()/ModelGridReader; static import
      test chặn TiffReader, PIL, tifffile, rasterio và source-reader backend.
- [ ] Không có hard-code tên model, output dtype, normalization, /65535,
      patch size hoặc tensor layout trong WarpBackend.
- [ ] Target grid, kernel, rounding, border, output layout/dtype và validity đều
      được resolve từ PreprocessingProfile.
- [ ] Warp nội bộ dùng float; cast chỉ xảy ra ở output boundary theo profile.
- [ ] validity_yx và validity_reason_yx độc lập với pixel color và giữ đủ
      nguyên nhân khi có nhiều lỗi.
- [ ] Normalization, band reorder, patching và HWC-to-NCHW chỉ nằm ở adapter.
- [ ] Artifact chỉ publish sau checksum và atomic manifest commit.
- [ ] Resource admission có cả preflight bound và runtime/commit check.
- [ ] Mọi lỗi đều có reason code và safe action RETAIN_FOR_GROUND.

## Gate dashboard

| Gate | Điều kiện mở gate | Status | Evidence / link |
|---|---|---|---|
| G0 - Contract/public API freeze | P0-01..P0-06 DONE; schema, facade, packaging và clean import đã review | "TODO" | - |
| G1 - Trust/admission | P1-01..P1-05 DONE; reject trước allocation và fail closed sau allocation | "TODO" | - |
| G2 - Geometric transform | P2-01..P3-04 DONE; golden parity và strip invariance | "TODO" | - |
| G3 - Atomic artifact/public call | P4-01..P4-05 DONE; crash/I/O an toàn và facade identity call hoạt động | "TODO" | - |
| G4 - Reference integration | P5-01..P5-08 DONE; artifact-only consumer, no image reader/private imports và parity | "TODO" | - |
| G5 - Recovery/decision | P6-01..P6-04 DONE; invalid/incomplete không thành decision | "TODO" | - |
| G6 - Hardware rollout | P7-01..P7-05 DONE; HIL, shadow mode và OBC review | "TODO" | - |

Không promote gate tiếp theo khi gate trước còn BLOCKED, trừ khi có decision
log ghi rõ phạm vi thử nghiệm bị giới hạn và safe action.

## Blockers và risks

| ID | Blocker/risk | Impact | Mitigation / next action | Owner | Status |
|---|---|---|---|---|---|
| B-01 | Chưa chốt calibration flight và chiều mapping | Không thể chứng minh geometric correctness | Cung cấp calibration bundle thật; chỉ dùng identity/affine cho boundary test | "TBD" | "OPEN" |
| B-02 | Chưa chốt output storage/codec cho từng compute profile | Không thể finalize disk bound và artifact writer | Chốt format/codec trong P0 và benchmark decode path | "TBD" | "OPEN" |
| B-03 | Chưa có engine manifest/input spec cho deployment | Chưa thể chạy adapter parity | Tạo manifest fixture; giữ core test độc lập engine | "TBD" | "OPEN" |
| B-04 | Chưa có RAM/disk/thermal benchmark trên board đích | Không được enable flight profile | Chạy HIL và ghi peak usage, latency, power, thermal margin | "TBD" | "OPEN" |
| B-05 | Reference hiện đọc/normalize/pad/transpose trực tiếp và repository chưa có package metadata | Có nguy cơ duy trì hai production preprocessing path | Tạo facade/pyproject ở P0.5, migrate reference ở P5 và khóa bằng import/call-count tests | "TBD" | "OPEN" |

## Decision log

| Date | ID | Decision | Rationale / impact | Owner |
|---|---|---|---|---|
| 2026-07-24 | D-01 | Tách PreprocessingProfile khỏi EngineInputSpec | Core nắn ảnh độc lập model/dataset; model-specific behavior ở adapter | "TBD" |
| 2026-07-24 | D-02 | Cho phép output dtype/layout/kernel/grid theo profile | Không biến cấu hình deployment hiện tại thành giới hạn thuật toán | "TBD" |
| 2026-07-24 | D-03 | Giữ validity mask và thêm validity-reason mask | Phân biệt pixel invalid và nguyên nhân để recovery/decision/audit | "TBD" |
| 2026-07-24 | D-04 | Giữ atomic artifact, trust bundle, admission hai tầng và state machine | Bảo đảm fail-closed, provenance và giữ source khi run không hoàn tất | "TBD" |
| 2026-07-24 | D-05 | Public import path là preprocessing; reference chỉ gọi facade | Biến core thành module dùng lại được và ngăn duplicate/private integration | "TBD" |
| 2026-07-24 | D-06 | Thêm reference_preprocessed_inference.py chỉ nhận PreprocessArtifact | Consumer mẫu không đọc raw/ảnh phụ và chỉ dùng output đã xác thực của module nắn ảnh | "TBD" |

## Update log

| Date | Change | Updated by |
|---|---|---|
| 2026-07-24 | Tạo tracker và đồng bộ backlog với plan module nắn ảnh | "TBD" |
| 2026-07-24 | Bổ sung public module, packaging và reference-integration tasks/gates | "TBD" |
| 2026-07-24 | Bổ sung artifact-only reference consumer và contract test | "TBD" |

Khi cập nhật tracker, sửa đồng thời Snapshot, Work items, Gate dashboard,
Blockers và Update log; không xóa lịch sử decision hoặc evidence cũ.
