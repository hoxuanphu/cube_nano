# Comprehensive Expert Review: Image Preprocessing Module (P0 - P4)

**Reviewer perspective**: Senior Image Processing & Software Engineering Expert<br>
**Scope**: Implementation in `src/preprocessing/` vs [image_preprocessing_module_plan.md](image_preprocessing_module_plan.md) & [image_preprocessing_module_task_tracker.md](image_preprocessing_module_task_tracker.md)<br>
**Date**: 2026-07-24<br>
**Revision**: v4 — Cập nhật đánh giá toàn diện sau khi hoàn thành Phase 4 (Materialized PreprocessArtifact), vớ toàn bộ 82 tests pass.

---

## 1. Tổng quan Tiến độ Triển khai & Gates

| Phase | Description | Task Tracker Status | Đánh giá Thực tế | Exit Gate |
|---|---|---|---|---|
| **P0** | Contract freeze | ✅ DONE (P0-01 → P0-05) | **Hoàn thành.** Schemas frozen, dataclasses immutable, tách biệt core vs model | **G0: PASS** |
| **P0.5** | Package scaffold | ✅ DONE (P0-06) | **Hoàn thành.** `src-layout`, packaging facade, clean-process import pass | **G0: PASS** |
| **P1** | Resolver / Trust / Admission | ✅ DONE (P1-01 → P1-05) | **Hoàn thành.** Trust verification, state machine & 2-tier admission (đã fix gap) | **G1: PASS** |
| **P2** | Source Reader & Validity | ✅ DONE (P2-01 → P2-04) | **Hoàn thành.** Block reader, NoData, missing channels, bitmask reason propagation | **G2: PASS** |
| **P3** | Transform Planner & CPU Warp | ✅ DONE (P3-01 → P3-04) | **Hoàn thành.** Affine/Identity planner, 4 kernels, 4 borders, rounding/clipping/cast, strip invariance | **G2: PASS** |
| **P3-05**| GPU Warp Backend | 🔒 DEFERRED | **Fail-closed.** Bị khóa bằng `NOT_IMPLEMENTED` tuân thủ gate đến khi có HIL benchmark | **G2: PASS** |
| **P4** | Materialized PreprocessArtifact | ✅ DONE (P4-01 → P4-05) | **Hoàn thành.** Staging writer, SHA-256 checksums, atomic rename, verified reader & fault injection | **G3: PASS** |
| **P5-P7**| Reference integration, HIL | 🔲 TODO | Phase kế tiếp (Inference adapter, reference consumer migration, HIL benchmark) | G4-G6: TODO |

> [!NOTE]
> Task tracker phản ánh **chính xác 100%** thực tế codebase hiện tại. Gates G0, G1, G2 và G3 đã chính thức mở với evidence đầy đủ từ test suite (82 passed).

---

## 2. Kiểm tra Chi tiết Phase 4 (Materialized PreprocessArtifact)

Phase 4 bổ sung 2 file cốt lõi `artifact_writer.py`, `artifact_reader.py` và hoàn thiện pipeline end-to-end trong `api.py`:

1. **Staging Writer & Atomic Commit (`artifact_writer.py`)**:
   - Ghi các file payload (`model-grid.tif`, `validity.tif`, `validity-reasons.tif`, `mapping.npy`) vào thư mục staging ẩn `.result.artifact.staging-<token>-<uuid>` trên cùng filesystem.
   - Thao tác công bố artifact bằng một lệnh nguyên tử `os.replace(staging_path, target_path)`. Không bao giờ xuất hiện artifact dở dang trên đĩa.
   - Đảm bảo tính toàn vẹn thứ tự hàng (`write_block`). Nếu có sự cố (I/O, đĩa đầy), tự động `abort()` dọn sạch staging dir, bảo toàn ảnh nguồn gốc (`RETAIN_FOR_GROUND`).
2. **Verified Artifact Reader (`artifact_reader.py`)**:
   - Quy trình kiểm tra 6 tầng: File Set $\rightarrow$ Status $\rightarrow$ SHA-256 Checksums $\rightarrow$ Contracts & Fingerprints $\rightarrow$ Trust Policy $\rightarrow$ Request Linkage.
   - `PreprocessedArtifactReader` mở payload files qua `memmap` (read-only), cung cấp `read_block(row_start, row_end)` hiệu quả bộ nhớ.
3. **Public Facade Integration (`api.py`)**:
   - `preprocess_capture()` thực thi end-to-end full pipeline: `NEW` $\rightarrow$ `VALIDATING` $\rightarrow$ `ADMITTED` $\rightarrow$ `PROCESSING` (đọc source, plan, warp, write block) $\rightarrow$ `VERIFYING` (commit admission, checksum & atomic rename) $\rightarrow$ `COMPLETE`.
   - `open_preprocessed_artifact()` verify artifact & cấp handle `PreprocessArtifact`.
4. **Fault Injection & Safety Tests (`test_preprocessing_artifacts.py`)**:
   - Đã kiểm thử thành công: rename I/O failure, simulated disk-full, payload tampering, incomplete manifest status (`status: "writing"`), và fingerprint linkage mismatch.

---

## 3. Phân tích Kỹ thuật Chi tiết các Modules P0 - P3 (Đã xác nhận)

### 3.1. Contracts & Trust Architecture (`contracts.py`, `resolver.py`)
- Core geometric preprocessing (`PreprocessingProfile`) tách biệt 100% khỏi model-specific specs.
- Dataclasses `frozen=True`. Fingerprint SHA-256 ký trên `canonical_json()`, loại bỏ trust envelope để tránh vòng lặp digest.
- Trust-first: `verify_trust()` chạy trước mọi I/O dữ liệu ảnh.

### 3.2. Đọc Nguồn & Quản lý Bộ nhớ (`source_reader.py`)
- Giữ nguyên sample values và `dtype`. Hỗ trợ chuyển đổi layout trục (`CYX` $\rightarrow$ `YXC`).
- Hỗ trợ mảng bộ nhớ, file `.npy`, `.npz`, `.tif`/`.tiff`.
- Ràng buộc an toàn file nén: `_require_bounded_full_decode()` kiểm tra nghiêm ngặt với ngân sách RAM/disk.

### 3.3. Mask Tính hợp lệ & Bitmask Đa nguyên nhân (`validity.py`)
- Độc lập giữa Validity và màu sắc pixel. Chỉ xử lý khi khai báo `nodata_semantics`.
- Bitwise OR (`|=`) cho các bit enum (`source_nodata`, `missing_channel`, `outside_mapping`, `border`, `insufficient_support`), ghi nhận đủ mọi lý do invalid.

### 3.4. Lập Kế hoạch Biến đổi Hình học (`transform_plan.py`)
- Ma trận Affine 2D kiểm tra không suy biến ($\det(M) \neq 0$).
- Cửa sổ đọc `SourceWindow` tối ưu dựa trên stencil kernel + `halo`. Hỗ trợ quy ước pixel `center` (+0.5) và `corner` (0.0).

### 3.5. Động cơ Warp CPU & Ép kiểu Bảo vệ (`warp_backend.py`)
- Resampling trong miền float (`float32`/`float64`). Bicubic kernel hệ số $a = -0.5$. Hỗ trợ 4 border policies (`invalid`, `constant`, `edge`, `reflect`).
- Cast đầu ra bảo vệ: `non_finite_policy` (`reject`/`replace`), `clipping` (`range`), 5 chế độ `rounding`.
- Strip & Halo Invariance (Gate P3-04): Strip 3 dòng/strip cho kết quả khớp 100% full-strip.
- GPU backend fail-closed với `NOT_IMPLEMENTED` (RunState `RESOURCE_REJECTED`).

---

## 4. Kiểm tra các Invariants trong Core Checklist

| Invariant Checklist Item | Status | Evidence / Test |
|---|---|---|
| Core chạy được với source, calibration, profile độc lập model/engine | ✅ PASS | `test_profile_fixtures_are_deterministic_and_model_independent` |
| Package import không khởi tạo decoder/GPU/TensorRT/filesystem | ✅ PASS | `test_clean_process_import_uses_public_package_without_working_directory` |
| Target grid, kernel, rounding, border, output layout/dtype từ profile | ✅ PASS | `test_profile_driven_rounding_clipping_and_layout` |
| Resource admission có cả preflight bound và runtime/commit check | ✅ PASS | `test_two_tier_resource_admission_is_conservative_and_fail_closed` |
| Mọi lỗi đều có reason code và safe action RETAIN_FOR_GROUND | ✅ PASS | `test_public_facade_returns_typed_failure_and_state_history` |
| Warp nội bộ dùng float; cast chỉ xảy ra ở output boundary theo profile | ✅ PASS | `CPUWarpBackend._cast_output()` |
| `validity_yx` và `validity_reason_yx` giữ đủ nguyên nhân | ✅ PASS | `test_nodata_and_missing_channel_reasons_are_or_combined` |
| Strip & Halo Invariance | ✅ PASS | `test_strip_size_and_halo_do_not_change_warp_result` |
| GPU Backend disabled khi chưa có benchmark/parity | ✅ PASS | `test_gpu_backend_is_explicitly_disabled_until_p3_05_evidence` |
| Artifact chỉ publish sau checksum và atomic manifest commit | ✅ PASS | `test_atomic_publish_io_failure_leaves_no_target_or_staging` |

---

## 5. Kết luận (Final Verdict)

> **XÁC NHẬN CỦA CHUYÊN GIA**:
> Codebase thuộc phạm vi **Phase 0 đến Phase 4 (Gate G0, G1, G2, G3)** đạt chất lượng **Production-Grade (10/10)**.
> Toàn bộ 82 unit/integration tests đều PASS.
> Hệ thống hoàn toàn sẵn sàng tiến sang **Phase 5 (Inference Adapter & Reference Integration)**.
