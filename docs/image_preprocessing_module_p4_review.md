# Đánh giá & Kiểm tra Implementation Phase 4: Materialized PreprocessArtifact

**Người kiểm tra**: Chuyên gia Xử lý Ảnh & Kỹ sư Phần mềm Cao cấp<br>
**Phạm vi kiểm tra**: Code Phase 4 (`artifact_writer.py`, `artifact_reader.py`, `api.py`) đối chiếu với [image_preprocessing_module_plan.md](image_preprocessing_module_plan.md) và [image_preprocessing_module_task_tracker.md](image_preprocessing_module_task_tracker.md)<br>
**Ngày kiểm tra**: 2026-07-24<br>

---

## 1. Tổng quan Trạng thái Triển khai Phase 4

Toàn bộ 5 work items thuộc **Phase 4 (P4-01 đến P4-05)** đã được triển khai hoàn chỉnh, vượt qua toàn bộ 82 unit/integration tests (bao gồm 7 fault-injection & lifecycle tests mới trong `test_preprocessing_artifacts.py`). Gate **G3 (Atomic Artifact / Public Call)** chính thức mở.

| Task ID | Phase | Nội dung Work Item | Trạng thái Code | Đánh giá |
|---|---|---|---|---|
| **P4-01** | P4 | Staging Writer cho Payload Files (`artifact_writer.py`) | ✅ DONE | Ghi `model-grid.tif`, `validity.tif`, `validity-reasons.tif`, `mapping.npy` qua memmap, đúng layout/dtype profile |
| **P4-02** | P4 | Checksum, Manifest Complete & Atomic Rename (`artifact_writer.py`) | ✅ DONE | Ghi `output.json`, `preprocess.json`, SHA-256 checksums, `status: complete` và `os.replace()` atomic rename |
| **P4-03** | P4 | Verified Artifact Reader (`artifact_reader.py`) | ✅ DONE | Verify đầy đủ file set, status, SHA-256 checksum, trust, fingerprints linkage và cấp `PreprocessedArtifactReader` |
| **P4-04** | P4 | Fault Injection Tests (`test_preprocessing_artifacts.py`) | ✅ DONE | Test rename failure, simulated disk-full, payload tampering, incomplete manifest và fingerprint mismatch |
| **P4-05** | P4 | Public Facade End-to-End Integration (`api.py`) | ✅ DONE | `preprocess_capture()` chạy end-to-end full pipeline từ source $\rightarrow$ warp $\rightarrow$ atomic artifact; `open_preprocessed_artifact()` verify & open qua facade |

---

## 2. Phân tích Kỹ thuật Chi tiết các Thành phần P4

### 2.1. [artifact_writer.py](file:///d:/AI20K/cube_nano/src/preprocessing/artifact_writer.py) (21.0 KB)

**Ưu điểm kỹ thuật:**
- **Thao tác AtomicRename trên cùng Filesystem**: `ArtifactWriter` tạo thư mục staging ẩn (ví dụ: `.result.artifact.staging-<token>-<uuid>`) nằm cùng parent directory với target. Việc công bố artifact được thực hiện bằng một lệnh duy nhất `os.replace(staging_path, target_path)`, đảm bảo hệ thống không bao giờ xuất hiện artifact dở dang.
- **Quản lý Bộ nhớ Memmap cho Payload Files**:
  - Ghi mảng ảnh `model-grid.tif` qua `tifffile.memmap` theo đúng layout (`YXC`, `CYX`, `YX`) và `output_dtype`.
  - Ghi mask hợp lệ `validity.tif` và mask nguyên nhân `validity-reasons.tif`.
  - Ghi mảng tọa độ ánh xạ `mapping.npy` qua `np.lib.format.open_memmap`.
- **Kiểm soát Thứ tự Ghi Nối tiếp (`write_block`)**: Kiểm tra nghiêm ngặt `row_start == self._next_row`. Nếu caller truyền lệch dòng hoặc ghi trùng, writer lập tức nâng lỗi `PreprocessError` (dùng `PATCH_RESULT_DUPLICATE` hoặc `ARTIFACT_INCOMPLETE`).
- **Checksum SHA-256 Không Nạp RAM (`sha256_path`)**: Đọc từng chunk 1MB để tính SHA-256 checksum cho toàn bộ payload files.
- **Tự động Dọn dẹp Rác khi Thất bại (`abort`)**: Nếu quá trình nắn hoặc finalize gặp sự cố (I/O error, đĩa đầy, exception), `abort()` được gọi để đóng toàn bộ memmap handle và xóa sạch thư mục staging, giữ nguyên ảnh nguồn gốc với safe action `RETAIN_FOR_GROUND`.

### 2.2. [artifact_reader.py](file:///d:/AI20K/cube_nano/src/preprocessing/artifact_reader.py) (22.0 KB)

**Ưu điểm kỹ thuật:**
- **Quy trình Verify 6 Tầng Rõ ràng (`verify_artifact`)**:
  1. *File Set Verification*: Kiểm tra sự tồn tại của đầy đủ 7 file bắt buộc (`REQUIRED_ARTIFACT_FILES`).
  2. *Status & Schema Verification*: Kiểm tra `manifest.json` có `status == "complete"` và `complete == True`.
  3. *Checksum Verification*: Tính toán lại SHA-256 của từng file payload và đối chiếu 100% với `manifest.json`.
  4. *Contract & Fingerprint Verification*: Phục hồi embedded contracts (`capture`, `profile`, `calibration`) từ `preprocess.json`, kiểm tra fingerprints khớp với manifest.
  5. *Trust Verification*: Xác minh chữ ký, issuer, generation, và hạn dùng theo `TrustPolicy` của caller.
  6. *Request Linkage Verification*: So sánh fingerprints thực tế với `expected_source_fingerprint`, `expected_profile_fingerprint`, `expected_calibration_fingerprint` trong request.
- **Đọc Block Hiệu quả Bộ nhớ (`PreprocessedArtifactReader`)**:
  - Mở payload files bằng `memmap` chế độ read-only (`mode="r"`).
  - Hàm `read_block(row_start, row_end)` trích xuất chính xác dải dòng được yêu cầu mà không nạp toàn bộ ảnh vào bộ nhớ RAM.

### 2.3. [api.py](file:///d:/AI20K/cube_nano/src/preprocessing/api.py) (Tích hợp End-to-End P4)

**Ưu điểm kỹ thuật:**
- **Chuyển đổi Trạng thái Đúng Quy trình (State Machine Lifecycle)**:
  - `preprocess_capture()` đi qua đủ 6 trạng thái: `NEW` $\rightarrow$ `VALIDATING` $\rightarrow$ `ADMITTED` $\rightarrow$ `PROCESSING` (đọc source, plan, warp, write block) $\rightarrow$ `VERIFYING` (commit admission check, finalize writer) $\rightarrow$ `COMPLETE`.
  - Trả về đối tượng `PreprocessArtifact` chứa `artifact_path` và `ArtifactManifest`.
- **`open_preprocessed_artifact()`**: Thực thi xác minh `verify_artifact()` qua facade và trả về `PreprocessArtifact` đã qua kiểm duyệt.
- **`PreprocessArtifact.open()`**: Phương thức tiện ích mở `PreprocessedArtifactReader` để consumer truy cập dữ liệu đã nắn.

---

## 3. Đánh giá Kiểm thử Fault Injection (Test Evidence)

Toàn bộ 82 tests trong test suite đều PASS. Các bài test kiểm tra khía cạnh an toàn của Phase 4 (`tests/test_preprocessing_artifacts.py`):

1. `test_artifact_metadata_stays_preprocessing_only`: Xác nhận `preprocess.json` và `output.json` hoàn toàn không chứa các thuộc tính của model/engine (`patch_size`, `normalization`, `tensor_dtype`, `engine`, `model_fingerprint`).
2. `test_checksum_mismatch_is_rejected_and_source_is_retained`: Giả lập làm sai lệch 1 byte dữ liệu trong `model-grid.tif` $\rightarrow$ `open_preprocessed_artifact()` từ chối mở với lỗi `ARTIFACT_CHECKSUM_MISMATCH`, nguồn ảnh gốc được bảo toàn.
3. `test_incomplete_or_non_complete_manifest_cannot_open`: Thư mục artifact dở dang / thiếu file từ chối mở với lỗi `ARTIFACT_INCOMPLETE`.
4. `test_manifest_status_is_fail_closed_even_when_payload_files_exist`: Artifact đủ file nhưng `status != "complete"` (ví dụ `status: "writing"`) bị từ chối fail-closed.
5. `test_artifact_expected_fingerprint_mismatch_is_rejected`: Yêu cầu mở artifact với fingerprint không khớp bị từ chối với lỗi `FINGERPRINT_LINKAGE_MISMATCH`.
6. `test_atomic_publish_io_failure_leaves_no_target_or_staging`: Giả lập lỗi `os.replace` (lỗi I/O khi rename) $\rightarrow$ trả lỗi `IO_ERROR`, giữ nguyên ảnh nguồn, dọn dẹp sạch staging dir, không để lại folder dở dang.
7. `test_disk_full_during_finalize_is_not_published`: Giả lập lỗi đĩa đầy trong quá trình đóng file $\rightarrow$ dọn dẹp staging, từ chối publish, giữ nguyên ảnh nguồn.

---

## 4. Kết luận & Khuyến nghị cho Phase 5

### Verdict
> **Phase 4 đã hoàn thành xuất sắc, đạt tiêu chuẩn Production-Grade (10/10 cho scope P4)**.
> Cơ chế materialization, atomic commit, checksum verification, fault-injection handling và public facade lifecycle hoạt động hoàn hảo, đảm bảo không bao giờ publish hoặc mở một artifact dở dang/bị sai lệch.

### Khuyến nghị cho Phase 5 tiếp theo (Inference Adapter Integration):
1. **Triển khai `InferenceAdapter` (`adapter.py`)**: Độc lập với `WarpBackend`, chỉ tiêu thụ `PreprocessArtifact` qua `open_preprocessed_artifact()` / `ModelGridReader`.
2. **Kiểm tra Compatibility Gate (P5-02)**: Đối chiếu `ModelCompatibilityProfile` (band order, patch size, tensor dtype) với `ArtifactManifest` trước khi cấp tensor cho TensorRT engine.
3. **Refactor Reference Script (`inference_large_image_trt.py`)**: Chuyển đổi consumer mẫu sang gọi public preprocessing facade, cấm trùng lặp logic xử lý ảnh thủ công.
