# Đưa cải thiện từ notebook về `src/`

Backport các cải tiến đã có trong `kaggle_train_cloud_model.ipynb` về các file trong `src/`, đồng thời thêm configurable threshold và tile-based reading cho ảnh lớn.

## Proposed Changes

### Ưu tiên 1: Backport scene-level split, pos_weight, seed

---

#### [MODIFY] split_dataset.py (`src/data/split_dataset.py`)

Thay đổi logic split từ **file-level random** sang **scene-level split** (giống notebook section 5):
- Thêm hàm `scene_id_from_patch()` — trích scene ID từ tên file patch (dựa trên `_p` separator)
- Thêm hàm `collect_scene_files()` — nhóm tất cả patch theo scene ID
- Sửa `main()` để shuffle và split theo danh sách scene, không phải danh sách file
- Giữ nguyên interface CLI (`--src_dir`, `--out_dir`, `--val_ratio`, `--test_ratio`, `--seed`, `--move`)
- Thêm output manifest JSON (giống notebook) để log scene nào vào split nào

---

#### [MODIFY] train.py (`src/train.py`)

Backport 3 cải tiến từ notebook:

**a) Seed** — Thêm hàm `set_seed(seed)` (random, numpy, torch, cuda) và gọi ở đầu `main()`. Thêm arg `--seed` (default 42).

**b) pos_weight** — Thêm arg `--use_pos_weight` (default True). Khi bật, tính `pos_weight = len(clear) / len(cloud)` từ dataset và truyền vào `BCEWithLogitsLoss(pos_weight=...)`.

**c) AMP** — Thêm arg `--amp` (default True khi CUDA). Dùng `torch.amp.autocast` + `GradScaler` giống notebook section 9.

---

### Ưu tiên 2: Configurable threshold

---

#### [MODIFY] train.py (`src/train.py`)

- Thêm arg `--threshold` (default 0.5) cho `calculate_metrics()`

#### [MODIFY] eval.py (`src/eval.py`)

- Thêm arg `--threshold` (default 0.5)
- Truyền threshold vào `calculate_metrics()` thay vì hardcode

#### [MODIFY] inference_tensorrt.py (`src/inference_tensorrt.py`)

- Thêm `threshold` parameter vào constructor `CloudTRTInfer` (default 0.5)
- Sử dụng `self.threshold` thay vì hardcode `0.5` ở `infer()` và `infer_batch()`

#### [MODIFY] inference_large_image_trt.py (`src/inference_large_image_trt.py`)

- Thêm arg `--threshold` (default 0.5) và truyền xuống `CloudTRTInfer`

---

### Ưu tiên 3: Tile-based reading cho ảnh lớn

---

#### [MODIFY] inference_large_image_trt.py (`src/inference_large_image_trt.py`)

Thay vì đọc toàn bộ ảnh vào RAM rồi sliding window, sẽ:
- Với file TIFF: dùng `tifffile.imread` với slicing theo page/tile để chỉ đọc vùng cần xử lý
- Với file khác (npy, HDF5, NetCDF): dùng memory-mapped read hoặc chunk read tương ứng
- Thêm hàm `_read_image_region(path, row_start, row_end, col_start, col_end)` — đọc chỉ 1 strip/tile
- Sửa `process_large_image()`: đọc metadata (shape) trước, rồi iterate theo row strip, mỗi strip chỉ đọc vùng cần thiết cho 1 hàng patch

> **Lưu ý:** Cách tiếp cận tile-based phụ thuộc vào cách file TIFF gốc được lưu (tiled vs stripped). Nếu file TIFF lưu stripped theo row, đọc theo row strip sẽ hiệu quả. Nếu file không hỗ trợ random access tốt (ví dụ compressed NetCDF), sẽ fallback về đọc toàn bộ nhưng in warning.

---

### Bổ sung: Cập nhật review

---

#### [MODIFY] project_review.md (`docs/project_review.md`)

Cập nhật lại bài review theo feedback của user:
- Bổ sung ngữ cảnh notebook đã xử lý pos_weight, seed, scene-level split
- Sửa nhận xét normalization cho chính xác (integer path dùng `iinfo`, chỉ float path dùng `max()`)
- Giữ nguyên các nhận xét đúng, đánh dấu rõ trạng thái "đã có trong notebook, cần backport về src"

---

## Sửa Lỗi Bổ Sung (Bugfixes)

Sau khi review chi tiết, đã triển khai thêm các bản vá lỗi sau để đảm bảo Production Ready:

1. **Fix Memory-efficient TIFF read** (`src/inference_large_image_trt.py`):
   Sử dụng `tiff.memmap(path)` thay vì load toàn bộ ảnh với `.imread()`. Điều này giúp khắc phục hoàn toàn rủi ro OOM trên Jetson Nano cho ảnh uncompressed TIFF.

2. **Fix Channel-first Slicing** (`src/inference_large_image_trt.py`):
   Các file dạng numpy, HDF5, NetCDF thường có dạng `(C, H, W)`. Thêm logic `_is_channel_first()` để cắt đúng trục (`arr[:, start:end, :]`) và trả về layout chuẩn `(H, W, C)` cho TensorRT xử lý.

3. **Ngăn Data Leakage do Tái sinh Split** (`src/data/split_dataset.py`):
   Thêm cờ `--force`. Khi thư mục output đã có các file patch `.npy` (chạy script trước đó), hệ thống sẽ cảnh báo lỗi và ngừng chạy trừ khi có `--force` (sẽ xoá sạch các file `.npy` cũ).

4. **Đảm bảo Tính Deterministic thực sự** (`src/train.py`):
   Set `torch.backends.cudnn.benchmark = False` và `torch.backends.cudnn.deterministic = True` để kết quả huấn luyện hoàn toàn giống nhau giữa các lần chạy với cùng một seed.

5. **Lưu Provenance cho Kết quả Eval** (`src/eval.py`):
   JSON output của quá trình test/eval đã được gom nhóm lại thành `config` (chứa model_path, threshold, channels) và `metrics`. Điều này giúp không bị lạc thông tin khi tuning threshold.

6. **Ngưỡng loại ảnh lớn theo cloud coverage** (`src/inference_large_image_trt.py`):
   Sau khi ghép mask từ các tile, tính tỷ lệ diện tích bị dự đoán là cloud. Mặc định ảnh bị đánh dấu loại khi tỷ lệ này đạt `0.60`.
   `--discard-cloudy` sẽ xóa mask output đã tạo cho ảnh bị loại; file ảnh nguồn không bị xóa.

---

## Verification Plan

### Automated Tests
- Chạy `python src/train.py --help` để verify args mới không conflict
- Chạy `python src/eval.py --help` để verify arg `--threshold`
- Chạy `python src/data/split_dataset.py --help` để verify scene-level split args
- Unit test: tạo dummy patches với tên `sceneA_p0.npy`, `sceneA_p1.npy`, `sceneB_p0.npy` → verify scene-level split giữ patches cùng scene trong cùng split
- Unit test: xác nhận cloud coverage `60%` bị loại và `59%` được giữ lại.

### Manual Verification
- Review diff để đảm bảo không break backward compatibility (tất cả args mới đều có default giữ hành vi cũ)
- Kiểm tra notebook vẫn hoạt động independent (không import từ `src/`)
