# Đánh giá mức độ sẵn sàng triển khai phương án đọc TIFF nén

## Kết luận

Bản kế hoạch hiện tại đã nhất quán và gần đủ điều kiện triển khai Phase 1. Các vấn đề lớn từ hai vòng review trước về đặc tả cũ/mới, channel semantics, RAM/disk budget, LRU cache, CLI defaults, streaming exit gate, path policy và verification plan hầu hết đã được xử lý.

Các điểm còn lại tập trung vào tính chính xác của TIFF bốn band, hành vi channel mapping, liên kết giữa sidecar và TensorRT engine, disk guard trên nhiều filesystem và cách đo memory budget.

## Phát hiện

### 1. Mức cao: Quy tắc `ExtraSamples` đang loại cả RGBNIR hợp lệ

Kế hoạch cho phép ảnh bốn band khi có sidecar hoặc `channel_mapping`, nhưng đồng thời từ chối mọi TIFF bốn sample có `ExtraSamples`.

Theo cấu trúc TIFF, sample thứ tư sau RGB thường được biểu diễn trong `ExtraSamples`. Kiểm tra với `tifffile 2026.6.1` cho một TIFF RGBNIR dạng `YXC` cho kết quả:

```text
SamplesPerPixel=4
ExtraSamples=UNSPECIFIED
PhotometricInterpretation=RGB
```

Do đó quy tắc hiện tại sẽ từ chối ảnh bốn band này ngay cả khi sidecar hoặc mapping đã xác nhận sample thứ tư là NIR.

Quy tắc đề xuất:

- Luôn từ chối `ASSOCALPHA` và `UNASSALPHA`.
- Cho phép đúng một `UNSPECIFIED` sample thứ tư chỉ khi sidecar hoặc mapping tường minh xác nhận nó là NIR.
- Không có sidecar/mapping thì mọi ảnh bốn sample đều bị từ chối.
- Các `ExtraSamples` khác hoặc nhiều hơn một extra sample phải bị từ chối nếu chưa có contract riêng.

Test cần phân biệt tối thiểu:

- `UNSPECIFIED + RGBNIR mapping` được chấp nhận.
- `UNASSALPHA` bị từ chối.
- `ASSOCALPHA` bị từ chối.
- `UNSPECIFIED` nhưng thiếu mapping/sidecar bị từ chối.

### 2. Mức cao: Chưa chốt mapping có reorder dữ liệu hay chỉ validation

Kế hoạch yêu cầu chuyển layout về HWC nhưng không tự động đổi thứ tự kênh. Đồng thời, CLI lại cho phép mapping tùy ý như:

```text
red=2,green=1,blue=0
```

Nếu mapping chỉ dùng để validation, output vẫn ở thứ tự vật lý và model nhận sai kênh. Nếu mapping được dùng để reorder, tài liệu cần nói rõ đây là reorder tường minh do người dùng yêu cầu, không phải suy đoán tự động.

Contract đề xuất:

- Không có mapping: giữ thứ tự vật lý và chỉ chấp nhận khi metadata chứng minh thứ tự đó đã canonical.
- Có mapping hoặc sidecar: reader select/reorder về `[R,G,B]` hoặc `[R,G,B,NIR]` trước khi trả dữ liệu.
- Reorder được thực hiện trên strip hoặc patch, không tạo bản sao toàn ảnh.
- `reader.band_order` mô tả thứ tự output sau mapping.
- Mapping identity và non-identity đều phải có test pixel correctness.

TensorRT contract cũng phải lấy từ binding thực tế, không chỉ tin tham số `channels` trên CLI. `CloudTRTInfer._prepare_input()` hiện có thể pad hoặc truncate kênh. Phase 1 nên đổi thành exact channel validation và fail khi không khớp, nếu không lỗi upstream vẫn có thể bị che giấu.

Test cần bao gồm:

- TIFF BGR với mapping về RGB tạo đúng pixel canonical.
- Mapping trùng index hoặc thiếu role bị từ chối.
- Reader output channel count khác engine binding bị từ chối.
- TensorRT runtime không tự pad/truncate channel mismatch.

### 3. Mức trung bình: Sidecar chưa được ràng buộc với input và engine

Sidecar mẫu có axes, band order, dtype và normalization nhưng chưa quy định cách xác minh sidecar thuộc đúng TIFF và đúng TensorRT input contract.

Nên bổ sung:

- `schema_version`.
- `input_spec_id` hoặc normalization contract ID chuẩn hóa.
- Kiểm tra axes, dtype, shape, số band và band order trong sidecar khớp metadata TIFF thực tế.
- Kiểm tra `input_spec_id`/normalization khớp sidecar hoặc manifest của TensorRT engine.
- Artifact ID hoặc checksum/fingerprint để tránh dùng nhầm sidecar của file khác.
- Quy tắc precedence khi có đồng thời `--input_sidecar` và `--channel_mapping`.
- Nếu hai nguồn metadata mâu thuẫn thì fail; không âm thầm ưu tiên một nguồn.

Nếu Phase 1 chưa triển khai fixed normalization theo `InputSpec`, kế hoạch phải nói rõ trường `normalization` chỉ được validate hay thực sự điều khiển `_normalize_patch()`. Không nên ghi nhận normalization ID nhưng vẫn âm thầm dùng heuristic normalization khác.

### 4. Mức trung bình: Disk guard cần tính riêng theo filesystem

`--tiff_cache_dir` có thể nằm trên filesystem khác với `out_mask`. User-provided mask cache cũng có thể nằm trên một filesystem thứ ba. Công thức hiện tại gộp decoded cache và output allowance thành một `disk_required`, nên có thể kiểm tra nhầm dung lượng của một device.

Cần nhóm allocation theo filesystem/device và kiểm tra riêng:

```text
cache filesystem:
    decoded source cache
    + cache-specific temporary files
    + cache headroom

output filesystem:
    output mask temp
    + internal mask cache nếu cùng device
    + output headroom

user mask-cache filesystem:
    mask cache requirement
    + headroom
```

Nếu các path resolve về cùng device thì có thể cộng nhu cầu và kiểm tra một lần. Nếu khác device thì mỗi device phải vượt guard riêng trước khi tạo hoặc truncate file.

Test cần dùng mock filesystem accounting hoặc injectable disk-usage provider để bao phủ trường hợp cache đủ chỗ nhưng output filesystem thiếu chỗ, và ngược lại.

### 5. Mức trung bình: Hai chế độ `auto` còn mơ hồ

CLI có cả:

```text
--tiff_read_mode auto|stream|full
--tiff_cache_mode auto|ram|disk
```

Phần cache-selection policy dùng cụm từ `chế độ auto` nhưng chưa nói rõ đang chỉ read mode hay cache mode.

Interaction contract đề xuất:

| Read mode | Hành vi | Vai trò của cache mode |
|---|---|---|
| `auto` | Thử memmap, sau đó block backend, cuối cùng one-time cache | Chỉ điều khiển loại cache của fallback |
| `full` | Bỏ qua memmap và block backend, dùng one-time cache | Chọn `auto`, `ram` hoặc `disk` |
| `stream` | Chỉ memmap hoặc true block backend | Không dùng decoded cache; cache mode khác `auto` nên bị từ chối |

Các cache-size option không áp dụng trong `stream` nên được bỏ qua có cảnh báo hoặc bị từ chối theo một policy thống nhất. Tránh chấp nhận option nhưng không có tác dụng mà không thông báo.

### 6. Mức trung bình: Memory guard cần định nghĩa cách đo duy nhất

Kế hoạch cho phép đo guard sau khi TensorRT/CUDA khởi tạo hoặc trừ `runtime_reserve`, trong khi công thức peak luôn cộng `runtime_reserve`. Nếu implementation vừa trừ available memory vừa cộng reserve vào peak thì reserve bị tính hai lần.

Nên chốt một công thức duy nhất:

```text
operation_peak_without_reserve
    <= MemAvailable_after_TensorRT_initialization - runtime_reserve
```

Hoặc công thức tương đương:

```text
operation_peak_without_reserve + runtime_reserve
    <= MemAvailable_after_TensorRT_initialization
```

Chỉ chọn một cách biểu diễn trong code và test.

Ngoài ra cần định nghĩa:

- Provider đo `MemAvailable` trên Jetson và provider giả cho unit test.
- `decoder_workers` cố định hoặc giới hạn rõ ràng.
- `compressed_block_buffer` lấy từ max `databytecounts` của block liên quan.
- `decoded_block_buffer` lấy từ block geometry và decoded dtype.
- Cách giới hạn hoặc đo `mapped_working_set` của disk-backed memmap.
- Guard được đánh giá lại nếu backend hoặc worker count thay đổi sau capability check.

Nếu để `tifffile` tự chọn số worker, `decoder_peak` không còn là một bound có thể kiểm chứng.

### 7. Mức thấp: Validation của numeric CLI chưa hoàn toàn nhất quán

CLI mô tả `max_ram_cache_gib`, `max_disk_cache_gib` và `runtime_reserve_gib` là positive value, nhưng phần validation chỉ nói giá trị âm, NaN và vô hạn bị từ chối. Cần nói rõ giá trị `0` có hợp lệ hay không.

Đề xuất:

- Các RAM/disk/reserve limit phải lớn hơn 0.
- `tiff_block_cache_mib` được phép bằng 0 theo điều kiện block-cache contract.
- Chuyển GiB/MiB sang byte trước khi so sánh với `decoded_bytes`; tránh so sánh byte trực tiếp với giá trị GiB dạng float.

### 8. Mức thấp: Runtime cache directory chưa được ignore

Cache directory mặc định là `.cube_nano-cache` trong thư mục cha của `out_mask`, nhưng `.gitignore` hiện chưa có rule cho directory này. Nếu output nằm trong repository, runtime cache sẽ xuất hiện dưới dạng untracked files.

Khi triển khai Phase 1, nên thêm:

```gitignore
.cube_nano-cache/
```

Cleanup vẫn phải hoạt động đúng; `.gitignore` không thay thế ownership và deletion policy.

## Các nội dung đã đạt

Kế hoạch hiện đã xử lý tốt các yêu cầu quan trọng sau:

- Chỉ còn một đặc tả thống nhất, không còn phần cũ và phụ lục mâu thuẫn.
- Input contract tách shape/axes khỏi channel semantics.
- RGBNIR bắt buộc có sidecar hoặc mapping.
- Reader lifecycle và backend selection dựa trên capability được định nghĩa rõ.
- Phase 1 không phụ thuộc Zarr/Rasterio.
- Streaming backend có exit gate cụ thể.
- LRU cache có byte budget, cache key, eviction và one-decode criterion.
- RAM cache và disk-backed cache có mô hình riêng.
- CLI defaults, cache directory và path ownership đã được chốt.
- Output temp nằm cùng filesystem với output và dùng atomic replace.
- Verification plan bao phủ pixel, backend, RAM/disk, fault injection và Windows cleanup.
- Benchmark có corpus metadata, cold/warm run, median/p95 và số lần lặp.
- Phase 1, Phase 2 và Phase 3 có ranh giới triển khai rõ ràng.

## Thứ tự chỉnh sửa cuối đề xuất

1. Sửa quy tắc `ExtraSamples` để phân biệt alpha và `UNSPECIFIED` NIR.
2. Chốt mapping là explicit reorder và thêm strict TensorRT binding validation.
3. Ràng buộc sidecar với TIFF và TensorRT `InputSpec`.
4. Tách disk guard theo filesystem/device.
5. Thêm interaction matrix cho read mode và cache mode.
6. Chốt một công thức memory reserve và provider đo available memory.
7. Làm rõ zero-value validation và thêm cache directory vào `.gitignore` khi triển khai.

## Đánh giá tổng thể

Kế hoạch hiện đạt khoảng 9.2/10. Sau khi sửa `ExtraSamples`, chốt explicit channel reorder, strict TensorRT input và làm rõ disk/CLI/memory interaction, tài liệu đủ chặt để bắt đầu Phase 1.
