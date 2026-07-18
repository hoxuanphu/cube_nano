# Trạng thái triển khai remediation TIFF nén

Cập nhật ngày 2026-07-18. Tài liệu này theo dõi implementation của
`compressed_tiff_reading_remediation_plan.md`; plan gốc vẫn là nguồn yêu cầu chuẩn.

## Kết luận hiện tại

Phase 1 đã hoàn thành ở mức code và kiểm thử cục bộ. Lỗi chính đã được loại bỏ:
TIFF nén không còn gọi full decode theo từng row strip. Reader thử memmap đúng một
lần; nếu không được thì decode đúng một lần vào RAM hoặc disk-backed cache sau khi
qua resource guard.

`stream` hiện chỉ chấp nhận TIFF memmap-compatible. True block streaming chưa được
enable vì chưa có backend nào qua đủ pixel, block-count, memory và benchmark gate
trên Jetson Nano. Đây là hành vi fail-closed theo plan, không phải fallback thiếu.

## Hạng mục đã hoàn thành

- `ImageBlockReader` và `TiffReader` là context manager theo vòng đời inference.
- Chọn series/level tường minh; từ chối multi-series, pyramid hoặc multi-page mơ hồ.
- Validate axes, HWC/CHW, planar contiguous/separate, dtype và số kênh.
- Validate photometric, samples-per-pixel, planar configuration và extra samples.
- RGB ba kênh có semantics rõ được chấp nhận; RGBNIR cần sidecar hoặc mapping.
- Alpha/RGBA bị từ chối; channel mapping có thể select/reorder về RGB/RGBNIR canonical.
- Sidecar kiểm tra SHA-256, axes, shape, dtype, band order, input spec và normalization.
- Engine manifest kiểm tra SHA-256, input shape/dtype, band order và optimization profile.
- `production_contract` bắt buộc engine manifest và TIFF input sidecar.
- TensorRT binding là nguồn shape/kênh/dtype thực; thiếu/thừa kênh bị từ chối.
- Normalization cố định theo `InputSpec`; heuristic dựa trên `patch.max()` đã bị loại bỏ.
- `auto`, `stream`, `full` và `auto|ram|disk` cache mode tuân theo interaction matrix.
- RAM/disk guard dùng integer bytes và chỉ cộng runtime reserve một lần.
- Disk allocations được group theo filesystem/device trước khi tạo file lớn.
- Source cache và mask cache nội bộ được xóa khi thành công hoặc có exception.
- User mask cache dùng exclusive create, không overwrite và được giữ lại khi có lỗi.
- Output TIFF được ghi qua temp cùng filesystem rồi atomic replace.
- Reader trả block metrics, read latency và provenance backend/codec/layout.
- Có benchmark harness ghi tối thiểu năm run sau warm-up, median/p95 và memory metrics.
- Có `create_tiff_sidecar.py` để đóng gói sidecar fingerprinted và từ chối alpha.

## Mode matrix đã triển khai

| Read mode | TIFF memmap-compatible | TIFF cần decoder | Cache mode |
|---|---|---|---|
| `auto` | memmap | one-time RAM, sau đó disk theo guard | `auto`, `ram`, `disk` |
| `full` | bỏ qua memmap | one-time RAM/disk theo guard | `auto`, `ram`, `disk` |
| `stream` | memmap | fail sớm | chỉ `auto` |

Disk cache hiện dùng guard bảo thủ `mapped_working_set = decoded_bytes`, vì đường
`tifffile.asarray(out=memmap)` chưa chứng minh được mapped-window bound trên Jetson.
Cách này có thể từ chối một số ảnh vẫn chạy được về lý thuyết, nhưng không đánh giá
thấp peak RAM khi chưa có số đo target.

## Kiểm thử đã có

- Pixel oracle cho memmap và TIFF Deflate one-time cache.
- HWC, CHW và planar separate.
- Strip overlap với `rowsperstrip` không chia hết patch size; block decode một lần.
- RGB, RGBNIR, BGR reorder, sidecar/mapping conflict và RGBA rejection.
- RAM/disk guard boundary, reserve một lần và multi-device disk guard.
- Codec capability failure trước decoder.
- Fault injection giữa inference, cache cleanup và atomic replace.
- Fake TensorRT integration, fixed normalization và edge tile không ghi quá biên.
- Strict TensorRT binding contract không pad/truncate channel.

## Exit gate còn lại

1. Chạy corpus production trên Jetson cho Deflate, LZW và JPEG. LZW/JPEG cần build
   `imagecodecs` tương thích ARM64; khi codec thiếu reader hiện fail trước inference.
2. Chạy `benchmark_tiff_inference.py` cho cold-cache và warm-cache, ít nhất năm lần
   sau warm-up, trên từng batch size/power mode được phát hành.
3. PoC `tifffile.aszarr` + Zarr trên cùng corpus; chỉ enable nếu pixel oracle,
   `blocks_decoded`, peak memory và latency đều đạt.
4. Nếu Zarr không đạt, đánh giá Rasterio/GDAL theo cùng gate.
5. Chốt production artifact format và dependency/version matrix trên JetPack mục tiêu.

Benchmark harness chỉ ghi nhận nhãn `cold` hoặc `warm`; nó không tự drop Linux page
cache. Cold-cache run phải được chuẩn bị bên ngoài harness theo quy trình vận hành có
quyền phù hợp.
