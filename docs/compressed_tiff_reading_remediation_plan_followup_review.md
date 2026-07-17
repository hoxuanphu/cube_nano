# Đánh giá lại phương án xử lý TIFF nén sau khi chỉnh sửa

## Kết luận

Bản kế hoạch đã cải thiện rõ rệt so với phiên bản đầu. Các nội dung về reader lifecycle, capability-based fallback, phân chia giai đoạn, axes validation, ownership, block metrics, dependency, fault-injection test, benchmark và đơn vị GiB đều đã được bổ sung đúng hướng.

Tuy nhiên, kế hoạch vẫn còn ba vấn đề mức cao trước khi có thể coi là sẵn sàng triển khai: đặc tả cũ và mới đang mâu thuẫn, hợp đồng bốn kênh chưa bảo đảm đúng semantics RGBNIR, và mô hình memory guard chưa tách RAM cache khỏi disk-backed cache.

## Phát hiện

### 1. Mức cao: Tài liệu đang có hai đặc tả mâu thuẫn

Phần cũ vẫn phân nhánh theo TIFF nén/không nén, trong khi phần bổ sung yêu cầu phân nhánh theo capability thực tế.

Các mâu thuẫn cụ thể gồm:

- Phần cũ chia `TIFF khong nen` và `TIFF nen`; phần mới yêu cầu không phân nhánh chỉ theo compression.
- Phần cũ chỉ tính `height * width * channels * dtype.itemsize`; phần mới yêu cầu tính peak memory.
- Phần cũ dùng `--max_full_read_gb`; phần mới dùng `--max_full_read_gib`.
- Tiêu chí cũ chỉ đếm số lần gọi `tiff.imread()`; tiêu chí mới yêu cầu đếm block thực sự được decode.
- Thứ tự triển khai cũ không phản ánh thiết kế ba giai đoạn mới.
- Phần cũ khuyến nghị chung `tiled/uncompressed TIFF`, trong khi phần mới đã phân biệt memmap và block streaming theo capability.

Phần `Bổ sung sau review chuyên gia` nên thay thế trực tiếp các mục cũ tương ứng, không nên tồn tại như một phụ lục song song. Nếu giữ cả hai, người triển khai có thể chọn một trong hai yêu cầu mâu thuẫn mà vẫn cho rằng mình tuân thủ kế hoạch.

Kế hoạch nên được tổ chức lại thành một đặc tả duy nhất theo thứ tự:

1. Input contract.
2. Reader contract.
3. Backend selection.
4. Memory và disk budget.
5. Ownership và cleanup.
6. CLI contract.
7. Test và benchmark.
8. Thứ tự triển khai và tiêu chí nghiệm thu.

### 2. Mức cao: Axes đúng chưa bảo đảm semantics của kênh đúng

Kế hoạch chấp nhận ảnh có 3 hoặc 4 kênh và giữ nguyên thứ tự kênh. Điều này chỉ xác nhận hình dạng dữ liệu, chưa xác nhận ý nghĩa quang phổ của từng kênh.

Model bốn kênh của repository được train với thứ tự:

```text
R, G, B, NIR
```

Trong khi đó, TIFF RGBA cũng có thể có shape `YXS`, bốn sample và vượt qua validation axes/số kênh hiện tại. Nếu không kiểm tra thêm, alpha có thể bị truyền vào TensorRT như NIR mà không phát sinh lỗi.

Input contract cần bổ sung:

- Kiểm tra `PhotometricInterpretation`.
- Kiểm tra `SamplesPerPixel`.
- Kiểm tra `ExtraSamples` để phát hiện alpha.
- Kiểm tra `PlanarConfiguration`.
- Từ chối RGBA/alpha, CMYK, palette và các photometric không được hỗ trợ.
- Với ảnh bốn band, yêu cầu manifest hoặc sidecar khai báo rõ `R,G,B,NIR`, hoặc yêu cầu channel mapping tường minh.
- Không coi tên axes là bằng chứng về ý nghĩa quang phổ của kênh.

Nếu production chỉ chấp nhận artifact đã được chuẩn hóa, kế hoạch nên quy định sidecar metadata là bắt buộc. Nếu cần nhận arbitrary GeoTIFF, cần có tùy chọn channel mapping và validation cụ thể.

Test bắt buộc phải có:

- RGB ba kênh hợp lệ.
- RGBNIR bốn kênh hợp lệ với metadata/mapping.
- RGBA bốn kênh bị từ chối.
- Sai thứ tự kênh hoặc thiếu mapping bị từ chối.
- Photometric không được hỗ trợ bị từ chối trước inference.

### 3. Mức cao: Memory model đang trộn RAM cache và disk-backed cache

Kế hoạch cho phép one-time decoded cache nằm trong RAM hoặc disk-backed memmap, nhưng công thức hiện tại luôn cộng toàn bộ `decoded_bytes` vào peak RAM. Cách tính này không mô tả đúng disk-backed cache và có thể từ chối những ảnh thực tế vẫn xử lý an toàn bằng memmap trên đĩa.

Cần tách ít nhất hai mô hình:

```text
RAM cache peak =
    decoded_bytes
    + decoder_peak
    + batch_and_copy_peak
    + tensorrt_reserve

Disk cache peak RAM =
    decoder_peak
    + mapped_working_set
    + batch_and_copy_peak
    + tensorrt_reserve

Disk required =
    decoded_bytes
    + temporary_output_allowance
    + filesystem_headroom
```

`decoder_peak` cần bao gồm:

- compressed input buffer;
- decoded tile/strip buffer;
- byte-order/layout conversion copy;
- số decoder worker tối đa chạy đồng thời.

`batch_and_copy_peak` phải bám theo pipeline thực tế. Code hiện normalize theo từng patch, giữ các patch trong `batch_patches`, sau đó gọi `np.stack(...).astype(np.float32)`. Việc stack/cast có thể tạo thêm một batch copy. Vì vậy `normalized_strip_bytes` không phải đại lượng phù hợp nếu implementation không normalize toàn bộ strip.

Guard cần được đánh giá sau khi TensorRT/CUDA đã được khởi tạo hoặc phải sử dụng một reserve đủ bảo thủ cho unified memory của Jetson.

### 4. Mức trung bình: LRU block cache chưa có ngân sách cụ thể

Cụm từ `LRU nhỏ` chưa đủ để triển khai và nghiệm thu. Kế hoạch cần định nghĩa:

- giới hạn byte hoặc số block;
- cache key gồm series, level, page, sample plane và physical block index;
- chính sách eviction;
- cách xử lý block lớn hơn toàn bộ cache budget;
- cache memory được cộng vào peak guard;
- giới hạn chính xác cho `blocks_decoded`.

Có thể bổ sung cấu hình nội bộ hoặc CLI:

```text
--tiff_block_cache_mib <non-negative-value>
```

Giá trị `0` có thể được định nghĩa là tắt LRU cache nếu backend/iteration bảo đảm không cần đọc lại block.

### 5. Mức trung bình: CLI chưa chốt default và cache-selection policy

Kế hoạch đã chốt đơn vị GiB nhưng chưa chốt:

- default của `max_full_read_gib`;
- default hoặc cách tính TensorRT/system reserve;
- cache directory;
- khi nào chọn RAM cache và khi nào chọn disk cache;
- hành vi khi RAM đủ nhưng disk không đủ, hoặc ngược lại;
- hành vi khi cache path đã tồn tại;
- quyền ghi và cleanup khi cache directory nằm trên filesystem khác.

Nên tách rõ các giới hạn:

```text
--max_ram_cache_gib <positive-value>
--max_disk_cache_gib <positive-value>
--tiff_cache_mode auto|ram|disk
--tiff_cache_dir <path>
```

Nếu không muốn mở rộng CLI, các giá trị tương đương vẫn phải được chốt thành constant/config với default có thể kiểm thử.

### 6. Mức trung bình: Backend proof-of-concept chưa có exit gate

Kế hoạch đã chọn `tifffile.aszarr` kết hợp Zarr làm backend proof-of-concept ưu tiên. Đây là lựa chọn hợp lý, nhưng cần định nghĩa kết quả khi PoC không đạt trên Jetson.

Exit gate đề xuất:

- Nếu Zarr đạt pixel correctness, block-decode count, peak memory và benchmark: bật backend trong `stream` và `auto`.
- Nếu Zarr không cài được hoặc không đạt benchmark: thử Rasterio/GDAL theo tiêu chí tương tự.
- Nếu cả hai không đạt: phát hành giai đoạn 1 không có true block streaming; `stream` báo unsupported và `auto` chuyển từ memmap sang guarded one-time cache.

Điều này giúp PoC streaming không chặn bản sửa lỗi lifecycle và repeated full decode.

Dependency cần được pin theo từng giai đoạn. Phase 1 không nên bắt buộc cài Zarr nếu chưa bật true block backend.

### 7. Mức trung bình: Cleanup còn thiếu policy cho path và atomic output

Ownership flags đã được bổ sung đúng hướng, nhưng cần chốt thêm:

- cache do người dùng chỉ định đã tồn tại thì fail, truncate hay reuse;
- source cache và mask cache có được dùng chung directory hay không;
- output temp phải nằm cùng filesystem/directory với `out_mask` để `os.replace()` có tính atomic;
- cleanup làm gì nếu ghi temp thành công nhưng replace thất bại;
- quyền truy cập file và hành vi trên Windows khi memmap chưa được đóng hoàn toàn.

Nên ưu tiên exclusive create cho cache nội bộ và từ chối overwrite cache do người dùng chỉ định nếu chưa có cờ explicit.

### 8. Mức trung bình: Test plan cần bao phủ các contract mới

Ngoài các test đã được bổ sung, cần thêm:

- RAM cache và disk-backed cache được test độc lập.
- Disk-space guard fail trước khi tạo/truncate file lớn.
- Batch peak có tính đến stack/cast copy.
- LRU cache eviction và block lớn hơn cache budget.
- RGBA không bị coi là RGBNIR.
- Unsupported photometric hoặc ambiguous channel semantics fail sớm.
- Output temp được atomic replace và file dở được cleanup.
- Backend PoC không đạt capability check không được quảng bá là `stream`.

## Các nội dung đã xử lý tốt

Những nhận xét từ vòng review đầu đã được tiếp thu tốt ở các phần sau:

- Reader có vòng đời bằng một inference session.
- Không coi mở `TiffFile` một lần là true block streaming.
- Backend selection dựa trên capability thay vì chỉ compression.
- Tách lifecycle fix, streaming PoC và input normalization thành ba giai đoạn.
- Dùng `series.axes` và từ chối axes/multi-dimensional layout không hỗ trợ.
- Kiểm tra số kênh trước patch đầu tiên.
- Có block metrics thay vì chỉ đếm `imread()`.
- Có ownership flags cho source cache, mask cache và output temp.
- Có capability check cho codec và dependency matrix.
- Có fault-injection test và benchmark cold/warm cache.
- Đã chốt đơn vị cấu hình là GiB.

## Thứ tự chỉnh sửa tài liệu đề xuất

1. Xóa hoặc viết lại các mục cũ mâu thuẫn; hợp nhất phần bổ sung vào thân kế hoạch.
2. Thêm input/channel semantics contract trước reader contract.
3. Tách công thức RAM cache, disk cache và disk-space guard.
4. Chốt cache-selection policy, default budget và LRU budget.
5. Chốt exit gate cho Zarr/Rasterio proof-of-concept.
6. Bổ sung test cho RGBNIR/RGBA, RAM/disk cache và atomic output.
7. Viết lại tiêu chí nghiệm thu cuối cùng để chỉ còn một nguồn yêu cầu chuẩn.

## Đánh giá tổng thể

Bản kế hoạch hiện đạt khoảng 8.5/10 về định hướng. Sau khi hợp nhất phần cũ với phần mới, bổ sung channel semantics và tách rõ RAM/disk budget, kế hoạch sẽ đủ chặt để bắt đầu triển khai.
