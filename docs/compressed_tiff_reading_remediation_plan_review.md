# Nhận xét phương án xử lý TIFF nén khi inference ảnh lớn

## Kết luận

Kế hoạch trong `compressed_tiff_reading_remediation_plan.md` xác định đúng nguyên nhân chính: khi `tifffile.memmap()` không sử dụng được, `_read_image_strip()` gọi `tiff.imread()` lại cho mỗi row strip và có thể liên tục giải mã toàn bộ ảnh. Hướng refactor thành reader có vòng đời bằng một phiên inference là hợp lý.

Tuy nhiên, kế hoạch hiện chưa đủ cụ thể để triển khai an toàn trên Jetson Nano. Các phần cần chốt trước khi viết code là backend đọc block, mô hình tính peak memory, hợp đồng shape/axes của TIFF, codec được hỗ trợ và ownership của tài nguyên tạm.

## Các nhận xét chính

### 1. Mức cao: Chưa xác định cách đọc từng tile/strip thực sự

Việc mở `TiffFile` một lần không tự động cung cấp spatial slicing:

- `TiffPage.asarray()` giải mã toàn bộ page.
- `TiffPage.segments()` mặc định duyệt và giải mã tất cả segment.
- Mở file một lần chỉ giảm chi phí mở file; nó không bảo đảm mỗi lần `read_strip()` chỉ giải mã vùng cần thiết.

Kế hoạch cần chọn rõ một backend cho true block streaming, ví dụ:

- `tifffile.aszarr` kết hợp Zarr slicing;
- Rasterio/GDAL windowed read;
- hoặc tự chọn `dataoffsets`/`databytecounts` và gọi decoder cấp thấp của `tifffile`.

Phương án decoder cấp thấp phụ thuộc mạnh vào API nội bộ và phiên bản `tifffile`, do đó chỉ nên chọn khi có adapter, version pin và test tương thích. Cần làm proof-of-concept với TIFF thực tế trên Jetson trước khi coi chế độ `stream` là khả thi.

Ngoài ra, nếu `patch_size` không khớp `rowsperstrip` hoặc tile height, nhiều lần `read_strip()` có thể giải mã lại cùng một TIFF block. Reader phải cache block đang dùng hoặc tổ chức vòng lặp theo block vật lý của TIFF.

### 2. Mức cao: Không nên phân loại chỉ theo nén/không nén

Khả năng memory-map phụ thuộc cả compression, layout, byte order và tính liên tục của dữ liệu. TIFF tiled không nén vẫn có thể không memory-map trực tiếp được.

Reader nên phân nhánh theo capability thực tế:

```text
memmap thành công
    -> dùng memmap
memmap thất bại và block-stream backend hỗ trợ
    -> đọc theo block
block-stream không hỗ trợ và one-time cache nằm trong giới hạn
    -> giải mã đúng một lần vào cache
các trường hợp còn lại
    -> fail sớm với lỗi rõ ràng
```

Khuyến nghị "tiled/uncompressed TIFF" cũng cần tách thành hai mục tiêu khác nhau:

- Contiguous hoặc stripped uncompressed TIFF phù hợp với memmap.
- Tiled TIFF phù hợp với block reader nhưng không nhất thiết memory-map được.

Không nên suy luận `memmap()` thất bại đồng nghĩa với TIFF nén.

### 3. Mức cao: Memory guard đang đánh giá thấp peak memory

Công thức:

```text
height * width * channels * dtype.itemsize
```

chỉ tính kích thước array đã giải mã. Peak memory thực tế còn có thể bao gồm:

- buffer dữ liệu nén và buffer giải nén của tile/strip;
- temporary copy khi đổi byte order hoặc layout;
- strip được chuyển thành `float32` trong normalization;
- các patch đang chờ trong batch;
- TensorRT engine, CUDA context và workspace dùng chung RAM trên Jetson.

Nên lấy shape, dtype và axes từ TIFF series đã chọn, tính kích thước bằng Python integer để tránh overflow, rồi cộng headroom và phần RAM dành riêng cho TensorRT. Guard phải áp dụng cho cả `auto` và `full`; chế độ `full` không được phép bỏ qua giới hạn.

Một fallback thực tế hơn là giải mã một lần vào disk-backed memmap do ứng dụng sở hữu. Cách này vẫn cần:

- giới hạn kích thước decoded cache;
- kiểm tra dung lượng đĩa trước khi tạo cache;
- giới hạn số worker/buffer giải nén;
- cleanup chắc chắn khi thành công hoặc có exception.

### 4. Mức cao: Hợp đồng shape và axes của TIFF chưa đầy đủ

Code hiện tại lấy `pages[0].shape`, nhưng TIFF có thể chứa:

- nhiều series hoặc nhiều page;
- image pyramid;
- planar contiguous hoặc planar separate;
- axes như `YXS`, `SYX`, `CYX`;
- các chiều bổ sung của OME-TIFF.

Reader cần quy định rõ:

- chọn series, page và pyramid level nào;
- dùng `series.axes` để xác định trục `Y`, `X` và `S/C`;
- chỉ chấp nhận đúng một ảnh 2-D có 3 hoặc 4 kênh;
- xác nhận số kênh trong file bằng tham số `channels` của TensorRT;
- trả về HWC mà không làm thay đổi thứ tự kênh;
- từ chối TIFF nhiều chiều hoặc axes mơ hồ bằng lỗi có hướng dẫn.

Heuristic dựa trên kích thước 3 hoặc 4 không đủ an toàn cho các ảnh có chiều cao hoặc chiều rộng nhỏ và cũng không mô tả được đầy đủ multi-page TIFF.

### 5. Mức trung bình: Cleanup cần hợp đồng ownership cụ thể

`try/finally` là cần thiết nhưng nên dùng context manager cho reader, ví dụ `with TiffStripReader(...) as reader`. Reader phải tự đóng chính xác những tài nguyên do nó tạo.

Đối với mask cache, cần lưu một ownership flag riêng thay vì suy luận lại từ giá trị `mask_cache`. Code hiện dùng `if mask_cache` khi tạo cache nhưng dùng `mask_cache is None` khi cleanup. Nếu `mask_cache=""`, hàm sẽ tạo cache nội bộ nhưng không xóa cache đó.

Hợp đồng nên quy định:

- cache nội bộ luôn được xóa khi thành công hoặc thất bại;
- cache do người dùng cung cấp được flush/close nhưng không bị xóa;
- memmap được flush và đóng explicit, đặc biệt trên Windows;
- output TIFF được ghi qua file tạm và atomic replace để tránh để lại file dở;
- hành vi khi đường dẫn cache đã tồn tại phải rõ ràng.

### 6. Mức trung bình: Dependency và codec chưa được đóng

`requirements.txt` hiện không pin `tifffile` và không khai báo `imagecodecs` hoặc `zarr`. Khả năng đọc TIFF nén phụ thuộc codec và phiên bản thư viện; kiểm thử trên máy phát triển không chứng minh rằng cùng codec có thể cài và chạy trên Jetson Nano ARM64.

Kế hoạch cần có:

- version hoặc khoảng version đã được kiểm thử;
- ma trận codec được hỗ trợ, ví dụ Deflate, LZW và JPEG;
- xác nhận khả năng cài dependency trên Jetson;
- capability check ngay khi khởi tạo reader;
- lỗi rõ ràng trước khi bắt đầu inference nếu codec không khả dụng.

### 7. Mức trung bình: Kiểm thử chưa đủ để chứng minh không decode lặp

Việc kiểm tra số lần gọi `tiff.imread()` là cần thiết nhưng chưa đủ. Backend có thể không gọi `imread()` mà vẫn giải mã lại cùng tile nhiều lần.

Nên bổ sung các test sau:

- So sánh pixel của từng `read_strip()` với full-read oracle.
- TIFF compressed dạng tiled và stripped.
- Tile height hoặc `rowsperstrip` không khớp `patch_size`.
- HWC/planar contiguous và CHW/planar separate.
- TIFF 3 kênh và 4 kênh.
- Codec không được hỗ trợ phải fail trước khi xử lý patch đầu tiên.
- Memory guard phải fail trước khi decoder được gọi.
- Exception sau khi tạo cache, giữa inference và trong lúc ghi output.
- Cache nội bộ được xóa và cache do người dùng cung cấp được giữ lại.
- Mỗi block vật lý cần thiết được giải mã tối đa một lần, hoặc số lần giải mã nằm trong giới hạn đã định nghĩa.

Integration test nên so sánh trực tiếp dữ liệu strip trước khi so sánh output mask. Mask có thể vẫn giống nhau dù kênh bị đảo hoặc pixel bị sai nhưng dự đoán chưa vượt threshold.

### 8. Mức trung bình: Tiêu chí benchmark cần tái lập được

Benchmark nên định nghĩa rõ:

- bộ TIFF mẫu, compression, tile/strip size, dtype và shape;
- cold-cache và warm-cache run;
- cách đo peak RSS và dung lượng disk cache;
- số lần decode theo block, không chỉ số lần gọi API cấp cao;
- số lần lặp và thống kê median/p95;
- cấu hình Jetson, TensorRT, batch size và power mode.

Nếu không cố định các điều kiện này, số liệu trước/sau khó so sánh và khó dùng làm tiêu chí nghiệm thu.

## Thứ tự triển khai đề xuất

### Giai đoạn 1: Sửa lỗi chắc chắn và giới hạn blast radius

1. Tạo `TiffReader` dạng context manager.
2. Dùng `tifffile.memmap()` một lần theo capability, không phân loại trước theo compression.
3. Khi memmap thất bại, giải mã đúng một lần vào RAM hoặc disk-backed cache có guard.
4. Thêm ownership flag và cleanup cho reader, source cache, mask cache và output tạm.
5. Thêm test pixel correctness, one-time decode, memory guard và exception cleanup.

Giai đoạn này giải quyết chắc chắn lỗi decode toàn ảnh lặp lại mà chưa phụ thuộc vào một thiết kế block decoder chưa được kiểm chứng.

### Giai đoạn 2: True block streaming

1. Kiểm kê TIFF production: compression, axes, planar config, tile/strip size và metadata.
2. Chọn một backend windowed/block read cụ thể.
3. Làm proof-of-concept và benchmark trên Jetson Nano.
4. Thêm block cache hoặc thay đổi iteration để không decode lại block giao nhau.
5. Chỉ bật `stream` khi capability check và test tương thích đều đạt.

### Giai đoạn 3: Chuẩn hóa dữ liệu triển khai

1. Chọn định dạng đầu vào chuẩn cho Jetson.
2. Nếu dùng TIFF uncompressed cho memmap, yêu cầu layout contiguous/stripped phù hợp.
3. Nếu dùng tiled TIFF, yêu cầu block-stream backend đã được xác nhận.
4. Nếu dùng `.npy`, lưu sidecar metadata cho dtype, axes, channel order, normalization và thông tin địa lý cần thiết.

## Ngữ nghĩa CLI đề xuất

```text
--tiff_read_mode auto|stream|full
--max_full_read_gb <positive-value>
```

- `auto`: thử memmap, sau đó block streaming, cuối cùng one-time cache nếu nằm trong giới hạn.
- `stream`: chỉ chấp nhận memmap hoặc true block streaming; không fallback full decode.
- `full`: chủ động dùng one-time decoded cache nhưng vẫn bắt buộc qua memory/disk guard.

Cần khai báo rõ đơn vị GB hay GiB, default phù hợp với Jetson và hành vi khi giá trị bằng 0 hoặc âm.

## Tiêu chí nghiệm thu đề xuất

- Không giải mã toàn ảnh lại theo số row strip.
- Với block streaming, một block cần thiết không bị decode lặp ngoài giới hạn đã định nghĩa.
- Mọi strip khớp pixel với full-read oracle và luôn có layout HWC.
- TIFF có axes hoặc số chiều không được hỗ trợ bị từ chối trước inference.
- Số kênh của ảnh phải khớp TensorRT engine.
- Fallback full decode luôn qua memory/disk guard trước khi cấp phát lớn.
- Codec không khả dụng bị phát hiện khi khởi tạo reader.
- Cache nội bộ được dọn khi thành công hoặc có exception; cache của người dùng được giữ lại.
- Output mask không thay đổi trên cùng input và fake TensorRT oracle.
- Benchmark trên Jetson chứng minh peak RAM và thời gian nằm trong ngân sách đã xác định.

## Đánh giá tổng thể

Kế hoạch đạt khoảng 7/10 về định hướng. Reader theo vòng đời inference, one-time fallback, memory guard và cleanup là các quyết định đúng. Trước khi triển khai, cần biến phần "nếu decoder hỗ trợ" thành một thiết kế backend cụ thể và bổ sung hợp đồng về TIFF axes, codec, peak memory và ownership tài nguyên.
