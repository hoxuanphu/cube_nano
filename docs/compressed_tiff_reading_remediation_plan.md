# Phuong an xu ly TIFF nen khi inference anh lon

## Van de

Trong `src/inference_large_image_trt.py`, `process_large_image()` doc tung row strip thong qua `_read_image_strip()`. Khi `tifffile.memmap()` khong hoat dong voi TIFF nen, fallback tai dong 382 goi `tiff.imread()` de doc toan bo anh moi lan.

Voi anh lon, cach nay co the:

- Lap lai viec decode toan anh cho moi row strip.
- Lam tang manh thoi gian xu ly.
- Tao nhieu lan phan bo bo nho va nguy co OOM tren Jetson Nano.

## Muc tieu

- Mo file TIFF mot lan trong mot phien inference.
- Khong decode lai toan anh cho moi row strip.
- Giu nguyen layout dau ra `(H, W, C)` cho TensorRT.
- Co gioi han bo nho va loi ro rang khi khong the doc an toan.
- Dam bao tai nguyen TIFF, memmap va cache tam duoc dong/xoa khi inference loi.

## Phuong an de xuat

### 1. Tao reader co vong doi theo phien inference

Refactor `_read_image_strip()` thanh mot reader duoc khoi tao mot lan truoc vong lap trong `process_large_image()`.

Reader can cung cap:

- `shape`: kich thuoc anh chuan hoa ve `(H, W, C)`.
- `read_strip(row_start, row_end)`: tra ve mot strip dang `(H, W, C)`.
- `close()`: dong file va giai phong tai nguyen.

`process_large_image()` dung mot reader duy nhat cho tat ca row strip thay vi mo va doc lai file trong moi vong lap.

### 2. Xu ly theo cac che do doc TIFF

#### TIFF khong nen

- Thu `tifffile.memmap()` mot lan.
- Dung memmap de cat strip truc tiep.
- Khong goi `tiff.imread()`.

#### TIFF nen

- Mo `TiffFile` mot lan.
- Kiem tra TIFF dang tiled hay stripped va kha nang doc tile/strip cua phien ban `tifffile`/`imagecodecs` dang su dung.
- Neu decoder ho tro doc tung tile/strip, chi decode vung can thiet.
- Neu khong ho tro, decode toan anh toi da mot lan va tai su dung ket qua cho cac row sau.

Fallback full-read phai duoc xem la phuong an co dieu kien, khong duoc lap lai trong moi lan goi `_read_image_strip()`.

### 3. Gioi han bo nho

Truoc khi fallback full-read, uoc tinh bo nho can thiet:

```text
height * width * channels * dtype.itemsize
```

Neu vuot nguong cho phep:

- Dung voi thong bao loi ro rang.
- Huong dan chuyen doi TIFF sang tiled/uncompressed TIFF hoac `.npy` memory-mapped.
- Khong tiep tuc doc de dan den OOM.

Co the them cau hinh CLI:

```text
--tiff_read_mode auto|stream|full
--max_full_read_gb <value>
```

Che do `auto` la mac dinh; `stream` tu choi fallback full-read; `full` chi dung cho anh nho.

### 4. Phuong an phu hop Jetson Nano

Uu tien chuan hoa anh dau vao truoc inference:

- Chuyen TIFF nen sang TIFF tiled/uncompressed.
- Hoac chuyen sang `.npy` co the memory-map.
- Ghi lai dtype, shape, channel order va normalization trong metadata.

Runtime tren Jetson chi doc strip/tile tu file da chuan hoa, tranh decode lai anh nen trong luc inference.

### 5. Quan ly tai nguyen

Dung `try/finally` trong `process_large_image()` de dam bao:

- `TiffFile` duoc dong.
- Memmap duoc giai phong.
- Cache tam duoc xoa neu do ham tao ra.
- Cache do nguoi dung cung cap duoc giu lai theo dung hop dong `mask_cache`.

## Kiem thu

### Unit test

- TIFF khong nen su dung memmap va khong goi `imread`.
- TIFF nen khong decode lap lai khi doc nhieu row strip.
- Fallback full-read chi duoc thuc hien toi da mot lan.
- Vuot gioi han bo nho phai fail som voi loi ro rang.
- HWC va CHW deu tra ve strip HWC dung shape.
- Anh co kich thuoc nho hon `patch_size` van duoc padding dung.

### Integration test

- Tao TIFF nen 3 kenh va 4 kenh co kich thuoc nho.
- Chay `process_large_image()` voi TensorRT fake inference.
- So sanh output mask truoc va sau refactor.
- Kiem tra cac tile o bien anh khong bi ghi ra ngoai kich thuoc that.

### Benchmark

Ghi nhan truoc va sau refactor:

- So lan decode TIFF.
- Thoi gian doc anh.
- Peak RAM.
- Thoi gian tong inference.
- Kich thuoc cache tam.

## Thu tu trien khai

1. Tao `TiffStripReader` va chuyen `process_large_image()` sang dung reader co vong doi.
2. Bao dam nhanh memmap khong thay doi.
3. Them compressed-TIFF fallback chi decode mot lan.
4. Them memory guard va cac tuy chon CLI.
5. Them `try/finally` cho file handle va cache.
6. Them unit test, integration test va benchmark.
7. Cap nhat documentation ve dinh dang TIFF khuyen dung cho Jetson Nano.

## Tieu chi nghiem thu

- Khong co lan `tiff.imread()` lap lai theo so row strip.
- TIFF nen nho duoc xu ly dung voi mot lan decode.
- Anh vuot gioi han bo nho bi tu choi truoc khi OOM.
- Output mask khong thay doi so voi pipeline hien tai tren cung input.
- Tai nguyen doc va cache duoc don dep khi thanh cong hoac khi inference loi.
