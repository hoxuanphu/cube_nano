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

## Bo sung sau review chuyen gia

### Quyet dinh pham vi

Mo file mot lan chi la buoc sua loi lifecycle. No khong duoc coi la true block streaming neu moi lan `read_strip()` van giai ma lai cung mot page hoac TIFF block.

Ke hoach duoc tach thanh ba giai doan:

1. Sua loi chac chan: reader co vong doi, one-time decoded cache va memory/disk guard.
2. Proof-of-concept true block streaming tren TIFF production va Jetson Nano.
3. Chuan hoa format dau vao sau khi backend streaming da duoc chung minh.

Trong giai doan 1, khong tuyen bo che do `stream` neu chua co test dem so lan decode theo block.

### Hop dong reader

Tao mot interface noi bo tuong tu `ImageBlockReader`:

- `shape`: kich thuoc chuan hoa `(H, W, C)`.
- `dtype`: dtype goc cua anh.
- `axes`: axes da duoc xac nhan, khong suy doan chi tu kich thuoc.
- `read_rows(row_start, row_end)`: tra ve HWC va dung pixel.
- `physical_blocks(row_start, row_end)`: thong tin block vat ly can doc.
- `close()`: giai phong tai nguyen do reader so huu.

`process_large_image()` chi phu thuoc vao interface nay, khong phu thuoc truc tiep vao `tifffile.memmap()` hoac `tiff.imread()`.

### Chon backend theo capability

Khong phan nhanh chi theo compression.

1. Kiem tra `memmap` thanh cong va layout contiguous/stripped phu hop: dung memmap.
2. Neu memmap that bai, thu true block backend da duoc kiem chung: doc theo block.
3. Neu khong co true block backend, giai ma dung mot lan vao decoded cache trong RAM hoac disk-backed memmap, sau khi qua guard.
4. Neu khong dap ung guard, axes hoac codec: fail som voi thong bao huong dan.

Backend true block uu tien cho proof-of-concept la `tifffile.aszarr` ket hop Zarr voi version duoc pin. Rasterio/GDAL la phuong an thay the neu stack Zarr khong phu hop voi Jetson. Khong dung API noi bo `dataoffsets`/`databytecounts` lam backend dau tien khi chua co adapter va version pin.

Moi backend phai co capability check va test pixel correctness truoc khi duoc bat trong che do `stream`.

### Block alignment va cache

`patch_size` khong nhat thiet trung voi `rowsperstrip` hoac tile height. Reader phai:

- Xac dinh cac block vat ly bi giao voi row request.
- Cache block dang dung bang LRU nho neu mot block phuc vu nhieu patch.
- Bao dam mot block vat ly khong bi decode lap qua gioi han da dinh nghia.
- Ghi metric `blocks_requested`, `blocks_decoded`, `cache_hits`, `cache_misses`.

Voi backend khong ho tro block streaming, decoded cache duoc tao mot lan va cac row sau chi slice cache.

### Hop dong TIFF va axes

Khong dung `pages[0].shape` va heuristic `shape[0] in (3, 4)` lam hop dong duy nhat.

Reader phai:

- Chon ro `series`, `page` va pyramid level.
- Doc `series.axes` de xac dinh truc `Y`, `X` va `C/S`.
- Xu ly ro planar contiguous va planar separate.
- Chi chap nhan mot anh 2-D spatial co 3 hoac 4 kenh sau khi chuan hoa.
- Giu nguyen thu tu kenh; khong tu dong reorder neu khong co mapping.
- Tu choi multi-page, OME/pyramid hoac axes mo ho neu chua co cau hinh tuong ung.
- Kiem tra so kenh voi `channels` cua TensorRT engine truoc khi doc patch dau tien.

Co the bo sung cac tuy chon:

```text
--tiff_series <index-or-name>
--tiff_level <index>
```

Neu khong co axes metadata va khong the xac dinh HWC/CHW an toan, phai fail som thay vi suy doan.

### Mo hinh peak memory

Memory guard phai tinh peak memory, khong chi tinh kich thuoc decoded array.

Voi fallback full-read, uoc tinh toi thieu:

```text
decoded_bytes = H * W * C * dtype.itemsize
normalized_strip_bytes = strip_h * W * C * 4
batch_bytes = batch_size * C * patch_size * patch_size * 4
peak_estimate = decoded_bytes + normalized_strip_bytes + batch_bytes + reserve_bytes
```

Voi block streaming, thay `decoded_bytes` bang kich thuoc block cache toi da. Tat ca phep tinh dung Python integer; khong de overflow.

`reserve_bytes` phai danh rieng cho CUDA context, TensorRT engine, workspace va phan RAM he thong. Guard ap dung cho ca `auto` va `full`; `full` khong duoc bo qua guard.

Neu dung disk-backed decoded cache, phai kiem tra dung luong disk truoc khi tao file va gioi han so buffer decoder ton tai cung luc.

### Ownership va cleanup

Dung context manager:

```python
with TiffReader(...) as reader:
    ...
```

Phan biet ro:

- `owns_source_cache`: cache do reader tao va phai xoa khi thanh cong hoac exception.
- `owns_mask_cache`: mask cache noi bo phai xoa; cache do user truyen vao chi flush/close, khong xoa.
- `owns_output_temp`: output tam phai duoc atomic replace sang `out_mask` sau khi ghi thanh cong.

Khong dung truthiness de suy ra ownership. Truong hop `mask_cache=""` phai bi tu choi tu dau hoac duoc chuan hoa thanh `None`, khong duoc tao cache noi bo roi bo quen file.

### Dependency va codec

Can lap ma tran codec toi thieu cho Deflate, LZW va JPEG:

- `tifffile` va version da kiem thu.
- `imagecodecs` neu backend can decoder ngoai.
- `zarr` neu bat `tifffile.aszarr`.
- Kha nang cai dat tren Jetson Nano ARM64.

Dependency phai duoc pin hoac gioi han version. Reader phai capability-check codec ngay khi khoi tao va bao loi truoc khi bat dau inference neu codec khong san sang.

### Test bo sung bat buoc

- So sanh pixel cua moi `read_rows()` voi full-read oracle.
- TIFF compressed tiled va compressed stripped.
- `rowsperstrip`/tile height khong chia het `patch_size`.
- HWC, CHW, planar contiguous va planar separate.
- Multi-page, pyramid va axes khong ho tro phai fail som.
- Codec khong ho tro phai fail truoc patch dau tien.
- Memory guard phai fail truoc khi decoder cap phat buffer lon.
- Exception sau khi tao source cache, trong inference va luc ghi output.
- Cache noi bo duoc xoa; cache do user cung cap duoc giu lai.
- Dem so lan decode block, khong chi dem so lan goi `tiff.imread()`.
- So sanh strip truoc, sau do moi so sanh output mask; mask giong nhau khong du de chung minh pixel dung.

### Benchmark co the lap lai

Benchmark phai ghi lai:

- File mau, codec, tiled/stripped, block size, dtype, axes va shape.
- Cold-cache va warm-cache.
- Peak RSS, CUDA/TensorRT memory va disk cache.
- `blocks_decoded`, `cache_hits`, `cache_misses`.
- Median va p95 cua thoi gian doc strip va tong inference.
- Jetson model, JetPack/TensorRT version, batch size va power mode.

Khong dung so lan goi API cap cao lam chi so duy nhat cho viec khong decode lap.

### Nghia CLI da chot

```text
--tiff_read_mode auto|stream|full
--max_full_read_gib <positive-value>
```

- `auto`: memmap -> true block backend -> one-time decoded cache neu qua guard.
- `stream`: chi chap nhan memmap hoac true block backend; khong fallback full decode.
- `full`: one-time decoded cache nhung van bat buoc qua memory/disk guard.

Don vi duoc chot la GiB. Gia tri bang 0, am hoac khong phai so duong phai bi tu choi.
