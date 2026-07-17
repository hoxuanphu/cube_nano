# Phuong an xu ly TIFF nen khi inference anh lon

## 1. Van de, muc tieu va pham vi

Trong `src/inference_large_image_trt.py`, `process_large_image()` doc tung row strip qua `_read_image_strip()`. Khi `tifffile.memmap()` that bai, fallback hien tai goi `tiff.imread()` cho moi row strip va co the giai ma toan bo anh lap lai nhieu lan.

Ke hoach nay nham:

- Khong giai ma toan anh lap theo so row strip.
- Duy tri peak RAM va disk usage trong budget da khai bao.
- Bao dam moi strip dung pixel, axes va semantics kenh ma model mong doi.
- Fail som khi TIFF, codec, metadata hoac tai nguyen khong dap ung hop dong.
- Don dep dung tai nguyen khi thanh cong hoac co exception.

Ke hoach chi ap dung cho TIFF duoc dua vao pipeline classification TensorRT 3 hoac 4 kenh. Cac TIFF segmentation mask, OME-TIFF nhieu chieu va pyramid khong duoc tu dong chap nhan.

## 2. Input contract

### 2.1. Shape va axes

Reader phai chon ro TIFF series, page va pyramid level truoc khi doc pixel. Shape phai lay tu series da chon, khong chi tu `pages[0].shape`.

Reader phai:

- Doc `series.axes` va xac dinh duy nhat truc `Y`, `X` va `C/S`.
- Chap nhan layout HWC nhu `YXC`, `YXS` va channel-first nhu `CYX`, `SYX` sau khi validation.
- Chi chap nhan dung hai chieu spatial va 3 hoac 4 kenh.
- Tu choi cac non-singleton axis bo sung, multi-page mo ho, pyramid level mo ho va axes khong xac dinh.
- Chuyen layout dau ra ve HWC ma khong tu dong doi thu tu kenh.
- Kiem tra so kenh voi cau hinh TensorRT truoc khi doc patch dau tien.

CLI co the chon ro series va level:

```text
--tiff_series <non-negative-index>
--tiff_level <non-negative-index>
```

Mac dinh ca hai la `None`. Reader chi tu dong chon index `0` khi file co dung mot series va series do co dung mot level. Neu co nhieu series/level, nguoi dung phai chon ro; reader khong duoc ngam chon.

### 2.2. Semantics kenh

Model trong repository co hai input contract:

```text
3 kenh: R, G, B
4 kenh: R, G, B, NIR
```

Axes va so sample khong chung minh duoc y nghia quang pho. Reader phai kiem tra:

- `PhotometricInterpretation`.
- `SamplesPerPixel`.
- `ExtraSamples`.
- `PlanarConfiguration`.

Quy tac chap nhan:

- TIFF `PhotometricInterpretation=RGB`, 3 sample va khong co extra sample duoc chap nhan la `R,G,B`.
- Anh 4 band chi duoc chap nhan khi co sidecar metadata hoac `channel_mapping` tuong minh khai bao `R,G,B,NIR`.
- RGBA/alpha, CMYK, palette va photometric khong ho tro phai bi tu choi.
- TIFF 4 sample co `ExtraSamples` phai bi tu choi; alpha khong duoc xem la NIR.
- TIFF 3 sample khong co semantics RGB ro rang phai co sidecar hoac mapping tuong minh.

Production artifact chuan nen co sidecar JSON:

```json
{
  "axes": "YXC",
  "band_order": ["red", "green", "blue", "nir"],
  "dtype": "uint16",
  "normalization": "training-input-contract-id"
}
```

Voi arbitrary TIFF, cung cap mapping:

```text
--channel_mapping red=0,green=1,blue=2,nir=3
```

Mapping phai co dung cac role ma model can, khong trung index va khong vuot so kenh cua anh.

## 3. Reader contract

Tao interface noi bo `ImageBlockReader` va implementation `TiffReader` dang context manager.

Reader cung cap:

- `shape`: `(H, W, C)` sau validation.
- `dtype`: dtype goc.
- `axes`: axes goc va axes da chuan hoa.
- `band_order`: thu tu band da xac nhan.
- `read_rows(row_start, row_end)`: tra ve HWC dung pixel.
- `physical_blocks(row_start, row_end)`: cac block vat ly can doc.
- `metrics`: so block request/decode va cache hit/miss.
- `close()`: dong dung tai nguyen do reader so huu.

Vi du lifecycle:

```python
with TiffReader(...) as reader:
    for row_start in range(0, reader.shape[0], patch_size):
        strip = reader.read_rows(row_start, row_end)
```

`process_large_image()` chi phu thuoc reader contract, khong goi truc tiep `tifffile.memmap()` hay `tiff.imread()` trong row loop.

## 4. Backend selection

Backend duoc chon theo capability thuc te, khong phan loai chi theo compression.

```text
memmap thanh cong va input contract hop le
    -> MemmapReader
memmap that bai va true-block backend da duoc chung minh
    -> BlockReader
khong co block backend va one-time cache qua guard
    -> DecodedCacheReader
con lai
    -> fail som
```

### 4.1. Memmap

- Thu `tifffile.memmap()` dung mot lan khi khoi tao reader.
- Chap nhan khi memmap thanh cong va axes/layout da validation.
- Slice row truc tiep tu mapped array.
- Memmap duoc xem la hop le cho `auto` va `stream`.

Khong suy luan truoc rang TIFF khong nen se memmap duoc hoac TIFF nen se khong memmap duoc.

### 4.2. One-time decoded cache

Phase 1 phai ho tro fallback chac chan:

- Giai ma toan anh dung mot lan vao RAM cache hoac disk-backed memmap.
- Moi row sau chi slice cache.
- Chon RAM/disk theo policy tai muc 5.
- Khong can Zarr/Rasterio cho phase 1.

### 4.3. True block streaming

Backend PoC uu tien la `tifffile.aszarr` ket hop Zarr voi version duoc pin. Rasterio/GDAL la backend thu hai. Khong uu tien API noi bo `dataoffsets`/`databytecounts` neu chua co adapter va compatibility tests.

Exit gate:

1. Neu Zarr dat pixel correctness, block-decode count, peak memory va benchmark tren Jetson: bat trong `stream` va `auto`.
2. Neu Zarr khong cai duoc hoac khong dat: danh gia Rasterio/GDAL theo cung gate.
3. Neu ca hai khong dat: phase 1 van duoc phat hanh; `stream` chi ho tro memmap va bao unsupported cho TIFF can block decoder, con `auto` fallback sang guarded one-time cache.

Dependency block backend chi duoc bat buoc khi backend do duoc enable.

### 4.4. Physical-block cache

Block cache dung LRU theo byte budget.

- Cache key: `(series, level, page, sample_plane, physical_block_index)`.
- Default budget: `64 MiB`.
- Evict block cu nhat den khi block moi vua budget.
- Reader phai tinh minimum overlap budget cho row-order traversal truoc khi decode.
- Neu budget khong du de tranh re-decode block giao nhau, `stream` phai fail hoac doi iteration sang block-aligned; khong duoc ngam decode lap.
- Gia tri `0` chi duoc tat LRU khi backend/iteration chung minh khong doc lai block.
- Block cache bytes va decoder working buffers phai duoc cong vao RAM guard.

Tieu chi cho traversal toan anh theo row tang dan: moi physical block can thiet duoc decode toi da mot lan. Neu backend khong the dam bao, no khong dat exit gate.

## 5. RAM va disk budget

Guard duoc tinh sau khi TensorRT/CUDA da khoi tao, hoac phai tru `runtime_reserve` bao thu. Moi phep tinh dung Python integer.

### 5.1. Cac dai luong chung

```text
decoded_bytes = H * W * C * dtype.itemsize

decoder_peak = decoder_workers * (
    compressed_block_buffer
    + decoded_block_buffer
    + byte_order_or_layout_copy
)

batch_bytes = batch_size * C * patch_size * patch_size * 4

batch_and_copy_peak =
    raw_strip_working_set
    + patch_list_bytes
    + stacked_batch_bytes
    + cast_copy_bytes
```

Code hien tai co the giu patch list, output cua `np.stack()` va output `.astype(np.float32)` cung luc. Guard phase 1 phai tinh ca ba ban sao cho den khi implementation loai bo va test chung minh copy du thua da bien mat.

### 5.2. RAM decoded cache

```text
ram_cache_peak =
    decoded_bytes
    + decoder_peak
    + batch_and_copy_peak
    + runtime_reserve
```

Chi chon RAM cache neu:

- `decoded_bytes <= max_ram_cache_gib`.
- `ram_cache_peak <= available_ram` sau khi TensorRT/CUDA khoi tao.

### 5.3. Disk-backed decoded cache

```text
disk_cache_peak_ram =
    decoder_peak
    + mapped_working_set
    + batch_and_copy_peak
    + runtime_reserve

disk_required =
    decoded_bytes
    + temporary_output_allowance
    + filesystem_headroom
```

Disk cache khong cong toan bo `decoded_bytes` vao peak RAM. `mapped_working_set` gom cac page mapped dang active va duoc gioi han boi implementation.

Chi chon disk cache neu:

- `decoded_bytes <= max_disk_cache_gib`.
- `disk_cache_peak_ram <= available_ram`.
- Free disk lon hon hoac bang `disk_required` truoc khi tao/truncate cache.

`temporary_output_allowance` toi thieu bang hai lan kich thuoc output mask uoc tinh. `filesystem_headroom` mac dinh la gia tri lon hon giua `1 GiB` va `10%` tong dung luong filesystem.

### 5.4. Default budget va cache-selection policy

Default ban dau cho Jetson Nano, can duoc xac nhan lai bang benchmark:

```text
tiff_cache_mode = auto
max_ram_cache_gib = 0.5
max_disk_cache_gib = 8.0
runtime_reserve_gib = 1.5
tiff_block_cache_mib = 64
```

Che do `auto`:

1. Chon RAM neu ca cache-size guard va available-RAM guard dat.
2. Neu RAM khong dat, chon disk neu RAM working-set guard va disk guard dat.
3. Neu ca hai khong dat, fail som.

Che do `ram` va `disk` khong tu dong chuyen sang loai cache khac. RAM du nhung disk thieu khong anh huong `ram`; disk du nhung RAM working set thieu van phai fail.

## 6. Ownership, path va cleanup

Reader va output writer phai dung context manager va ownership flag ro rang:

- `owns_source_cache`: source cache noi bo luon xoa khi thanh cong hoac exception.
- `owns_mask_cache`: mask cache noi bo xoa; mask cache do user chi dinh chi flush/close.
- `owns_output_temp`: output temp xoa neu chua replace thanh cong.

Path policy:

- Cache directory mac dinh: `.cube_nano-cache` trong thu muc cha cua `out_mask`.
- Source cache va mask cache dung ten file rieng va exclusive create.
- Cache path do user chi dinh neu da ton tai thi fail; phase 1 khong overwrite/reuse file cu.
- `mask_cache=""` bi tu choi nhu input khong hop le.
- Kiem tra writable va disk guard truoc khi tao/truncate file lon.
- Output temp nam cung directory/filesystem voi `out_mask`.
- Ghi, flush va dong output temp truoc khi `os.replace()`.
- Neu replace that bai, giu nguyen output cu va xoa output temp neu co the.
- Tren Windows, dong explicit moi memmap/file mapping truoc cleanup hoac replace.

## 7. CLI contract

```text
--tiff_read_mode auto|stream|full              default: auto
--tiff_cache_mode auto|ram|disk                default: auto
--max_ram_cache_gib <positive-value>           default: 0.5
--max_disk_cache_gib <positive-value>          default: 8.0
--runtime_reserve_gib <positive-value>         default: 1.5
--tiff_block_cache_mib <non-negative-value>    default: 64
--tiff_cache_dir <path>                        default: <out-mask-parent>/.cube_nano-cache
--tiff_series <non-negative-index>             default: none; auto 0 only when unique
--tiff_level <non-negative-index>              default: none; auto 0 only when unique
--channel_mapping <mapping>                    default: none
--input_sidecar <path>                         default: none
```

Ngu nghia read mode:

- `auto`: memmap -> block backend da duoc chung minh -> one-time cache qua guard.
- `stream`: chi memmap hoac true block backend; khong full decode.
- `full`: one-time decoded cache qua RAM/disk guard.

Don vi la GiB va MiB. Gia tri am/NaN/vo han bi tu choi. `tiff_block_cache_mib=0` chi hop le theo quy tac tai muc 4.4.

## 8. Dependency va codec

Can pin hoac gioi han version cho tung backend:

- Phase 1: `tifffile`; `imagecodecs` neu codec can.
- Zarr backend: `tifffile`, `zarr`, `imagecodecs` theo ma tran da test.
- Rasterio backend: Rasterio/GDAL build da test tren Jetson ARM64.

Ma tran codec toi thieu: Deflate, LZW va JPEG. Reader capability-check codec khi khoi tao va fail truoc patch dau tien neu decoder khong kha dung.

Moi release artifact phai ghi version cua backend, codec va TensorRT trong provenance.

## 9. Verification plan

### 9.1. Pixel va input contract

- Moi `read_rows()` khop pixel voi full-read oracle.
- RGB 3 kenh hop le.
- RGBNIR 4 kenh hop le voi sidecar/mapping.
- RGBA bi tu choi va alpha khong bi xem la NIR.
- Sai band order, thieu mapping va photometric khong ho tro bi tu choi som.
- HWC, CHW, planar contiguous va planar separate.
- Multi-page, pyramid va axes mo ho bi tu choi truoc inference.

### 9.2. Backend va block cache

- Memmap success path khong full decode.
- TIFF compressed tiled va stripped.
- Tile height/`rowsperstrip` khong chia het `patch_size`.
- Moi physical block decode toi da mot lan trong traversal chuan.
- LRU eviction, budget bang 0 va budget nho hon minimum overlap requirement.
- Backend PoC that bai capability check khong duoc quang ba la `stream`.

### 9.3. RAM, disk va copy peak

- RAM cache va disk-backed cache duoc test rieng.
- Guard fail truoc decoder/cap phat lon.
- Disk guard fail truoc create/truncate file lon.
- Batch peak tinh patch list, stack va cast copy.
- Runtime reserve va block cache duoc cong vao guard.

### 9.4. Fault injection va cleanup

- Exception sau khi tao source cache.
- Exception giua inference.
- Exception khi ghi, flush va atomic replace output.
- Cache noi bo xoa; cache user giu lai.
- Existing user cache path bi tu choi.
- File output cu khong bi hu khi replace that bai.
- Test Windows bao dam memmap duoc dong truoc xoa/replace.

### 9.5. Integration

- Chay `process_large_image()` voi fake TensorRT oracle.
- So sanh strip data truoc, sau do moi so sanh output mask.
- Kiem tra edge tile/padding khong ghi ngoai kich thuoc anh.
- Test ca `auto`, `stream`, `full` va tung cache mode.

## 10. Benchmark plan

Moi benchmark ghi lai:

- Input file, codec, tiled/stripped, block size, dtype, axes va shape.
- Cold-cache va warm-cache run.
- Peak RSS, CUDA/TensorRT memory va disk cache.
- `blocks_requested`, `blocks_decoded`, `cache_hits`, `cache_misses`.
- Median/p95 cua read latency va total inference latency.
- Jetson model, JetPack/TensorRT version, batch size va power mode.
- It nhat nam lan lap cho moi cau hinh sau mot warm-up run.

## 11. Thu tu trien khai

### Phase 1: Lifecycle fix va guarded one-time cache

1. Tao `ImageBlockReader`/`TiffReader` context manager.
2. Them input contract va channel-semantics validation.
3. Dung memmap mot lan theo capability.
4. Them one-time RAM/disk decoded cache va guard tach biet.
5. Them ownership, exclusive cache create va atomic output.
6. Them CLI phase 1 va tests tuong ung.

Phase 1 sua chac chan repeated full decode. `stream` chi ho tro memmap cho den khi phase 2 dat exit gate.

### Phase 2: True block streaming PoC

1. Thu `tifffile.aszarr` + Zarr tren corpus TIFF production.
2. Chay pixel, block count, peak memory va benchmark tren Jetson.
3. Neu khong dat, thu Rasterio/GDAL.
4. Chi enable backend dat toan bo exit gate.
5. Them LRU block cache va block metrics.

### Phase 3: Chuan hoa production input

1. Chot format artifact: memmap-compatible TIFF, block-stream-compatible TIFF hoac `.npy`.
2. Bat buoc sidecar cho multispectral RGBNIR.
3. Luu dtype, axes, band order, normalization va provenance.
4. Cong bo ma tran codec/backend duoc ho tro tren Jetson.

## 12. Tieu chi nghiem thu

- Tai lieu chi co mot hop dong, khong con dac ta cu/moi mau thuan.
- RGBNIR duoc xac nhan bang metadata/mapping; RGBA bi tu choi.
- Moi strip khop pixel oracle va luon tra HWC.
- Khong full-decode lap theo so row strip.
- Memmap/block/cache backend duoc chon theo capability.
- RAM cache, disk cache va disk-space guard dung mo hinh rieng.
- Guard fail truoc allocation/decoder lon.
- Block streaming chi duoc enable sau exit gate; moi block decode toi da mot lan trong traversal chuan.
- Codec khong ho tro bi phat hien truoc inference.
- Cache noi bo va output temp duoc cleanup; user cache va output cu duoc bao toan dung policy.
- Benchmark tren Jetson dat budget RAM, disk va latency da phe duyet.
