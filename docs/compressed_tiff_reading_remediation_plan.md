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
- Chuan hoa layout dau ra ve HWC; viec select/reorder kenh tuan theo muc 2.2.
- Kiem tra so kenh output voi input binding thuc te va engine manifest truoc khi doc patch dau tien.

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
- `ExtraSamples=ASSOCALPHA` va `ExtraSamples=UNASSALPHA` luon bi tu choi; alpha khong duoc xem la NIR.
- Dung mot `ExtraSamples=UNSPECIFIED` o sample thu tu chi duoc chap nhan khi sidecar hoac mapping tuong minh xac nhan sample do la NIR.
- TIFF 4 sample khong co sidecar/mapping bi tu choi, ke ca khi extra sample la `UNSPECIFIED`.
- Nhieu hon mot extra sample hoac gia tri `ExtraSamples` khac bi tu choi neu chua co contract rieng.
- TIFF 3 sample khong co semantics RGB ro rang phai co sidecar hoac mapping tuong minh.

Quy tac select/reorder:

- Neu khong co sidecar va mapping, reader giu thu tu vat ly va chi chap nhan khi TIFF metadata chung minh thu tu do da la canonical.
- Neu co sidecar hoac mapping, reader select/reorder tren tung strip hoac patch ve `[red, green, blue]` hoac `[red, green, blue, nir]` truoc khi tra du lieu; khong tao ban sao reorder toan anh.
- `reader.band_order` luon mo ta thu tu output sau mapping, khong mo ta thu tu vat ly trong file.
- Mapping phai co dung cac role ma engine can, khong trung index, khong thieu role va khong vuot so kenh cua anh.
- Mapping identity va non-identity dung chung contract; reader khong tu suy doan BGR/RGB hoac NIR.

Production TIFF phai co input sidecar JSON voi schema duoc version hoa. Vi du:

```json
{
  "schema_version": 1,
  "source_fingerprint": {
    "algorithm": "sha256",
    "digest": "<sha256-of-tiff>"
  },
  "axes": "YXC",
  "shape": [10000, 10000, 4],
  "band_order": ["red", "green", "blue", "nir"],
  "dtype": "uint16",
  "input_spec_id": "<input-spec-id>",
  "normalization": "<normalization-contract-id>"
}
```

Voi arbitrary TIFF, cung cap mapping:

```text
--channel_mapping red=0,green=1,blue=2,nir=3
```

Input sidecar phai duoc validate truoc khi doc pixel:

- `source_fingerprint` phai khop TIFF dang mo; khong duoc dung sidecar cua file khac. Fingerprint duoc tao trong buoc dong goi artifact, khong suy ra chi tu ten file.
- `input_sidecar.band_order` mo ta role cua cac sample theo thu tu vat ly trong source; `reader.band_order` mo ta output canonical sau reorder.
- `axes`, `shape`, `dtype`, so band va moi semantics TIFF khai bao phai khop sidecar; band mapping phai bao phu dung cac physical index va khong mau thuan voi photometric/extra-sample metadata.
- `input_spec_id` va `normalization` phai khop engine manifest.
- Neu co ca `--input_sidecar` va `--channel_mapping`, hai nguon phai resolve thanh cung role-to-index mapping; mau thuan thi fail, khong nguon nao ngam override nguon kia.

Phase 1 thay `_normalize_patch()` bang dispatch theo fixed normalization cua `InputSpec`. Truong `normalization` chon dung contract do; khong chi duoc ghi nhan/validate trong khi runtime van dung heuristic theo dtype hoac `patch.max()`.

### 2.3. TensorRT input contract

So kenh, layout, spatial shape va dtype model lay tu input binding thuc te. Engine manifest co `schema_version`, `engine_fingerprint`, `input_spec_id`, `normalization`, `band_order` va optimization profile. Fingerprint phai khop engine bytes dang load; shape/dtype/profile phai khop binding; cac truong `InputSpec` phai khop input sidecar.

- Tham so CLI `channels` neu con duoc giu de tuong thich chi la assertion; no khong duoc override binding. Mismatch phai fail khi khoi tao.
- Reader output phai co exact channel count va canonical `band_order` ma engine manifest yeu cau.
- `CloudTRTInfer._prepare_input()` phai fail khi channel count khong khop; khong pad kenh thieu va khong truncate kenh thua. Batch padding theo fixed engine batch size la contract rieng va van co the duoc ho tro.
- Engine/manifest khong co du thong tin de xac minh `InputSpec` bi tu choi trong production mode.

## 3. Reader contract

Tao interface noi bo `ImageBlockReader` va implementation `TiffReader` dang context manager.

Reader cung cap:

- `shape`: `(H, W, C)` sau validation.
- `dtype`: dtype goc.
- `axes`: axes goc va axes da chuan hoa.
- `band_order`: thu tu output canonical sau select/reorder.
- `read_rows(row_start, row_end)`: tra ve HWC dung pixel va dung thu tu kenh canonical.
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

Select/reorder chi duoc thuc hien tren view/array cua strip hoac patch dang xu ly. Memory guard phai tinh moi copy cuc bo do advanced indexing tao ra; implementation khong duoc materialize mot ban sao toan anh chi de doi thu tu kenh.

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

Moi memory guard dung cung mot cong thuc, duoc danh gia sau khi TensorRT/CUDA da khoi tao:

```text
operation_peak_without_reserve + runtime_reserve_bytes
    <= MemAvailable_after_TensorRT_initialization
```

`runtime_reserve_bytes` chi xuat hien o ve trai va chi duoc tinh mot lan. Implementation khong duoc vua tru reserve khoi `MemAvailable` vua cong reserve vao operation peak.

`MemoryInfoProvider.available_bytes()` tren Jetson doc `MemAvailable` tu `/proc/meminfo` sau khi engine, execution context va CUDA buffers da duoc cap phat. Unit test dung provider gia co gia tri xac dinh. Neu backend, `decoder_workers`, batch size hoac mapped-window policy thay doi sau capability check, guard phai duoc tinh lai truoc allocation/decode tiep theo. Moi phep tinh dung Python integer bytes.

### 5.1. Cac dai luong chung

```text
decoded_bytes = H * W * C * dtype.itemsize

compressed_block_buffer = max(databytecounts cua cac block lien quan)

decoded_block_buffer = max(
    block_height * block_width * samples_in_decoded_block * dtype.itemsize
)

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

`decoder_workers` phai duoc truyen tuong minh va co bound co dinh, default phase 1 la `1`; khong de `tifffile` tu chon worker. `compressed_block_buffer` lay tu gia tri lon nhat trong `databytecounts` cua page/plane dang duoc doc. `decoded_block_buffer` lay tu block geometry, planar configuration va decoded dtype. `byte_order_or_layout_copy` bao gom copy do byteswap, chuyen axes va channel reorder tren working strip.

Voi true block backend:

```text
stream_operation_peak =
    decoder_peak
    + block_cache_bytes
    + batch_and_copy_peak
```

`block_cache_bytes` bang byte budget thuc te cua LRU, khong chi la tong payload hien co khi do guard.

True block backend chi duoc chon khi:

```text
stream_operation_peak + runtime_reserve_bytes
    <= MemAvailable_after_TensorRT_initialization
```

### 5.2. RAM decoded cache

```text
ram_cache_operation_peak =
    decoded_bytes
    + decoder_peak
    + batch_and_copy_peak
```

Chi chon RAM cache neu:

- `decoded_bytes <= max_ram_cache_bytes`.
- `ram_cache_operation_peak + runtime_reserve_bytes <= MemAvailable_after_TensorRT_initialization`.

### 5.3. Disk-backed decoded cache

```text
disk_cache_operation_peak =
    decoder_peak
    + mapped_working_set
    + batch_and_copy_peak
```

Disk cache khong mac dinh cong toan bo `decoded_bytes` vao peak RAM. Implementation phai gioi han read/write mapping theo mot `mapped_window_bytes` cu the va cong ca OS readahead margin da do tren target:

```text
mapped_working_set = mapped_window_bytes + mapped_readahead_margin
```

Moi window phai duoc flush/release truoc khi tien qua bound. Neu backend/OS path khong the enforce hoac chung minh bound nay, guard phai bao thu dat `mapped_working_set = decoded_bytes` hoac tu choi disk backend; khong duoc gia dinh memmap co working set nho ma khong do.

Chi chon disk cache neu:

- `decoded_bytes <= max_disk_cache_bytes`.
- `disk_cache_operation_peak + runtime_reserve_bytes <= MemAvailable_after_TensorRT_initialization`.
- Tat ca filesystem guard tai muc 5.4 dat truoc khi tao/truncate cache.

### 5.4. Disk guard theo filesystem/device

Moi file se duoc tao co mot allocation record `(resolved_parent_device, required_bytes, purpose)`. Toan bo record duoc group theo device thuc te truoc khi tao file:

```text
source_cache_bytes = decoded_bytes
source_cache_temp_bytes = backend-specific worst-case temporary allocation
output_temp_bytes = output_height * output_width * output_dtype.itemsize
mask_cache_bytes = mask_height * mask_width * mask_dtype.itemsize
```

Backend khong tao source-cache temporary file dat `source_cache_temp_bytes=0`; backend co tao temp phai khai bao bound truoc capability check.

```text
cache filesystem:
    decoded source cache
    + source-cache temporary files neu backend can

output filesystem:
    output mask temp
    + internal mask cache neu cung device

user mask-cache filesystem:
    user-provided mask cache neu nam tren device rieng
```

Neu nhieu path resolve ve cung device, nhu cau cua chung duoc cong va chi them headroom mot lan. Neu khac device, tung device phai dat guard doc lap:

```text
device_required = sum(required_bytes tren device)
    + max(1 GiB, 10% tong dung luong device)

free_bytes_on_device >= device_required
```

`FilesystemInfoProvider` resolve parent dang ton tai thanh device ID (`st_dev` tren Linux, volume ID tren Windows) va cung cap total/free bytes; unit test dung provider gia. Guard cho moi device phai chay thanh cong het truoc allocation dau tien, de khong de lai file lon mot phan khi device sau thieu cho. Race/`ENOSPC` luc ghi van phai duoc bat va cleanup theo muc 6.

Disk guard ap dung cho moi read mode, ke ca RAM cache hoac stream, vi output temp/mask cache van can disk. `max_disk_cache_bytes` chi gioi han decoded source cache, khong thay the guard cho output va mask.

### 5.5. Default budget va cache-selection policy

Default ban dau cho Jetson Nano, can duoc xac nhan lai bang benchmark:

```text
tiff_cache_mode = auto
max_ram_cache_gib = 0.5
max_disk_cache_gib = 8.0
runtime_reserve_gib = 1.5
tiff_block_cache_mib = 64
```

Policy duoi day chi ap dung khi read mode can one-time decoded cache. `tiff_cache_mode=auto`:

1. Chon RAM neu ca cache-size guard va available-RAM guard dat.
2. Neu RAM khong dat, chon disk neu RAM working-set guard va disk guard dat.
3. Neu ca hai khong dat, fail som.

Che do `ram` va `disk` khong tu dong chuyen sang loai cache khac. Thieu cho cho decoded source cache khong anh huong `ram`, nhung disk guard cho output/mask van bat buoc trong moi mode. Disk du nhung RAM working set thieu van phai fail.

Gia tri GiB/MiB tu CLI duoc parse tu decimal string va chuyen mot lan thanh integer bytes (`max_ram_cache_bytes`, `max_disk_cache_bytes`, `runtime_reserve_bytes`, `block_cache_bytes`) truoc moi phep so sanh. Khong so sanh `decoded_bytes` truc tiep voi float GiB/MiB.

## 6. Ownership, path va cleanup

Reader va output writer phai dung context manager va ownership flag ro rang:

- `owns_source_cache`: source cache noi bo luon xoa khi thanh cong hoac exception.
- `owns_mask_cache`: mask cache noi bo xoa; mask cache do user chi dinh chi flush/close.
- `owns_output_temp`: output temp xoa neu chua replace thanh cong.

Path policy:

- Cache directory mac dinh: `.cube_nano-cache` trong thu muc cha cua `out_mask`.
- Repository ignore `.cube_nano-cache/`; ignore rule khong thay the ownership va cleanup policy.
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

Interaction contract:

| Read mode | Backend selection | Vai tro cua `tiff_cache_mode` |
|---|---|---|
| `auto` | Thu memmap, sau do block backend da dat exit gate, cuoi cung one-time cache | Chi chon `auto`, `ram` hoac `disk` cho one-time-cache fallback |
| `full` | Bo qua memmap va block backend; dung one-time cache | `auto` thu RAM roi disk; `ram`/`disk` ep dung loai tuong ung |
| `stream` | Chi memmap hoac true block backend; khong full decode | Phai la `auto`; `ram` hoac `disk` bi tu choi vi decoded cache khong duoc dung |

Trong `stream`, neu user truyen tuong minh `--max_ram_cache_gib` hoac `--max_disk_cache_gib` thi CLI fail vi option khong co tac dung; gia tri default khong duoc consult. `--tiff_block_cache_mib` van ap dung cho true block backend va `--runtime_reserve_gib` van ap dung cho memory guard. Khong option nao duoc chap nhan roi bo qua im lang.

Validation numeric:

- `max_ram_cache_gib`, `max_disk_cache_gib` va `runtime_reserve_gib` phai finite va lon hon `0`; gia tri `0` bi tu choi.
- `tiff_block_cache_mib` phai finite va lon hon hoac bang `0`; gia tri `0` chi hop le theo quy tac tai muc 4.4.
- GiB/MiB duoc chuyen sang integer bytes ngay sau parse nhu muc 5.5; moi guard so sanh byte voi byte.
- `tiff_series` va `tiff_level` phai la integer lon hon hoac bang `0` neu duoc cung cap.

Neu CLI `--channels` hien tai duoc giu trong phase 1, gia tri do phai khop input binding va engine manifest; mismatch fail thay vi pad/truncate du lieu.

## 8. Dependency va codec

Can pin hoac gioi han version cho tung backend:

- Phase 1: `tifffile`; `imagecodecs` neu codec can.
- Zarr backend: `tifffile`, `zarr`, `imagecodecs` theo ma tran da test.
- Rasterio backend: Rasterio/GDAL build da test tren Jetson ARM64.

Ma tran codec toi thieu: Deflate, LZW va JPEG. Reader capability-check codec khi khoi tao va fail truoc patch dau tien neu decoder khong kha dung.

Moi release artifact phai ghi version cua backend, codec va TensorRT trong provenance.

## 9. Verification plan

### 9.1. Pixel va input contract

- Moi `read_rows()` khop pixel canonical voi full-read oracle sau cung select/reorder.
- RGB 3 kenh hop le.
- RGBNIR co mot `UNSPECIFIED` extra sample va mapping/sidecar dung duoc chap nhan.
- `UNSPECIFIED` thieu mapping/sidecar, `UNASSALPHA` va `ASSOCALPHA` deu bi tu choi.
- TIFF BGR voi `red=2,green=1,blue=0` tra dung RGB pixel; mapping identity va non-identity deu co oracle test.
- Mapping trung index, thieu role, role du, index vuot range va sai channel count bi tu choi som.
- `reader.band_order` phan anh output canonical sau mapping.
- Sidecar sai source fingerprint, axes, shape, dtype, band order hoac so band bi tu choi.
- Sidecar va CLI mapping mau thuan bi tu choi; cung mapping thi duoc chap nhan.
- Sidecar `input_spec_id`/normalization khong khop engine manifest bi tu choi.
- Fixed normalization tu `InputSpec` tao dung tensor oracle; runtime khong dung heuristic theo `patch.max()`.
- Photometric khong ho tro bi tu choi som.
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
- Fake `MemoryInfoProvider` test boundary pass/fail cua cong thuc duy nhat va chung minh `runtime_reserve` khong bi tinh hai lan.
- Guard fail truoc decoder/cap phat lon va duoc tinh lai khi backend hoac `decoder_workers` thay doi.
- `compressed_block_buffer` va `decoded_block_buffer` duoc tinh tu `databytecounts`, block geometry, planar configuration va dtype.
- Disk-backed cache enforce `mapped_working_set`; backend khong chung minh duoc bound phai tinh bao thu toan `decoded_bytes` hoac bi tu choi.
- Batch peak tinh patch list, stack va cast copy.
- Stream peak tinh day du decoder buffers va byte budget cua block cache.
- Fake `FilesystemInfoProvider` test cache device du cho/output device thieu cho va truong hop nguoc lai.
- Test cache, output va user mask tren ba device rieng; cac path cung device duoc cong requirement va headroom dung mot lan.
- Moi device fail guard truoc create/truncate file dau tien.
- Gia tri GiB/MiB duoc convert thanh bytes truoc so sanh voi `decoded_bytes`.

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
- Reader output channel count khac input binding bi tu choi truoc patch dau tien.
- `_prepare_input()` khong pad/truncate channel mismatch; batch padding duoc test rieng.
- So sanh strip data truoc, sau do moi so sanh output mask.
- Kiem tra edge tile/padding khong ghi ngoai kich thuoc anh.
- Test day du interaction matrix cua `auto`, `stream`, `full` va tung cache mode.
- `stream` tu choi cache mode `ram`/`disk` va explicit decoded-cache size option.
- Numeric CLI test `0`, so am, NaN va vo han theo dung contract cua tung option.

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
2. Them `ExtraSamples` validation va explicit channel select/reorder tren strip/patch.
3. Them input-sidecar fingerprint, engine manifest, fixed `InputSpec` normalization va strict TensorRT binding validation.
4. Dung memmap mot lan theo capability.
5. Them one-time RAM/disk decoded cache, `MemoryInfoProvider` va guard theo tung filesystem/device.
6. Them ownership, exclusive cache create va atomic output.
7. Them CLI interaction matrix, numeric validation va tests tuong ung.

Phase 1 sua chac chan repeated full decode. `stream` chi ho tro memmap cho den khi phase 2 dat exit gate.

### Phase 2: True block streaming PoC

1. Thu `tifffile.aszarr` + Zarr tren corpus TIFF production.
2. Chay pixel, block count, peak memory va benchmark tren Jetson.
3. Neu khong dat, thu Rasterio/GDAL.
4. Chi enable backend dat toan bo exit gate.
5. Them LRU block cache va block metrics.

### Phase 3: Chuan hoa production input

1. Chot format artifact: memmap-compatible TIFF, block-stream-compatible TIFF hoac `.npy`.
2. Bat buoc sidecar/fingerprint cho multispectral RGBNIR va engine manifest cho TensorRT artifact.
3. Luu dtype, axes, shape, band order, `input_spec_id`, normalization va provenance.
4. Cong bo ma tran codec/backend duoc ho tro tren Jetson.

## 12. Tieu chi nghiem thu

- Tai lieu chi co mot hop dong, khong con dac ta cu/moi mau thuan.
- RGBNIR `UNSPECIFIED` dung duoc chap nhan khi co mapping/sidecar; moi alpha va anh 4 sample mo ho bi tu choi.
- Mapping/sidecar reorder ve canonical band order tren strip/patch; moi strip khop pixel oracle va luon tra HWC.
- Input sidecar khop TIFF fingerprint va cung `InputSpec`/normalization voi engine manifest.
- TensorRT binding la nguon shape/channel thuc te; channel mismatch fail va khong bi pad/truncate.
- `_normalize_patch()` dung fixed normalization cua `InputSpec`, khong dung heuristic ngam.
- Khong full-decode lap theo so row strip.
- Memmap/block/cache backend duoc chon theo capability.
- RAM reserve duoc tinh dung mot lan sau TensorRT init; decoder worker/buffer va mapped working set co bound test duoc.
- Disk requirement duoc group va guard doc lap theo filesystem/device truoc allocation.
- Read/cache mode interaction va zero-value CLI validation dung ma tran tai muc 7.
- Block streaming chi duoc enable sau exit gate; moi block decode toi da mot lan trong traversal chuan.
- Codec khong ho tro bi phat hien truoc inference.
- Cache noi bo va output temp duoc cleanup; user cache va output cu duoc bao toan dung policy.
- Benchmark tren Jetson dat budget RAM, disk va latency da phe duyet.
