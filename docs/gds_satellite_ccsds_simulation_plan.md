# Ke hoach he thong mo phong ve tinh va GDS giao tiep CCSDS

> Trang thai: Ban ke hoach ky thuat da cap nhat sau expert review
> Ngay lap: 2026-07-19
> Ngay cap nhat: 2026-07-19
> Pham vi mac dinh: Software-in-the-loop, mo phong duong truyen o muc byte/frame, chua bao gom RF/SDR.
> Co so cap nhat: [expert review](review_gds_simulation_plan.md), [phan bien ky thuat](review_gds_simulation_plan_rebuttal.md) va doi chieu source F Prime v4.1.0.

## 1. Muc tieu

Xay dung mot he thong mo phong gom hai module chinh:

1. **Satellite Simulator** mo phong ve tinh, nhan telecommand (TC), phat telemetry (TM), chay model phat hien may va tao san pham anh theo scene/ROI.
2. **Ground Data System (GDS)** co webapp de quan sat trang thai, chon scene, keo-tha ROI, cau hinh nguong may, gui command va nhan ket qua qua luong TC/TM thuc su.

Muc tieu nghiem thu quan trong nhat la giao dien web khong duoc goi truc tiep ham inference. Moi thao tac nghiep vu phai di theo luong:

```text
Web UI -> GDS command service -> CCSDS TC -> Link Simulator
       -> Satellite decoder/dispatcher -> AI inference
       -> CCSDS TM/data product -> GDS decoder -> Web UI
```

## 2. Hien trang repository

### 2.1 Model va inference

- Model hien tai la `MobileNetV3-Small`, dau ra mot logit cloud/clear cho moi patch, khong phai segmentation pixel-level: [mobilenetv3.py](../src/models/mobilenetv3.py#L5).
- Checkpoint [best_model.pth](../checkpoints/best_model.pth) thuc te nhan anh RGB **3 kenh**, patch `256 x 256`. Moi entry point cua Satellite Simulator phai lay `channels`, `band_order`, `patch_size` va normalization tu Model/InputSpec manifest; khong duoc dua vao default 4 kenh cua CLI/helper hien tai.
- `CloudTorchInfer` la diem tich hop phu hop de nap model mot lan va goi batch inference: [inference_large_image.py](../src/inference_large_image.py#L20).
- Pipeline anh lon da ho tro TIFF, JP2, NumPy, HDF va NetCDF; chia anh thanh patch, chay batch va tao mask: [inference_large_image_trt.py](../src/inference_large_image_trt.py#L559).
- Ket qua hien co gom `accepted`, `cloud_coverage`, nguong, duong dan mask, reader metrics va latency: [inference_large_image_trt.py](../src/inference_large_image_trt.py#L770).
- `cloud_coverage` hien tai la ty le dien tich cac patch duoc phan loai la may. Day la coarse tile coverage, khong phai ty le pixel may chinh xac.
- Pipeline hien quet toan scene. `TiffReader.read_rows()` doc tron chieu ngang va chua co API doc mot cua so ROI: [tiff_reader.py](../src/tiff_reader.py#L592).

### 2.2 Du lieu scene

- Scene mau `T48PYS_RGB.tif` co kich thuoc `10980 x 10980 x 3`, kieu `uint16`.
- File TIFF hien tai khong con CRS/geotransform. MVP phai dinh nghia ROI bang toa do pixel `x, y, width, height`.
- Neu can chon theo kinh do/vi do, pipeline ingest phai giu GeoTIFF metadata va bo sung GDAL/rasterio truoc khi mo tinh nang nay.
- Phan raster thu cua scene co kich thuoc `10980 x 10980 x 3 x uint16 = 723,362,400` byte, tuong duong `723.36 MB` hoac `689.85 MiB`; file hien tai la `723,362,624` byte. Webapp khong duoc tai truc tiep file goc; phai dung quicklook va/hoac tile pyramid da duoc downlink ve GDS.

### 2.3 F Prime va CCSDS

- Repository hien chi co dictionary sinh tu F Prime, chua co source `.fpp`, flight deployment hay CCSDS encoder/decoder trong repo.
- Dictionary dang su dung F Prime `v4.1.0`, deployment `ReferenceDeployment`: [fprime_dictionary.json](../fprime_dictionary.json#L3).
- Cau hinh hien tai co Spacecraft ID `68` va TM frame co dinh `1024` byte: [fprime_dictionary.json](../fprime_dictionary.json#L1025).
- Dictionary co 18 command, 43 telemetry channel, 76 event va khong co parameter cho cloud inference/ROI.
- Cac command data handling hien co chi phuc vu dong file va downlink file; can bo sung component va dictionary cho payload cloud.
- Trong F Prime v4.1.0 mac dinh, `FprimeRouter` dien giai truc tiep APID thanh `Fw::ComPacketType`, con `ComQueue` sinh APID tu packet descriptor. Vi vay cac APID mission tuy y nhu `0x120..0x123` khong the dung voi topology stock neu khong co adapter/router rieng.
- `FW_FILE_BUFFER_MAX_SIZE` mac dinh la `512` byte. Voi descriptor 2 byte va FilePacket DATA header 11 byte, moi DATA packet mac dinh chi mang toi da `499` byte file; frame size 1024 byte khong tu dong lam chunk tang len gan 1 KiB.

### 2.4 Baseline kiem thu

- Tai thoi diem lap ke hoach, 53 test hien co deu pass.
- Test PyTorch large-image dang mock model; can them smoke/regression test dung checkpoint that.
- Worktree hien co cac thay doi va file untracked lien quan inference. Can on dinh baseline truoc khi tao nhanh implementation.

## 3. Pham vi va nguyen tac thiet ke

### 3.1 Pham vi MVP

- Software-in-the-loop tren mot may hoac Docker Compose.
- CCSDS Space Packet, TC Transfer Frame Type-BD va TM Transfer Frame.
- Mo phong latency, jitter, loss, duplicate, corruption, bandwidth va blackout.
- Chon ROI theo pixel tren quicklook/tile pyramid.
- Chay PyTorch tren PC; co profile rieng de chuyen sang TensorRT/Jetson.
- TM trang thai, event, command acknowledgement va downlink data product.
- GDS ho tro command delivery `immediate` va persisted `next_contact`; mode mac dinh la `immediate`.
- Mission adapter MVP chi nhan scene TIFF memmap-compatible, single-series/single-level, co InputSpec sidecar hop le. Cac backend JP2, HDF, NetCDF va TIFF can full decode van la cong cu development cho toi khi co profile backend da benchmark va duoc enable ro rang.

### 3.2 Chua nam trong MVP

- RF modulation/demodulation, antenna, SDR va link budget.
- TC Type-AD day du voi COP-1/FOP/FARM/CLCW.
- CCSDS Space Data Link Security (SDLS).
- Chon ROI theo lat/lon khi input chua co georeference.
- Cloud segmentation pixel-level.

### 3.3 Nguyen tac

- Command, telemetry va data product phai co `RequestKey`/correlation identity xuyen suot.
- Khong dung float truc tiep cho cac nguong trong wire contract. Dung `U16` basis point, mien `0..10000`.
- Khong tao crop TIFF tam truoc inference. Doc truc tiep ROI tu source.
- Model chi duoc nap mot lan; GPU worker co bounded queue.
- Moi output phai co checksum, model version va config snapshot da su dung.
- Dictionary/protocol profile la single source of truth; UI khong hard-code opcode hay payload layout.
- Model artifact va InputSpec la mot bundle bat bien. Satellite khong duoc vao `READY` neu thieu manifest hoac checkpoint/manifest khong khop.
- Runtime khong tu suy doan channels, band order, patch size hay normalization tu dtype, filename hoac default CLI.
- Moi queue phai co capacity, overflow policy, metric va error code; khong thanh phan nao duoc silent drop.
- Moi command dung `RequestKey = {ground_instance_id: U64, request_id: U32}`. GDS persist ca namespace va sequence qua restart; satellite persist idempotency journal it nhat bang cua so retry/retention, khong chi cache trong RAM.
- Moi job snapshot config tai thoi diem validation/admission. Analysis command phai mang ca hai threshold va `config_epoch/config_revision`; worker khong doc lai global config khi bat dau chay.
- Scene catalog tren satellite la authority cho onboard inventory; GDS chi la verified replica co catalog epoch/revision, scene revision va source checksum.
- MVP chi downlink mot deterministic product bundle cho moi `product_id`; khong mo nhieu reassembly session ngam dinh cho cac artifact cung ID.
- Fault decision phai la ham deterministic cua seed, direction, frame ID va fault type hoac duoc ghi day du de replay; shared PRNG theo timing thread la khong du.

## 4. Kien truc de xuat

```text
+------------------- GDS --------------------+
| React Web                                  |
|   | REST/WebSocket                         |
| FastAPI GDS Backend                        |
|   |- Scene/Product API                     |
|   |- Command Ledger                        |
|   |- Transactional Outbox                  |
|   |- TC Encoder / TM Decoder               |
|   |- Telemetry/Event Store                 |
+---|----------------------------------------+
    | CCSDS frame bytes
+---v----------------------------------------+
| Link Simulator                             |
| latency | jitter | loss | corrupt | bandwidth |
+---|----------------------------------------+
    | CCSDS frame bytes
+---v-------------- Satellite ---------------+
| F Prime CCSDS Stack                        |
|   -> Stock APID Router -> Command Dispatcher|
|   -> CloudPayload component                |
|   -> AI Worker (PyTorch/TensorRT)           |
|   -> Scene/Product Store                    |
|   -> TM/Event/File -> MissionComScheduler   |
|      -> SpacePacketFramer -> TmFramer       |
+--------------------------------------------+
```

### 4.1 Cau truc thu muc muc tieu

```text
flight/                         # F Prime deployment va CloudPayload component
sat_ai/                         # Adapter quanh model/inference hien co
protocol/
  mission_profile.yaml          # Instance ID, SCID, APID, VCID, frame size, CRC, endian
  schemas/                      # Command, TM, event va product schemas
  golden_vectors/               # Packet/frame mau byte-exact
  conformance_matrix.md         # Boundary va muc ho tro theo tung standard/version
link_sim/                       # Transport va fault injection
gds/
  backend/                      # FastAPI, command ledger, TM decoder
  web/                          # React/TypeScript GDS
data/
  satellite/scenes/             # Storage phia satellite
  satellite/products/
  ground/products/              # Chi chua du lieu da downlink
deploy/                         # Docker va Jetson profiles
```

## 5. CCSDS mission profile

Can tao mot tai lieu conformance matrix ghi ro tung chuan, phien ban, tinh nang da ho tro va tinh nang chua ho tro.

| Lop | MVP | Nang cap |
|---|---|---|
| Application | F Prime command/event/channel dictionary; timestamp trong application payload | ECSS PUS neu interoperability o application layer la bat buoc |
| Packet | CCSDS Space Packet; sequence count modulo 16384 theo APID; secondary header absent | CCSDS CUC/CDS secondary header trong profile moi |
| TC link | TC Transfer Frame Type-BD, FECF/CRC, VC0 | Type-AD, ground FOP, onboard FARM va COP-1 |
| TM link | TM Transfer Frame 1024 byte, SCID 68, OCF absent | Nhieu VC, shared master-channel counter, OCF va CLCW |
| Sync/channel coding | Ngoai MVP; UDP datagram mang mot transfer frame | TC CLTU/BCH va TM ASM/coding profile khi noi serial/SDR |
| Data product | Mot deterministic product bundle qua F Prime FileDownlink/FilePacket trong CCSDS TM; full-file retry | CFDP acknowledged mode/selective recovery |
| Security | `local_sil`: host loopback hoac Compose internal network, exact Host/Origin/peer allowlist, khong auth/TLS | Networked GDS co OIDC/RBAC/TLS/CSRF; CCSDS SDLS va anti-replay |

### 5.1 Canh bao ve muc do tuan thu

- CCSDS quy dinh packet va data-link protocol, khong tu dinh nghia command set nghiep vu cloud. Command schema phai duoc dinh nghia boi mission/F Prime dictionary hoac PUS.
- F Prime `TcDeframer` hien tai chi xu ly Expedited Service, Type-BD va khong thuc hien FARM check cho Type-A.
- F Prime `TmFramer` hien tai khong ho tro Operational Control Field, do do chua the phat CLCW day du.
- MVP khong co FOP, FARM, CLCW hay COP-1 retransmission. TC frame sequence khong duoc dung de cam ket delivery.
- GDS retry o application layer: moi lan encode lai command co Space Packet Sequence Count moi nhung giu cung business `RequestKey`. TC Transfer Frame Sequence Number cua Type-BD khong duoc dung de sequence-control delivery. TM command ACK la ACK nghiep vu, khong phai CLCW/link acknowledgement.
- Vi vay MVP duoc mo ta la **CCSDS profile voi TC Type-BD**, khong duoc tuyen bo da ho tro full COP-1. Nang cap Type-AD phai bo sung dong bo ground FOP, onboard FARM, sequence control, OCF/CLCW va return path; khong phai chi doi mot flag.
- Boundary MVP ket thuc tai transfer-frame bytes. CLTU/BCH va ASM/channel coding chi vao scope khi mission chon CCSDS 231/131 sync/channel-coding profile; viec noi serial/SDR tu than khong bat buoc cac profile nay. Khong hard-code ASM truoc khi chon profile.
- MVP giu mapping APID stock cua F Prime v4.1.0. Cloud commands duoc phan biet bang opcode; channel/event/file duoc phan biet bang dictionary ID va F Prime packet descriptor, khong cap APID rieng cho tung nghiep vu cloud.
- Neu mission bat buoc APID tuy y, day la mot profile nang cap gom custom ingress mapper, egress APID selector, topology moi va golden vector rieng; khong duoc chi sua `mission_profile.yaml`.

Nguon tham khao:

- [CCSDS Blue Books](https://ccsds.org/publications/bluebooks/)
- [F Prime v4.1.0 SpacePacketFramer](https://github.com/nasa/fprime/tree/v4.1.0/Svc/Ccsds/SpacePacketFramer)
- [F Prime v4.1.0 TcDeframer](https://github.com/nasa/fprime/tree/v4.1.0/Svc/Ccsds/TcDeframer)
- [F Prime v4.1.0 TmFramer](https://github.com/nasa/fprime/tree/v4.1.0/Svc/Ccsds/TmFramer)
- [F Prime v4.1.0 FprimeRouter](https://github.com/nasa/fprime/tree/v4.1.0/Svc/FprimeRouter)
- [F Prime v4.1.0 ComQueue](https://github.com/nasa/fprime/tree/v4.1.0/Svc/ComQueue)

### 5.2 APID/VCID MVP tuong thich F Prime v4.1.0

| Direction | APID | Noi dung |
|---|---:|---|
| TC | `0x000` | `FW_PACKET_COMMAND`; cloud/scene/ROI phan biet bang opcode |
| TM | `0x001` | `FW_PACKET_TELEM`; health, config, progress va result channels |
| TM | `0x002` | `FW_PACKET_LOG`; events va mission command acknowledgement |
| TM | `0x003` | `FW_PACKET_FILE`; product bundle FilePacket |

- TC su dung VC0.
- Local mission profile co `spacecraft_instance_id: U64` stable, khac voi CCSDS SCID `68`; GDS link bind dung instance nay. Reset/migrate satellite durable state bat buoc tao instance ID moi va GDS full rebaseline. Khi bind sang instance moi, GDS terminal hoa moi outbox/lease nonterminal nham instance cu thanh `DELIVERY_FAILED` voi reason `TARGET_INSTANCE_RETIRED`, khong auto-retry; operator muon chay lai phai submit body moi de cap `RequestKey` moi. `spacecraft_boot_id: U32` tang bang durable transaction truoc moi process start, khong reuse trong instance; `product_id: U32` chi unique trong boot. Allocator missing/corrupt hoac sap wrap deu giu service o `FAULT` cho toi khi migration co chu dich, khong silent reset.
- MVP su dung mot TM VC. `MissionComScheduler` dat truoc `SpacePacketFramer`, schedule theo F Prime packet, sau do moi packet tao dung mot TM frame; scheduler duoc mo ta tai muc 8.1.
- Neu nang cap nhieu TM VC, moi VC co Virtual Channel Frame Count rieng nhung Master Channel Frame Count phai dung chung qua master-channel multiplexer/shared counter.
- Wire format dung big-endian. Space Packet Sequence Count la 14 bit, rollover modulo 16384 doc lap theo APID: `16383 -> 0`.
- GDS phai phan biet rollover hop le voi packet gap. Reset som chi duoc phep sau process/spacecraft restart va phai co boot/event marker de receiver rebaseline.
- TM Master/Virtual Channel Frame Count la counter 8 bit modulo 256, rieng voi Space Packet Sequence Count; `Fw::FilePacket.sequenceIndex` la counter transfer file va khong duoc dung thay sequence count cua Space Packet.
- `0x000..0x003` la mapping mac dinh cua profile MVP va phai khop `ComCfg.Apid`/`Fw::ComPacketType` tai build time. `mission_profile.yaml` mirror cac gia tri da build, khong duoc override doc lap.
- APID `0x120..0x123` duoc loai khoi MVP. Neu duoc yeu cau sau nay, phase custom routing phai chung minh ca ingress va egress mapping truoc khi thay mapping tren.

### 5.3 TM frame budget va time contract

- MVP giu TM frame co dinh `1024` byte de khop dictionary F Prime v4.1.0 hien tai. Project topology bypass `ComAggregator` va noi `MissionComScheduler -> SpacePacketFramer -> TmFramer`, tao mot Space Packet moi TM frame. Voi TM primary header 6 byte, OCF absent va FECF 2 byte, TM Data Field la `1016` byte. Tru Idle Space Packet toi thieu 7 byte, Space Packet header 6 byte, `FwPacketDescriptorType` 2 byte va FilePacket DATA header 11 byte, tran raw file la `990` byte/frame.
- Profile MVP dat `FW_FILE_BUFFER_MAX_SIZE=1003`: `1003 - 2 - 11 = 990` byte DATA, Space Packet sau framing la `1009` byte va con dung 7 byte cho Idle Space Packet. Build-time assertion va golden boundary test phai reject moi cau hinh/packet lon hon. Neu giu default F Prime `512`, max DATA chi la `499` byte va goodput/SLO phai tinh theo con so nay.
- Giu `FW_COM_BUFFER_MAX_SIZE=512` cho command/event/telemetry; project override rieng file buffer. File/SP buffer-manager bin phai co element size it nhat `1009` byte, TM-frame bin it nhat `1024` byte, va count du cho scheduler ownership/return path; exact bin count duoc khoa cung queue capacity va co allocation-failure test.
- Khong tang frame len `2048` chi dua tren kich thuoc anh. Default MVP da khoa `1024`; neu stakeholder override theo coding/link profile thi phai tao profile, budget va golden vectors moi. Phase 3 benchmark goodput, frame count va recovery cost o cac fault rate muc tieu.
- `space_packet_secondary_header` cua MVP la `absent`. Timestamp nam trong F Prime/application payload.
- Moi TM/event/result duoc GDS gan `source_spacecraft_instance_id`, `sender_boot_id` va `link_session_id` tu validated local-transport envelope/handshake, khong tu mutable current-link state; payload luu `satellite_event_time` va GDS luu `gds_receive_time`. Monotonic clock chi dung tinh duration/latency. `mission_profile.yaml` phai khoa time base, epoch, resolution, byte order va hanh vi khi clock reset.

### 5.4 Canonical scalar representation

- Moi field semantic `U64` (`ground_instance_id`, `gds_installation_epoch`, `spacecraft_instance_id`, `link_generation`, `link_session_id`, `simulation_run_id`, `link_frame_id`, `sender_frame_id`, `file_epoch_id`, `control_request_id`, `event_id`, `sim_time_ns`, `raw_draw`, area/counter U64...) la 8 byte unsigned big-endian tren binary wire va fixed replay fields; rieng payload deterministic-CBOR dung canonical unsigned CBOR integer trong `0..2^64-1` (khong float/tag), khong dung bstr co do dai khac 8 byte cho U64. Trong JSON/JCS/REST/WebSocket/structured log no la **string dung 16 chu so hex lowercase, khong `0x`**. U32/U16 van la JSON integer. Validator reject decimal, uppercase, leading/trailing space va sai length; RFC 8785 canonicalize string, khong dua U64 vao JSON Number.
- SQLite luu U64 bang fixed `BLOB(8)` big-endian; lexical BLOB order chinh la unsigned order va keyset cursor dung byte nay. Khong ep U64 vao signed `INTEGER`. C++/Python convert bang checked integer; TypeScript giu opaque string hoac parse `BigInt`, tuyet doi khong qua `Number`. HTML data/cache key va URL path dung cung 16-hex representation.
- Golden round-trip/JCS/digest/SQLite-order/React tests bao phu `0`, `2^53-1`, `2^53`, `2^63-1`, `2^63`, `2^64-1`; bat ky conversion mat bit nao la protocol error, khong silent truncate.

## 6. Command contract

MVP chi co mot GDS writer. Moi command co `RequestKey = {ground_instance_id: U64, request_id: U32}`. `ground_instance_id` la CSPRNG namespace duoc persist; `request_id` la sequence cap phat trong database. Khi request U32 sap wrap hoac database duoc khoi tao lai, GDS tao namespace moi sau khi drain/terminal moi command cu; U64 namespace trung voi lich su local bi regenerate. ID cua command hien tai khong bao gio duoc tai su dung lam ID job dich; cac command truy van/huy mang `target_request_key` rieng.

Moi command co common envelope `{target_spacecraft_instance_id: U64, request_key: RequestKey}` va Satellite reject `TARGET_INSTANCE_MISMATCH` truoc opcode-specific validation. Moi scene command dung `SceneRef = {catalog_epoch: U32, scene_id: U32, scene_revision: U32}`; revision khong bao gio duoc so sanh tach khoi epoch. Trong GDS, identity day du la `ScopedSceneRef = {spacecraft_instance_id: U64, scene_ref: SceneRef}`. Moi product command/API dung `ProductRef = {spacecraft_instance_id: U64, origin_boot_id: U32, product_id: U32}` de boot/product counter co the cap phat lai sau migration/restart ma khong tro nham bundle cu. Moi GDS row/event phat sinh tu satellite phai mang `source_spacecraft_instance_id`; moi command/outbox/attempt phai mang `target_spacecraft_instance_id`.

`JobKey` chinh la `RequestKey` cua `SCENE_ANALYZE`/`ROI_REQUEST` da tao job. `JOB_GET_STATUS` va `JOB_CANCEL` co `RequestKey` moi cua chinh command, con `target_request_key` mang `JobKey`; MVP khong tao them mot `job_id` ngam dinh.

Common envelope nam trong FPP command arguments; cac bang payload ben duoi viet tat `request_key` nhung deu bao gom target instance. GDS chi encode/send khi target van khop instance cua bound link; link migration khong duoc viet lai target trong payload da admission. `RequestKey` la business identity, doc lap voi F Prime `cmdSeq`, CCSDS Space Packet Sequence Count, TC Transfer Frame Sequence Number va WebSocket `event_id`; khong counter nao trong so nay duoc tai su dung thay `RequestKey`.

| Command | Payload | Validation | Ket qua |
|---|---|---|---|
| `CLOUD_SET_CONFIG` | `request_key; expected_config_epoch, expected_config_revision: U32; model_threshold_bp, coverage_limit_bp: U16` | Hai nguong hop le va CAS identity khop | TM config snapshot voi revision moi |
| `SCENE_REQUEST_CATALOG` | `request_key` | Satellite san sang doc inventory | Catalog snapshot co epoch/revision |
| `SCENE_REQUEST_PREVIEW` | `request_key; scene_ref: SceneRef` | SceneRef khop catalog hien hanh | Quicklook/tile product |
| `SCENE_ANALYZE` | `request_key; scene_ref: SceneRef; expected_config_epoch, expected_config_revision: U32; model_threshold_bp, coverage_limit_bp: U16` | SceneRef va config snapshot khop | Full-scene result |
| `ROI_REQUEST` | `request_key; scene_ref: SceneRef; x, y, width, height: U32; expected_config_epoch, expected_config_revision: U32; model_threshold_bp, coverage_limit_bp: U16` | SceneRef, ROI va config snapshot khop | ROI result/product |
| `JOB_GET_STATUS` | `request_key; target_request_key: RequestKey` | JobKey ton tai trong full journal/range marker | TM job status hoac `TARGET_RETIRED` |
| `JOB_CANCEL` | `request_key; target_request_key: RequestKey` | JobKey ton tai; terminal/compacted duoc tra outcome ro | Cancel ACK, `ALREADY_TERMINAL` hoac `TARGET_RETIRED` |
| `PRODUCT_REQUEST_DOWNLINK` | `request_key; origin_request_key: RequestKey; product_ref: ProductRef` | Bundle con retention, khong co attempt active | Downlink bundle voi `transfer_id` moi |
| `PRODUCT_CANCEL_DOWNLINK` | `request_key; product_ref: ProductRef; transfer_id: U32` | Dung active global attempt hoac da terminal | Cancel requested, `ALREADY_TERMINAL` hoac completion-wins outcome |

- `SceneRef` chan ca catalog namespace reset va scene replacement. Neu epoch/revision khong khop, satellite tra `CATALOG_EPOCH_MISMATCH`/`SCENE_REVISION_MISMATCH` ma khong tao job; catalog/product manifest mang source SHA-256 day du va GDS phai refresh catalog.
- Moi analysis command mang ca hai threshold va config identity. Satellite phai nap persisted config row `(epoch, revision, model_threshold_bp, coverage_limit_bp)` truoc khi vao `READY`, validate ca identity **va byte-equality cua hai threshold**; identity dung nhung value khac bi reject `CONFIG_SNAPSHOT_MISMATCH`. Sau do Satellite copy thresholds, model SHA-256, InputSpec ID, threshold-mapping ID/LUT SHA-256 va config identity vao immutable job record trong cung transaction admission truoc khi tra `COMMAND_ACCEPTED`; reorder giua SET va analysis chi tao `CONFIG_REVISION_MISMATCH`, khong tao ket qua voi config ngam dinh khac.
- `CLOUD_SET_CONFIG` cap nhat nguyen tu ca hai threshold bang compare-and-swap. Config row moi, idempotency journal/terminal command result va ACK payload phai commit trong cung durable onboard transaction; crash truoc commit khong doi config, crash sau commit replay cung `RequestKey` tra snapshot da cache va khong tang revision lan hai. Moi update tang `config_revision`; truoc revision U32 wrap, Satellite atomic cap `config_epoch` durable chua tung dung, reset revision va phat event/TM de GDS rebaseline. Truoc `config_epoch` U32 wrap, service fail closed va bat buoc migrate spacecraft instance; khong reuse config identity trong instance.

### 6.1 Hai nguong runtime bat buoc phai tach rieng

1. `model_threshold`: nguong sigmoid score de mot patch duoc danh dau la cloud; chi goi la xac suat sau khi co calibration report.
2. `coverage_limit`: gioi han cho chi so tile-level ben duoi de scene/ROI duoc chap nhan.

Boundary model dung byte-canonical LUT `logit-bp-f32-lut-v1`: logit non-finite lam job fail; `model_threshold_bp=0` danh dau moi logit finite la cloud, `10000` danh dau tat ca clear, con `1..9999` dung `cloud iff float32_logit >= threshold_lut[bp]`; equality la cloud. Moi entry LUT duoc sinh offline tu `ln(bp / (10000 - bp))` bang arithmetic high-precision da pin va round IEEE-754 binary32 round-to-nearest-ties-even, roi dong goi big-endian cung SHA-256 trong release artifact. Runtime khong duoc tu tinh `ln` bang math library cua platform. Science acceptance dung strict `< coverage_limit`, nen equality voi coverage limit la reject, ke ca endpoint.

Nguong `cloud_ratio_threshold` dung de tao nhan khi training khong phai tham so runtime va khong nen dua len command console van hanh.

Ten wire/manifest chinh thuc cua ket qua la `cloud_positive_tile_area_ratio_bp`, khong phai pixel cloud percentage. Tu so la tong dien tich valid giao ROI cua cac scene-anchored patch duoc classifier danh dau cloud; mau so la tong valid ROI area. Quyet dinh dung checked `U64` cho phep so sanh integer `cloud_positive_area * 10000 < coverage_limit_bp * analyzed_area`, reject overflow truoc nhan va khong dung ratio float da lam tron.

### 6.2 Tinh idempotent

- GDS cap phat `RequestKey` trong cung transaction admission/outbox va khong reset namespace/sequence khi process restart.
- Mission digest la `SHA-256(ASCII "mission-command-v1\0" || protocol_schema_version:U16BE || opcode:U32BE || argument_length:U32BE || exact dictionary/FPP mission-argument bytes)`. Argument bytes gom target instance, RequestKey va payload theo wire order, khong gom F Prime `cmdSeq`, CCSDS packet/frame header hay transport counter. HTTP idempotency digest doc lap dung RFC 8785 JCS cua validated semantic API body: `delivery_mode` luon hien dien; `expires_at` explicit duoc normalize thanh UTC RFC 3339, con field bi omit duoc canonical hoa thanh literal sentinel `"DEFAULT"`. Effective expiry tinh tu server clock chi duoc materialize tai commit dau tien va khong nam trong digest cua sentinel; ca hai digest co golden vectors.
- Satellite persist idempotency journal tren disk, khoa boi `RequestKey`, gom mission digest, command state, immutable config snapshot, job/product reference va terminal result. Local profile giu full result `7 ngay`, lon hon command TTL toi da `24 gio` va retry window; sau khi moi entity lien quan terminal, per-key marker duoc merge thanh contiguous/sparse `retired_request_ranges` cho tung ground namespace va giu suot vong doi `spacecraft_instance_id`. Request trong range tra `DUPLICATE_REQUEST_RETIRED`, khong chay lai business effect; khong co namespace-retire command ngam dinh.
- Khi full journal con retention, TC lap lai cung `RequestKey` va cung mission digest tra lai ACK/status/ket qua cu; cung key nhung digest khac bi reject `DUPLICATE_REQUEST_CONFLICT`. Sau compact, range marker khong con per-key digest, vi vay **moi** payload co key nam trong retired range deu tra `DUPLICATE_REQUEST_RETIRED`, khong co gang phan biet same/different payload. Khong nhanh nao dispatch hay chay business effect lan hai, ke ca sau satellite process restart. Cung `request_id` trong `ground_instance_id` khac la command moi.
- Voi `PRODUCT_REQUEST_DOWNLINK`, moi transfer attempt co chu dich phai dung `RequestKey` moi; `origin_request_key` giu identity cua originating command da tao product (`ANALYSIS`, `PREVIEW` hoac `CATALOG`). Retry do mat ACK giu cung `RequestKey` va khong duoc tao attempt moi.
- `origin_request_key` phai khop product record duoc tham chieu. Command journal, transfer ID/attempt row va admission vao transfer queue phai commit atomic; restart co chinh xac mot attempt da terminal/reconcile, khong duoc co zero-attempt sau ACK hay hidden attempt thu hai.
- Voi moi product command, common target instance, `ProductRef.spacecraft_instance_id`, product row/origin RequestKey va active transfer instance phai bang nhau. Satellite reject `PRODUCT_TARGET_INSTANCE_MISMATCH` truoc product lookup/path access; GDS cung validate nhung khong thay the onboard check.
- `PRODUCT_CANCEL_DOWNLINK` dung RequestKey moi va idempotent nhu command khac. Command chi `EXECUTED` sau khi coordinator da persist `CANCEL_REQUESTED` hoac outcome `ALREADY_TERMINAL`; neu final DATA da duoc stock component chot de phat END thi `SEND_COMPLETED` thang cancel race.
- Journal chi duoc compact sau khi command, job va moi product/transfer lien quan deu terminal. GDS khong retry ID da qua full-result retention; GET_STATUS/CANCEL target trong retired range tra `TARGET_RETIRED` (REST `410 Gone`) va operator phai tao command moi ro rang. Retired range van chan business effect cu trong suot spacecraft instance.

## 7. Telemetry, event va data product

### 7.1 Telemetry channels

| Nhom | Truong chinh |
|---|---|
| Satellite health | state, uptime, queue depth, storage free, last error |
| Link health | TC received, CRC errors, invalid APID/VCID, TM frames sent, drops |
| Model config | model release/hash/assurance, channel count, patch size, config epoch/revision, model threshold, coverage limit |
| Active job | RequestKey, SceneRef, job state, progress, start time, elapsed time |
| Last result | ROI, cloud-positive tile-area ratio, science decision, ProductRef, error code |
| Queue/scheduler | queue depth theo priority, oldest ACK age, queue rejects, high-priority preemptions, file frames sent |
| Transfer | ProductRef, transfer ID/attempt, bytes/ranges received, state, timeout va checksum status |

### 7.2 Events

- `COMMAND_RECEIVED`, `COMMAND_REJECTED`, `COMMAND_ACCEPTED`.
- `JOB_QUEUED`, `JOB_STARTED`, `JOB_PROGRESS`, `JOB_COMPLETED`.
- `SCIENCE_DECISION_ACCEPTED`, `SCIENCE_DECISION_REJECTED`, `JOB_FAILED`, `JOB_CANCELED`.
- `PRODUCT_CREATED`, `PRODUCT_DOWNLINK_STARTED`, `PRODUCT_SEND_COMPLETED`.
- `PRODUCT_TRANSFER_INCOMPLETE`, `PRODUCT_CHECKSUM_FAILED`, `PRODUCT_RETRY_STARTED`.
- `WORKER_LOST`, `WORKER_RESTARTED`, `QUEUE_FULL`.
- `INVALID_CRC`, `INVALID_APID`, `INVALID_ROI`, `MODEL_CONTRACT_MISMATCH`.

### 7.3 Cac state machine doc lap

```text
Command delivery (GDS):
CREATED -> ADMITTED
ADMITTED -> HELD_NO_CONTACT | OUTBOX_PENDING | CANCELED
HELD_NO_CONTACT -> OUTBOX_PENDING | EXPIRED | CANCELED
OUTBOX_PENDING -> DISPATCHING | EXPIRED | CANCELED
DISPATCHING -> OUTBOX_PENDING | SENT | EXPIRED | DELIVERY_FAILED
SENT -> OUTBOX_PENDING | RECEIPT_CONFIRMED | EXPIRED | DELIVERY_FAILED

Command execution (Satellite, replicated to GDS):
RECEIVED -> VALIDATED | EXECUTION_FAILED
VALIDATED -> COMMAND_REJECTED | COMMAND_ACCEPTED | EXECUTION_FAILED
COMMAND_ACCEPTED -> DISPATCHED | EXECUTED | EXECUTION_FAILED

Job execution (Satellite/GDS replica):
QUEUED -> RUNNING | CANCEL_REQUESTED | FAILED | TIMEOUT
RUNNING -> SUCCEEDED | FAILED | TIMEOUT | CANCEL_REQUESTED
CANCEL_REQUESTED -> CANCELED | SUCCEEDED | FAILED | TIMEOUT

Science decision (field on SUCCEEDED job):
NOT_EVALUATED | ACCEPTED | REJECTED

Product generation (Satellite):
STAGING -> READY | FAILED

Transfer attempt (Satellite):
QUEUED -> SENDING | ABORTING | CANCELED
SENDING -> SEND_COMPLETED | CANCEL_REQUESTED | ABORTING
CANCEL_REQUESTED -> CANCEL_DRAINING | SEND_COMPLETED | ABORTING
CANCEL_DRAINING -> COOLDOWN | ABORTING
ABORTING -> COOLDOWN
COOLDOWN -> SEND_FAILED | CANCELED

Transfer attempt (GDS):
RECEIVING -> VERIFIED | INCOMPLETE | CHECKSUM_FAILED | CANCELED
VERIFIED -> PUBLISHING
PUBLISHING -> PUBLISHED | PUBLISH_FAILED
```

- Migration fence cua GDS delivery la transition chung: moi state nonterminal (`CREATED`, `ADMITTED`, `HELD_NO_CONTACT`, `OUTBOX_PENDING`, `DISPATCHING`, `SENT`) deu co the terminal hoa thanh `DELIVERY_FAILED(reason=TARGET_INSTANCE_RETIRED)` sau khi write-fence dong admission generation cu; day la mot delivery terminal, khong phai state `TARGET_INSTANCE_RETIRED` rieng. Late ACK chi duoc audit `LATE_RECEIPT`, khong viet nguoc terminal state.
- `ACCEPTED`/`REJECTED` cua tile-area ratio la `science_decision` trong job `SUCCEEDED`, khong phai command state. Command duoc `ACK_ACCEPTED` khi da validate va mutation/work row bat buoc cua opcode da admission durable; chi opcode tao job moi admission vao job queue.
- Cac canh `RECEIVED/VALIDATED -> EXECUTION_FAILED`, `QUEUED job -> FAILED|TIMEOUT`, `QUEUED transfer -> ABORTING` chi dung cho reason allow-list nhu restart, service fault, deadline hoac invariant da neu; terminal van immutable. Moi delivery state GDS nonterminal co canh chung toi `DELIVERY_FAILED(reason=TARGET_INSTANCE_RETIRED)` khi migration.
- `DISPATCHED` la terminal command-success cho command tao async job/product/transfer sau khi work row da commit atomic; `EXECUTED` la terminal command-success cho mutation/query dong bo. `COMMAND_REJECTED` va `EXECUTION_FAILED` cung terminal. Job/product/transfer fail ve sau khong ghi de command-success.
- `JOB_CANCEL` chi `EXECUTED` sau khi `CANCEL_REQUESTED`/`CANCELED` hoac outcome `ALREADY_TERMINAL` da persist. Neu worker ket thuc truoc cancel fence, state that `SUCCEEDED`/`FAILED`/`TIMEOUT` thang; cancel lap lai tra outcome hien co va khong viet nguoc terminal state.
- Nhanh quay lai `OUTBOX_PENDING` la retry sau lease/send khong xac dinh; no giu nguyen `RequestKey`/payload, tao `command_attempt` va Space Packet sequence moi. `RECEIPT_CONFIRMED` chi den tu mission ACK trung `RequestKey`, khong suy ra tu UDP send thanh cong.
- Moi delivery state nonterminal co transition migration allow-list toi terminal `DELIVERY_FAILED(reason=TARGET_INSTANCE_RETIRED)`. ACK den sau state nay duoc audit `LATE_RECEIPT` nhu sau expiry/contact loss, co the cap nhat replica cua old instance nhung khong viet nguoc delivery terminal.
- `Fw::CmdResponse` chi phan anh dispatch noi bo F Prime. CloudPayload phat mission ACK co `RequestKey`, stage va error code tren TM/event; GDS khong suy dien job success tu framework command response.
- `SEND_COMPLETED`/`PRODUCT_SEND_COMPLETED` chi duoc dat sau END terminal comStatus + drain fence, va chi noi sender/link simulator da xu ly delivered-or-dropped moi frame; chi GDS duoc dat `VERIFIED`/`PUBLISHED` sau integrity checks.
- GDS persist `VERIFIED` truoc publish. Atomic rename vao final directory la idempotent; startup gap giua verify/rename/DB commit phai kiem tra lai path + hash va tiep tuc `PUBLISHING`, tao dung mot final directory hoac terminal `PUBLISH_FAILED` ro rang.
- Moi transition co bang allow-list, timestamp va reason code. Delivery, onboard command, job, product va transfer co khoa/foreign key rieng; API co the aggregate chung nhung khong ghi de state cua nhau.

### 7.4 Data product

Moi command tao product (`ANALYSIS`, `PREVIEW`, `CATALOG`) tao mot thu muc staging va mot generic manifest envelope gom `schema_version`, `product_type`, `ProductRef`, `origin_request_key`, artifact list `{path, size, sha256}` va timestamps. Artifact list phai normalize path theo ASCII POSIX, reject duplicate/case-collision, va sort tang dan theo normalized path truoc khi hash/JCS; cac map key tuan RFC 8785, con array co thu tu ngu nghia thi giu nguyen thu tu da khai bao. Moi `product_type` co section bat buoc rieng:

- `ANALYSIS`: SceneRef/source SHA-256, ROI, patch-grid/coverage algorithm ID, thresholds/config identity, threshold-mapping ID/LUT SHA-256, cloud-positive tile-area ratio, science decision, model release/assurance, display profile; artifact co the gom `crop.tif`/`crop.jp2`, `quicklook.webp` va `cloud_mask.tif`.
- `PREVIEW`: SceneRef/source SHA-256, display profile va quicklook/tile index.
- `CATALOG`: catalog epoch/revision va `snapshot_sha256`, trong do hash nay la SHA-256 cua canonical catalog payload artifact, khong phai hash bundle hay manifest tu tham chieu. Catalog payload sort scene entry theo `(scene_id U32, scene_revision U32)`; moi set domain/capability sort ASCII, chi cac array nhu `band_order` giu thu tu semantic.
- `manifest.json` khong tu liet ke chinh no trong artifact list.

Mac dinh, ROI bi reject chi downlink metadata va quicklook/mask; khong downlink crop day du de mo phong tiet kiem bang thong.

Sau khi moi artifact duoc atomic publish trong staging, Satellite dong goi uncompressed POSIX USTAR byte-canonical trong namespace cua `origin_boot_id`: chi regular-file entry, ASCII relative path `<=100` byte sort tang dan, khong directory/PAX/GNU header, `mtime=0`, `uid/gid=0`, `uname/gname/linkname/prefix=""`, `mode=0644`, `devmajor/minor=0`, `typeflag='0'`, magic/version `ustar\0`/`00`. Numeric field dung zero-padded ASCII octal + terminal NUL (`size/mtime` 11 digit, cac field khac theo USTAR width); checksum dung 6 octal digit + NUL + space va tinh khi 8 checksum byte tam la space. Moi entry zero-pad toi 512 byte va archive dung dung hai block zero 512 byte ket thuc. `manifest.json` la RFC 8785 JCS UTF-8 + mot LF; timestamps lay tu immutable command/job/product snapshot, khong lay wall clock moi khi rebuild. Relative source filename truyen cho stock FileDownlink la `b/{origin_boot_id:08x}/{product_id:08x}.tar`, dai `23` byte; FileDownlink process co fixed working root la satellite product store, khong dua absolute host path len port. Manifest khong chua hash cua chinh no hay TAR. SHA-256 cua bundle duoc tinh sau khi dong goi va luu ngoai archive trong product record/catalog/event; bundle SHA da bao phu ca bytes cua manifest.

### 7.5 File transfer va reassembly

- Wire contract dung truc tiep F Prime v4.1.0 `Fw::FilePacket` START/DATA/END/CANCEL; khong tao sequence protocol canh tranh. START bat buoc `sequenceIndex=0`; DATA bat dau `1` va tang dung mot moi packet; END/CANCEL dung next index. End checksum la F Prime `CFDP::Checksum`: cong modulo `2^32` cac word 4-byte big-endian theo absolute file offset, zero-pad trai/phai tai bien word, carry bi bo; day khong phai CRC/SHA va co Python/C++ golden vectors, gom partial/non-4-byte offset.
- Stock DATA/END khong mang path/transfer ID, vi vay MVP chi cho **mot global FilePacket attempt tren wire** moi spacecraft instance. Link Simulator parse APID `3` + FilePacket type va cap `file_epoch_id` trong transport sideband. START phai duoc GDS consume hoac fault model danh dropped truoc khi admit DATA; tat ca DATA phai consume/dropped truoc END/CANCEL; terminal phai consume/dropped va epoch phai dong truoc START ke tiep. `FRAME_CONSUMED` chi xac nhan GDS da ingest/reject frame, khong chung minh file verified.
- FilePacket START dung canonical ASCII destination `p/{origin_boot_id:08x}/{product_id:08x}/{transfer_id:08x}/{bundle_sha256}.tar`, dai dung `97` byte. Ca source `23` byte va destination `97` byte deu nho hon stock F Prime v4.1.0 `FILE_ENTRY_FILENAME_LEN=101`. `transfer_id: U32` do durable global allocator cap phat, khong reset khi sender restart va khong tai su dung; truoc wrap phai dung admission, drain attempt va migration sang spacecraft instance moi/GDS rebaseline.
- Transport reassembly key la `{spacecraft_instance_id, link_session_id, file_epoch_id}`; START path bind no voi globally unique `(spacecraft_instance_id, transfer_id)`, ProductRef, size va SHA. GDS prepend instance-owned storage root thay vi dung destination nhu absolute host path; session van khoi tao duoc khi `PRODUCT_DOWNLINK_STARTED` bi mat/reorder va khi downlink READY product cua boot cu. GDS ghi `.part`, byte ranges va sequence map durable.
- Duplicate START trong cung epoch chi idempotent neu sequence `0`, source/destination/size byte-identical; khac thi `START_CONFLICT`. DATA out-of-order duoc dat theo offset, nhung duplicate sequence phai cung offset/length/bytes va duplicate range phai byte-identical; sequence/range conflict terminal `FILE_PACKET_CONFLICT`. END lap chi idempotent neu sequence/checksum giong; CANCEL lap cung sequence giong la idempotent. END-vs-CANCEL conflict, terminal packet sai next-sequence, hoac DATA sau terminal bi reject. DATA/END/CANCEL khong co START tao epoch tombstone `INCOMPLETE(reason=MISSING_START)`; closed/stale epoch bi reject, khong gan vao active attempt moi.
- `PRODUCT_DOWNLINK_STARTED` lap lai spacecraft/sender boot, ProductRef, transfer ID, expected size, Fw transport checksum va bundle SHA-256 de quan sat/cross-check; event khong phai dieu kien duy nhat de parse START.
- `PRODUCT_REQUEST_DOWNLINK` bi reject `TRANSFER_BUSY` cho toi khi attempt truoc terminal o sender **va** file-epoch fence da dong. Late packet A mang old session/epoch va GDS reject theo tombstone, nen khong the gan vao B. Neu can concurrent/interleaved transfer thi chuyen CFDP hoac custom envelope co identity tren moi packet.
- `FileDownlinkCoordinator` boc stock `Svc::FileDownlink` v4.1.0. Chi coordinator goi guarded typed `SendFile`; stock `SendFile`, `SendPartial`, `Cancel` opcodes khong nam trong external `MissionCommandGate` allow-list nen TC khong bypass fixed path/global attempt. Coordinator co internal route vao CommandDispatcher sau gate de goi stock `Cancel` va doi correlated `Fw::CmdResponse`. Build bat buoc `FILEDOWNLINK_COMMAND_FAILURES_DISABLED=false`; `fileQueueDepth=1`. Build/golden test fail neu macro/opcode wiring khac.
- Coordinator luu attempt tag o side metadata, **khong sua `Fw::Buffer.context/data/size`**; stock context phai round-trip byte-for-byte. No so huu moi `Run` tick: khi stock o WAIT/co buffer outstanding, coordinator khong tick stock timeout ma dung watchdog rieng; nhu vay late return khong the den stock o COOLDOWN va assert. Config validate `cycleTime>0`; khi FileComplete den, coordinator drive dung `ceil(cooldown/cycleTime)+1` Run tick theo stock check-before-increment logic truoc khi release global slot.
- Neu packet terminal adapter failure/watchdog, coordinator: (1) persist `ABORTING`, dong output gate; (2) enqueue stock Cancel va doi response OK **truoc** khi return held buffer; (3) return held buffer dung mot lan. Stock co the tao CANCEL neu START/non-final DATA dang outstanding, END neu final DATA da chot, hoac khong tao packet neu END/CANCEL dang outstanding; coordinator intercept va return output nay noi bo, khong admit no len wire, va normalize FileComplete thanh failure intent; (4) goi idempotent `ABORT_FILE_EPOCH`, doi `FILE_EPOCH_CLOSED`; (5) drive cooldown roi terminal `SEND_FAILED`. Khong co "next DATA" queued trong stock vi chi mot buffer outstanding.
- Voi mission `PRODUCT_CANCEL_DOWNLINK`, coordinator persist `CANCEL_REQUESTED`, enqueue stock Cancel va doi response truoc held-buffer return. Neu stock tao CANCEL thi vao `CANCEL_DRAINING`, gui/drain no va terminal `CANCELED`; neu final DATA da chot, stock tao END va completion wins `SEND_COMPLETED`. Cancel/END adapter failure dung internal abort fence, terminal intent van duoc ghi ro. Retry cancel cung RequestKey tra outcome cu.
- Startup/control handshake `{spacecraft_instance_id, sender_boot_id}` phai close old session/epoch truoc `READY`. Satellite process cu mat thi pointer RAM cung mat; process moi chi reconcile durable row va drain Link Simulator. Link Simulator restart phat `SESSION_RESET`; ke ca cung sender boot, Satellite bat buoc abort attempt hien tai va khong tiep tuc DATA khong co START. START moi chi admit sau session ready + old epoch closed.
- GDS xac minh ca F Prime CFDP end checksum va bundle SHA-256 lay tu canonical START path. Khi extract vao staging, entry set sau normalize phai bang chinh xac `{manifest.json} union manifest.artifacts`; reject file thua, duplicate/case-collision, non-regular entry, symlink, traversal, vuot path/size/file-count quota va moi artifact mismatch. Chi sau do moi atomic rename ca product directory; khong artifact nao duoc publish rieng.
- Timeout, gap hoac checksum sai chuyen transfer sang `INCOMPLETE`/`CHECKSUM_FAILED`; `satellite finished sending` khong dong nghia `GDS verified complete`.
- Local SIL dung `transfer_inactivity_timeout=30 s` khi contact dang mo. Blackout khong pause/resume FileDownlink: link simulator drop frame theo profile trong khi sender co the van ket thuc attempt. GDS tam dung wall-clock timeout luc contact dong, nhung khi contact mo lai phai dong attempt con gap thanh `INCOMPLETE(reason=NO_CONTACT)` va yeu cau full-file attempt moi; khong resume attempt cu.
- TM MVP khong co reliable retransmission. Recovery mac dinh la gui TC Type-BD `PRODUCT_REQUEST_DOWNLINK` de downlink lai toan file bang attempt moi; selective range retry/resume de danh cho CFDP hoac phase nang cap.
- Reassembly state va staging metadata duoc persist de GDS restart co the khoi phuc hoac ket thuc transfer theo timeout; file `.part` het retention phai duoc don dep.

## 8. Satellite Simulator

### 8.1 State machine

Satellite service state va job state phai tach rieng:

```text
BOOTING -> SELF_TEST -> STANDBY -> READY
                |         ^       |
                v         |       v
              FAULT <- DEGRADED <-+
```

- Job lifecycle tiep tuc dung state machine tai muc 7.3; `PROCESSING`, `PRODUCT_READY` va `DOWNLINKING` khong phai service state.
- Chi nhan inference command khi service o `READY` va AI job queue con cho. O `DEGRADED`/`FAULT`, satellite phai tra ACK loi ro rang, khong drop command.
- Mot GPU/Jetson worker chi chay mot job tai mot thoi diem, tru khi benchmark chung minh co the tang concurrency.
- Cau hinh khoi diem: `max_pending_jobs=4` (khong tinh active RUNNING job), `ack_mailbox_capacity=32`, `control_queue_capacity=64`, `file_queue_capacity=16`, `worker_heartbeat_interval_ms=1000`, `worker_heartbeat_timeout_ms=5000`, `max_worker_restarts=3` trong `restart_window_ms=300000` va exponential restart backoff. Restart counter reset sau mot full window khong co worker failure. Worker/job values duoc freeze tu Phase 1/2b benchmark; egress capacity/burst chi provisional o Phase 2b va freeze sau Phase 3 target-bitrate benchmark.
- Mat heartbeat/IPC hoac qua job deadline: active job chuyen `FAILED: WORKER_LOST`/`TIMEOUT`, staging product khong duoc publish va service chuyen `DEGRADED`.
- Supervisor chi dua service ve `READY` sau khi restart worker va chay lai checkpoint/manifest validation, IPC health check va inference smoke test. Active job bi fail; pending jobs duoc giu trong bounded queue khi recovery con retry. Qua `max_worker_restarts` thi chuyen `FAULT` va fail pending jobs voi `SERVICE_FAULT`; MVP chi recover `FAULT` bang satellite process restart, chua co remote reset command.
- Satellite process restart tao `spacecraft_boot_id` moi. Truoc khi vao `READY`, Satellite recover durable store va reconcile theo bang sau; operator muon chay lai business work da fail phai tao `RequestKey` moi.

| Durable entity khi restart | Reconciliation bat buoc |
|---|---|
| Config transaction | Rollback neu chua commit; neu da commit thi config row, journal va cached ACK cung ton tai atomic |
| Command `RECEIVED/VALIDATED` chua terminal | `EXECUTION_FAILED: SATELLITE_RESTARTED`; duplicate tra terminal result nay |
| `COMMAND_ACCEPTED` nhung thieu mutation/work row ma opcode bat buoc co | Invariant corruption: giu service `FAULT`, khong binh thuong hoa thanh restart failure |
| `COMMAND_ACCEPTED` co async work row hop le | Atomic reconcile command thanh `DISPATCHED`, sau do reconcile job/product/transfer theo row ben duoi; sync mutation khong duoc ton tai o state nay, neu co thi `FAULT` |
| Command `DISPATCHED/EXECUTED` | Giu terminal command-success; khong suy dien lai tu job/product state |
| Job `QUEUED/RUNNING/CANCEL_REQUESTED` | `FAILED: SATELLITE_RESTARTED`, cleanup staging, khong auto-rerun |
| Product `STAGING` | `FAILED: SATELLITE_RESTARTED`, cleanup; product `READY` van duoc giu theo retention |
| Transfer `QUEUED/SENDING` | Durable row vao `ABORTING`; pointer/buffer RAM cua process cu da mat nen khong duoc "return" gia. Startup handshake resolve old sender epoch, process moi qua `COOLDOWN` logic roi persist `SEND_FAILED: SATELLITE_RESTARTED`; khong resume attempt cu |
| Catalog/config state va transfer allocator | Recover persisted epoch/revision/transfer counter; `product_id` co the reset chi trong `origin_boot_id` moi, khong alias ProductRef cu |

Durable onboard store cua local profile la SQLite rieng `satellite_state.db`, khong share voi GDS: `journal_mode=WAL`, `synchronous=FULL`, `foreign_keys=ON`, forward-only migration va dung mot writer task. Config, allocator, command journal, retired ranges, job/product/transfer metadata va outbox event can atomic deu nam trong cung transaction domain; product bytes dung staging + fsync file/directory + atomic rename, DB chi tro toi artifact da publish.

- Startup chay WAL recovery + `quick_check`, verify schema, allocator high-watermark, foreign key va product-path/hash reconciliation truoc `READY`. Missing/corrupt DB, allocator rewind, journal/range overlap hoac READY row tro toi artifact sai giu service `FAULT`; khong tao DB/ID moi ngam dinh.
- Local cap cho DB+WAL+journal la `2 GiB`, high/hard watermark `80%/90%`, `max_nonterminal_commands=1024`; full journal 7 ngay duoc compact thanh retired ranges ngay khi du dieu kien. DB volume co emergency reserve `128 MiB`. Admission estimate row/work/result worst case va reject **truoc** `ACK_ACCEPTED` bang `ONBOARD_STORAGE_FULL` neu quota/headroom khong du; khong ACK roi moi hy vong ghi duoc.
- Tai hard watermark, dung admission/job/product moi, release reserve chi de ghi terminal ACK/status, cancel/restart reconciliation va audit; recreate reserve khi xuong duoi 80%. Critical commit/fsync fail sau admission chuyen service `FAULT`, giu input de retry commit neu con ownership va khong phat success gia. Product-store reservation 7 ngay va DB reservation debit cung per-volume ledger neu chung filesystem.
- Crash-injection test tai truoc/sau WAL fsync/commit/ACK, torn WAL, corrupt page/index, disk-full truoc/sau admission, reserve release/recreate va journal-compaction boundary phai chung minh khong co ACK-without-work, allocator reuse hay duplicate business effect.

- Job deadline duoc tinh tu ROI/patch count va benchmark p99 co safety factor, khong hard-code mot timeout chung cho moi ROI.

`Svc::ComQueue` stock khong co ACK reservation hay fairness can cho SLO ben duoi. MVP vi vay implement `MissionComScheduler` thay cho stock `ComQueue` tren egress, truoc `SpacePacketFramer`, voi ba class nhung ACK co storage rieng:

1. Command ACK/terminal status, `32` per-RequestKey mailbox duoc reserve.
2. Control `fault/health/progress`, queue `64` slot; progress coalesce va fault khong duoc dung ACK slot.
3. File/data product, queue `16` slot.

TC ingress reserve mot mailbox/token truoc dispatch. `COMMAND_RECEIVED` chi la durable/control event, khong tao ACK packet rieng. Moi mailbox co toi da mot buffer in-flight va mot pending stage enum khong cap buffer: `ACK_ACCEPTED|ACK_REJECTED`, sau do terminal `DISPATCHED|EXECUTED|EXECUTION_FAILED`; stage moi ghi de pending stage cu theo monotonic order, nen accepted+terminal den nhanh khong an hai queue slot. Token chi release sau rejected hoac terminal packet co downstream completion; producer crash sau reserve phai duoc reconciliation tao `EXECUTION_FAILED`, khong release im lang. Het mailbox thi dung dequeue TC. Sau terminal send failure, ACK rebuild **buffer moi** tu durable journal voi backoff va giu mailbox; control ghi failure, file vao abort policy. Fault profile van co the drop sau simulator admission va GDS retry cung RequestKey se nhan latest durable stage.

Arbitration tai moi downstream completion la executable va co dinh. ACK duoc uu tien toi da `ack_burst=8` packet lien tiep; neu non-ACK dang cho, scheduler sau do phat toi da mot control roi mot file (skip class rong) truoc khi quay lai ACK, nen ACK flood khong starve hai class con. Khi ACK rong, control duoc chon toi da `control_burst=4` packet lien tiep khi file dang cho, sau do bat buoc mot file; file co the chay toi da `file_burst=8` khi control rong, nhung control moi den preempt tai completion ke tiep. GDS local profile them command token bucket `4 command/s`, burst `8`; admission satellite van bi chan boi ACK token. Do MVP mot packet/frame, packet burst va frame burst co cung gia tri.

Topology phai noi du ownership, frame completion va link readiness. MVP dung custom `MissionUdpAdapter` implement `Svc.Com` contract va chua completion-gate behavior; khong wire raw stock v4.1.0 `ComStub.comStatusOut`, vi `ComStub::drvConnected` co the phat SUCCESS khong gan frame va assert data khi dang reinitialize. Adapter co state `LINK_NOT_READY -> READY -> FRAME_IN_FLIGHT`; connection/session event chi phat `linkStateOut`, khong di vao frame `comStatusOut`. Scheduler khong send truoc initial `SESSION_READY` hay sau `SESSION_RESET`/admission failure.

```text
data:       MissionComScheduler.dataOut -> SpacePacketFramer.dataIn -> TmFramer.dataIn -> MissionUdpAdapter
frame back: MissionUdpAdapter.dataReturnOut -> TmFramer.dataReturnIn
SP back:    TmFramer.dataReturnOut -> SpacePacketFramer.dataReturnIn
input back: SpacePacketFramer.dataReturnOut -> MissionComScheduler.dataReturnIn
status:     MissionUdpAdapter.comStatusOut -> TmFramer.comStatusIn -> SpacePacketFramer.comStatusIn -> MissionComScheduler.comStatusIn
link state: MissionUdpAdapter.linkStateOut -> MissionComScheduler.linkStateIn
```

Adapter cap durable/session-scoped sender frame ID, gui UDP va chi complete success sau idempotent `FRAME_ACCEPTED` cua Link Simulator; send syscall khong du. Voi moi frame, no forward `dataReturnOut` dung mot lan **truoc** dung mot correlated terminal `comStatusOut`; status-den-som bi giu. Reject/control-timeout cung return roi FAILURE, sau do link state ve NOT_READY neu session khong con hop le. Readiness SUCCESS khong bao gio duoc forward nhu frame status.

Scheduler co state `LINK_NOT_READY | READY | IN_FLIGHT` va chi mot packet in-flight. Completion no quan sat la `{own_dataReturnIn, propagated_terminal_comStatus}`; lower layer dam bao status chi den sau ownership cua layer do dong. Truoc khi goi synchronous `dataOut_out`, scheduler phai persist current item/producer, set `IN_FLIGHT` va initialize flags, vi return/status chain co the re-enter ngay trong cung call stack; handler phai reentrant-safe hoac serialize status ma khong mat callback. `own_dataReturnIn` den som chi duoc giu, **chua return upstream producer**, cho toi khi ca cap completion den.

Neu `SESSION_RESET/LINK_NOT_READY` den khi `IN_FLIGHT`, scheduler latch `not_ready_after_completion` nhung khong xoa current item/flags. Adapter van phai return buffer + FAILURE dung mot lan (bang control timeout neu can); scheduler hoan tat ownership/failure policy roi moi vao `LINK_NOT_READY`. Reset khi READY chuyen ngay NOT_READY. Unit fake phai bao phu synchronous reentrant return, reset truoc/giua/sau accepted ACK va duplicate/late control response.

Sau success, scheduler return upstream dung mot lan roi moi arbitrate packet ke tiep. Sau terminal failure, no khong doc/requeue pointer sau khi da return: ACK duoc rebuild vao buffer moi tu durable journal; control terminal theo policy; file chuyen buffer con dang giu cho `FileDownlinkCoordinator` abort sequence o muc 7.5, coordinator cancel producer truoc khi return. Adapter chi duoc retry internal frame trong khi con ownership. Moi component co invariant mot input buffer duoc return dung mot lan, mot frame co dung mot terminal status, khong UAF/corrupt/leak/double-return hay gui packet tiep qua som.

Muc tieu MVP la `oldest_ack_age <= 1 s`, `health_max_latency <= 2 s` va file van co non-zero service trong fault-free local SIL ke ca khi file downlink/control lien tuc; latency tinh tu admission qua scheduler, framing va link serialization. Phase 3 benchmark va dong bang lai theo target bitrate.

### 8.2 Model manifest

Tao file manifest co version va checksum, toi thieu gom:

```yaml
schema_version: 1
model_id: cloud-mobilenetv3-small-rgb
model_release_id: cloud-mobilenetv3-small-rgb-r1
checkpoint_sha256: "generated-from-release-artifact"
framework: pytorch
assurance_level: demo_non_validated
assurance_profile_id: demo-v1
model_card_sha256: null
evaluation_report_sha256: null
supported_domains:
  sensor_ids: []
  platform_ids: []
  product_types: []
  processing_levels: []
input_spec:
  input_spec_id: rgb-legacy-dtype-range-v1
  channels: 3
  band_order: [red, green, blue]
  patch_size: 256
  source_dtype: uint16
  tensor_dtype: float32
  tensor_layout: NCHW
  input_shape: [null, 3, 256, 256]
  batch_axis: 0
  normalization:
    id: legacy-dtype-range-v1
    kind: dtype-range
    integer_scale: 65535
  padding:
    id: scene-edge-constant-raw-v1
    kind: constant
    value_space: source
    values: [0, 0, 0]
output:
  kind: binary_cloud_logit
  calibrated_probability: false
  threshold_mapping_id: logit-bp-f32-lut-v1
  threshold_lut_sha256: "generated-from-release-artifact"
```

Manifest phai mo ta phep bien doi bang tham so cu the, khong chi bang ten mo ho. Voi checkpoint hien tai, contract legacy la `uint16 -> float32 / 65535` de giu tuong thich voi training; khong duoc tu y doi rieng inference sang `/10000`, percentile stretch hoac mean/std khac.

Contract legacy chi duoc phep qua Gate 0 khi training provenance va golden input/output xac nhan cung phep bien doi. InputSpec/golden transform compatibility la bat buoc ke ca voi `demo_non_validated`; nhan demo chi mien scientific-performance gate, khong mien input-contract gate. Neu khong the xac minh, checkpoint khong duoc vao mission runtime va phai retrain/fine-tune voi InputSpec moi, sau do evaluate va calibrate threshold/probability.

Satellite phai fail-fast neu checkpoint SHA-256, first-convolution channels, band order, tensor layout/shape/dtype hoac normalization ID/tham so khong khop manifest. Manifest thieu/mismatch giu service o `FAULT`, khong vao `READY`.

Gioi han runtime phu thuoc target, do do nam trong deployment profile rieng:

```yaml
schema_version: 1
target_id: jetson-nano-tensorrt
runtime: tensorrt
runtime_version: "pinned-at-build"
deployable: false
benchmark_artifact_id: null
batch_size: 1
max_batch_size: 1
progress_min_delta_bp: 100
progress_min_interval_ms: 1000
progress_max_silence_ms: 5000
```

Block tren chi la non-deployable template; `benchmark_artifact_id=null` khong bao gio duoc vao `READY`. `batch_size=1` chi la candidate dau tien de benchmark, khong phai mien gate. Phase 1 phai sinh artifact va materialize profile `local-cpu-pytorch` deployable cho reference MVP; Jetson/TensorRT phai co artifact rieng dung target/runtime truoc khi READY. Effective maximum la min cua deployment benchmark va TensorRT optimization-profile maximum. Benchmark phai tinh activation, TensorRT/CUDA workspace, allocator, image buffers va shared system memory; khong suy ra batch chi tu kich thuoc input tensor.

### 8.3 Model assurance va scientific promotion gate

- Golden transform/output va checksum chi chung minh reproducibility, khong chung minh model phu hop sensor hay quyet dinh nghiep vu.
- Checkpoint hien tai mac dinh mang `assurance_level=demo_non_validated`. TM/result/manifest phai lap lai assurance level; khong duoc goi sigmoid score la calibrated probability hoac goi model flight-ready.
- De promote sang `validated_decision`, release bundle phai co model card; dataset/sensor/label provenance; split theo scene khong leakage; PR-AUC, precision, recall, false-clear/false-reject va confusion matrix tren held-out scenes; calibration report neu dung tu `probability`; threshold sweep theo mission cost matrix; va domain-shift report tren sensor/season/region muc tieu.
- Moi metric promotion phai kem sample/event count, confidence interval va subgroup/domain breakdown, khong chi point estimate. Model release phai khoa `model_release_id`, `assurance_profile_id`, model-card/evaluation hash va supported sensor/platform/product/processing-level domains.
- Admission doi chieu scene-domain metadata voi supported domain. `validated_decision` mismatch bi reject `MODEL_DOMAIN_MISMATCH`; `demo_non_validated` co the chay de demo nhung result bat buoc mang `DOMAIN_UNVERIFIED` va khong duoc nang assurance.
- Gate 0 phai dong bang assurance profile va minimum metric/SLO. Neu mission chua cung cap cost matrix/validation corpus, MVP van duoc demo protocol/workflow nhung science decision chi co gia tri mo phong.
- Moi thay doi InputSpec, checkpoint, sensor mapping hoac threshold calibration tao model release ID moi va bat buoc chay lai promotion gate; khong ke thua assurance ngam dinh.

### 8.4 ROI inference

Can refactor core inference thanh cac API:

```python
read_window(scene, x, y, width, height)
infer_region(scene, roi, model_config, progress_callback)
progress_callback(processed_patches, total_patches, elapsed_ms)
build_products(result, output_directory)
```

- Worker co the cap nhat progress sau moi batch. Non-terminal TM chi phat khi dong thoi dat `progress_min_interval_ms` va tang toi thieu `progress_min_delta_bp`; neu khong dat delta thi van phat heartbeat progress khi den `progress_max_silence_ms`. STARTED, terminal 100% va FAILED/CANCELED luon duoc phat.
- Progress phai monotonic, duoc coalesce va khong block inference hay lam day ACK/health queue.

Quy tac xu ly:

1. Validate full `SceneRef`, config identity/snapshot va ROI half-open `[x, x+width) x [y, y+height)`; chan overflow, path traversal, vuot bien va `width/height < patch_size`.
2. MVP chi mo TIFF khi `tifffile.memmap()` va sidecar validation thanh cong. Sidecar versioned phai bind source SHA/shape/dtype/bands va `validity.kind=all_valid|nodata_value|mask`; `nodata_value` khai bao vector + any/all-band rule. `mask` phai khai bao relative path, shape, `dtype=uint8`, `semantics=1_valid_0_invalid` va SHA-256; file la TIFF mot band cung H/W, uncompressed-contiguous va `tifffile.memmap()` duoc. Ingest validate gia tri chi `0/1` theo bounded chunk; runtime memmap mask va doc dung cung X/Y window voi source. Mask can full decode/compression, sai shape/dtype/value/path/hash bi danh `UNSUPPORTED_VALIDITY_MASK`/`INVALID` truoc admission. Compressed source TIFF, JP2/HDF/NetCDF hoac input can full decode bi reject `UNSUPPORTED_SCENE_FORMAT`, tru khi deployment enable mot backend da qua pixel-oracle, full-decode-spy, target peak-RSS va p95-latency gate.
3. Dung scene-anchored grid co origin `(0, 0)` va patch size tu InputSpec (`256`). Chon moi patch co giao voi ROI; khong khoi tao lai grid tu goc ROI.
4. Doc dung cua so patch theo ca X/Y, khong doc tron chieu ngang va khong tao crop tam. Model nhan ca patch scene de giu context; manifest ghi ro context ngoai ROI co the anh huong classification.
5. Chi pad o bien scene de dat input shape. Padding constant lay tu InputSpec, ap dung trong `value_space` da khai bao roi moi normalization; padding ngoai scene khong nam trong validity denominator hay coverage. Gia tri pixel `0` khong tu dong la NoData.
6. Voi moi patch, dat `weight = valid_pixel_count(patch intersect ROI)`. Coverage la `sum(predicted_cloud * weight) / sum(weight)`, khong tinh context ngoai ROI hay padding.
7. Local profile dung strict valid-data policy `10000 bp`: **moi in-scene pixel cua toan bo model-input patch da chon**, ke ca context ngoai ROI, phai valid; denominator validity tai bien chi gom in-scene pixels. NoData trong ROI hoac context deu tra `INSUFFICIENT_VALID_DATA`. Policy cho fill/bo qua NoData sau nay phai co algorithm ID moi va scientific validation rieng.
8. So sanh integer theo contract muc 6.1; bang nguong la science reject, nhung job van `SUCCEEDED`.
9. Tao deterministic product staging va TM result tu immutable job/config/model/scene snapshot.

Scene full-image va ROI phai dung cung scene-anchored grid. Test dich ROI 1 pixel phai chung minh prediction cua cac patch scene khong bi tinh lai theo grid moi; chi intersection weight o bien ROI duoc thay doi.
`tiling_algorithm_id`, `coverage_algorithm_id`, padding policy va valid-data policy la mot phan cua immutable job config snapshot va phai xuat hien trong TM result/manifest.
Gia tri TM/manifest `cloud_positive_tile_area_ratio_bp` duoc serialize bang `floor(cloud_positive_area * 10000 / analyzed_area)`; science decision van dung cross-multiply khong lam tron tai muc 6.1.

Reader ghi rieng `logical_source_bytes_read` bang tong unique in-scene source bytes cua expanded scene-anchored patch windows va `logical_validity_bytes_read` bang unique mask bytes cua cung windows (`0` neu khong co mask); `logical_bytes_read` la tong hai metric, metadata I/O ghi rieng. Local CPU reference guard cho ROI `256 x 256`, sau model/scene warmup, la `max_window_rss_delta_bytes=268435456` va p95 latency tren scene co cung dtype/layout/validity nhung dien tich `4x` khong vuot `1.25x` canonical scene. Target profile co the thay so bang benchmark artifact, khong dung mo ta `khong tang` khong do duoc.

## 9. Link Simulator

### 9.1 Transport abstraction

- `InMemoryTransport` cho unit/integration test.
- `UdpTransport` cho Docker/multi-process simulation; moi UDP datagram chua local sideband envelope + dung mot transfer frame. Envelope big-endian v1 la `{magic=0x43534c31:U32, version=1:U8, direction:U8, reserved=0:U16, spacecraft_instance_id:U64, sender_boot_id:U32, link_session_id:U64, sender_frame_id:U64, link_frame_id:U64, file_epoch_id:U64, frame_length:U16, frame_bytes}`. Receiver exact-validate magic/version/reserved/direction/peer/session/length va strip envelope truoc CCSDS decode; envelope khong phai CCSDS wire bytes, khong bi fault injection va khong tinh vao simulated link bitrate. `InMemoryTransport` mang cung metadata bang typed sideband.
- Co hai hop contract tren cung schema: **ingress sender -> Link Simulator** bat buoc `direction=0`, `spacecraft_instance_id=target`, `sender_boot_id=0`, `link_frame_id=0`, `file_epoch_id=0`; **egress Link Simulator -> endpoint** bat buoc `direction=1`, `spacecraft_instance_id=source`, `sender_boot_id=boot hien tai`, `link_frame_id>0`, va `file_epoch_id>0` chi cho APID `3` FilePacket (`0` cho packet khac). Link Simulator gan `link_frame_id` sau ordered ingress; khi START duoc admit, no gan `file_epoch_id` va tra mapping trong `FRAME_ACCEPTED`, sau do map DATA/END/CANCEL ingress zero-field vao cung epoch duy nhat dang active. Sender khong tu phat ID do Link Simulator cap.
- `sender_frame_id` bat dau `1`, tang trong session va retry UDP giu nguyen ID/bytes. Stale/closed session, sai boot/instance, nonzero reserved/ingress-assigned field hoac ID reuse voi bytes khac bi reject/audit. `ABORT_FILE_EPOCH`/`FILE_EPOCH_CLOSED` tren LinkControl dung epoch mapping da tra ve, khong dua epoch vao UDP ingress.
- `LinkControl` la kenh TCP length-prefixed JCS rieng tren loopback/Compose internal network, khong qua fault model. Message allow-list: `OPEN_SESSION/SESSION_READY/SESSION_RESET`, `FRAME_ACCEPTED/FRAME_REJECTED`, `FRAME_CONSUMED`, `ABORT_FILE_EPOCH/FILE_EPOCH_CLOSED`. Moi request co `control_request_id` U64, session va relevant sender/link-frame/copy/epoch IDs; `FRAME_ACCEPTED` cua START tra `file_epoch_id` mapping (va duplicate tra lai mapping cu). Link Simulator durable-deduplicate va replay response byte-identical. Timeout `5 s`, retry moi `500 ms`, toi da `20`; het retry dong session, tombstone moi epoch, sender vao LINK_NOT_READY/abort va GDS reject UDP muon.
- Link Simulator chi tra `FRAME_ACCEPTED` sau khi serialize admission, cap `link_frame_id`, snapshot profile va durable append replay decision; duplicate sender frame tra mapping cu, khong inject frame lan hai. `MissionUdpAdapter` giu frame cho toi response nay. O egress, GDS append/reject frame idempotent theo `{simulation_run_id,direction,link_frame_id,copy_index}`, roi tra `FRAME_CONSUMED`; Link Simulator chi goi copy la delivered sau ACK nay. Intentional fault loss/overflow la dropped ngay. Neu consume timeout, Link Simulator retry cung UDP envelope; het retry thi `SESSION_RESET`, mark unresolved copy `CONTROL_TIMEOUT_DROP`, close/tombstone epoch/session va khong mo START moi tren session do.
- `simulation_run_id` va `link_session_id` do Link Simulator sinh CSPRNG U64, collision-check voi durable lifetime registry + artifact paths; retry toi da `128` draw roi fail closed. Registry khong prune trong GDS-installation epoch. `link_frame_id` monotonic trong run; `sender_frame_id`, `control_request_id`, `file_epoch_id` monotonic trong session; sap U64 wrap thi stop run/close session truoc reuse. Restart recover registry/counters/idempotency ledger/event queue, hoac phat explicit `SESSION_RESET` va close old epochs; khong tiep tuc ngam dinh voi counter reset.
- `StreamTransport` hoac CLTU/ASM adapter chi bo sung khi can serial/SDR.

### 9.2 Fault model

Moi chieu uplink/downlink co cau hinh rieng:

- Latency va jitter.
- Frame loss va duplicate rate.
- Bit/byte corruption truoc CRC validation.
- Reordering window.
- Bandwidth va queue capacity.
- Contact window/blackout schedule.
- Queue capacity va overflow policy rieng cho moi chieu; overflow phai tang counter va theo policy `DROP_NEWEST`/`DROP_OLDEST` da cau hinh, khong silent drop.

Schema MVP khoa probability bang `rate_ppm: U32` trong `0..1_000_000`: `0` luon false, `1_000_000` luon true, gia tri giua true khi U64 draw nho hon `floor(rate_ppm * 2^64 / 1_000_000)` tinh bang U128/big-int. Bounded integer `0..N-1` dung high-half U128 `floor(draw_u64 * N / 2^64)`, khong overflow U64. Duplicate tao toi da mot copy them. Corruption MVP la `corrupt_frame_rate_ppm + bits_per_corrupt_frame`, validate `0 <= bits_per_corrupt_frame <= frame_bits`; bit offset trung bi bo va tang `draw_index` toi khi du unique offset. Day khong duoc goi la BER doc lap moi bit.

Moi time la checked integer `sim_time_ns`. `signed_jitter = bounded(2*jitter_abs_ns+1)-jitter_abs_ns`, `reorder_slots=bounded(reorder_window_slots+1)`; profile bat buoc `base_latency_ns >= jitter_abs_ns`, nen `due = ingress + base_latency + signed_jitter + reorder_slots * reorder_slot_ns` khong o qua khu. Bandwidth la strict serializer, khong goi token bucket: `tx_start=max(due, link_available)`, `duration=ceil(frame_bits * 1e9 / bitrate_bps)` bang U128, `release=tx_start+duration`, roi `link_available=release`. Moi cong/nhan/time overflow, bounded range qua U64, hoac bitrate `<=0` bi reject profile. Profile khac phai co schema/algorithm version va golden vectors moi.

Determinism contract:

1. Link Simulator serialize admission va cap `link_frame_id: U64` monotonic theo **ordered ingress trace**; moi envelope da qua envelope/session validation duoc cap ID truoc contact/overflow decision (record bi reject van co audit ID, khong co egress copy). UDP ingress envelope van mang sentinel `link_frame_id=0`, egress moi mang ID da cap. Thu tu race giua concurrent caller la mot phan cua input, khong co cam ket hai call order khac nhau sinh cung trace.
2. Moi draw dung counter PRF `SHA-256("link-fault-v1" || seed:U64BE || simulation_run_id:U64BE || profile_revision:U32BE || direction:U8 || link_frame_id:U64BE || copy_index:U16BE || stage_code:U8 || draw_index:U32BE)`. First U64BE duoc map vao threshold/range da dinh nghia; khong dung shared PRNG phu thuoc thread timing. V1 khoa `stage_code`: `1=LOSS`, `2=DUPLICATE`, `3=CORRUPT_DECISION`, `4=CORRUPT_BIT`, `5=JITTER`, `6=REORDER`; contact/overflow/bandwidth khong draw. Moi stage bat dau `draw_index=0`; loss/duplicate dung `copy_index=0`, con corruption/jitter/reorder dung `0` cho original va `1` cho duplicate. CORRUPT_BIT tang draw cho toi du unique offset. Bit offset `0` la MSB (`0x80`) cua frame byte 0, mask `1 << (7-(offset mod 8))`, corruption dung XOR.
3. Thu tu ap dung co dinh: contact/blackout admission -> ingress queue overflow -> loss -> duplicate -> corruption tung copy -> latency/jitter -> reorder scheduling -> bandwidth serialization/egress.
4. FilePacket START/DATA/END drain fence dung `file_epoch_id` muc 7.5/9.1; priority queue khong reorder qua epoch. Epoch scope boi `{spacecraft_instance_id, link_session_id, file_epoch_id}`; new START chi admit sau moi old copy `FRAME_CONSUMED`/fault-dropped va `FILE_EPOCH_CLOSED`. Closed epoch co durable tombstone, nen UDP muon bi reject. DATA trong epoch van dung day du loss/duplicate/corruption/reorder.
5. `InMemoryTransport` dung simulation clock va mot priority event queue co tie-breaker `(due_time, direction, link_frame_id, copy_index)`.
6. `UdpTransport` live ghi decision log day du; replay test nap frame bytes + decision log, khong co gang tai tao wall-clock thread schedule.
7. Fault-profile update commit nguyen tu va chi co hieu luc tu `link_frame_id` ingress ke tiep; moi frame giu revision da snapshot khi admission. Process restart phai recover next ID/profile revision hoac tao `simulation_run_id` moi kem explicit reset marker; khong tiep tuc ngam dinh voi counter reset.

Voi TC Type-BD, frame mat khong duoc tu dong retransmit boi COP-1. GDS phai timeout va cho phep operator retry voi cung `RequestKey` va canonical payload.

### 9.3 Observability

- Structured JSON log gom `service`, UTC timestamp, severity, event name, request/boot/product/transfer ID, APID/VCID/frame sequence va fault-profile revision.
- Metrics gom effective throughput, frames queued/dropped/duplicated/corrupted, queue age/depth, blackout state va high-priority preemptions.
- Decision log phai co `simulation_run_id`, seed, profile/algorithm revision, direction, `link_frame_id`, input frame SHA-256, `copy_index`, tung `(stage_code, draw_index, raw_draw, decision)`, corruption bit offset/mask, due/release simulation time va output frame SHA-256. Day la replay oracle; structured service log co the chi tham chieu decision-record ID.
- Replay binary schema `link-replay-v1` dung segment `segment-{index:08x}.lrp`. Segment header la `magic="LRP1"`, `schema_version:U16BE=1`, `segment_index:U32BE`. Moi record la `{record_type:U8, flags=0:U8, payload_length:U32BE, deterministic-CBOR payload, crc32c:U32BE}`; CRC-32C dung reflected polynomial `0x82f63b78`, init/xorout `0xffffffff`, bao phu header record + payload, khong gom CRC. Payload theo RFC 8949 Core Deterministic Encoding, chi definite length/no float/tag; schema CDDL hash duoc pin trong mission profile.
- Record type/key map v1: `1 PROFILE {0:revision U32,1:JCS-profile-bytes bstr}`, `2 FRAME {0:run U64,1:direction U8,2:link_frame U64,3:sender_frame U64,4:sideband bstr,5:input_frame bstr,6:decisions array,7:outputs array}`, `3 SESSION {0:event_code U8,1:session U64,2:instance U64,3:boot U32}`, `4 FENCE {0:event_code U8,1:session U64,2:file_epoch U64,3:reason U16}`. Decision map la `{0:stage U8,1:copy U16,2:draw_index U32,3:raw_draw U64,4:decision_code U8,5:result integer-or-bstr}`; output map la `{0:copy U16,1:disposition U8,2:due_ns U64,3:release_ns U64,4:frame bstr}`. Arrays giu execution order; profile JCS bytes khong LF va dung scalar rule muc 5.4.
- Moi run co self-contained append-only `replay/{simulation_run_id-16hex}/`: segment toi da `256 MiB`, record chua ordered ingress sideband/frame + decisions/profile snapshot. Replay khong tham chieu raw-frame rolling file. `replay_manifest` bat dau `OPEN`; finalize fsync segment, ghi ordered `{name,size,sha256}`, tong byte va `replay_artifact_sha256 = SHA-256("link-replay-v1\0" || moi U64BE(size) || sha256-bytes theo thu tu segment)`, roi atomic rename manifest thanh `FINAL`. Startup validate header/CRC, truncate torn tail; run crash khong du state de tiep tuc deterministic thi terminal `INCOMPLETE_CRASH`.
- Local profile dat `replay_artifact_max_bytes=10 GiB/run`, global replay cap `20 GiB` **bao gom ca artifact pinned va unpinned** va reserve max-run truoc khi start; khong du cho thi reject `507 REPLAY_STORAGE_FULL`. Artifact `OPEN` khong bao gio bi evict. Cham cap giua run thi dung admission frame moi, finalize phan da ghi thanh `INCOMPLETE_STORAGE` va fail run ro rang, khong silent rotate/continue. Khi vao bat ky terminal state, reservation 10 GiB chuyen thanh actual-size charge va release phan chua dung. Artifact `FINAL` giu toi thieu `30 ngay`; `INCOMPLETE_*` giu `7 ngay`; sau retention, oldest unpinned moi duoc evict va moi state deu tinh vao global cap. Pin quota replay mac dinh `10 GiB` la **quota logic nam trong** global cap, khong phai dung luong cong them; pin van debit global/per-volume headroom va bi reject neu vuot quota hoac headroom. Run manifest phai tham chieu `replay_state=PRESENT|PINNED|EVICTED`, size/SHA va ly do eviction; prune raw frame `24 gio` khong anh huong replay artifact khi artifact con `PRESENT`/`PINNED`.
- Finalize khong hua transaction phan tan: filesystem `replay_manifest` la source of truth cho artifact bytes. Link Simulator durable-rename no thanh `FINAL/INCOMPLETE_*` truoc, sau do GDS writer commit SQLite row va run manifest tham chieu hash trong mot transaction. Crash giua hai buoc duoc startup reconciliation quet artifact by run ID va hoan tat/ha cap GDS row; GDS khong bao gio ghi `FINAL` neu artifact chua durable. Prune retire DB reference truoc khi xoa artifact, giong raw-segment protocol.
- Khong dung `RequestKey` lam metric label co cardinality cao; dung no cho trace/log.

## 10. GDS Backend

### 10.1 Stack

- FastAPI va Pydantic cho API/schema.
- SQLite cho MVP voi mot serialized writer task; PostgreSQL khi can multi-instance/multi-writer hoac retention lon.
- WebSocket cho telemetry/event/progress realtime.
- Background service rieng cho TC/TM link, khong chay inference hay decode blocking tren event loop.
- Cac queue GDS uplink, product reassembly va WebSocket client deu co capacity; overflow tra loi/metric ro rang. WebSocket client buffer mac dinh toi da `1000` event hoac `4 MiB`, cham moc nao truoc thi disconnect va yeu cau resync; khong duoc lam RAM tang vo han.
- Local profile dat persisted outbox capacity `1024` command. Day la admission capacity trong database, khong phai buffer RAM.

### 10.2 API de xuat

| Method | Endpoint | Muc dich |
|---|---|---|
| `GET` | `/api/spacecraft/{spacecraft_instance_id}/scenes` | Verified catalog replica cua dung instance, kem epoch/revision/stale |
| `GET` | `/api/spacecraft/{spacecraft_instance_id}/scenes/{catalog_epoch}/{scene_id}/{scene_revision}` | Metadata theo full ScopedSceneRef va product availability |
| `POST` | `/api/commands` | Validate, admission vao ledger va sinh/giu TC |
| `GET` | `/api/commands/{ground_instance_id}/{request_id}` | GDS command lifecycle va linked job/product |
| `GET` | `/api/products/{spacecraft_instance_id}/{origin_boot_id}/{product_id}` | Product manifest theo ProductRef, hoac retention tombstone `410` |
| `GET` | `/api/products/{spacecraft_instance_id}/{origin_boot_id}/{product_id}/download` | Tai product da downlink, `410` neu da evict |
| `GET` | `/api/products/{spacecraft_instance_id}/{origin_boot_id}/{product_id}/tiles/{z}/{x}/{y}` | Tile quicklook immutable theo ProductRef |
| `GET` | `/api/state` | Snapshot authoritative kem `as_of_event_id` |
| `GET` | `/api/frames` | P2: packet/frame da phan trang va filter |
| `GET` | `/api/link/profile` | Cau hinh mo phong link |
| `PUT` | `/api/link/profile` | Cap nhat fault profile co audit |
| `WS` | `/ws/telemetry?last_event_id=...` | TM, event, ACK va progress co cursor |

`POST /api/commands` chi tao va gui TC. Endpoint nay khong duoc goi truc tiep AI worker. GDS command envelope co `delivery_mode=immediate|next_contact` va `expires_at`:

- `immediate` khi khong co contact bi reject ro rang.
- `next_contact` duoc persist o GDS voi trang thai `HELD_NO_CONTACT`, qua han thi khong uplink.
- Validate TTL trong `1 s..24 gio`; default `immediate=5 phut`, `next_contact=1 gio`. Khi client omit `expires_at`, semantic digest dung sentinel `"DEFAULT"`; chi transaction insert dau tien materialize `effective_expires_at=server_now+default`, va moi retry cung key tra lai chinh gia tri da luu, khong tinh lai theo clock. Local outbox dung `lease_duration=10 s`, mission-ACK timeout `5 s` contact-open time, exponential retry `min(500 ms * 2^(attempt-1), 30 s)` va `max_attempts=20`. Retry giu RequestKey/payload, tiep tuc den ACK, max-attempt hoac effective expiry, moc nao den truoc.
- Lease het han dua `DISPATCHING -> OUTBOX_PENDING`. ACK timeout dua `SENT -> OUTBOX_PENDING` neu con attempt/TTL. Voi `next_contact`, contact dong pause ACK timer/backoff nhung khong pause absolute expiry; voi `immediate`, contact episode dong truoc receipt thi terminal `DELIVERY_FAILED: CONTACT_LOST` thay vi doi contact sau. ACK den sau `EXPIRED/DELIVERY_FAILED` duoc luu/audit la `LATE_RECEIPT` va co the link onboard job, nhung khong viet nguoc terminal delivery state.
- Browser khong tu giu command queue.
- Client gui `Idempotency-Key` HTTP cho moi submission. Scope MVP la `{gds_installation_epoch, principal, Idempotency-Key}` (`principal=local-operator`); networked profile thay principal bang tenant/subject da xac thuc. Backend parse/schema-validate va tinh semantic digest khong phu thuoc clock, sau do trong serialized transaction lookup key **truoc** contact/capacity/admission validation: cung key + digest tra row/effective expiry da commit du response truoc bi mat; cung key + digest khac tra `409 IDEMPOTENCY_CONFLICT`. Concurrent same-key chi duoc tao mot RequestKey/outbox row.
- Full response/command row co the prune theo retention. Lightweight `http_idempotency_retired` chi giu key, digest va original RequestKey trong `90 ngay` sau terminal-row prune, voi scope `{gds_installation_epoch, principal, Idempotency-Key}` (`principal=local-operator` trong MVP); retry trong TTL tra `410 IDEMPOTENCY_KEY_RETIRED` kem original RequestKey, digest khac van `409`. Het TTL marker duoc xoa va cung HTTP key co the tao command moi voi RequestKey moi; client phai tao key moi neu muon tranh nham retry. Process restart khong doi `gds_installation_epoch`; operator-authorized GDS reinitialize/corruption recovery tao epoch moi va invalidates old keys.
- Neu khong co row cu, backend validate capacity, cap `RequestKey`, insert `commands` va `command_outbox`, hoac rollback ca hai trong mot SQLite transaction. Outbox capacity `1024` chi dem row nonterminal/held/pending/in-flight, khong dem lich su terminal. API chi tra `202 Accepted` sau commit. Validation loi tra `422`, immediate/no-contact tra `409 NO_CONTACT`, va outbox day tra `429 QUEUE_FULL` kem `Retry-After`; khong duoc co command row mo coi khong co outbox.
- Moi link binding co durable monotonic `link_generation: U64` va `link_session_id`. Outbox worker claim row, snapshot generation/session, kiem tra expiry/target, cap Space Packet sequence va persist `command_attempt` raw bytes truoc send. Ngay truoc UDP send no acquire generation read-fence, revalidate target/generation/session va giu fence qua send syscall; mismatch ghi attempt `NOT_SENT_REBIND`. Migration acquire write-fence, dong admission old generation, doi moi read-fence dang active ket thuc, terminal hoa held/pending/leased/sent row cu thanh `DELIVERY_FAILED: TARGET_INSTANCE_RETIRED`, invalidate lease/close old session, roi moi publish binding/generation moi. Packet da send truoc fence chi mang old instance/session envelope va khong the route vao B. Generation U64 sap wrap thi fail closed/recreate GDS instance co chu dich.
- UDP/database khong the dam bao exactly-once network send. Contract la at-least-once transmission va exactly-once business effect nho satellite durable idempotency journal.

### 10.3 Du lieu luu tru

- `spacecraft_instances`: instance ID, link binding, link generation/session, active/retired state, first/last seen va rebaseline reason. GDS metadata cung luu mot `gds_installation_epoch: U64` CSPRNG duoc tao mot lan cho database lifecycle; restart giu nguyen, reinitialize/co corruption recovery tao epoch moi trong transaction migration.
- `catalog_snapshots`/`scenes`: `source_spacecraft_instance_id`, catalog epoch/revision/hash, scene revision, domain/capability metadata, sidecar/source SHA-256, immutable source/mask stat tuple, preview status, `active_preview_product_ref` (nullable full ProductRef) va sync time; scene primary identity la full ScopedSceneRef.
- `commands`: `target_spacecraft_instance_id`, `RequestKey`, HTTP idempotency key/digest, opcode, canonical args/mission digest, effective expiry, delivery state va timestamps.
- `command_outbox`: target instance, availability/expiry, attempt count, lease, last error va capacity state.
- `command_attempts`: target instance + link generation/session, immutable encoded TC bytes, APID/packet/frame sequence, send timestamp va result.
- `telemetry_samples`: source instance + `source_boot_id`, `simulation_run_id`, direction, `link_session_id`, `link_frame_id`, `copy_index`, `sample_ordinal`, APID/channel, satellite timestamp, GDS receive timestamp, raw value va decoded value. Dedupe canonical theo `{simulation_run_id,direction,link_frame_id,copy_index,sample_ordinal}` kem source instance; duplicate byte-identical khong tao sample hai lan, conflict bi audit/reject.
- `telemetry_rollups`: source instance, channel, 1-minute bucket, count/min/max/mean/last va source retention revision.
- `events`: monotonic event ID, source/target instance, server time, boot ID, severity, `RequestKey`, dictionary version va message.
- `link_frames`: source/target instance/session, direction, APID/VCID, sequence, CRC, fault decision va raw rolling-file segment/offset/length; khong luu raw frame BLOB vo han trong SQLite.
- `jobs`: source spacecraft instance, JobKey, ROI, thresholds, model version, progress, result va error.
- `products`/`product_transfers`: full ProductRef, origin RequestKey, bundle/artifact metadata, sender boot va `(spacecraft_instance_id, transfer_id)`, size/checksums, lifecycle va instance-owned local path.
- `simulation_runs`/`replay_segments`: run/release/profile identity, `OPEN|FINAL|INCOMPLETE_CRASH|INCOMPLETE_STORAGE`, `replay_state=PRESENT|PINNED|EVICTED`, ordered segment size/SHA, replay artifact size/SHA, reservation/pin va retention timestamps; segment bytes nam ngoai SQLite.
- `http_idempotency_retired`/`product_tombstones`: HTTP key digest + original RequestKey trong retention huu han cua cung `gds_installation_epoch`/principal, va full ProductRef/eviction reason/checksum metadata can de tra `410`.
- `audit_log`: user, action, old/new values va timestamp.

SQLite local profile:

- Startup bat `journal_mode=WAL`, `synchronous=FULL`, `foreign_keys=ON`, `busy_timeout=5000`; schema dung forward-only versioned migrations va fail readiness neu binary/schema khong tuong thich.
- Dung dung mot SQLite writer connection trong writer task; link worker, reassembly, retention va HTTP handler chi gui mutation intent qua bounded IPC, khong tu mo writer khac. `writer_queue_capacity=4096`, reserve `256` slot high-priority cho ledger/outbox/ACK ingest/status-cancel/audit. Queue full tra loi ro cho request; low-priority telemetry/frame batch co the bi reject voi counter/event, khong silent drop. Telemetry batch toi da `100 row` hoac `100 ms`, moc nao den truoc. Reader dung connection rieng.
- Khi ca high-priority reserve day, HTTP mutation tra `503 WRITER_BACKPRESSURE` + `Retry-After`; InMemory/stream link worker pause dequeue/propagate backpressure. UDP khong backpressure duoc thi drop datagram, tang synchronous metric va ghi fallback log `GDS_INGEST_OVERFLOW` truc tiep vao bounded rotated stderr/file, khong enqueue them database event de tranh overflow de quy. ACK mat theo nhanh nay dan den GDS command timeout/retry; durable raw-frame inbox nam ngoai MVP.
- Dat `wal_autocheckpoint=1000` page; read transaction/API stream toi da `2 s` roi page/reopen. App canh bao tai WAL `128 MiB`, tai `256 MiB` tam admission low-priority, ket thuc reader qua han va checkpoint; readiness fail neu khong the giam WAL. `TRUNCATE` checkpoint chi chay khi khong co reader active.
- Unique/index toi thieu cho `RequestKey`, HTTP idempotency `(gds_installation_epoch, principal, Idempotency-Key)`, event cursor, ScopedSceneRef, ProductRef, telemetry dedupe `(source_spacecraft_instance_id, simulation_run_id, direction, link_frame_id, copy_index, sample_ordinal)`, `(source_spacecraft_instance_id, channel, time)`, `(source_spacecraft_instance_id, direction, receive_time, APID)` va `(spacecraft_instance_id, transfer_id)`. Foreign key khong duoc noi entity qua instance boundary. API list dung keyset pagination, khong dung unbounded offset scan.
- Raw-frame segment gom version/length/CRC cho moi record, fsync append truoc khi commit DB reference. Startup scan tu offset DB cuoi, truncate torn/orphan tail; prune transaction xoa/retire DB references truoc roi moi xoa file, orphan file duoc startup sweep de khong tao dangling row.
- Retention mac dinh: raw telemetry `24 gio`, rollup 1 phut `30 ngay`, replayable event `24 gio`, frame metadata `24 gio`, command/job/transfer metadata va audit `90 ngay`, `.part` `24 gio`; catalog giu current + `10` snapshot da verify cho tung spacecraft instance. HTTP idempotency metadata/tombstone giu `90 ngay` sau terminal-row prune trong scope `{gds_installation_epoch, principal, Idempotency-Key}` (`principal=local-operator` trong MVP); het TTL thi xoa marker va cung key co the tao command moi voi RequestKey moi. Process restart khong doi epoch; operator-authorized GDS reinitialize/corruption recovery tao `gds_installation_epoch` moi va client phai tao key moi. Product tombstone giu `90 ngay` sau eviction, trong thoi gian do API tra `410`, het tombstone moi tra `404`. Rotated log dung earlier-of `7 ngay`/`1 GiB`; raw-frame segment roll `256 MiB` va dung earlier-of retention/`5 GiB` cap. Replay artifact co reservation/cap/retention rieng tai muc 9.3 va khong bi raw-frame pruning xoa theo.
- Final ground product da admission duoc dam bao `30 ngay`: truoc downlink, GDS reserve expected bundle + extraction overhead trong `20 GiB` cap unpinned, khong du thi reject `507`; khong evict product tre hon retention de nhan product moi. Sau 30 ngay, oldest unpinned bi evict truoc va API tra tombstone/`410` trong 90 ngay, sau do `404`; pinned product co quota operator rieng nhung van nam trong global/per-volume headroom va pin request bi reject neu thieu cho. Satellite cung reserve truoc product creation de dam bao READY product `7 ngay`; thieu cap thi reject admission `STORAGE_FULL`, khong tao product roi xoa som.
- Watermark tinh tren tung filesystem volume chua DB/WAL, raw segment, staging/product, replay va log. Moi product/replay/staging reservation debit cung mot durable per-volume headroom ledger, khong duoc overbook giua cac quota logic. High watermark `80%` prune theo thu tu: orphan/expired `.part` -> raw-frame qua retention/cap -> raw telemetry da rollup -> derived cache -> expired incomplete replay -> expired unpinned product/replay -> log qua retention. Hard watermark `90%` dung raw/replay capture, terminal run dang mo thanh `INCOMPLETE_STORAGE`, va reject command tao product/run moi voi `507 STORAGE_FULL`.
- Tren **DB/WAL volume**, profile tao emergency reserve file `256 MiB`; tai hard watermark release reserve de bao dam terminal ACK/status-cancel, ledger va audit ghi duoc, sau do chi recreate reserve khi usage xuong duoi `80%`. Moi volume staging/product/log khac co admission reservation/headroom cua chinh no. Neu write quan trong van fail, readiness fail va khong admission command moi; khong duoc hua van ghi khi khong con reserved headroom.

### 10.4 Scene catalog authority va sync

- Satellite inventory la authority. Catalog identity gom `catalog_epoch`, `catalog_revision` va `snapshot_sha256`; moi scene entry gom immutable `scene_id`, `scene_revision`, source/sidecar SHA-256, shape/bands/format, sensor/platform/product-type/processing-level domain, reader backend/version, `analysis_capability=VERIFIED|UNSUPPORTED|INVALID` kem reason va `active_preview_product_ref` nullable (full ProductRef).
- Ingest phai chay memmap/sidecar/domain capability check truoc khi publish catalog revision. Ingest-only writer copy source + validity sidecar vao content-addressed, read-only package theo SHA-256, fsync va atomic publish; runtime chi mount package read-only. GDS chi enable Analyze/ROI khi capability `VERIFIED`; Satellite revalidate khi command den bang catalog identity, capability va cheap stat tuple `(file_id/device,inode-or-file-id,size,mtime_ns)` cua source/mask, khong hash lai toan file trong command path. Full SHA/value scan chi chay luc ingest va startup scrub. Neu stat/hash/value mismatch hoac file bi mutate out-of-band, mark scene `INVALID`, tao scene revision moi voi reason, va reject command; khong doc byte khong co trong catalog. Backend thay the chi duoc danh VERIFIED sau pixel oracle, full-decode spy, target RSS va p95 benchmark artifact.
- Add/remove/replace scene tang revision; thay bytes du giu cung `scene_id` van bat buoc tang `scene_revision`. `scene_id` dung durable allocator trong epoch; `catalog_epoch` durable allocate va khong reuse trong spacecraft instance. Truoc khi `catalog_revision`, `scene_id` allocator hoac bat ky `scene_revision` U32 sap wrap, Satellite atomic cap epoch chua tung dung, rebuild full snapshot voi ID/revision moi tu dau va phat full-resync event; khong publish gia tri wrap trong epoch cu. Truoc khi `catalog_epoch` U32 wrap, service fail closed va bat buoc migrate sang spacecraft instance moi. Reboot satellite khong doi catalog epoch neu inventory khong reset; reset namespace cung tao epoch moi va full-resync.
- MVP `SCENE_REQUEST_CATALOG` tao full deterministic catalog snapshot bundle. GDS chi atomic activate replica sau transfer/checksum/manifest verification; snapshot partial khong duoc phuc vu nhu current.
- Health TM quang ba spacecraft instance + catalog epoch/revision. `/api/spacecraft/{spacecraft_instance_id}/scenes` tra `synced_at` va `stale`; GDS restart co the phuc vu cached replica dung instance nhung danh stale cho toi khi revision duoc xac nhan. Replica cua instance retired van read-only theo retention, khong duoc dung de build command nham instance active moi.
- Preview/analyze/ROI mang full `SceneRef`; catalog epoch mismatch tra `CATALOG_EPOCH_MISMATCH`, scene revision mismatch tra `SCENE_REVISION_MISMATCH`, deu khong tao job/product. Sau PREVIEW publish thanh cong, Satellite/GDS CAS-update `active_preview_product_ref` neu van dung full SceneRef/source SHA va tang `catalog_revision`; tile route chi doc theo ProductRef day du, cache key gom full ProductRef + `{z,x,y}`. Preview version moi khong ghi de byte/product cu va pointer cu van truy cap duoc theo ProductRef retention. Khi active preview bi evict/invalid, chi CAS-clear neu pointer hien tai dung ProductRef bi evict, tang catalog revision truoc/ke cung tombstone; route ProductRef cu van tra `410` trong retention. Shared-volume demo khong duoc tu y bypass catalog contract.

### 10.5 Realtime state va reconnect

- Database/GDS backend la authority cho delivery ledger va replica da ingest; Satellite van la authority cua onboard catalog, command validation va job/product origin. WebSocket chi la dong event cap nhat.
- `GET /api/state` tra snapshot kem `as_of_event_id`. WebSocket nhan `last_event_id`, replay trong retention window hoac tra `RESYNC_REQUIRED` neu cursor qua cu.
- Client deduplicate va apply event theo `event_id`. Sau disconnect, client phai reconnect voi exponential backoff, replay tu cursor hoac lay snapshot moi; reconnect khong duoc chi giu state cu trong RAM.
- Moi TM hien `gds_receive_time` va age; boot ID moi tao mot epoch state moi, khong noi nham event truoc/sau satellite restart.

### 10.6 Backend metrics va logging

- Metrics toi thieu: command status/latency, TC encode time, queue depth/rejects, worker restarts, WebSocket clients/drops, database write backlog, product transfer timeout/gap/checksum failure.
- Cung cap `/healthz`, `/readyz` va `/metrics`; readiness phan anh database, link worker va decoder process, khong chi HTTP process. Scheduled `NO CONTACT`/`BLACKOUT` la trang thai nghiep vu khoe manh va khong lam `/readyz` fail.
- Log co rotation/retention, khong ghi auth token hay product payload.

## 11. GDS Webapp

### 11.1 Bo cuc van hanh

- Thanh tren: connection state, satellite state, link mode, queue depth va current time.
- Ben trai: scene catalog, search/filter va product availability.
- Trung tam: OpenLayers viewer, quicklook/tile pyramid, ROI va cloud-mask overlay.
- Ben phai: command form theo command dictionary va config nguong.
- Phia duoi: TC/TM/event timeline, product transfer progress va packet inspector P2.

Giao dien nen dam, de scan va toi uu cho thao tac lap lai; khong dung bo cuc landing page hoac card trang tri.

### 11.2 Cong cu ROI

- Segmented control `Pan` / `Select ROI`.
- Keo hinh chu nhat, resize canh/goc va di chuyen ROI.
- Dong bo hai chieu voi input `x`, `y`, `width`, `height`.
- Hien kich thuoc pixel va dien tich ROI.
- Khoa ROI trong bien scene va enforce `width,height >= patch_size` giong backend; scene `UNSUPPORTED/INVALID` khong enable Analyze/ROI.
- Nut reset dung icon va tooltip.
- Slider kem numeric input cho model threshold va coverage limit.
- Hai nguong duoc commit nguyen tu bang `CLOUD_SET_CONFIG`; UI khong phat hai update doc lap co the tao config lai.
- Truoc khi uplink, hien command preview gom target spacecraft instance, full SceneRef, ROI, hai threshold, config epoch/revision, HTTP `Idempotency-Key` va estimated downlink time/canh bao catalog stale/fault profile/contact state. `RequestKey` chi hien sau khi backend admission tra `202 Accepted`; UI khong tu cap phat hay dua truoc mot RequestKey.

Frontend co the luu ROI normalized cho view state khi resize, nhung gia tri authoritative cua command la pixel integer tren scene goc voi mien half-open. Voi drag float, `x/y=floor(min_corner)` va `end_x/end_y=ceil(max_corner)`, sau do clamp vao scene va tinh `width=end_x-x`, `height=end_y-y`; numeric input da la integer thi giu nguyen. Backend tinh/validate lai cung quy tac; sai so viewer-to-scene toi da 1 pixel.

### 11.3 Anh lon va tile

- Satellite tao quicklook va/hoac tile pyramid khi ingest scene.
- Quicklook/tile phai duoc downlink sang GDS truoc khi browser su dung.
- Browser chi truy cap GDS storage, khong doc truc tiep satellite volume.
- Neu MVP cho phep shared volume de tang toc demo, phai dat feature flag va ghi ro day la low-fidelity mode.
- Nguon analytic `uint16` duoc Satellite product generator tone-map theo display profile cap scene/ROI thanh quicklook/tile `8-bit sRGB`; browser khong tu tone-map TIFF goc.
- Display profile gom band order, NoData, black/white point, gamma/tone curve va algorithm version. AI normalization va display tone mapping la hai contract doc lap.
- Khong tinh percentile rieng tung tile vi se tao seam. Cloud mask la categorical overlay, dung nearest-neighbor va khong tone-map.
- Quicklook/tile co checksum, immutable product version va HTTP `ETag`. Tile mac dinh `256 x 256`; client cache co memory cap/LRU, huy request ngoai viewport va mac dinh toi da `8` concurrent tile requests.
- Local reference profile gioi han derived-tile cache phia GDS o `5 GiB` theo LRU/retention; cache eviction khong duoc xoa authoritative downlinked product.
- Canvas/WebGL duoc chon qua benchmark; khong mac dinh WebGL neu chua co bang chung. Tile/quicklook khong thay the crop TIFF/JP2 analytic.

### 11.4 Degraded/blackout UX

- UI hien rieng trang thai Browser-GDS, GDS-satellite/contact va tuoi TM: `CONNECTED`, `NO CONTACT`, `BLACKOUT`, `STALE TM`, `SATELLITE DEGRADED`.
- Cached telemetry luon hien `last_updated`/age; du lieu stale khong duoc trinh bay nhu realtime. Local profile dat `tm_stale_after=max(3 * health_period, 5 s)`.
- Catalog stale va model `demo_non_validated` duoc hien nhu status cua ket qua; UI khong trinh bay tile-area proxy nhu pixel cloud percentage.
- UI khong hard-disable moi command chi dua tren telemetry `READY` co the stale. UI canh bao va gui request den backend; backend/satellite la authority tra `ACCEPTED`, `BUSY`, `QUEUE_FULL` hoac `NO_CONTACT`.
- `delivery_mode=next_contact` chi duoc queue trong persisted GDS ledger, khong trong local browser. Confirmation phai hien expiry va fault profile dang active.

### 11.5 Frontend state va packet inspector

- Server state dung mot normalized store tap trung va duoc reconcile bang REST snapshot + WebSocket event cursor; ROI chua gui va editing state la local UI state rieng.
- Packet inspector P2 co parsed view va hex dump, filter APID/direction/time/CRC, server-side pagination va virtualized rows.
- Frame retention nam o GDS storage/rolling files; browser khong giu buffer vo han. Export offline la P2 sau khi core workflow on dinh.

## 12. Bao mat va an toan van hanh

- Security profile MVP la `local_sil`, khong TLS/auth va chi co identity `local-operator`; profile nay khong production-ready. No co hai deployment mode hop le:
  - `host_local_sil`: moi process chay tren host va HTTP/WebSocket/UDP chi bind/peer `127.0.0.1`.
  - `compose_sil`: service bind interface container tren Compose network `internal: true`; chi GDS/UI duoc publish ra host tai `127.0.0.1`, Link Simulator va Satellite khong co host port va khong noi external network.
- Startup/deployment validation fail neu bind/published-port/network topology lech mode da chon. Exact Host/HTTP Origin/WebSocket Origin allowlist bat buoc cho GDS; UDP peer allowlist bat buoc cho Link/Satellite. Request JSON phai dung Content-Type, body-size/rate limit va download quota.
- Local limits mac dinh: HTTP header `16 KiB`, JSON body `64 KiB`, general API token bucket `30 request/s` burst `60`, command `4/s` burst `8`, toi da `8` WebSocket/client, `2` concurrent product download/client va `50 MiB/s` download shaping; mot bundle toi da `1 GiB`, extract toi da `2 GiB`/`32` regular file. State-changing HTTP va WebSocket handshake thieu/sai Origin bi reject; safe GET co the thieu Origin neu Host + peer dung profile, nhung Origin neu co van phai exact-match.
- `local_sil` tin cay moi local OS user/process co the truy cap loopback hoac Compose daemon/network; Host/Origin khong thay the authentication. May multi-user khong duoc coi la security boundary, va phai chuyen networked profile neu can tach operator.
- Role `Observer`/`Operator`/`Admin`, OIDC/session, TLS, CSRF, secret storage va remote download authorization thuoc networked profile ngoai MVP; khong duoc expose LAN/Internet truoc khi profile nay duoc implement va negative-test.
- Audit moi command va thay doi nguong.
- Validate payload theo dictionary truoc encode va sau decode.
- Gioi han ROI, queue length, product size va retention.
- Sanitize scene/product path, cam path tuy y tu TC.
- Dung `RequestKey`, HTTP idempotency key va command confirmation de han che thao tac lap/sai.
- Khong tu viet lop ma hoa tuy bien. Neu can bao mat data-link thuc, lap ke hoach SDLS rieng.

Release manifest chi ghi artifact build bat bien: `release_id`, Git commit/dirty flag, dependency-lock/SBOM hash, container digest, compiler va Python/PyTorch/NumPy/tifffile/CUDA/cuDNN/TensorRT/JetPack versions, F Prime/dictionary/mission-profile hash, model/InputSpec release hashes va threshold-mapping ID/LUT SHA-256. Reproducible build pin `SOURCE_DATE_EPOCH`, canonical archive/file/SBOM ordering, normalized owner/mode/mtime va OCI creation/history metadata. Release chinh thuc khong duoc mang dirty flag.

Moi simulation run tao run manifest `OPEN` gom release ID, full ScopedSceneRef/source/catalog snapshot, config/model/deployment/fault-profile revisions, seed, `simulation_run_id`, simulation-clock epoch va replay reservation/state. Chi khi stop/finalize atomic moi dong command set, ghi command-set hash + `replay_artifact_sha256/size` va chuyen `FINAL`; run bi crash/storage-cap thanh `INCOMPLETE_CRASH|INCOMPLETE_STORAGE`, khong duoc claim replay-complete. Manifest luu `replay_state=PRESENT|PINNED|EVICTED`; `FINAL` chi mo ta command set da ket thuc, con replay chi kha dung khi state la `PRESENT`/`PINNED`. Artifact self-contained van replay duoc sau khi raw rolling frames da prune, nhung retention eviction phai chuyen state `EVICTED` va tra loi khong con replay bytes. Protocol/fault replay byte-exact chi duoc claim trong pinned algorithm/runtime/platform profile; logits giua backend dung tolerance da khoa, con decision golden phai giong nhau va khong dat score sat threshold tolerance.

## 13. Lo trinh trien khai

### Phase 0 - Dong bang baseline va contract, 6-10 person-days

- On dinh cac thay doi inference hien co.
- Tao package layout; dong goi checkpoint + InputSpec manifest va normalization contract cu the.
- Loai bo viec mission runtime dua vao default 4 kenh; them smoke test checkpoint 3 kenh va negative test sai channel/SHA/patch size/normalization.
- Implement Model/InputSpec schema parser, SHA verification va mission adapter truyen InputSpec vao `CloudTorchInfer`; runtime khong tu khoi tao model tu CLI defaults.
- Pin F Prime v4.1.0 theo dictionary hien tai hoac lap upgrade task rieng; khong tron source/dictionary v4.1.0 voi tai lieu `latest`.
- Tao dependency lock co hash cho Python/Node, pin container/base image digest, ghi F Prime/dictionary hash va them dev/test dependencies de baseline test co the tai lap.
- Xac nhan stakeholder chap nhan baseline Type-BD/pixel/byte-frame SIL; chot policy downlink ROI bi reject. Yeu cau full COP-1, lat/lon hoac RF/SDR la change request phai re-open scope/estimate, khong phai toggle trong baseline.
- Dong bang mission profile stock APID `0/1/2/3`, VCID, endian, sequence/time contract, `FW_COM_BUFFER_MAX_SIZE=512`, `FW_FILE_BUFFER_MAX_SIZE=1003`, descriptor 2 byte, buffer-manager bins, one-packet-per-frame topology va exact TM budget 990 byte raw file.
- Dong bang `RequestKey`/JobKey, ScopedSceneRef/ProductRef, target/source spacecraft instance, config/catalog epoch/revision/wrap, sau state machine doc lap + science-decision enum, durable idempotency/restart policy va error codes trong schema/golden vectors.
- Dong bang scene-anchored grid, window-readable validity mask/strict NoData/padding, memmap-only runtime raster, `logit-bp-f32-lut-v1`, deterministic TAR/checksum ownership va catalog authority/capability/domain.
- Dong bang `MissionComScheduler` + completion-gate behavior ben trong `MissionUdpAdapter` (khong tao stock adapter rieng), initial queue capacities + quy tac tune/freeze sau benchmark, ACK token, overflow policy, watchdog/restart, FileDownlink abort/cooldown, tone-map/display va numeric budget.
- Dong bang `demo_non_validated` scientific status hoac metric/cost thresholds de promote; tao conformance matrix ngay trong Phase 0.
- Dong bang `host_local_sil`/`compose_sil`, SQLite writer/WAL/retention/watermark va canonical deterministic fault-stage + self-contained replay artifact contract.

**Exit criteria:** baseline test pass; checkpoint/InputSpec golden va mismatch tests pass; dependency/model/dictionary hashes tai lap duoc; mission profile/conformance matrix duoc review; command/config/state/catalog/bundle/ROI/security/storage/fault schemas khong con ambiguity/implementation placeholder. Non-deployable benchmark template duoc phep `null`, nhung phai bi schema/startup chan khoi `READY`.

### Phase 1 - ROI inference core, 7-10 person-days

- Them `read_window(x, y, width, height)` slice ca X/Y va `infer_region()` theo scene-anchored grid.
- Nap model singleton, bounded queue, batch config va progress co throttle.
- Tao crop, quicklook 8-bit/tile, mask, manifest va deterministic TAR co display/model assurance profile.
- Them regression test dung checkpoint that, threshold LUT, ROI shift/NoData-context/padding oracle, sidecar checksum va negative test compressed TIFF/JP2/runtime mask khong memmap duoc.
- Spy test chung minh mission inference khong goi full decode/`TiffFile.asarray` cho ca source va validity mask, khong doc tron chieu ngang va khong tao crop tam.
- Chay local CPU/PyTorch batch-1 benchmark theo patch count/ROI size, sinh non-null benchmark artifact + deployable reference profile, khoa RSS/p95/logical-read guards va dat job-deadline model ban dau; Phase 6 revalidate tren tung target.

**Exit criteria:** ROI hop le tren canonical memmap TIFF chay dung; scene grid, patch bien, strict valid-data, integer threshold, normalization, deterministic bundle va progress throttle co test; local CPU profile co benchmark artifact va ROI nho dat exact logical-read, `256 MiB` RSS-delta, `1.25x` p95 scene-scale guard; browser artifact khong phu thuoc file `uint16` goc.

### Phase 2a - F Prime skeleton, dictionary va protocol, 5-8 person-days (conditional)

Native F Prime source/toolchain is not present in this repository or local
environment. The Python reference artifacts below are not a native-delivery
claim. Phase 2a native exit and final DoD remain conditional until the pinned
source, `fprime`/FPP generation toolchain, native deployment, and native vector
and UDP E2E evidence are available. See
`docs/gds_satellite_ccsds_fprime_scope_decision_20260721.md`.

- Tao flight deployment va `CloudPayload` FPP component.
- Dinh nghia commands, telemetry, events va opcodes.
- Tich hop stock APID router, scheduler forwarding skeleton `READY/IN_FLIGHT` va `MissionUdpAdapter` voi completion-gate behavior, full dataReturn/comStatus chain voi `SpacePacketFramer -> TmFramer`; TC/TM framing, CRC va APID sequence; bypass `ComAggregator` theo profile MVP.
- Sinh dictionary va golden binary vectors, gom APID/descriptor route `0/1/2/3`, unknown/mismatch reject, DATA `990` byte va oversize, idle padding, Space Packet rollover `16382, 16383, 0, 1`, TM MCFC/VCFC `254, 255, 0, 1` va timestamp application payload.

**Exit criteria:** TC APID `0` tu GDS dispatch dung va APID unknown khong dispatch; TM descriptor/APID `1/2/3` dung; dictionary/profile/F Prime build constants khop; binary/rollover/exact frame/source+destination path-length tests pass; hai packet nho tao dung hai TM frame; status-before-frame-return bi completion gate trong adapter tri hoan, scheduler chi complete khi co own return + propagated status, khong gui packet thu hai som, UAF/leak hay double-return buffer.

### Phase 2b - AI worker, watchdog, queue va TM scheduler, 7-10 person-days

- Ket noi AI worker qua IPC co heartbeat, timeout, bounded restart va staging cleanup.
- Implement durable onboard transactions cho command/job/config/transfer, idempotency journal+retired ranges, sau state machine + science enum, cancel race, `QUEUE_FULL`, job deadline va reconciliation moi entity khi restart.
- Hoan thien `MissionComScheduler` voi ACK storage/token rieng, control/file queue, upstream-return hold, single-in-flight ownership, `ack_burst=8`, `control_burst=4`, `file_burst=8`, backpressure va progress coalesce.
- Tich hop deterministic bundle FileDownlink, `FileDownlinkCoordinator` ABORTING/COOLDOWN va `PRODUCT_REQUEST_DOWNLINK`.

**Exit criteria:** worker/process-kill, config/attempt crash matrix, cancel race, full-journal/retired-range, control/ACK flood, upstream-return hold, mid-file terminal failure/no-late-DATA, queue saturation va continuous-file-downlink tests pass; business effect dung mot lan cho duplicate `RequestKey`; service khong publish partial product; ACK/health/file service dat bound da chot.

### Phase 3 - Link Simulator, 8-12 person-days

- Implement in-memory va UDP transport.
- Them latency, loss, duplicate, corruption, bandwidth va blackout.
- Them virtual clock/event queue, canonical counter PRF/distribution, serialized ingress, stage order, UDP self-contained segmented replay artifact, quota/retention, queue overflow policy, structured log va link metrics.
- Them FilePacket START/DATA/END/abort drain fence, sender boot handshake/restart resolution va prove khong cross-attempt/boot reorder.
- Benchmark goodput cua TM frame 1024 byte va recovery cost theo fault profile muc tieu.

**Exit criteria:** cung ordered ingress bytes, run ID, seed va profile sinh decision log/byte stream giong nhau du downstream worker scheduling khac; concurrent ingress co logged admission order; profile-update/restart va UDP replay byte-exact; OPEN/finalize/crash/cap artifact dung state, raw-frame prune khong lam mat artifact co `replay_state=PRESENT|PINNED`, va retention/pin eviction chuyen dung sang `EVICTED`; overflow counter khop; attempt/boot A khong cross barrier vao B; ACK/health latency bound van dat khi file downlink.

### Phase 4a - GDS ledger, API va SQLite core, 8-12 person-days

- Atomic command ledger + transactional outbox/lease/attempts scoped target instance, HTTP idempotency/default-expiry ordering, TC sequence allocator va audit.
- Single-writer IPC, priority/capacity, WAL checkpoint/migrations, keyset pagination, telemetry rollup va storage watermark/reserve.
- Command timeout/retry, HTTP/satellite idempotency, spacecraft-instance migration fence va boot/restart ledger recovery.

**Exit criteria:** admission/outbox crash matrix khong mat command hay tao row mo coi; concurrent same-key, omitted-expiry retry, finite HTTP idempotency TTL/installation epoch, target migration, capacity, WAL/long-reader, writer saturation, raw-segment crash/prune va reserved-headroom tests pass.

### Phase 4b - TM, catalog, file, realtime va local deployment, 10-15 person-days

- Implement TM decoder, instance-scoped catalog capability/snapshot sync, immutable scene-package/stat-scrub va active preview ProductRef CAS, ScopedSceneRef/ProductRef REST state va WebSocket cursor/replay.
- Implement FilePacket reassembly, generic manifest, safe extraction, transport/bundle/artifact verification, retry va crash-safe atomic publish.
- Implement rolling-frame/log/product va replay retention (`PRESENT|PINNED|EVICTED`), backend/link/transfer metrics, structured logging va health/readiness.
- Enforce `host_local_sil`/`compose_sil` topology, Host/Origin/peer/body/rate/path policy va startup guard.

**Exit criteria:** round trip command -> ACK -> progress -> result -> verified bundle hoat dong khong can web UI; catalog old-epoch/domain/capability, active-preview version/cache identity, out-of-band scene mutation, lost-frame/retry, publish/restart, retention, local profile negative/connectivity, slow WebSocket va cursor-resync tests pass.

### Phase 5 - GDS Webapp core, 8-12 person-days

- P0: scene viewer/tile, ROI interaction va command confirmation.
- P0: catalog/config/model-assurance stale state, degraded/blackout va immediate/next-contact delivery UX.
- P1: realtime telemetry/event timeline va product transfer progress.
- P1: product preview/download, cache budget va Canvas/WebGL benchmark.
- P2 sau MVP core: packet inspector nang cao va export offline.

**Exit criteria:** operator hoan thanh full workflow chi bang webapp; blackout, stale TM, reconnect/resync va fault-profile warning co E2E test; packet inspector P2 khong chan MVP.

### Phase 6 - E2E va hardening, 10-16 person-days

- Playwright desktop/mobile cho ROI va command flow.
- Fault/reconnect/file-epoch/replay-artifact lifecycle tests.
- Benchmark batch `1, 2, 4, 8...` tren CPU/GPU/Jetson va dat resource guards tu ket qua.
- Soak test queue, WebSocket, log/metric/replay retention-cap va product staging cleanup.
- Docker CPU profile va Jetson/L4T profile.
- Release/run manifest, clean-build reproducibility, SBOM, runbook va demo scenario; revalidate conformance matrix da tao o Phase 0.

**Exit criteria:** tat ca acceptance criteria muc 14 dat va demo co the lap lai.

### Uoc luong

- Tong uoc luong: `69-105 person-days` truoc risk contingency, la tong cac phase MVP o tren.
- Hai ky su: khoang `11-17 tuan`, da tinh dependency va 25-40% contingency; khong suy ra bang cach chia doi person-days don gian.
- Mot ky su: khoang `18-29 tuan`, gom 25-40% contingency cho F Prime integration, raster I/O, storage va fault/restart recovery.
- Full COP-1, CFDP, SDLS hoac RF/SDR can duoc uoc luong thanh phase rieng sau MVP.
- True windowed compressed TIFF/JP2 dung `5-10 person-days` chi cho discovery/PoC va phai re-estimate implementation sau benchmark. Networked security uoc luong them `8-15`; scientific validation them `8-15` neu da co labeled holdout, chua gom thu thap nhan/retraining.

## 14. Ke hoach kiem thu va tieu chi nghiem thu

### 14.1 Unit tests

- Stock APID/descriptor golden routes: TC command `0`, TM telemetry `1`, event/ACK `2`, file `3`; unknown va descriptor/APID mismatch bi reject.
- Space Packet primary header, packet length, APID va sequence vectors `16382, 16383, 0, 1`; TM MCFC/VCFC `254, 255, 0, 1` rollover dong bo trong single VC.
- TC/TM frame encode/decode, CRC, malformed length, exact FilePacket DATA `990` byte, Space Packet `1009` byte, idle padding 7 byte, oversize reject, FileDownlink source path `23 < 101` va START destination `97 < 101` byte.
- `RequestKey`, JobKey, target instance, ScopedSceneRef va ProductRef envelope; U64 ground namespace, request U32 wrap/restart, old catalog epoch collision va product ID reuse o boot/instance moi khong alias identity cu. Boot/transfer allocator missing/corrupt/near-wrap phai fail closed va migration doi spacecraft instance.
- U64 canonical scalar golden vectors: JSON/JCS/REST/WS/log fixed 16-hex lowercase string, binary/fixed replay fields 8-byte BE (deterministic-CBOR exception uses canonical unsigned integer), SQLite BLOB(8), TypeScript BigInt/string; boundary `0`, `2^53`, `2^63`, `2^64-1`, malformed decimal/uppercase/space reject.
- Config CAS epoch/revision/wrap, threshold byte-equality va immutable snapshot: job accept tai config N van dung N sau khi global config len N+1; identity dung/value sai tra `CONFIG_SNAPSHOT_MISMATCH`.
- `logit-bp-f32-lut-v1` generation/SHA/golden binary32 tai `1..9999`; model threshold tai `0`, `1`, equality, `9999`, `10000`, non-finite logit; runtime spy khong goi platform `ln`. Coverage equality/endpoints va checked-U64/floor serialization khop oracle.
- Legal transition table cho GDS delivery, onboard command, job, product generation, satellite transfer va GDS transfer; science enum rieng, terminal immutable, cancel-vs-complete allow-list va non-job command khong tao job ngam dinh.
- Threshold basis point, integer cross-multiply tai boundary bang nguong va exact `floor(... * 10000 / ...)` TM serialization; khong dua tren float ratio da lam tron.
- ROI half-open/rounding/min-size/overflow, scene-anchored cell selection, grid-boundary/scene-edge/1-pixel shift; sidecar schema/hash, NoData trong ROI, NoData chi o patch context, padding value-space va mask mismatch. Validity mask compressed/non-memmap, sai H/W/dtype/value/hash/path bi reject; zero-valued source pixel van valid neu sidecar khong khai NoData.
- Weighted `cloud_positive_tile_area_ratio_bp` chi tinh valid `patch intersect ROI`, khong tinh context, padding hay NoData.
- Model/InputSpec golden transform va checkpoint mismatch: wrong channel, SHA, patch size va normalization.
- Model assurance/domain schema, promotion artifact/hash/sample-count/confidence-interval validation va PyTorch/CPU/GPU/TensorRT golden logits theo tolerance; decision vector khong duoc nam sat threshold tolerance.
- Deployment profile validation: `batch_size > 0`, `batch_size <= effective_max_batch_size`, benchmark artifact dung target/runtime va khong vuot TensorRT optimization profile.
- Mission digest exact-byte gom target instance va HTTP RFC 8785 digest golden gom explicit expiry/`"DEFAULT"` sentinel. Khi full journal con: same key/same digest replay, same key/different digest conflict; sau compact: ca same/different payload trong retired range deu `DUPLICATE_REQUEST_RETIRED`, GET/CANCEL deu `TARGET_RETIRED`.
- Deterministic `ANALYSIS/PREVIEW/CATALOG` POSIX USTAR golden byte qua process/toolchain; manifest timestamp bat bien, JCS+LF, khong self-reference, exact entry set, duplicate normalized path/case collision, non-regular/PAX/GNU/symlink/traversal/artifact mismatch bi reject.
- `PRODUCT_REQUEST_DOWNLINK`: origin RequestKey phai khop ProductRef; attempt moi bat buoc RequestKey moi, retry cung key khong tao transfer thu hai.
- `MissionComScheduler`/completion gate ben trong `MissionUdpAdapter`: progress coalescing, ACK slot cach ly fault flood, ACK/control/file burst oracle, single-in-flight, status-before-frame-return bi delay, own return/status theo moi thu tu, upstream buffer chua return truoc terminal completion va ownership return dung mot lan; late failure khong doc/requeue buffer da return. Fake downstream phai callback reentrant trong cung call stack va reset session o truoc/giua/sau `FRAME_ACCEPTED`.
- FileDownlink stock contract: build macro `FILEDOWNLINK_COMMAND_FAILURES_DISABLED=false`, only typed internal SendFile/Cancel route duoc phep, external TC opcode bypass bi reject, side metadata khong ghi de `Fw::Buffer.context/data/size`, cooldown tick dung `ceil` va watchdog khong de stock WAIT timeout assert.
- File reassembly byte ranges, duplicate identical, conflicting overlap, gap, transport checksum, bundle SHA va artifact checksum.
- Catalog epoch/revision/add/remove/replace, capability/domain va stale SceneRef rejection; catalog/scene revision near-wrap atomic bump epoch, epoch near-wrap fail closed/migrate, khong SceneRef nao reuse trong instance. Catalog payload permutation golden, PREVIEW pointer CAS voi hai ProductRef version va tile cache full-ProductRef key; immutable source/mask out-of-band mutation -> INVALID/new revision.
- Canonical fault PRF/distribution golden vectors gom rate `0/1_000_000`, U128 bounded map, invalid corruption-bit count, latency/jitter/time overflow va ceil serializer; event-queue tie-break/decision-log round-trip. Replay record length/CRC, ordered segment hash/tree hash va torn-tail recovery co golden; cung ordered ingress nhung downstream scheduling khac phai cung trace.
- UDP sideband ingress/egress exact vectors: ingress ID fields zero/sender boot zero, Link Simulator assigns link/file epoch only after `FRAME_ACCEPTED`, egress carries assigned IDs, stale session/ID reuse/length mismatch reject; LinkControl epoch mapping and `FRAME_CONSUMED` timeout/session reset.
- HTTP idempotency finite-retention vectors: same key/body replay within 90 days, different body conflict, retired marker expiry creates a new RequestKey, GDS installation epoch change isolates old keys; telemetry duplicate/conflict dedupe includes source boot/run/frame/copy/sample ordinal.

### 14.2 Integration tests

- GDS TC APID `0` -> link -> stock F Prime route -> command ACK APID `2`; unknown APID khong dispatch.
- Hai TM packet nho tao dung hai frame. Ep transport status den truoc adapter frame return de chung minh completion gate ben trong `MissionUdpAdapter` tri hoan propagated status; dao moi thu tu own dataReturn/status va xac nhan scheduler chua return upstream/khong admit packet thu hai cho toi completion, callback reentrant trong cung call stack khong lam mat flags, khong retry pointer cu, leak hay double-return.
- Reorder `CLOUD_SET_CONFIG(N -> N+1)` va ROI expected N+1 de ROI den truoc -> `CONFIG_REVISION_MISMATCH`, khong tao job. Inject crash truoc/sau onboard config transaction commit, reboot va replay cung RequestKey -> dung mot revision increment va cached snapshot N+1.
- ROI command -> scene-anchored inference -> TM progress -> `SUCCEEDED` + science decision; compressed TIFF/JP2 source va compressed/non-memmap validity mask bi reject truoc full decode. Full-decode spy va RSS oracle phai bao phu rieng source/mask.
- Accepted ROI -> deterministic bundle -> file packets -> verify transport/SHA/manifest/artifact -> mot atomic published directory.
- Drop mot FilePacket -> transfer `INCOMPLETE`, khong publish final; full-file retry tao dung mot final product.
- Duplicate byte-identical va out-of-order FilePacket -> bat buoc tao byte-exact final product; conflicting overlap -> terminal error, khong publish.
- FileDownlink stock opcode/compile contract: external `SendFile`/`SendPartial`/`Cancel` TC bypass bi reject; chi internal coordinator route duoc goi stock typed command, `FILEDOWNLINK_COMMAND_FAILURES_DISABLED=false`, buffer context/data/size round-trip byte-exact, watchdog + cooldown khong de late-return assert.
- Downlink READY product cua boot cu sau sender reboot va drop/reorder `PRODUCT_DOWNLINK_STARTED` -> START path + global transfer ID van bind dung session. Hai attempt cung product co transfer ID khac nhung bundle byte/hash giong nhau.
- Rejected ROI -> job `SUCCEEDED`, bundle khong co full crop va van qua artifact policy/checksum.
- Satellite process restart tai tung state command/job/product/transfer trong bang reconciliation -> config/journal atomic, nonterminal work terminal hoa dung policy, READY product/catalog/allocator duoc giu, khong publish partial; `COMMAND_ACCEPTED` khong work row bi detect corruption/FAULT va khong bao gio la crash outcome binh thuong.
- Race `JOB_CANCEL` voi queue dequeue/job complete/timeout/restart -> chi mot terminal job state, cancel ACK tra `CANCEL_REQUESTED`, `CANCELED` hoac `ALREADY_TERMINAL` dung allow-list.
- Race `PRODUCT_CANCEL_DOWNLINK` voi final DATA/END -> non-final vao `CANCEL_DRAINING`/`CANCELED`, final DATA completion-wins `SEND_COMPLETED`, duplicate cancel replay outcome cu; no transfer attempt moi duoc tao.
- Kill AI worker khi dang inference -> `WORKER_LOST`, khong publish staging; restart worker thanh cong ve `READY`, qua retry limit vao `FAULT`.
- Giu worker busy, fill du `max_pending_jobs` pending slots roi gui them mot job -> job du nhan `QUEUE_FULL`, khong mat hoac chay trung.
- Concurrent outbox admission tai capacity -> dung `1024` row nonterminal commit, phan du `429`, khong co command/outbox row mo coi. Concurrent same HTTP key chi tao mot row; retry row cu duoc tra truoc no-contact/full validation; same key/body khac `409`. Omit `expires_at`, inject crash sau commit/tru response roi retry muon hon phai tra cung RequestKey/effective expiry va `"DEFAULT"` digest, khong tao expiry moi; sau 90-ngay marker expiry cung key duoc cap RequestKey moi, con installation epoch moi tach hoan toan namespace.
- Outbox crash injection sau commit/tru HTTP, sau lease, sau persist raw attempt, sau UDP send truoc status va sau ACK ingest -> khong mat admitted command; co the send lap nhung onboard effect dung mot lan.
- Virtual-clock outbox test `lease=10 s`, ACK timeout `5 s`, exponential backoff/max-attempt/TTL boundary, next-contact pause vs immediate contact-loss va late ACK sau terminal; capacity duoc release dung allow-list.
- `HELD_NO_CONTACT` het han qua GDS restart khong duoc gui; Space Packet sequence allocator persist va retry sinh sequence moi tru rollover hop le.
- Migrate bound link tu spacecraft instance A sang B khi A con PENDING/SENT TC va con retained catalog/product. Moi outbox A terminal `DELIVERY_FAILED(reason=TARGET_INSTANCE_RETIRED)`, khong packet nao duoc rewrite/gui sang B; route/API full identity van tra dung artifact A va khong alias catalog/product B. Operator submit lai tren B nhan RequestKey moi.
- Catalog add/remove/replace tang revision; GDS chi activate verified snapshot. Command cu co cung scene ID/revision nhung epoch cu bi `CATALOG_EPOCH_MISMATCH`; capability/domain mismatch khong silent inference. Inject catalog/scene revision `0xffffffff` boundary phai bump durable epoch + full resync; epoch exhaustion fail closed/migrate instance. Publish hai PREVIEW version phai CAS pointer dung full ProductRef, catalog revision tang, tile cache khong alias; mutate source/mask sau ingest phai bi startup scrub/command stat check danh `INVALID` va reject.
- Link loss/corruption/duplicate/blackout; frame gui trong blackout bi drop, attempt cu khong resume va full-file retry sau contact tao final product dung.
- Delay DATA/END cua file attempt A toi boundary, yeu cau B trong khi A chua drain -> `TRANSFER_BUSY`; START B chi admit sau barrier va khong nhan nham packet A. Inject terminal adapter failure khi DATA hien tai va DATA ke tiep dang queued: coordinator vao ABORTING, khong them DATA/END, return moi buffer dung mot lan va chi release slot sau abort fence/COOLDOWN. START/DATA/END loss van khong duoc coi drain fence la delivery ACK.
- Kill Satellite khi DATA boot A dang nam trong latency/reorder queue, khoi dong boot B: startup handshake phai deliver/drop + close epoch A truoc READY/START B. Restart Link Simulator giua fence phai recover queue hoac explicit-drop old records, khong packet A nao cross sang session B.
- Cung ordered frame bytes/run ID/seed/profile voi downstream scheduling khac sinh cung decision log/byte stream; concurrent ingress replay theo recorded order, profile update/restart va UDP artifact replay byte-exact, doi seed tao trace khac. Test OPEN crash/torn-tail, atomic FINAL, cap exhaustion `INCOMPLETE_STORAGE`, pin nam trong global cap, retention chuyen `PRESENT/PINNED -> EVICTED`, va replay FINAL sau khi raw-frame segment da prune.
- File downlink + continuous command/ACK va control/fault flood -> `oldest_ack_age <= 1 s`, `health_max_latency <= 2 s`, fault khong chiem ACK slot va file van co minimum service theo `ack_burst/control_burst` oracle.
- Crash transfer attempt truoc/sau allocation/send start khong tao zero/hidden second attempt. GDS restart giua transfer hoac sau `VERIFIED`/rename -> khoi phuc/terminal dung state va dung mot final directory.
- SQLite migration/WAL recovery, writer queue saturation/priority gom high-reserve full, HTTP `503`, stream backpressure va UDP fallback overflow log; long reader/WAL cap, raw append-before-DB crash, prune crash, guaranteed product/replay retention-admission cap, log quota, per-volume emergency reserve va keyset pagination pass.
- `host_local_sil` reject public bind/foreign peer; `compose_sil` ket noi duoc noi bo nhung Link/Satellite khong co host/LAN exposure. Ca hai test exact header/body/API+command rate/WS/download/product limits, missing/foreign Origin theo method, foreign Host va path traversal.

### 14.3 Web E2E

- Chon scene capability VERIFIED, zoom/pan va keo ROI; scene unsupported/domain-unverified hien dung status/admission policy.
- ROI khong thay doi toa do khi resize viewport; drag rounding va min `patch_size` khop backend.
- Gui command va theo doi rieng command/job/science/product/transfer lifecycle qua `RequestKey`.
- Catalog stale/old epoch, scene/config snapshot mismatch, `demo_non_validated`/`DOMAIN_UNVERIFIED` hien dung status; tile proxy khong bi gan nhan pixel cloud percentage.
- WebSocket reconnect theo cursor khong mat/duplicate state; cursor qua retention bat buoc snapshot resync.
- Event phat sinh giua luc lay snapshot va mo WebSocket duoc replay tu `as_of_event_id`.
- Blackout hien `NO CONTACT`/`STALE TM`; immediate command bi reject ro rang, next-contact command duoc persist den `expires_at`.
- Fault profile active va estimated downlink time hien trong command confirmation.
- Slow WebSocket client bi disconnect/resync ma khong lam RAM server tang vo han.
- `/readyz` van healthy trong scheduled blackout, nhung fail khi database/link worker/decoder process hong.
- Hien loi decode, timeout va reject ro rang.
- Khong co text/control overlap tren desktop va mobile.

### 14.4 Performance/resource tests

- Scene `10980 x 10980` khong vuot resource guard; peak process memory khong vuot 75% memory budget duoc cap cho profile.
- ROI `256 x 256` doc X/Y expanded-patch window that cho ca source va validity mask; `logical_source_bytes_read`, `logical_validity_bytes_read` va tong khop oracle, RSS delta `<=256 MiB`, p95 tren scene-area `4x` `<=1.25x` va khong goi full decode.
- Model chi nap mot lan trong vong doi worker.
- Benchmark `batch_size=1,2,4,8...` den gioi han target; ghi throughput, p95 latency, peak RSS/shared memory va OOM. Chi cau hinh batch da co artifact benchmark.
- Health/ACK TM dat `oldest_ack_age <= 1 s`, `health_max_latency <= 2 s` va file co non-zero goodput trong local SIL khi file/control transfer bao hoa.
- Browser khong request source TIFF/JP2; tone mapping deterministic, golden pixels/checksum khop va khong co seam giua tile.
- Warm-tile p95 `<= 200 ms` tren local reference profile; browser tile cache `<= 256 MiB`, GDS derived-tile cache `<= 5 GiB` va ca hai khong tang vo han trong soak test.
- Mot request E2E truy vet duoc qua GDS, Link Simulator va Satellite log bang `{spacecraft_instance_id, RequestKey}`; deterministic fault counters/replay artifact khop byte-frame oracle va log/SQLite/raw-frame/replay retention khong vuot disk budget.
- SQLite sustained-write soak giu writer/WAL backlog, DB/raw/log/product size trong budget; hard watermark/reserve khong lam mat terminal ledger/audit.
- Clean release build hai lan voi cung `SOURCE_DATE_EPOCH` co cung pinned artifact/SBOM/OCI/LUT hashes; run manifest chi claim complete o state `FINAL`, tham chieu self-contained replay artifact size/SHA + replay state, va replay sau raw prune tao cung protocol/fault outputs trong supported platform profile khi artifact chua `EVICTED`.
- Ghi latency rieng cho no-fault local, CPU, GPU va Jetson; khong dat SLO inference truoc khi co benchmark tren target hardware.

### 14.5 Definition of Done

- UI khong co duong tat goi inference truc tiep.
- APID/descriptor `0/1/2/3`, packet/frame size va bytes khop golden vectors/conformance matrix/build constants.
- Sai so ROI tu viewer den scene goc khong qua 1 pixel.
- Cung `RequestKey` replay truoc/sau satellite restart hoac sau full-journal compaction trong active ground namespace khong chay trung inference hay tao transfer attempt thu hai; payload khac sau compact cung bi retired, khong bi nhan nham la conflict co digest.
- `202 Accepted` dong nghia command + outbox da commit atomic; outbox send at-least-once va satellite business effect exactly-once.
- Moi command/entity/route GDS scope spacecraft instance; migration terminal hoa outbox cu va khong retry business effect sang instance moi.
- TM result chua full SceneRef, ROI/grid/validity/padding algorithm, config epoch/revision + hai nguong, floor-serialized tile-area ratio, science decision, model release/domain/assurance, latency va ProductRef/bundle checksum.
- Full workflow `scene -> ROI -> TC -> ACK -> inference -> TM -> product` chay thanh cong.
- Moi profile duoc phep vao `READY` co non-null benchmark artifact dung target/runtime; reference MVP co deployable local CPU/PyTorch profile.
- Worker crash, queue saturation, file packet loss va blackout deu co terminal state/ma loi va recovery test.
- Web reconnect tao state trung khop database snapshot; telemetry stale co age ro rang.
- Product chi duoc publish sau khi bundle du byte, Fw checksum, bundle SHA, exact archive entry set va moi artifact hash dung; catalog stale/old epoch/capability mismatch khong silent run.
- Stock FilePacket chi co mot global attempt; scheduler upstream-return hold, coordinator ABORTING/COOLDOWN va START/DATA/END + sender-boot drain barrier ngan cross-attempt/restart misbinding ma khong bi hieu nham thanh delivery ACK.
- `host_local_sil` chi loopback; `compose_sil` chi publish GDS tren host loopback va co internal-only Link/Satellite. Release va run manifest tach rieng; moi run `FINAL` ghi replay state/hash/revision, replay duoc sau raw-frame prune khi artifact con `PRESENT`/`PINNED`, va `EVICTED` khong duoc claim replay bytes.
- 53 test cu tiep tuc pass va cac test moi pass.
- Co runbook khoi dong, cau hinh fault profile va demo scenario.

## 15. Rui ro chinh va bien phap

| Rui ro | Tac dong | Bien phap |
|---|---|---|
| Model la classifier patch-level | Tile-area proxy tho, khong dung cho cloud boundary chinh xac | Dung ten metric chinh xac, science assurance; dung segmentation neu yeu cau pixel-level |
| Model chua co scientific validation | Demo bi hieu nham la quyet dinh flight/operational | `demo_non_validated` mac dinh; promotion gate/model card/cost metrics rieng |
| Checkpoint 3 kenh nhung CLI default 4 | Runtime failure hoac sai input | Model manifest va fail-fast validation |
| Normalization training/runtime khong khop | Model cho score sai silent | InputSpec cu the, golden transform/output va khong cho runtime heuristic |
| ROI-anchored grid | Dich ROI 1 pixel lam thay toan bo patch input | Scene-anchored grid, intersection weighting va shift oracle |
| Config SET/ROI reorder hoac crash giua mutation/journal | Job dung threshold sai hoac revision tang hai lan | Atomic onboard transaction, config identity+value equality va immutable job snapshot |
| Anh mat georeference | Khong chon ROI theo lat/lon | MVP dung pixel; sua ingest de giu CRS/geotransform |
| Compressed/JP2 runtime full decode | ROI nho van OOM/latency theo toan scene | MVP memmap TIFF fail-closed; backend moi phai qua resource gate |
| Anh gan 690 MiB qua lon cho browser | Treo tab, ton bang thong | Quicklook/tile 8-bit, tone-map server-side va bounded cache |
| APID mission khong khop F Prime router | TC khong toi dispatcher, TM route sai | MVP stock APID `0/1/2/3`, compile/golden route tests |
| Bypass ComAggregator nhung thieu reverse status/ownership | `TmFramer` assert, leak/double-return buffer | Completion gate ben trong MissionUdpAdapter, observable completion tuple, upstream-return hold va delayed/failure tests |
| F Prime MVP chi Type-BD | Mat TC khong co COP-1 retransmission | Timeout/retry idempotent; lap phase COP-1 rieng |
| Mat TM packet khi downlink file | File thieu nhung bi coi la hoan chinh | Persist reassembly, gap/checksum detection va full-file retry attempt moi |
| DATA/END attempt/boot cu den sau START moi | Stock FilePacket khong co transfer ID de demux | Mot global attempt, coordinator abort/cooldown, START/DATA/END + boot-epoch drain barrier; CFDP neu can concurrency |
| Product gom nhieu artifact | Reassembly collision/checksum self-reference | Mot deterministic TAR/product, bundle hash ngoai manifest, atomic directory publish |
| GDS crash quanh send | Mat command hoac send lap | Transactional outbox + lease, at-least-once send va durable onboard idempotency |
| Outbox cu retry sang spacecraft instance moi | Business effect cu chay lai, catalog/product alias | Target instance trong wire/ledger, migration terminal fence va instance-scoped API/FK |
| Catalog GDS stale/epoch reset | Inference nham bytes du cung scene ID/revision | Satellite authority, full SceneRef/source SHA va mismatch-resync |
| Shared filesystem lam giam do trung thuc | Web co the vo tinh bo qua downlink | Tach satellite/ground storage va feature flag low-fidelity |
| GPU bi goi dong thoi | OOM, latency khong on dinh | Singleton runtime va bounded single-worker queue |
| AI worker crash/treo | Service mac ket PROCESSING hoac publish partial | Heartbeat, deadline, bounded restart va atomic staging |
| Queue/WS client bi bao hoa | Mat command, tang RAM, health TM bi doi | Capacity/error policy, scheduler bound, disconnect/resync slow client |
| SQLite/WAL/raw/log/product/replay tang vo han | Het disk lam mat ledger/product hoac run replay | Single writer, WAL cap, rolling files, admission reservation, quotas, retention, watermark va emergency reserve |
| Concurrent ingress/thread timing khac | Fault test khong lap lai | Serialize/log admission order, counter PRF, virtual clock/tie-break va self-contained replay artifact |
| Compose bind sai network | Service khong noi duoc nhau hoac bi expose LAN | Profile-specific topology guard, internal network va published-port negative test |
| Output filename/ID chung qua reboot | Race, ghi de hoac tro nham product | RequestKey/SceneRef/ProductRef namespace, fixed canonical path, atomic write va checksum |

## 16. Gate 0 va change control

### 16.1 Default ky thuat da khoa cho MVP

1. Pin F Prime v4.1.0, TC Type-BD, OCF absent, stock APID `0/1/2/3`; khong custom APID mapper/FOP/FARM/COP-1 trong MVP.
2. Mot TM VC, `MissionComScheduler` single-in-flight + completion gate ben trong `MissionUdpAdapter` co full ownership/status chain va upstream-return hold, bypass `ComAggregator`, mot Space Packet/frame, `FW_FILE_BUFFER_MAX_SIZE=1003`, raw file toi da 990 byte/frame.
3. Pixel ROI half-open, scene-anchored grid, strict full-patch valid-data `10000 bp`, explicit window-readable validity sidecar/padding, `logit-bp-f32-lut-v1` va runtime input chi memmap-compatible TIFF.
4. `RequestKey`/JobKey/ScopedSceneRef/ProductRef, target/source spacecraft instance, exact mission/HTTP digest/default-expiry, atomic config CAS/snapshot, catalog wrap policy, sau state machine + science enum, durable journal/retired ranges va restart reconciliation moi entity.
5. Mot byte-canonical typed USTAR moi product, fixed source `23`/destination `97` byte, mot global FilePacket attempt + abort/cooldown va attempt/boot drain barrier, full-file retry va crash-safe atomic product-directory publish.
6. Satellite catalog authority; GDS verified versioned replica va moi scene command mang full SceneRef/capability/domain contract.
7. `demo_non_validated` model assurance cho toi khi qua target-domain scientific promotion gate.
8. `host_local_sil` hoac `compose_sil`, SQLite WAL/single-writer/retention-watermark/reserve va release/run manifest tach rieng.
9. Fault profile dung serialized ingress, canonical counter PRF/distribution, virtual clock va self-contained segmented replay artifact co quota/retention.

### 16.2 Input stakeholder can chot trong baseline

1. ROI bi science-reject co downlink crop day du hay chi metadata/quicklook/mask?
2. Analytic output dung TIFF hay JP2; quicklook WebP co bat buoc khong?
3. Target benchmark/release chinh la PC, Jetson Nano hay ca hai?
4. Sau benchmark, queue capacities, watchdog/deadline/restart limit va ACK/health/file-goodput bound chot bao nhieu?
5. Display tone-map profile, tile/cache quota va renderer profile nao duoc dung?
6. `demo_non_validated` co du cho muc tieu demo hay can `validated_decision`; neu can, supported domain, metric threshold va false-clear/false-reject cost matrix la gi?
7. Product/log retention quota va danh sach product nao duoc operator pin?

### 16.3 Lua chon ngoai baseline, bat buoc re-plan

- Full COP-1/Type-AD/CLCW, ECSS PUS, CFDP/selective recovery hoac SDLS.
- ROI lat/lon/georeference, RF/SDR/channel coding hoac network-exposed GDS.
- Nang F Prime khoi v4.1.0, custom APID, compressed TIFF/JP2 input backend hoac segmentation pixel-level.

Chon bat ky muc 16.3 nao phai re-open Phase 0, conformance matrix, golden vectors, threat model va estimate; khong duoc bat bang feature flag trong baseline.

Mac dinh cho input chua duoc stakeholder override: khong downlink crop bi science-reject, TIFF analytic + WebP quicklook 8-bit, CPU/PyTorch la reference truoc Jetson/TensorRT, queue/SLO dung gia tri khoi diem trong plan va product unpinned theo retention local profile.
