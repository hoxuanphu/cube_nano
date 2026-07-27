# Runbook van hanh GDS va Satellite CCSDS SIL

Ngay cap nhat: 2026-07-21
Pham vi da xac minh: ` host_local_sil `, CPU/PyTorch, mot may, khong RF/SDR.

Tai lieu nay mo ta cach he thong dang chay trong repository hien tai, cac lenh co
the copy-paste, contract HTTP/command va cach doc ket qua. Day la runbook van
hanh cho local SIL, khong phai huong dan tuyen bo he thong da san sang cho
flight hoac production network.

## 1. Ket luan nhanh

Duong chay duoc xac minh day du la ` host_local_sil `:

```
GDS HTTP/Web -> command ledger/outbox -> Space Packet APID 0
             -> TC Transfer Frame Type-BD -> MissionLink in-process
             -> satellite decoder/CloudPayload -> worker inference
             -> TM/FilePacket -> GDS reassembly/checksum
             -> atomic product publish -> HTTP/Web
```

Trang thai release hien tai:

| Profile | Trang thai | Pham vi |
|---|---|---|
| ` host_local_sil ` | READY khi benchmark/hash hop le | Duong chay reference, da xac minh scene -> ROI -> TC -> ACK -> inference -> TM/FilePacket -> ` PUBLISHED ` |
| ` compose_sil ` | Profile va UDP bridge hop le | Topology Compose duoc validate; chua claim E2E HTTP nhieu container |
| ` jetson-l4t-tensorrt ` | BLOCKED | Thieu benchmark tren Jetson va TensorRT optimization profile |

Khong duoc suy ra tu runbook nay rang da co COP-1/FOP/FARM/CLCW, RF/SDR,
SDLS, TLS/OIDC/RBAC, pixel-level cloud segmentation hay scientific validation
cua model. Model mac dinh co ` assurance_level=demo_non_validated ` va output
la patch/tile-area decision.

## 2. Dieu kien tien quyet

Thuc hien tu root cua repository:

```
Set-Location D:\AI20K\cube_nano
```

Can co:

- Python 3.11 hoac 3.12. Moi lenh phai dung cung mot interpreter da cai FastAPI, Uvicorn, PyTorch CPU, ` tifffile `, NumPy va PyYAML.
- Node.js/npm neu muon chay webapp hoac Playwright.
- Docker Desktop va Linux engine neu muon build/chay Compose.
- Cac file model va benchmark: ` checkpoints/best_model.pth `, ` sat_ai/model_manifest.yaml `, ` sat_ai/deployment_profile.yaml ` va artifact tuong ung trong ` artifacts/benchmarks/ `.

Runtime reference nen cai theo [requirements-lock.txt](../requirements-lock.txt),
khong theo requirements training khong pin trong
[requirements.txt](../requirements.txt):

```
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    python -m venv .venv
}
& $py -m pip install --upgrade pip
& $py -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-lock.txt
& $py -c "import fastapi, uvicorn, websockets, yaml, numpy, tifffile, torch; print('runtime dependencies: OK'); print(torch.__version__)"
```

Neu ` .venv ` da ton tai nhung thieu ` fastapi `/` uvicorn `, khong dung no de
chay demo cho den khi cai lai lock file. Kiem tra interpreter dang duoc goi:

```
& $py -c "import sys; print(sys.executable)"
```

Trong cac lenh ben duoi, thay ` python ` bang ` & $py ` neu muon ep dung
interpreter trong ` .venv `.

## 3. Cau hinh va readiness gate

` SatelliteDeployment ` khong vao ` READY ` chi vi process da khoi dong. Khi tao
` SatelliteSimulator `, no validate theo thu tu thuc te sau:

1. [protocol/mission_profile.yaml](../protocol/mission_profile.yaml) va F Prime dictionary v4.1.0.
2. Model manifest, checkpoint SHA-256, ` InputSpec ` RGB 3 kenh va threshold LUT.
3. [sat_ai/deployment_profile.yaml](../sat_ai/deployment_profile.yaml) co ` deployable=true `.
4. Benchmark JSON ton tai, dung ` target_id `/runtime/CPU threads, va SHA-256 trong deployment profile khop byte file.
5. SLO profile va cac guard RSS/logical-read/scene-scale.
6. Satellite scene catalog, journal SQLite va reconciliation sau restart.
7. Worker process singleton khoi dong va gui heartbeat.

Gia tri reference hien tai:

| Truong | Gia tri |
|---|---|
| Runtime profile | ` host_local_sil ` |
| SCID | ` 68 ` |
| Spacecraft instance | ` 0000000000000001 ` |
| F Prime | ` v4.1.0 ` |
| TC APID / TM APID | ` 0 ` / ` 1 ` |
| TM event/file APID | ` 2 ` / ` 3 ` |
| TM frame | ` 1024 ` bytes |
| Model release | ` cloud-mobilenetv3-small-rgb-r1 ` |
| Model assurance | ` demo_non_validated ` |
| InputSpec | RGB, ` uint16 `, patch ` 256 `, NCHW ` float32 ` |
| Config mac dinh | ` config_epoch=0 `, ` config_revision=0 `, threshold ` 5000 bp `, coverage ` 6000 bp ` |
| Queue job | toi da ` 4 ` job cho reference worker |

Boot ID, RequestKey va ProductRef co the thay doi sau moi lan khoi dong; khong
hard-code cac gia tri nay trong script van hanh. Lay chung tu ` /api/state ` hoac
output cua command.

## 4. So do luong xu ly

### 4.1 Khoi dong

```
runtime profile
    |
    v
MissionProfile + F Prime constants + model/InputSpec + LUT
    |
    v
benchmark artifact SHA + SLO + scene catalog
    |
    v
satellite journal reconcile + worker process/heartbeat
    |
    v
READY
```

Mot loi manifest, benchmark, checkpoint, catalog hoac worker la loi startup;
khong sua ` deployable `/` SHA ` chi de bo qua gate.

### 4.2 Command va product

` POST /api/commands ` duoc ghi vao command ledger va transactional outbox trong
mot commit. GDS cap RequestKey va tinh mission_digest; browser chi cung cap
` Idempotency-Key `, khong tu cap RequestKey. Outbox sau do:

1. Encode command payload thanh Space Packet APID ` 0 `.
2. Boc Space Packet vao TC Transfer Frame Type-BD, VC ` 0 `, CRC/FECF.
3. Gui frame qua ` MissionLink ` va nhan business ACK.
4. Satellite router kiem tra SCID/VC/APID, sau do ` CloudPayload ` validate target, request digest, catalog/config/ROI.
5. Analysis duoc ghi journal voi config snapshot bat bien, day vao worker process bounded queue.
6. Worker tao product staging; chi product da duoc verify moi duoc rename atomic sang final directory.
7. Local HTTP adapter tu dong tao ` PRODUCT_REQUEST_DOWNLINK ` sau khi analysis ` SUCCEEDED `.
8. FilePacket TM di qua scheduler, GDS reassemble START/DATA/END, kiem tra file checksum, bundle SHA, manifest va tung artifact hash.
9. Ground product chi chuyen sang ` PUBLISHED ` sau khi tat ca kiem tra pass.

Trinh tu terminal binh thuong:

```
COMMAND_ADMITTED -> OUTBOX_PENDING -> ACKED
                 -> JOB_QUEUED -> RUNNING -> SUCCEEDED
                 -> PRODUCT_READY -> transfer VERIFIED -> PUBLISHED
```

` ACKED ` chi la business command acknowledgement. No khong phai CLCW hay
link-layer delivery acknowledgement cua COP-1.

### 4.3 Identity phai duoc giu nguyen

- ` spacecraft_instance_id `: U64, chuoi hex thuong 16 ky tu, vi du ` 0000000000000001 `.
- ` RequestKey `: ` {ground_instance_id: U64, request_id: U32} `.
- ` SceneRef `: ` {catalog_epoch, scene_id, scene_revision} `.
- ` ProductRef `: ` {spacecraft_instance_id, origin_boot_id, product_id} `.
- Threshold dung basis point ` 0..10000 `, khong dung float trong wire payload.
- ROI la toa do pixel ` x,y,width,height `; width/height phai duong va patch toi thieu la ` 256 `.

Neu cung Idempotency-Key duoc gui lai voi body khac, GDS tra ` 409 ` conflict.
Retry cung body se tra lai command da commit va khong tao business effect thu hai.

## 5. Smoke khong co HTTP

### 5.1 Health gate

```
python -m flight.satellite_simulator --root . --health-once
```

Lenh phai in JSON co cac truong toi thieu:

```
state=READY
worker_state=READY
worker_heartbeat_age_ms gan 0
spacecraft_instance_id=0000000000000001
model_release_id=cloud-mobilenetv3-small-rgb-r1
assurance_level=demo_non_validated
```

` sender_boot_id ` thay doi theo boot. Neu lenh raise ` satellite deployment is
not READY `, xem [Xu ly su co](#10-xu-ly-su-co).

### 5.2 ROI smoke qua worker va FilePacket

```
python -m flight.satellite_simulator --root . --roi-smoke
```

Lenh in health JSON truoc, sau do result JSON. Dieu kien pass:

```
job_state=SUCCEEDED
downlink_frame_count > 0
transfer_state=SEND_COMPLETED
```

Smoke nay goi ROI ` (0,0,256,256) ` tren ` SceneRef(1,1,1) `, cho worker chay,
tao downlink va drain frame o satellite. No khong mo HTTP va khong kiem tra
browser rendering.

### 5.3 Demo E2E disposable, khuyen nghi

```
python scripts/demo_scenario.py --root . --timeout 90
```

Script tao ` LocalSilMission ` va state directory tam trong OS temp, sau do
tu dong cleanup. Ket qua pass can co:

```
status=PASS
job_state=SUCCEEDED
product.state=PUBLISHED
product.verified=true
product.checksum_status=SHA256_MATCH
product.transfer_state=VERIFIED
```

Trace trong output phai cho thay ` opcode=65541 ` (` ROI_REQUEST `), ` outbox_state `
` ACKED `, ` product_state=PUBLISHED ` va ` transfer_id ` khong null.
` science_status ` co the la ` DOMAIN_UNVERIFIED `; day la dung voi model demo
hien tai va khong duoc doc thanh scientific acceptance.

### 5.4 Round trip cap protocol

De kiem tra khong co browser va khong dung HTTP adapter:

```
python scripts/p4b_roundtrip.py --root .
```

Script dung state tam rieng cho satellite va GDS, catalog sync, ROI analysis,
downlink TM FilePacket va reassembly. Ket qua da xac minh truoc day la
` analysis=SUCCEEDED `, ` frame_count=526 `, ground ` PUBLISHED ` va
` shared_volume_bypass=false `. So frame phu thuoc bundle fixture va co the
thay doi neu artifact thay doi.

## 6. Chay backend HTTP

Truoc khi chay host profile, xoa override Compose neu no dang ton tai:

```
Remove-Item Env:CUBE_NANO_RUNTIME_PROFILE -ErrorAction SilentlyContinue
```

Mo terminal 1 tai repository root:

```
$env:PYTHONPATH = (Get-Location).Path
python -m gds.http_app --root . --host 127.0.0.1 --port 8000
```

Startup tao mot mission trong process gom GDS, ` MissionLink ` va Satellite
Simulator. Mac dinh state nam tai ` .cube_nano-cache/p6-http/ `:

```
.cube_nano-cache/p6-http/
  satellite/satellite.sqlite3
  satellite/products/
  ground/gds.sqlite3
  ground/products/
  ground/reassembly/
```

Kiem tra tu terminal 2:

```
$base = "http://127.0.0.1:8000"
Invoke-RestMethod "$base/healthz" | ConvertTo-Json -Depth 8
Invoke-RestMethod "$base/readyz" | ConvertTo-Json -Depth 8
Invoke-RestMethod "$base/api/state" | ConvertTo-Json -Depth 12
```

Dieu kien san sang:

- ` /healthz ` tra HTTP ` 200 `, ` status=ok `, satellite ` state=READY `.
- ` /readyz ` tra HTTP ` 200 `; database, decoder/worker va satellite khong loi.
- ` /api/state ` co mot spacecraft key, mot scene trong catalog, config hien tai va ` runtime.gds_satellite=CONNECTED `.

` BLACKOUT ` la transport state, khong phai process fault; trong scheduled
blackout ` /readyz ` van co the ` 200 ` trong khi command immediate bi tu choi.

Dung Ctrl+C o terminal backend de shutdown binh thuong. Khong xoa
` .cube_nano-cache/p6-http ` khi process van dang chay.

## 7. Chay webapp

Mo terminal 3:

```
Set-Location D:\AI20K\cube_nano\gds\web
npm install
$env:VITE_API_BASE_URL = "http://127.0.0.1:8000"
npm run dev -- --host 127.0.0.1 --port 4173
```

Mo http://127.0.0.1:4173. Vite chi phuc vu UI; inference va command van chay o
backend. UI dung REST snapshot va /ws/telemetry; browser khong doc TIFF, khong
import sat_ai va khong cap RequestKey.

Playwright tu khoi dong backend/UI disposable, nen tat server thu cong truoc:

```
Set-Location D:\AI20K\cube_nano\gds\web
npm test -- --run
npm run build
npm run test:e2e
```

### 7.1 Goi ROI truc tiep

Tat ca POST can Origin trong allowlist; /api/commands can Idempotency-Key. Vi du
lay instance/config tu snapshot va gui ROI:

```
$base = "http://127.0.0.1:8000"
$state = Invoke-RestMethod "$base/api/state"
$instance = ($state.state.spacecraft.PSObject.Properties | Select-Object -First 1).Name
$config = $state.state.configs.PSObject.Properties[$instance].Value
$body = @{
    target_spacecraft_instance_id = $instance
    opcode = 65541
    payload = @{
        scene_ref = @{ catalog_epoch = 1; scene_id = 1; scene_revision = 1 }
        roi = @{ x = 0; y = 0; width = 256; height = 256 }
        expected_config_epoch = $config.config_epoch
        expected_config_revision = $config.config_revision
        model_threshold_bp = $config.model_threshold_bp
        coverage_limit_bp = $config.coverage_limit_bp
    }
    delivery_mode = "immediate"
}
$headers = @{
    Origin = "http://127.0.0.1:8000"
    "Idempotency-Key" = "runbook-roi-$([guid]::NewGuid())"
}
$accepted = Invoke-RestMethod -Method Post -Uri "$base/api/commands" -Headers $headers -ContentType "application/json" -Body ($body | ConvertTo-Json -Depth 10)
$accepted | ConvertTo-Json -Depth 10
```

Response la HTTP 202, co request_key, mission_digest, command_state ADMITTED va
outbox_state OUTBOX_PENDING (hoac HELD_NO_CONTACT). Poll command:

```
$uri = "$base/api/commands/$($accepted.request_key.ground_instance_id)/$($accepted.request_key.request_id)"
Invoke-RestMethod $uri | ConvertTo-Json -Depth 12
```

Cho den khi product published va download:

```
$published = $null
for ($i = 0; $i -lt 180 -and $null -eq $published; $i++) {
    Start-Sleep -Milliseconds 500
    $snapshot = Invoke-RestMethod "$base/api/state"
    $published = $snapshot.state.products.PSObject.Properties.Value | Where-Object state -eq "PUBLISHED" | Select-Object -First 1
}
if ($null -eq $published) { throw "product was not published before timeout" }
$ref = $published.product_ref
Invoke-WebRequest "$base/api/products/$($ref.spacecraft_instance_id)/$($ref.origin_boot_id)/$($ref.product_id)/download" -OutFile "artifacts\runbook-product.tar"
```

Route chinh gom /healthz, /readyz, /api/state, catalog scenes, POST
/api/commands, command status, product metadata/download, /admin/contact/{state}
va WebSocket /ws/telemetry. Opcode 65541 la ROI_REQUEST; UI dung opcode nay va
HTTP adapter tu tao PRODUCT_REQUEST_DOWNLINK sau analysis thanh cong. Body toi
da 1 MiB, header toi da 16 KiB, rate 120 request/phut.

## 8. Blackout, next-contact va restart

```
$base = "http://127.0.0.1:8000"
$headers = @{ Origin = "http://127.0.0.1:8000" }
Invoke-RestMethod -Method Post -Uri "$base/admin/contact/BLACKOUT" -Headers $headers
```

Trong blackout, immediate bi tu choi voi HTTP 409 NO_CONTACT; next_contact van
duoc admit HTTP 202 voi HELD_NO_CONTACT. Mo contact de dispatcher tiep tuc:

```
Invoke-RestMethod -Method Post -Uri "$base/admin/contact/CONTACT_OPEN" -Headers $headers
```

State mac dinh:

| Entry point | State |
|---|---|
| flight.satellite_simulator | data/satellite/state/ va data/satellite/products/ |
| gds.http_app | .cube_nano-cache/p6-http/ |
| scripts/demo_scenario.py | OS temp, disposable |
| scripts/p4b_roundtrip.py | OS temp, disposable |

Dung Ctrl+C va doi worker/SQLite dong sach truoc khi archive/xoa state. Khong xoa
DB khi outbox dang OUTBOX_PENDING/SENT hoac product dang RECEIVING. Startup se
reconcile journal va cleanup staging; demo script tranh alias product cu.

## 9. Compose va fault profile

Kiem tra Compose syntax:

```
docker compose -f deploy/docker-compose.yml config --quiet
```

Neu Docker daemon dang chay:

```
docker compose -f deploy/docker-compose.yml build
docker compose -f deploy/docker-compose.yml up
```

Compose publish GDS tren 127.0.0.1:8000, dat link/satellite tren network
mission_internal va danh dau network internal. Gioi han quan trong: gds/http_app
van tao LocalSilMission voi MissionLink in-process; CUBE_NANO_LINK_MODE chua
wiring HTTP qua UDP bridge. UDP bridge cua link co ton tai, nhung E2E nhieu
container chua duoc xac minh. Dung host_local_sil cho workflow E2E da verify.

Neu docker info loi named pipe, Docker Linux daemon chua chay. Shutdown:

```
docker compose -f deploy/docker-compose.yml down
```

Khong dung down -v neu muon giu volume ground_state.

deploy/fault_profiles/lossless.yaml va degraded.yaml hien la fixture khai bao;
khong co CLI --fault-profile va link_sim.__main__ chua load truc tiep. Dung test:

```
python -m pytest tests/test_link_simulator.py tests/test_link_simulator_blackout.py tests/test_phase6_recovery.py -q
```

FaultProfile duoc truyen vao MissionLink/LinkSimulator trong code va dung counter
PRF deterministic. Muon van hanh degraded bang YAML can them loader/CLI/E2E wiring.

## 10. Xu ly su co

| Trieu chung | Xu ly |
|---|---|
| Thieu fastapi/uvicorn | Cai lai requirements-lock bang dung interpreter; kiem tra sys.executable. |
| Satellite khong READY | Kiem tra benchmark artifact/hash, checkpoint, LUT va catalog; chay validator, khong bo qua gate. |
| PUBLIC_BIND_FORBIDDEN | Host phai 127.0.0.1; xoa CUBE_NANO_RUNTIME_PROFILE neu dang override Compose. |
| HOST_FORBIDDEN/PEER_FORBIDDEN | Dung loopback, khong dung LAN IP. |
| ORIGIN_FORBIDDEN | Them Origin dung profile cho POST, vi du http://127.0.0.1:8000 hoac :4173. |
| NO_CONTACT/HELD_NO_CONTACT | Mo CONTACT_OPEN hoac dung next_contact co expires_at hop le. |
| QUEUE_FULL/HTTP 429 | Doi job terminal va poll; khong tao idempotency key moi lien tuc. |
| PRODUCT_NOT_VERIFIED | Chi download sau ground state PUBLISHED va checksum SHA256_MATCH. |
| Web khong noi API | Kiem tra /healthz, VITE_API_BASE_URL va CORS/port. |
| Playwright bi chiem port | Tat server thu cong; Playwright dung reuseExistingServer=false. |
| docker info loi | Start Docker Desktop/Linux engine. |
| DOMAIN_UNVERIFIED | Trang thai dung cua model demo; khong doc la scientific validation. |

## 11. Kiem thu va release gate

```
python -m pytest -q
python -m compileall -q gds flight link_sim scripts
python scripts/validate_deploy_profiles.py --root .
python scripts/demo_scenario.py --root . --timeout 90
python scripts/soak_test.py --iterations 100
```

Web:

```
Set-Location gds/web
npm test -- --run
npm run build
npm run test:e2e
```

Evidence trong worktree dang phat trien:

```
Set-Location D:\AI20K\cube_nano
python scripts/generate_release_manifest.py --allow-dirty
```

Official release phai worktree sach va chay khong co --allow-dirty:

```
python scripts/generate_release_manifest.py
```

Khong promote Jetson va khong suy ra GPU SLO tu benchmark CPU.

## 12. Chung cu da xac minh

Tai thoi diem cap nhat:

| Lenh | Ket qua |
|---|---|
| python scripts/validate_deploy_profiles.py --root . | Host/Compose pass; Jetson ready=false |
| python -m flight.satellite_simulator --health-once | Satellite/worker READY, instance 0000000000000001 |
| python -m flight.satellite_simulator --roi-smoke | Job SUCCEEDED, 526 frame trong run hien tai, SEND_COMPLETED |
| python scripts/demo_scenario.py --root . --timeout 90 | Product PUBLISHED, SHA256_MATCH, 518656 bytes trong run hien tai |
| docker compose -f deploy/docker-compose.yml config --quiet | Compose config pass |
| docker info | Chua xac minh container runtime vi Docker Linux daemon khong chay |

So frame, boot ID, product ID, bundle SHA va RequestKey thay doi theo run. Xem them
[phase6_completion_report.md](phase6_completion_report.md),
[phase6_conformance_checklist.md](phase6_conformance_checklist.md) va
[quick runbook](../deploy/local_sil_runbook.md).
