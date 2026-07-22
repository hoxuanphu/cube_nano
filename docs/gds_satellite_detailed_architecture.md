# So do chi tiet GDS va Satellite CCSDS SIL

Tai lieu nay mo ta profile `host_local_sil` dang duoc xac minh trong
repository. Mermaid duoc dung de co the render truc tiep trong GitHub, VS Code
hoac cac cong cu documentation co ho tro Mermaid.

## 1. Kien truc logic

```mermaid
flowchart LR
  subgraph OP["Operator / Web"]
    UI["React/Vite Mission Control<br/>gds/web"]
  end

  subgraph GROUND["GDS / Ground"]
    HTTP["gds.http_app<br/>FastAPI + LocalSilMission"]
    API["GDSApi<br/>contract-neutral admission/status"]
    LEDGER["AtomicCommandLedger<br/>JCS digest + idempotency<br/>RequestKey + atomic admission"]
    OUTBOX["OutboxService<br/>lease + contact + retry + ACK<br/>binding generation fence"]
    DB[("SQLiteWriter + gds.sqlite3<br/>WAL + FULL + migrations")]
    INGEST["TmIngestService<br/>Validated envelope + TMDecoder"]
    EVENTS["EventStore + RealtimeHub<br/>cursor / replay / resync"]
    REASM["FilePacketReassembler<br/>START / DATA / END<br/>gap + epoch tracking"]
    STORE["ProductStore<br/>safe extract + checksum/SHA/manifest<br/>atomic publish"]
    CATALOG["CatalogReplicaStore + PreviewService<br/>scene snapshot + quicklook tiles"]

    HTTP --> API
    API --> LEDGER
    API --> OUTBOX
    LEDGER --> DB
    OUTBOX --> DB
    INGEST --> DB
    INGEST --> EVENTS
    INGEST --> REASM
    REASM --> STORE
    STORE --> DB
    CATALOG --> DB
    HTTP --> CATALOG
    HTTP --> EVENTS
    STORE --> HTTP
    EVENTS --> HTTP
  end

  subgraph LINK["Link / CCSDS transport"]
    MISSION["MissionLink<br/>MissionComScheduler + MissionUdpAdapter<br/>ACK / CONTROL / FILE queues<br/>one frame in flight"]
    SIM["LinkSimulator<br/>session + sideband envelope + transport<br/>loss / duplicate / corruption / blackout / bandwidth"]
    CLOCK["VirtualClock + ReplayManager<br/>deterministic time and fault replay"]
    MISSION --> SIM
    CLOCK -.-> SIM
  end

  subgraph FLIGHT["Satellite / Flight reference boundary"]
    SAT["SatelliteSimulator<br/>transport endpoint"]
    ROUTER["StockApidRouter<br/>SCID / VC / APID routing"]
    PAYLOAD["CloudPayload<br/>command validation + dispatch<br/>journal + worker + downlink"]
    SJOURNAL[("SatelliteJournal<br/>durable command / job / product / transfer state")]
    WCLIENT["WorkerProcessClient<br/>bounded queue + heartbeat<br/>deadline / cancel / restart watchdog"]
    WORKER["sat_ai.worker_process<br/>isolated process"]
    INFER["Validated memmap scene + ROI<br/>model runtime + ThresholdLUT<br/>deterministic product build"]
    DOWNLINK["FileDownlinkCoordinator<br/>FilePacket framing + file epoch<br/>single active transfer + abort fence"]

    SAT --> ROUTER --> PAYLOAD
    PAYLOAD --> SJOURNAL
    PAYLOAD --> WCLIENT
    WCLIENT --> WORKER --> INFER
    INFER -->|WorkerResult| WCLIENT
    WCLIENT --> PAYLOAD
    PAYLOAD --> DOWNLINK
    DOWNLINK -->|TM FilePacket frames| SIM
  end

  subgraph CONTRACTS["Mission contracts"]
    PROFILE["mission_profile.yaml<br/>SCID 68 | APID 0/1/2/3<br/>TM frame 1024 bytes | VC 0"]
    SCHEMAS["protocol/schemas.py + YAML schemas<br/>Command / telemetry / identity"]
    CODECS["protocol/ccsds.py + messages.py + file_packet.py<br/>SpacePacket / TC-BD / TM / FilePacket<br/>CRC and FECF"]
    PROFILE --> CODECS
    SCHEMAS --> CODECS
  end

  UI -->|REST + WebSocket| HTTP
  HTTP -->|TC bytes| MISSION
  SIM -->|validated TC envelope| SAT
  SIM -->|validated TM envelope| INGEST
  SAT -->|TM ACK and event frames| SIM
  CODECS -.-> HTTP
  CODECS -.-> ROUTER
  CODECS -.-> INGEST

  READY["SatelliteDeployment readiness gate<br/>profile + F Prime dictionary<br/>model manifest/checkpoint/LUT<br/>benchmark + SLO + scene catalog + worker heartbeat"]
  READY -.-> SAT
  READY -.-> WCLIENT
  READY -.-> WORKER

  classDef operator fill:#e8f1ff,stroke:#3568a8,color:#16325c
  classDef ground fill:#edf8ee,stroke:#3b7f4b,color:#183b20
  classDef link fill:#fff4df,stroke:#ad7420,color:#4a310b
  classDef flight fill:#f8eafa,stroke:#7b4a88,color:#3d2145
  classDef contract fill:#f1f3f5,stroke:#68727d,color:#27313a
  classDef readiness fill:#ffe7e7,stroke:#a43d3d,color:#571c1c

  class UI operator
  class HTTP,API,LEDGER,OUTBOX,DB,INGEST,EVENTS,REASM,STORE,CATALOG ground
  class MISSION,SIM,CLOCK link
  class SAT,ROUTER,PAYLOAD,SJOURNAL,WCLIENT,WORKER,INFER,DOWNLINK flight
  class PROFILE,SCHEMAS,CODECS contract
  class READY readiness
```

### Boundary va quyen so huu

| Boundary | So huu chinh | Dieu khong duoc lam |
|---|---|---|
| Web | Hien thi catalog, quicklook, ROI, command, product va realtime state | Khong doc TIFF, khong import `sat_ai`, khong tu cap `RequestKey` |
| GDS | Admission, idempotency, outbox, TM ingest, reassembly, verified ground product | Khong goi inference truc tiep |
| Link | Session, sideband envelope, virtual time, fault model, replay va completion gate | Khong an fault vao satellite runtime |
| Flight | Decode TC, route APID, validate mission command, journal, worker admission va TM/FilePacket | Khong bo qua identity, digest hoac readiness gate |
| AI worker | Memmap scene, ROI inference, threshold LUT va staged product | Khong tu sua durable journal |
| Protocol | Wire bytes, schema, fixed-width identity, CRC/FECF va golden vectors | Khong thay doi contract theo UI |

## 2. Sequence ROI den san pham ground

```mermaid
sequenceDiagram
  autonumber
  actor OP as Operator
  participant UI as Web UI
  participant HTTP as FastAPI routes
  participant MISSION as LocalSilMission
  participant API as GDSApi
  participant LEDGER as AtomicCommandLedger
  participant OUTBOX as OutboxService
  participant WIRE as CCSDS codecs
  participant LINK as MissionLink + LinkSimulator
  participant SAT as SatelliteSimulator
  participant PAYLOAD as CloudPayload
  participant SJ as SatelliteJournal
  participant WC as WorkerProcessClient
  participant WORKER as Isolated AI worker
  participant TM as GDS TM ingest
  participant REASM as FilePacketReassembler
  participant STORE as ProductStore

  OP->>UI: Chon scene va ve ROI
  UI->>HTTP: POST /api/commands + Idempotency-Key
  HTTP->>MISSION: submit(body, idempotency_key)
  MISSION->>API: post_commands(body, headers)
  API->>LEDGER: Validate opcode, target, payload, expiry
  LEDGER->>LEDGER: Tinh JCS digest va cap RequestKey
  LEDGER->>LEDGER: BEGIN IMMEDIATE
  LEDGER->>LEDGER: Ghi command + outbox + audit
  LEDGER-->>API: 202 ADMITTED + OUTBOX_PENDING
  API-->>MISSION: Accepted command
  MISSION-->>HTTP: Trace command lifecycle
  HTTP-->>UI: 202 + request_key + mission_digest

  MISSION->>OUTBOX: claim_next()
  OUTBOX->>OUTBOX: Kiem tra contact, lease, generation, session
  OUTBOX-->>MISSION: Lease + bound link
  MISSION->>WIRE: encode Command -> SpacePacket APID 0
  WIRE-->>MISSION: TC Type-BD + CRC/FECF
  MISSION->>OUTBOX: persist_attempt() then mark_sent()
  MISSION->>LINK: send_uplink(TC frame)
  LINK->>LINK: Apply session, clock, fault profile, bandwidth
  LINK->>SAT: Validated ingress transport frame
  SAT->>PAYLOAD: Decode TC and route by SCID/VC/APID
  PAYLOAD->>PAYLOAD: Check target, digest, catalog, config, ROI
  PAYLOAD->>SJ: COMMAND_ACCEPTED + JOB_QUEUED
  PAYLOAD->>WC: Submit immutable job snapshot
  SAT-->>LINK: TM event/ACK packet
  LINK-->>TM: Validated egress transport envelope
  TM->>TM: Decode APID 2, validate session/boot/CRC
  TM->>OUTBOX: ingest_ack(request_key)
  OUTBOX-->>MISSION: ACKED or delivery failure

  WC->>WORKER: WorkerRequest via serialized process boundary
  WORKER->>WORKER: Open memmap TIFF + sidecar
  WORKER->>WORKER: Validate InputSpec, domain, deadline, cancel
  WORKER->>WORKER: ROI inference + ThresholdLUT
  WORKER->>WORKER: Build staged product + bundle/artifact hashes
  WORKER-->>WC: WorkerResult SUCCEEDED
  WC->>PAYLOAD: Callback after result ownership check
  PAYLOAD->>SJ: JOB SUCCEEDED + PRODUCT READY

  MISSION->>WIRE: Build PRODUCT_REQUEST_DOWNLINK command
  MISSION->>LINK: Send downlink request as TC APID 0
  LINK->>SAT: Dispatch downlink command
  SAT->>PAYLOAD: Route to CloudPayload
  PAYLOAD->>SJ: Allocate transfer and file epoch
  PAYLOAD->>PAYLOAD: Start FileDownlinkCoordinator
  PAYLOAD-->>LINK: TM FilePacket START/DATA/END frames
  LINK-->>TM: Deliver egress frames through fault model
  TM->>REASM: Decode and persist START/DATA/END
  REASM->>REASM: Track ranges, duplicates, order, file epoch
  REASM->>STORE: Verify file checksum, bundle SHA, manifest, artifacts
  STORE->>STORE: Atomic publish only after all checks pass
  STORE-->>HTTP: Product state PUBLISHED
  HTTP-->>UI: Product metadata, progress, verified download/tile
  UI-->>OP: PUBLISHED + SHA256_MATCH
```

## 3. Lifecycle va recovery

### 3.1 Command va outbox

```mermaid
stateDiagram-v2
  [*] --> ADMITTED
  ADMITTED --> OUTBOX_PENDING: atomic commit complete
  OUTBOX_PENDING --> HELD_NO_CONTACT: contact closed
  HELD_NO_CONTACT --> OUTBOX_PENDING: contact opens
  OUTBOX_PENDING --> DISPATCHING: lease claimed
  DISPATCHING --> SENT: raw attempt persisted
  SENT --> ACKED: business ACK accepted
  SENT --> OUTBOX_PENDING: ACK timeout / retryable fault
  SENT --> DELIVERY_FAILED: TTL or max attempts
  ACKED --> EXECUTED: target handled command
  ACKED --> FAILED: target rejected command
  ADMITTED --> REJECTED: validation / capacity / storage failure
```

`ACKED` trong so do la business command acknowledgement. No khong phai
CLCW, COP-1 hay link-layer delivery acknowledgement.

### 3.2 Job va product tren satellite

```mermaid
stateDiagram-v2
  [*] --> QUEUED
  QUEUED --> RUNNING: WorkerRequest dispatched
  QUEUED --> CANCELED: cancel while pending
  RUNNING --> SUCCEEDED: WorkerResult accepted
  RUNNING --> CANCEL_REQUESTED: operator cancel / deadline
  CANCEL_REQUESTED --> CANCELED: worker or watchdog terminalizes
  RUNNING --> TIMEOUT: deadline exceeded
  RUNNING --> FAILED: worker or protocol failure
  RUNNING --> REJECTED: insufficient valid data
  SUCCEEDED --> PRODUCT_READY: product identity and summary verified
  PRODUCT_READY --> TRANSFER_QUEUED: PRODUCT_REQUEST_DOWNLINK
```

Worker chi tao product trong staging. `CloudPayload` giu ownership callback,
kiem tra `ProductRef`, sau do moi ghi terminal state vao `SatelliteJournal`.

### 3.3 Transfer va ground publish

```mermaid
stateDiagram-v2
  [*] --> QUEUED
  QUEUED --> SENDING: FileDownlinkCoordinator starts
  SENDING --> SEND_COMPLETED: all satellite frames consumed
  SENDING --> CANCEL_REQUESTED: cancel request
  CANCEL_REQUESTED --> CANCEL_DRAINING: stop new frame leases
  CANCEL_DRAINING --> COOLDOWN: abort fence closed
  COOLDOWN --> CANCELED: cooldown complete
  SEND_COMPLETED --> RECEIVING: GDS receives START
  RECEIVING --> RECEIVING: DATA, duplicate, out-of-order tracking
  RECEIVING --> VERIFIED: END + ranges + checksum + SHA + manifest pass
  RECEIVING --> FAILED: gap, bad checksum, epoch conflict or storage limit
  VERIFIED --> PUBLISHED: atomic rename and durable DB update
```

## 4. Readiness gate va profile

```mermaid
flowchart TB
  P["MissionProfile + F Prime dictionary"] --> M["Model manifest + checkpoint SHA"]
  M --> I["InputSpec RGB / uint16 / 256 / NCHW"]
  I --> L["Threshold LUT + protocol golden vectors"]
  L --> B["Benchmark target/runtime + SHA"]
  B --> S["SLO + resource guards + topology"]
  S --> C["Scene catalog + journal reconcile"]
  C --> W["Worker process + heartbeat"]
  W --> READY["READY"]
  P -.-> FAIL["Any mismatch => fail closed"]
  M -.-> FAIL
  I -.-> FAIL
  L -.-> FAIL
  B -.-> FAIL
  S -.-> FAIL
  C -.-> FAIL
  W -.-> FAIL
```

| Profile | Trang thai | So do ap dung |
|---|---|---|
| `host_local_sil` | Duong chay reference da verify | `FastAPI -> LocalSilMission -> MissionLink in-process -> SatelliteSimulator -> GDS ingest` |
| `compose_sil` | Topology va UDP bridge da validate; chua claim E2E HTTP nhieu container | `gds -> link UDP bridge -> satellite` tren network internal |
| `jetson-l4t-tensorrt` | Blocked | Can benchmark target va TensorRT optimization profile |

## 5. Identity va durable trace

Mot request phai giu nguyen cac khoa sau tu Web den ground product:

| Khoa | Vai tro | Noi su dung |
|---|---|---|
| `spacecraft_instance_id` | Dinh danh instance spacecraft, U64 hex 16 ky tu | GDS binding, router, TM decoder, ProductRef |
| `RequestKey` | `{ground_instance_id, request_id}` | Idempotency, command, job, event va origin product |
| `SceneRef` | `{catalog_epoch, scene_id, scene_revision}` | Catalog snapshot va science input |
| `ProductRef` | `{spacecraft_instance_id, origin_boot_id, product_id}` | Product staging, downlink, reassembly, publish |
| `link_session_id` + `link_generation` | Dinh danh binding hien hanh | Outbox fence, envelope validation, TM ingest |
| `file_epoch_id` | Dinh danh transfer epoch | FilePacket reassembly, duplicate/out-of-order protection |

Trace mong doi:

```text
Idempotency-Key
    -> RequestKey + mission_digest
    -> command/outbox lease + TC attempt
    -> satellite journal command/job
    -> WorkerRequest/WorkerResult
    -> ProductRef + transfer_id + file_epoch_id
    -> TM/FilePacket reassembly
    -> verified ground product PUBLISHED
```

## 6. Traceability den source code

| Khoi trong so do | File chinh |
|---|---|
| Web operator | `gds/web/src/App.tsx`, `gds/web/src/api/client.ts`, `gds/web/src/api/realtime.ts` |
| HTTP adapter va orchestration local SIL | `gds/http_app.py`, `gds/local_sil.py` |
| Admission va durable outbox | `gds/api.py`, `gds/ledger.py`, `gds/outbox.py`, `gds/writer.py` |
| TM ingest va product publish | `gds/tm.py`, `gds/ingest.py`, `gds/file_reassembly.py`, `gds/product_store.py` |
| Link va deterministic faults | `link_sim/mission_link.py`, `link_sim/link_simulator.py`, `link_sim/transport.py`, `link_sim/replay_manager.py` |
| Flight command/TM boundary | `flight/satellite_simulator.py`, `flight/stock_router.py`, `flight/cloud_payload.py` |
| Worker va inference boundary | `flight/worker_client.py`, `sat_ai/worker_contract.py`, `sat_ai/worker_process.py`, `sat_ai/inference.py` |
| Wire contract | `protocol/profile.py`, `protocol/ccsds.py`, `protocol/messages.py`, `protocol/file_packet.py`, `protocol/schemas.py` |
| Runtime gate | `flight/deployment.py`, `protocol/mission_profile.yaml`, `protocol/runtime_profile.yaml`, `sat_ai/deployment_profile.yaml` |

