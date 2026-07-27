# Phase 4A Completion Report

**Ngay**: 2026-07-19  
**Giai doan**: Phase 4A - GDS ledger, delivery, API va SQLite core  
**Trang thai**: COMPLETE, 14/14 task (100%)

---

## 1. Executive Summary

Phase 4A da hoan tat tu schema/storage foundation den command delivery runtime.
GDS hien co mot write path duy nhat, idempotency durable, delivery at-least-once,
contact-aware outbox, target migration fence va ingest observability:

```text
HTTP body + Idempotency-Key
          |
          v
JCS semantic digest -> durable lookup truoc validation thay doi
          |
          v
SQLiteWriter / BEGIN IMMEDIATE
          |
          +-- RequestKey + command + outbox + admission audit
          |
          +-- claim lease -> persist raw attempt -> read-fence -> send
          |                                  |
          |                                  +-- ACK / retry / terminal state
          v
COMMIT -> 202 Accepted
```

Crash sau commit truoc HTTP response chi tao retry cung row. Crash truoc send
giu raw attempt `PERSISTED_NOT_SENT`; retry cap packet sequence moi. SQLite
khong hua exactly-once network send, nhung command business effect duoc bao ve
boi RequestKey va onboard durable idempotency contract.

## 2. Storage va Migration

`gds/migrations/001_initial.sql` tao entity schema cho command, outbox, attempt,
telemetry, rollup, event, link frame, product, run/replay, tombstone va audit.
`gds/migrations/002_phase4a_runtime.sql` bo sung:

- ACK deadline, delivery reason, attempt ACK time va sequence epoch.
- Contact state/change timestamp tren spacecraft instance.
- Durable binding snapshot trong `gds_metadata`.
- APID-scoped TC sequence allocator.
- Monotonic event sequence.
- Per-volume storage reservations.

`gds/schema.py` fail closed khi schema moi hon binary, migration checksum sai,
database unversioned co application table, foreign key sai hoac `quick_check`
that bai. Khong co downgrade path.

SQLite runtime profile:

| Setting | Gia tri |
|---------|---------|
| Journal | WAL |
| Synchronous | FULL |
| Foreign keys | ON |
| Busy timeout | 5000 ms |
| WAL autocheckpoint | 1000 pages |
| WAL warning/throttle | 128/256 MiB |
| Writer queue/high reserve | 4096/256 |

## 3. Delivery Core

`gds/outbox.py` va `gds/sequence.py` implement state machine sau:

| Su kien | Chuyen trang thai |
|---------|-------------------|
| Admit co contact | `OUTBOX_PENDING` |
| Admit next contact khi dong link | `HELD_NO_CONTACT` |
| Claim | `DISPATCHING` voi lease 10s |
| Persist attempt | raw TC + SHA-256 + APID sequence, `PERSISTED_NOT_SENT` |
| Send thanh cong | `SENT`, ACK deadline 5s |
| ACK thanh cong | `ACKED` |
| ACK timeout | pending voi backoff 500ms, exponential, cap 30s |
| TTL het han | `EXPIRED` |
| Immediate mat contact/NACK/max attempt | `DELIVERY_FAILED` |

Attempt khong rewrite encoded TC. Moi retry cap sequence Space Packet moi theo
`(spacecraft_instance_id, APID)`, doc lap voi RequestKey, frame sequence va
F Prime `cmdSeq`. Sequence rollover tang `sequence_epoch` va phat reset marker.

`send_with_fence` giu binding read fence qua send syscall. Migration write fence
doi read fence ket thuc, terminal hoa old-target rows voi
`TARGET_INSTANCE_RETIRED`, invalidates lease va danh dau persisted attempt
`NOT_SENT:REBIND`. Semantic body/mission digest cu khong bi rewrite.

## 4. Contact va API Contract

`immediate` khong co contact tra `409 NO_CONTACT`; contact loss sau admission
terminal hoa `CONTACT_LOST`. `next_contact` duoc persist, absolute expiry van
chay, nhung ACK timer va retry backoff pause trong blackout; late ACK duoc audit
`LATE_RECEIPT` va khong overwrite terminal state.

`gds/api.py` la adapter framework-neutral:

| Tinh huong | HTTP |
|------------|------|
| Admission commit | `202` |
| Semantic idempotency conflict/no contact | `409` |
| Retired target/key | `410` |
| Body/header validation | `422` |
| Outbox capacity | `429` + `Retry-After` |
| Writer backpressure | `503` + `Retry-After` |
| Storage hard watermark | `507 STORAGE_FULL` |
| Command khong ton tai | `404` |

API khong tu cap RequestKey. `GET` status doc row authoritative tu ledger.
Admission ghi `COMMAND_ADMITTED` trong cung transaction voi command/outbox.

## 5. Event, Telemetry, Audit va Raw Bytes

- `gds/events.py`: event ID BLOB(8) monotonic, keyset cursor, source/target
  instance, boot, RequestKey va dictionary version.
- `gds/telemetry.py`: dedupe theo source instance + simulation run + direction
  + link frame + copy + ordinal; duplicate byte-identical ignored; conflict
  audit/reject; rollup bucket 1 phut co count/min/max/mean/last.
- `gds/audit.py`: append/list durable, dung cho admission, migration, late ACK,
  telemetry conflict va cau hinh/system action.
- `gds/raw_segments.py`: record version/length/CRC, append fsync truoc DB
  reference, startup scan truncate torn tail, DB reference prune truoc file.
- `gds/storage.py`: per-volume hard watermark va reservation durable; reject
  admission voi `507 STORAGE_FULL` khi headroom khong con.

## 6. Verification

Focused suite:

```text
python -m pytest tests/test_phase4a_gds.py tests/test_phase4a_runtime.py -q
19 passed

python -m compileall -q gds protocol
pass
```

`tests/test_phase4a_gds.py` bao phu migration, U64, writer reserve, WAL reader,
RequestKey, JCS, concurrency, rollback, capacity va retired idempotency.
`tests/test_phase4a_runtime.py` bao phu lease crash/retry, contact pause,
sequence rollover, API mapping/status, migration fence, event cursor,
telemetry conflict/rollup, raw torn-tail recovery va storage reservation.

Mot lan chay full suite co 190 test pass, 19 subtest pass va mot failure flaky
cua Phase 2 worker (startup/callback process-spawn timeout); chay rieng test do
sau do pass. Day la flaky test/moi truong, khong nam trong P4A path. Can chay
lai full suite tren CI voi process isolation truoc release.

## 7. Residual Scope

Phase 4A da dong. Cac rui ro con lai thuoc Phase 4B/5/6:

- TM decoder va CCSDS transport envelope integration.
- Catalog authority, file reassembly, product publish va retention workflow.
- REST state/catalog/product surface, WebSocket replay/resync.
- End-to-end mission round trip, soak, security va release hardening.

---

**Nguon**: simulation plan, task tracker, implementation va test suite trong workspace  
**Tac gia**: Codex
