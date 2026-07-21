# GDS Satellite CCSDS Remediation Plan

- Date: 2026-07-20
- Scope: remediation for findings in `docs/gds_satellite_ccsds_code_review_findings.md`.
- Goal: remove all P1 blockers, close the relevant P2 correctness gaps, and complete the release DoD.
- Constraint: keep the current MVP boundary. Do not expand this work into COP-1, CFDP acknowledged mode, SDLS, OIDC/RBAC/TLS, or RF/SDR.

## Target Architecture

The deployed web path must exercise the same transport boundary as the simulation path:

```text
HTTP/UI
  -> GDS Outbox
  -> MissionUdpAdapter
  -> UDP + LinkControl
  -> LinkSimulator
  -> UDP + MissionUdpAdapter
  -> Satellite

Satellite TM/File
  -> LinkSimulator
  -> GDS TM decoder/reassembler
  -> ProductStore + RealtimeHub + UI
```

HTTP must not call `satellite.receive_tc_frame()` or `gds.ingest_tm()` directly.
Local SIL should use `InMemoryTransport`; Compose should use `UdpTransport` through the same adapter interface.

## Phase 1: P1 Blockers

### 1. Real LinkSimulator integration

Affected areas: `gds/http_app.py`, `gds/local_sil.py`, `link_sim/link_simulator.py`, `link_sim/transport.py`, `flight/mission_udp_adapter.py`, and `deploy/docker-compose.yml`.

- Add a mission link orchestration layer shared by local SIL and Compose.
- Use `MissionUdpAdapter` for TC and TM/file traffic, preserving the completion gate and single-in-flight behavior.
- Add the LinkControl channel with `OPEN_SESSION`, `SESSION_READY`, `FRAME_ACCEPTED`, `FRAME_CONSUMED`, `ABORT_FILE_EPOCH`, and `SESSION_RESET`.
- Replace LinkSimulator-local session/epoch state with `SessionManager` and `FileEpochManager`.
- Make uplink/downlink direction explicit instead of hard-coding ingress.
- Apply fault decisions independently to each copy and carry the correct `copy_index`.
- Assign the current boot ID on egress and assign `file_epoch_id` only to APID 3 FilePacket traffic.
- Validate UDP peer, magic, version, direction, session, frame length, and all egress envelope fields.
- Remove the Compose link service placeholder and add internal health/readiness checks.

Acceptance criteria:

- HTTP command -> LinkSimulator -> satellite -> ACK works in local SIL and Compose.
- Loss, duplicate, corruption, latency, bandwidth, blackout, session reset, and file epoch behavior are observable in the real path.
- `FRAME_CONSUMED` is required before the LinkSimulator considers an egress copy delivered.
- No direct web-to-satellite or web-to-GDS bypass remains.

### 2. TC SCID/VCID and packet validation

Affected areas: `flight/cloud_payload.py`, `flight/stock_router.py`, `protocol/ccsds.py`, and `protocol/profile.py`.

- Decode and CRC-check the TC frame first.
- Validate SCID against `MissionProfile.spacecraft_id`.
- Validate VCID against `MissionProfile.tc_virtual_channel`.
- Validate the expected packet type, APID, and packet flags before routing.
- Record rejected frames in the audit trail even when no `Command` exists.
- Do not route a CRC-valid frame for another spacecraft or virtual channel.

Acceptance criteria:

- Wrong SCID, wrong VCID, wrong flags, bad CRC, and wrong target instance are rejected with stable error codes.
- Rejected frames produce an audit/event record and no command/job/product side effect.

### 3. FilePacket allocation and product recovery

Affected areas: `gds/file_reassembly.py` and `gds/product_store.py`.

- Validate `file_size` against bundle, extraction, artifact-count, and filesystem headroom limits before opening or truncating a file.
- Add a durable storage reservation for each active reassembly.
- Stream DATA writes and enforce every range against the reserved logical size.
- Release reservations on cancel, timeout, checksum failure, restart reconciliation, and all terminal states.
- During product reconciliation, verify the bundle checksum, manifest schema, expected product identity, F Prime checksum, exact artifact set, and every artifact size/hash before publishing.
- Reconcile through `VERIFIED -> PUBLISHED`; never repair directly to `PUBLISHED` from manifest parsing alone.
- Record reconciliation failures as durable audit events.

Acceptance criteria:

- A crafted near-4 GiB START is rejected before file allocation.
- Storage exhaustion returns a deterministic error and leaves no unbounded `.part` file.
- A corrupt or incomplete product is never repaired into `PUBLISHED`.
- A valid product is repaired exactly once after a crash between rename and database commit.

### 4. Worker terminal-state handling

Affected areas: `flight/worker_client.py`, `flight/cloud_payload.py`, and `flight/journal.py`.

- Keep active-request ownership until the completion callback has durably handled the result.
- Convert malformed result, result identity mismatch, and callback exceptions into a terminal `FAILED` job with `WORKER_PROTOCOL_ERROR` or a more specific stable code.
- Safely discard any staging product produced by an invalid result.
- Catch `queue.Full` during cancellation.
- Use a reserved cancellation path or retry watchdog so a durable `CANCEL_REQUESTED` job cannot remain indefinite.
- Preserve terminal-state immutability and make completion/cancel races use an explicit allow-list.

Acceptance criteria:

- Worker loss, malformed result, identity mismatch, callback exception, full control queue, deadline, and cancel race all produce one terminal job state.
- No durable job remains in `RUNNING` or `CANCEL_REQUESTED` without an active bounded recovery path.

## Phase 2: P1/P2 Correctness

### 5. Product timestamps and preview retention

Affected areas: `gds/product_store.py`, `gds/preview.py`, `gds/local_sil.py`, and migrations.

- Keep `created_at`, `verified_at`, `published_at`, and `retention_until` as separate values.
- Set verification and publication timestamps from an injected clock at the actual transition.
- Compute default retention as `published_at + 30 days`.
- Never use `received_at` as a retention deadline.
- Add a forward-only migration or repair routine for rows containing the old incorrect values.
- Wire `PreviewService` into local SIL and implement the tile route, or explicitly disable preview capability in the API and UI until the full workflow is available.

Acceptance criteria:

- Fixed-clock tests prove timestamps and retention boundaries independently.
- Preview products remain available for the configured retention period and do not expire immediately.

### 6. Realtime replay and browser state

Affected areas: `gds/realtime.py`, `gds/http_app.py`, `gds/web/src/api/realtime.ts`, and `gds/web/src/state/store.ts`.

- Separate initial replay results from the live client queue.
- Register the client at a deterministic cursor boundary so events are not lost between snapshot and replay.
- Translate stale cursors into an explicit `RESYNC_REQUIRED` WebSocket message.
- Keep only a bounded recent-ID dedup window in the browser; consume normal events after dispatch.
- Ensure server and client use one event envelope shape without replaying the same event through both `events` and `event` paths.

Acceptance criteria:

- Reconnect does not duplicate events or lose events around the snapshot boundary.
- A stale cursor always causes snapshot resync rather than a silent socket failure.
- Normal high event volume does not fill the browser buffer solely because events were never consumed.

### 7. TM session/generation and clock binding

Affected areas: `gds/tm.py`, `gds/local_sil.py`, and `flight/cloud_payload.py`.

- Bind the production decoder to the active `link_session_id` and `link_generation` from the mission snapshot.
- Rebind atomically on session open/reset; reject stale session or generation packets.
- Inject an explicit simulation or TAI clock into the satellite payload.
- Keep satellite event time and GDS receive time as separate fields.
- Never label Unix wall-clock nanoseconds as TAI or simulation time.

Acceptance criteria:

- Old-session and old-generation TM are rejected.
- Identical simulation inputs produce deterministic satellite event times.
- Restart creates a new boot epoch without merging old events into the new state.

## Phase 3: AI and Package Contract

### 8. Fail-closed AI validation

Affected areas: `sat_ai/roi.py`, `sat_ai/worker_process.py`, `sat_ai/inference.py`, and `flight/cloud_payload.py`.

- Accept only binary validity-mask values `{0, 1}`; reject all other values.
- Reject unknown `nodata` rules instead of defaulting to all-band behavior.
- Require exact `input_spec_id`, channel, normalization, patch-size, and dtype compatibility.
- Validate scene domain and capability at catalog admission and again at satellite command admission.
- Reject incompatible domains before creating a job.
- Emit `science_status=DOMAIN_UNVERIFIED` for results that are not scientifically validated; never present them as validated science.

Acceptance criteria:

- Non-binary masks, unknown nodata rules, input-spec mismatch, and domain mismatch fail closed.
- Non-validated results are visible as `DOMAIN_UNVERIFIED` in TM, product metadata, API, and UI.

### 9. Canonical scene package hashing

Affected area: `flight/scene_package.py`.

- Copy source, sidecar, and validity mask first.
- Rewrite and fsync the copied sidecar before computing package identity.
- Compute the package hash from the final canonical content and manifest inputs, without self-reference.
- Reopen and validate all content before atomic publication.
- When a package directory already exists, validate its manifest, source, sidecar, mask, hashes, and stat contract; rebuild or quarantine stale content instead of returning it blindly.

Acceptance criteria:

- `package_sha256` represents the final package content.
- A stale or mutated existing package is detected and cannot be silently reused.

### 10. Legacy inference paths

Affected areas: `src/inference.py` and `src/inference_large_image_trt.py`.

- Route all production inference through the manifest-bound `sat_ai` pipeline.
- Mark legacy four-channel/float-threshold entry points as development-only, or require an explicit legacy flag outside production deployment.
- Add an architecture test ensuring deployment code does not import legacy entry points.
- If a legacy path must remain supported, migrate it to the current three-channel, basis-point LUT, scene-ROI contract first.

## Verification and Release Gate

Add focused tests for link integration, SCID/VCID rejection, allocation limits, product reconciliation, worker failures, timestamps, preview, WebSocket replay, TM binding, AI contracts, package hashing, HTTP status, and frontend telemetry freshness.

Run the existing suite plus:

```text
python -m pytest -q
frontend Vitest
npm run build
python -m compileall -q ...
python scripts/p4b_roundtrip.py --root .
python scripts/demo_scenario.py --root . --timeout 120
run_soak(20)
```

The final sign-off requires:

- Zero unresolved P1 findings.
- Full HTTP and Compose workflows through LinkSimulator.
- Product publication only after complete verification.
- Correct realtime replay/resync and telemetry freshness.
- P6-13 complete with clean reproducible build, SBOM, and `source_dirty=false`.
- P6-15 conformance matrix complete.
- CPU release evidence separated from CUDA/Jetson claims until target hardware benchmarks exist.

Recommended critical path:

```text
Link boundary
  -> TC/session/file/worker P1
  -> product/realtime/TM correctness
  -> AI/package contract
  -> regression and release DoD
```

## Execution Record (2026-07-21)

Implemented in the shared workspace:

- `MissionLink`, `MissionUdpAdapter`, `SessionManager`, `FileEpochManager`, strict sideband validation, direction-specific fault copies, and LinkControl lifecycle messages are active in the local-SIL path.
- TC frame validation now rejects CRC, SCID, VCID, packet, and target-instance violations with durable audit events.
- File reassembly has pre-allocation limits, durable reservations, streamed range writes, restart cleanup, and verified product reconciliation before `VERIFIED -> PUBLISHED`.
- Worker callback/protocol/cancel failures are terminalized; product timestamps and retention are clock-bound; preview tiles are wired through `GDSApi`.
- Realtime replay/resync, TM session/generation binding, fail-closed AI contracts, canonical scene package hashing, stale-package quarantine, and the legacy inference architecture boundary are covered by code and tests.
- Compose now builds one named CPU image for GDS, Link, and Satellite and runs a real UDP LinkSimulator bridge with readiness checks.

Verification completed:

- `python -m pytest -q`: 216 passed, 19 subtests passed.
- `npm test -- --run`: 9 tests passed; `npm run build`: passed with only the existing large-chunk warning.
- `python -m compileall -q flight gds link_sim protocol sat_ai src`: passed.
- `scripts/p4b_roundtrip.py`, `scripts/demo_scenario.py`, `scripts/soak_test.py --iterations 20`, and `scripts/validate_deploy_profiles.py`: passed on the CPU/local-SIL profile.

Release conditions still open:

- P6-13 is evidence-only because the shared worktree is dirty; an official manifest requires `source_dirty=false` from a clean intended release tree.
- CUDA/Jetson evidence is unavailable and remains fail-closed.
- The full multi-container Compose HTTP -> UDP -> LinkSimulator -> Satellite -> GDS workflow is not claimed yet; the current verified E2E is local-SIL through the same in-process mission-link boundary.
