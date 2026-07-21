# GDS Satellite CCSDS Code Review Findings

- Review date: 2026-07-20
- Scope: simulation plan, task tracker, CCSDS protocol, flight software, link simulator, satellite AI, GDS, frontend, legacy inference paths, and tests.
- Conclusion: the nominal round-trip path works, but the implementation is not ready for final sign-off because of the P1/P2 findings below.

## Findings

### 1. P1 - The deployed web path bypasses LinkSimulator

- Evidence: [gds/http_app.py:119](D:/AI20K/cube_nano/gds/http_app.py:119) to [gds/http_app.py:142](D:/AI20K/cube_nano/gds/http_app.py:142) calls `self.satellite.receive_tc_frame(frame)` directly.
- Evidence: [gds/http_app.py:197](D:/AI20K/cube_nano/gds/http_app.py:197) to [gds/http_app.py:229](D:/AI20K/cube_nano/gds/http_app.py:229) drains telemetry directly into GDS.
- Evidence: [deploy/docker-compose.yml:17](D:/AI20K/cube_nano/deploy/docker-compose.yml:17) to [deploy/docker-compose.yml:22](D:/AI20K/cube_nano/deploy/docker-compose.yml:22) starts a link service that only prints a status message.
- Impact: the actual HTTP and compose workflow does not exercise loss, duplication, corruption, latency, bandwidth, link IDs, replay, or session behavior.
- Recommendation: connect the web and deployment paths through `MissionUdpAdapter` and `LinkSimulator`, with separate control, sideband, and consume/ACK flows.

### 2. P1 - TC SCID and VCID are not enforced before command execution

- Evidence: [flight/cloud_payload.py:124](D:/AI20K/cube_nano/flight/cloud_payload.py:124) to [flight/cloud_payload.py:126](D:/AI20K/cube_nano/flight/cloud_payload.py:126) decodes `TcTypeBdFrame` but dispatches only the packet bytes.
- Evidence: [protocol/ccsds.py:247](D:/AI20K/cube_nano/protocol/ccsds.py:247) to [protocol/ccsds.py:269](D:/AI20K/cube_nano/protocol/ccsds.py:269) exposes SCID/VCID without profile enforcement.
- Evidence: [flight/stock_router.py:23](D:/AI20K/cube_nano/flight/stock_router.py:23) to [flight/stock_router.py:34](D:/AI20K/cube_nano/flight/stock_router.py:34) routes by APID only.
- Impact: a CRC-valid frame for the wrong spacecraft or virtual channel can reach command routing.
- Recommendation: compare frame SCID/VCID and packet flags with the active `MissionProfile` before routing, and record rejected frames in the audit trail.

### 3. P1 - LinkSimulator session, direction, epoch, and duplicate semantics are incomplete

- Evidence: [link_sim/link_simulator.py:196](D:/AI20K/cube_nano/link_sim/link_simulator.py:196) to [link_sim/link_simulator.py:205](D:/AI20K/cube_nano/link_sim/link_simulator.py:205) hard-codes ingress direction.
- Evidence: [link_sim/link_simulator.py:237](D:/AI20K/cube_nano/link_sim/link_simulator.py:237) to [link_sim/link_simulator.py:248](D:/AI20K/cube_nano/link_sim/link_simulator.py:248) reuses the same decision for a duplicate instead of using `copy_index=1`.
- Evidence: [link_sim/link_simulator.py:272](D:/AI20K/cube_nano/link_sim/link_simulator.py:272) to [link_sim/link_simulator.py:284](D:/AI20K/cube_nano/link_sim/link_simulator.py:284) copies an ingress boot ID into egress and hard-codes `file_epoch=0`.
- Evidence: [link_sim/transport.py:218](D:/AI20K/cube_nano/link_sim/transport.py:218) to [link_sim/transport.py:238](D:/AI20K/cube_nano/link_sim/transport.py:238) does not validate the UDP sender peer or the complete egress envelope.
- Impact: sideband control, file downlink, session isolation, and deterministic per-copy fault behavior do not match the plan.
- Recommendation: wire `SessionManager` and `FileEpochManager`, implement open/close/control handling, assign the APID 3 file epoch, and validate peer plus envelope on both directions.

### 4. P1 - FilePacket START can cause a resource exhaustion attack

- Evidence: [gds/file_reassembly.py:151](D:/AI20K/cube_nano/gds/file_reassembly.py:151) to [gds/file_reassembly.py:165](D:/AI20K/cube_nano/gds/file_reassembly.py:165) configures a bundle limit.
- Evidence: [gds/file_reassembly.py:114](D:/AI20K/cube_nano/gds/file_reassembly.py:114) to [gds/file_reassembly.py:115](D:/AI20K/cube_nano/gds/file_reassembly.py:115) performs only a U32 range check.
- Evidence: [gds/file_reassembly.py:247](D:/AI20K/cube_nano/gds/file_reassembly.py:247) to [gds/file_reassembly.py:254](D:/AI20K/cube_nano/gds/file_reassembly.py:254) truncates the file before enforcing the configured maximum.
- Impact: a crafted START can request near-4 GiB sparse allocation and later trigger excessive reads.
- Recommendation: reject the declared size before opening or truncating the file, reserve durable headroom, stream writes, and enforce limits before allocation.

### 5. P1 - Crash reconciliation can publish an unverified product

- Evidence: [gds/product_store.py:448](D:/AI20K/cube_nano/gds/product_store.py:448) to [gds/product_store.py:476](D:/AI20K/cube_nano/gds/product_store.py:476) scans recovered products and marks them `PUBLISHED` after manifest and SHA parsing without calling `verify_bundle()` or verifying all artifacts.
- Impact: a corrupt or incomplete artifact can be repaired into a published state after restart.
- Recommendation: verify the manifest, product checksum, and every required artifact before database repair; record reconciliation failures instead of publishing them.

### 6. P1 - Worker callback failures can leave durable jobs stuck in RUNNING

- Evidence: [flight/worker_client.py:247](D:/AI20K/cube_nano/flight/worker_client.py:247) to [flight/worker_client.py:264](D:/AI20K/cube_nano/flight/worker_client.py:264) clears `_active` before the completion callback finishes.
- Evidence: [flight/worker_client.py:194](D:/AI20K/cube_nano/flight/worker_client.py:194) to [flight/worker_client.py:205](D:/AI20K/cube_nano/flight/worker_client.py:205) handles worker loss, while [flight/cloud_payload.py:463](D:/AI20K/cube_nano/flight/cloud_payload.py:463) to [flight/cloud_payload.py:472](D:/AI20K/cube_nano/flight/cloud_payload.py:472) can raise while processing malformed results or identity mismatches.
- Evidence: [flight/worker_client.py:156](D:/AI20K/cube_nano/flight/worker_client.py:156) to [flight/worker_client.py:175](D:/AI20K/cube_nano/flight/worker_client.py:175) can raise `queue.Full` for cancellation; [flight/cloud_payload.py:300](D:/AI20K/cube_nano/flight/cloud_payload.py:300) to [flight/cloud_payload.py:319](D:/AI20K/cube_nano/flight/cloud_payload.py:319) does not catch that failure.
- Impact: a durable job may remain `RUNNING`, and cancellation can fail with an uncaught exception when the control queue is full.
- Recommendation: convert callback errors into a terminal job result, reserve or catch control-queue capacity, and test malformed results, identity mismatch, and full control queues.

### 7. P1/P2 - Product timestamps and preview retention use the wrong values

- Evidence: [gds/product_store.py:336](D:/AI20K/cube_nano/gds/product_store.py:336) to [gds/product_store.py:340](D:/AI20K/cube_nano/gds/product_store.py:340) defaults retention to `now + 30 days`.
- Evidence: [gds/product_store.py:359](D:/AI20K/cube_nano/gds/product_store.py:359) to [gds/product_store.py:385](D:/AI20K/cube_nano/gds/product_store.py:385) stores the retention expiry as verified and published timestamps.
- Evidence: [gds/preview.py:100](D:/AI20K/cube_nano/gds/preview.py:100) to [gds/preview.py:104](D:/AI20K/cube_nano/gds/preview.py:104) passes `received_at` as the retention expiry.
- Impact: audit and UI timestamps can be in the future, while previews can expire immediately.
- Recommendation: keep `published_at`, `verified_at`, and the future retention deadline as separate fields and add boundary tests.

### 8. P1/P2 - Realtime replay can duplicate events and stale cursors do not receive RESYNC_REQUIRED

- Evidence: [gds/realtime.py:136](D:/AI20K/cube_nano/gds/realtime.py:136) to [gds/realtime.py:143](D:/AI20K/cube_nano/gds/realtime.py:143) pre-enqueues replay events and returns the same replay list.
- Evidence: [gds/http_app.py:520](D:/AI20K/cube_nano/gds/http_app.py:520) to [gds/http_app.py:531](D:/AI20K/cube_nano/gds/http_app.py:531) sends replay and then drains the queue; the stale-cursor exception is not translated to a `RESYNC_REQUIRED` event.
- Evidence: [gds/web/src/api/realtime.ts:105](D:/AI20K/cube_nano/gds/web/src/api/realtime.ts:105) to [gds/web/src/api/realtime.ts:113](D:/AI20K/cube_nano/gds/web/src/api/realtime.ts:113) only appends to the browser buffer and does not consume entries after normal dispatch.
- Impact: clients can receive duplicate events, stale cursors can fail without a protocol response, and healthy clients can be forced into resync after hitting the buffer limit.
- Recommendation: separate initial replay from the live queue, send an explicit resync event, and consume or clear the browser buffer after dispatch. Add WebSocket end-to-end tests.

### 9. P2 - TM session/generation binding and satellite time are not enforced

- Evidence: [gds/local_sil.py:62](D:/AI20K/cube_nano/gds/local_sil.py:62) to [gds/local_sil.py:65](D:/AI20K/cube_nano/gds/local_sil.py:65) configures only the expected TM instance.
- Evidence: [gds/tm.py:167](D:/AI20K/cube_nano/gds/tm.py:167) to [gds/tm.py:196](D:/AI20K/cube_nano/gds/tm.py:196) supports expected session and generation but leaves them optional.
- Evidence: [flight/cloud_payload.py:524](D:/AI20K/cube_nano/flight/cloud_payload.py:524) to [flight/cloud_payload.py:533](D:/AI20K/cube_nano/flight/cloud_payload.py:533) labels `time.time_ns()` as `satellite_event_time`.
- Impact: telemetry from an old session or generation can be accepted, and Unix wall-clock time is mislabeled as TAI or simulation time.
- Recommendation: bind expected session and generation from the active mission snapshot and use an explicit simulation or TAI clock.

### 10. P2 - AI validity, mask, input contract, and domain handling fail open

- Evidence: [sat_ai/roi.py:138](D:/AI20K/cube_nano/sat_ai/roi.py:138) to [sat_ai/roi.py:143](D:/AI20K/cube_nano/sat_ai/roi.py:143) treats only the value `1` as valid.
- Evidence: [sat_ai/roi.py:243](D:/AI20K/cube_nano/sat_ai/roi.py:243) to [sat_ai/roi.py:257](D:/AI20K/cube_nano/sat_ai/roi.py:257) does not reject non-binary mask values; [sat_ai/roi.py:217](D:/AI20K/cube_nano/sat_ai/roi.py:217) to [sat_ai/roi.py:230](D:/AI20K/cube_nano/sat_ai/roi.py:230) silently maps an unknown nodata rule to all-band behavior.
- Evidence: [sat_ai/worker_process.py:104](D:/AI20K/cube_nano/sat_ai/worker_process.py:104) to [sat_ai/worker_process.py:110](D:/AI20K/cube_nano/sat_ai/worker_process.py:110) does not compare the scene sidecar `input_spec`; [flight/cloud_payload.py:233](D:/AI20K/cube_nano/flight/cloud_payload.py:233) to [flight/cloud_payload.py:262](D:/AI20K/cube_nano/flight/cloud_payload.py:262) ignores the scene domain.
- Evidence: [sat_ai/inference.py:220](D:/AI20K/cube_nano/sat_ai/inference.py:220) to [sat_ai/inference.py:224](D:/AI20K/cube_nano/sat_ai/inference.py:224) emits only `science_status=demo_non_validated`, without an explicit `DOMAIN_UNVERIFIED` state.
- Impact: malformed masks, mismatched normalization, or out-of-domain scenes can produce misleading science results.
- Recommendation: reject unknown or non-binary values, require exact input and domain compatibility, and emit `DOMAIN_UNVERIFIED` for non-validated results.

### 11. P2 - Scene package content hash does not represent final package content

- Evidence: [flight/scene_package.py:90](D:/AI20K/cube_nano/flight/scene_package.py:90) to [flight/scene_package.py:101](D:/AI20K/cube_nano/flight/scene_package.py:101) computes `package_sha` from the original sidecar hash.
- Evidence: [flight/scene_package.py:124](D:/AI20K/cube_nano/flight/scene_package.py:124) to [flight/scene_package.py:145](D:/AI20K/cube_nano/flight/scene_package.py:145) rewrites the copied mask sidecar and records a new hash while retaining the old package hash.
- Evidence: [flight/scene_package.py:105](D:/AI20K/cube_nano/flight/scene_package.py:105) to [flight/scene_package.py:116](D:/AI20K/cube_nano/flight/scene_package.py:116) returns an existing package without revalidating its content.
- Impact: the final package identity is not the hash of the final canonical content, and stale packages can be reused.
- Recommendation: compute the package hash from the final canonical manifest and content, and validate or rebuild existing package directories.

### 12. P2 - HTTP status, preview, telemetry freshness, and frontend state are inconsistent

- Evidence: [gds/http_app.py:454](D:/AI20K/cube_nano/gds/http_app.py:454) to [gds/http_app.py:460](D:/AI20K/cube_nano/gds/http_app.py:460) returns only `.body` and drops `ApiResponse` status codes.
- Evidence: [gds/http_app.py:493](D:/AI20K/cube_nano/gds/http_app.py:493) to [gds/http_app.py:500](D:/AI20K/cube_nano/gds/http_app.py:500) always returns `PREVIEW_UNAVAILABLE`; [gds/local_sil.py:53](D:/AI20K/cube_nano/gds/local_sil.py:53) to [gds/local_sil.py:59](D:/AI20K/cube_nano/gds/local_sil.py:59) does not wire `PreviewService`.
- Evidence: [gds/http_app.py:318](D:/AI20K/cube_nano/gds/http_app.py:318) refreshes the last telemetry time on every snapshot, while [gds/http_app.py:378](D:/AI20K/cube_nano/gds/http_app.py:378) returns an empty telemetry list.
- Evidence: [gds/web/src/state/store.ts:172](D:/AI20K/cube_nano/gds/web/src/state/store.ts:172) to [gds/web/src/state/store.ts:180](D:/AI20K/cube_nano/gds/web/src/state/store.ts:180) updates telemetry time for every event.
- Impact: REST errors may appear as HTTP 200, the default UI cannot display actual quicklook or tile data, and stale telemetry is reported as fresh.
- Recommendation: preserve response status codes, wire preview/downlink behavior or gate the feature explicitly, and update telemetry age only for telemetry events.

### 13. P2 - Legacy inference entry points diverge from the current AI contract

- Evidence: [src/inference.py:51](D:/AI20K/cube_nano/src/inference.py:51) and [src/inference.py:149](D:/AI20K/cube_nano/src/inference.py:149) default to four channels.
- Evidence: [src/inference_large_image_trt.py:559](D:/AI20K/cube_nano/src/inference_large_image_trt.py:559) and [src/inference_large_image_trt.py:795](D:/AI20K/cube_nano/src/inference_large_image_trt.py:795) also default to four channels and use float thresholds or coarse cloud coverage.
- Impact: if these paths are exposed as production entry points, they are incompatible with the current three-channel, basis-point LUT, scene-ROI contract in `sat_ai`.
- Recommendation: mark them development-only or route all production inference through the manifest-bound `sat_ai` pipeline.

## Verification Evidence

- `python -m pytest -q`: 214 passed, 19 subtests passed.
- Frontend Vitest: 9 passed.
- `npm run build`: passed; bundle-size warning is over 500 kB.
- `python -m compileall -q ...`: passed.
- `git diff --check`: passed; only line-ending warnings were reported.
- `python scripts/p4b_roundtrip.py --root .`: passed; analysis `SUCCEEDED`, 526 frames, ground product `PUBLISHED`.
- `python scripts/demo_scenario.py --root . --timeout 120`: passed; product `PUBLISHED`, SHA256 matched.
- `run_soak(20)`: guards passed; deployment profile validation passed.
- `ruff` and `mypy` were not available in the review environment.

## Release and DoD Gaps

- Tracker items P6-13 and P6-15 remain unchecked: [docs/gds_satellite_ccsds_task_tracker.md:818](D:/AI20K/cube_nano/docs/gds_satellite_ccsds_task_tracker.md:818) and [docs/gds_satellite_ccsds_task_tracker.md:830](D:/AI20K/cube_nano/docs/gds_satellite_ccsds_task_tracker.md:830).
- `artifacts/release/phase6-release-manifest.json` reports `source_dirty: true`.
- CUDA and Jetson targets are unavailable in `artifacts/benchmarks/phase6-batch-matrix-v1.json`.
- The baseline documents a Python F Prime-compatible reference skeleton or placeholder, not native F Prime source. This is a release or assumption gap rather than necessarily a runtime defect.
- This review created only this report and did not modify runtime modules.

## Suggested Remediation Order

1. Wire the real LinkSimulator path and enforce TC SCID/VCID validation.
2. Fix FilePacket allocation, product reconciliation, and worker job terminal-state handling.
3. Correct timestamps, preview retention, realtime replay, and stale telemetry behavior.
4. Enforce TM session/generation, AI input/domain contracts, and package content hashing.
5. Decide the supported status of legacy inference entry points and complete the remaining tracker DoD items.

## Follow-up Review Findings (2026-07-21)

This follow-up review was performed against the implementation currently referenced by
`docs/gds_satellite_ccsds_simulation_plan.md`. The nominal unit and frontend tests pass,
but the findings below remain open and block final Definition-of-Done sign-off.

### F-01. P1 - The GDS HTTP host path bypasses the CCSDS/TM boundary

- Evidence: [gds/http_app.py:103](D:/AI20K/cube_nano/gds/http_app.py:103) to [gds/http_app.py:110](D:/AI20K/cube_nano/gds/http_app.py:110) activates the catalog by reading the embedded satellite payload directly.
- Evidence: [gds/http_app.py:190](D:/AI20K/cube_nano/gds/http_app.py:190) to [gds/http_app.py:206](D:/AI20K/cube_nano/gds/http_app.py:206) reads the satellite journal directly while completing a command, and [gds/http_app.py:320](D:/AI20K/cube_nano/gds/http_app.py:339) builds state from satellite internals.
- Evidence: the flight side emits an ACK in [flight/satellite_simulator.py:103](D:/AI20K/cube_nano/flight/satellite_simulator.py:103) to [flight/satellite_simulator.py:126](D:/AI20K/cube_nano/flight/satellite_simulator.py:126) and [flight/cloud_payload.py:661](D:/AI20K/cube_nano/flight/cloud_payload.py:675), but no APID 1 telemetry/progress/result producer feeds the GDS path.
- Impact: the HTTP workflow does not prove command execution, telemetry correlation, loss/replay behavior, or TM-driven state transitions. The E2E test polls `/api/state` rather than validating the wire path ([gds/web/e2e/mission-control.spec.ts:22](D:/AI20K/cube_nano/gds/web/e2e/mission-control.spec.ts:22)).
- Required fix: remove direct satellite-payload reads from the GDS boundary. Route TC bytes through the configured link adapter and derive command/catalog/state completion only from decoded TM and durable GDS state.

### F-02. P1 - Compose does not run a real three-process UDP transport

- Evidence: [deploy/docker-compose.yml:23](D:/AI20K/cube_nano/deploy/docker-compose.yml:23) to [deploy/docker-compose.yml:48](D:/AI20K/cube_nano/deploy/docker-compose.yml:48) starts separate containers, but [gds/http_app.py:58](D:/AI20K/cube_nano/gds/http_app.py:95) always embeds `SatelliteSimulator` and `MissionLink` and does not honor the link host/mode configuration.
- Evidence: [flight/satellite_simulator.py:140](D:/AI20K/cube_nano/flight/satellite_simulator.py:248) only prints health and sleeps; it does not bind a UDP endpoint. [link_sim/__main__.py:116](D:/AI20K/cube_nano/link_sim/__main__.py:118) reports READY without checking a live socket.
- Evidence: [scripts/validate_deploy_profiles.py:25](D:/AI20K/cube_nano/scripts/validate_deploy_profiles.py:27) validates configuration only, while [deploy/local_sil_runbook.md:37](D:/AI20K/cube_nano/deploy/local_sil_runbook.md:37) to [deploy/local_sil_runbook.md:40](D:/AI20K/cube_nano/deploy/local_sil_runbook.md:40) documents that fault profiles are not loaded by the CLI/HTTP path.
- Impact: Compose is not an integration test of the deployed topology; link faults, peer validation, sessions, and process isolation are untested.
- Required fix: add real UDP endpoints for GDS and flight, connect both through the LinkSimulator service, load the selected fault profile, and make health checks prove socket reachability and a command/TM exchange.

### F-03. P1 - Durable outbox leases can strand commands in `DISPATCHING`

- Evidence: [gds/http_app.py:143](D:/AI20K/cube_nano/gds/http_app.py:146) calls `claim_next()` and silently returns when the claimed lease is not the requested key. Concurrent executor tasks can therefore steal each other's lease.
- Evidence: [gds/http_app.py:279](D:/AI20K/cube_nano/gds/http_app.py:280) schedules only a newly admitted command; a replayed command after restart is not rescheduled and there is no startup/background outbox pump.
- Evidence: [gds/http_app.py:164](D:/AI20K/cube_nano/gds/http_app.py:167) marks a command sent and ingests a direct satellite result, while [gds/ingest.py:109](D:/AI20K/cube_nano/gds/ingest.py:127) records TM without driving the outbox state machine.
- Impact: a crash, duplicate dispatch, or executor race can leave a durable command permanently in `DISPATCHING` with no recovery path.
- Required fix: claim by request key or atomically return the claimed work item, reconcile expired leases on startup, run a retry pump, and make correlated TM ACK/result ingestion the only path that advances outbox state.

### F-04. P1 - Runtime CCSDS bytes do not match persisted sequence/profile metadata

- Evidence: [gds/http_app.py:147](D:/AI20K/cube_nano/gds/http_app.py:156) encodes a TC using `lease.attempt_count` and the default packet type, while [gds/outbox.py:679](D:/AI20K/cube_nano/gds/outbox.py:703) allocates the durable packet sequence after encoding. Wire bytes and persisted metadata can diverge.
- Evidence: the command router expects packet type 0 ([flight/cloud_payload.py:63](D:/AI20K/cube_nano/flight/cloud_payload.py:68)), while the golden generator emits packet type 1 ([protocol/generate_vectors.py:33](D:/AI20K/cube_nano/protocol/generate_vectors.py:39)).
- Evidence: TM MCFC/VCFC remain at defaults in [flight/satellite_simulator.py:114](D:/AI20K/cube_nano/flight/satellite_simulator.py:119) and [flight/file_downlink.py:190](D:/AI20K/cube_nano/flight/file_downlink.py:191); APID 3 packet sequence starts at zero for every transfer ([flight/file_downlink.py:55](D:/AI20K/cube_nano/flight/file_downlink.py:56).
- Impact: conformance vectors, replay/deduplication, rollover/gap detection, and audit records cannot be trusted across retries or transfers.
- Required fix: allocate sequence values before encoding in one durable transaction, define one packet-type/profile source of truth, persist TM master/VC counters, persist APID 3 transfer counters, and add GDS gap/rollover tracking with golden-byte tests.

### F-05. P1 - Product downlink derives a synthetic RequestKey and bypasses the GDS ledger

- Evidence: [gds/http_app.py:221](D:/AI20K/cube_nano/gds/http_app.py:244) derives a downlink key by OR-ing `0x80000000` into the originating request ID and dispatches the product request directly.
- Impact: high-bit request IDs can collide, the transfer has no independent durable admission/outbox record, and restart/retry/correlation behavior is not defined.
- Required fix: allocate a fresh durable `RequestKey` for every downlink, admit it through the same ledger/outbox path as a command, and correlate APID 3 transfer state and TM completion to that key.

### F-06. P1/P2 - Frontend U64 hexadecimal event cursors are parsed as decimal

- Evidence: [gds/web/src/api/realtime.ts:131](D:/AI20K/cube_nano/gds/web/src/api/realtime.ts:131) calls `BigInt(String(event.event_id))`, but the protocol emits fixed-width lowercase hexadecimal IDs. IDs containing `a`-`f` fail to parse (the first failure is event 10).
- Evidence: [gds/realtime.py:65](D:/AI20K/cube_nano/gds/realtime.py:68) and [gds/realtime.py:151](D:/AI20K/cube_nano/gds/realtime.py:157) mark slow clients closed, but [gds/http_app.py:553](D:/AI20K/cube_nano/gds/http_app.py:559) does not close the websocket or send a resynchronization response.
- Impact: realtime dispatch and cursors stop at hexadecimal IDs, and slow/stale sockets can remain open without a deterministic recovery signal.
- Required fix: parse and compare cursors as hex, add tests with `a`-`f` IDs and stale cursors, and close slow clients explicitly with `RESYNC_REQUIRED` or a documented reconnect code.

### F-07. P2 - ROI and file paths violate the memory and streaming bounds

- Evidence: [sat_ai/roi.py:181](D:/AI20K/cube_nano/sat_ai/roi.py:190) hashes the entire source TIFF for every job; [sat_ai/worker_process.py:113](D:/AI20K/cube_nano/sat_ai/worker_process.py:113) enables that verification by default.
- Evidence: [sat_ai/products.py:163](D:/AI20K/cube_nano/sat_ai/products.py:180) and [sat_ai/products.py:220](D:/AI20K/cube_nano/sat_ai/products.py:238) materialize full artifacts and tar archives in memory. [flight/file_downlink.py:109](D:/AI20K/cube_nano/flight/file_downlink.py:109), [gds/file_reassembly.py:428](D:/AI20K/cube_nano/gds/file_reassembly.py:444), and [gds/product_store.py:181](D:/AI20K/cube_nano/gds/product_store.py:220) repeat whole-file reads.
- Impact: a canonical roughly 723 MB scene and the configured 1 GiB reassembly ceiling can exceed the planned 256 MiB RSS limit and cause OOM or unacceptable ROI latency.
- Required fix: use ingest-time/package fingerprints, stream artifact/tar/downlink/reassembly verification through bounded buffers, and add RSS and latency tests for the canonical scene and download size limits.

### F-08. P1 - Native F Prime deliverable is still a placeholder

- Evidence: [flight/CloudPayload.fpp:1](D:/AI20K/cube_nano/flight/CloudPayload.fpp:5) explicitly describes a placeholder with only ports listed at [flight/CloudPayload.fpp:7](D:/AI20K/cube_nano/flight/CloudPayload.fpp:11).
- Evidence: [flight/README.md:3](D:/AI20K/cube_nano/flight/README.md:7) describes a Python reference because the F Prime source checkout is absent.
- Impact: the Phase 2a requirements in [docs/gds_satellite_ccsds_simulation_plan.md:810](D:/AI20K/cube_nano/docs/gds_satellite_ccsds_simulation_plan.md:810) to [docs/gds_satellite_ccsds_simulation_plan.md:817](D:/AI20K/cube_nano/docs/gds_satellite_ccsds_simulation_plan.md:817) cannot be validated for actual component ownership, routing, buffer sizes, stock F Prime components, or build constants.
- Required fix: pin the F Prime source/SDK version, provide the real C++/FPP deployment and dictionary, build the stock router/deframer/framer stack, and run the same golden vectors and 990-byte boundary tests against the native deployment. If native F Prime is intentionally out of scope, mark Phase 2a and the related DoD items as conditional instead of complete.

## Follow-up Verification Evidence

- `python -m pytest -q`: 218 passed; 19 subtests passed.
- Frontend Vitest: 9 passed.
- `npm run build`: passed with the existing large-bundle warning.
- `python -m compileall -q protocol link_sim sat_ai flight gds src`: passed.
- Docker Compose live transport and Playwright browser E2E were not run in this follow-up; they remain release gates.

## Follow-up Release Gate

Do not mark the simulation plan complete until F-01 through F-05 and F-08 are closed, F-06 and F-07 have passing regression/performance evidence, and the Compose/Playwright gates are executed against the real three-process topology.
