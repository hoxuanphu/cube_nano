# GDS Satellite CCSDS Follow-up Remediation Plan

- Date: 2026-07-21
- Source: `docs/gds_satellite_ccsds_code_review_findings.md`, section `Follow-up Review Findings (2026-07-21)`
- Scope: F-01 through F-08 and the follow-up release gate
- Status: F-01 through F-07 implementation/regression work has evidence;
  F-08 and external release gates remain conditional

## Scope and Current Context

This plan treats F-01 through F-08 as the findings recorded on 2026-07-21. The older findings from 2026-07-20 remain covered by `docs/gds_satellite_ccsds_remediation_plan.md` and are not reopened here unless a regression is found.

The local-SIL path and its nominal tests are useful baselines, but they do not close the follow-up findings. The release gate still requires the real three-process HTTP -> UDP -> LinkSimulator -> Satellite -> GDS path, Compose and Playwright evidence, and a clean reproducible release build.

## Execution Status (2026-07-21)

The runtime now has a transport-neutral GDS host, durable outbox/attempt and
downlink identities, persisted TM counters, bounded realtime/file paths, and
the scheduler preserves durable TM MCFC order across APID 2 and APID 3 queues.
The latter prevents an ACK-priority APID 2 frame from overtaking an already
allocated APID 3 START frame and causing a false stale-counter rejection.

Current regression evidence is `171 passed, 10 subtests` for the CCSDS/GDS
core suite, `61 passed, 9 subtests` for the artifact/AI suite, `9 passed` for
Phase 6 hardening, and `13 passed` for frontend Vitest. The JUnit artifacts are
`artifacts/ccsds-core.xml`, `artifacts/ml-artifact.xml`, and
`artifacts/phase6-hardening.xml`.

These results do not close the release gate. `docker compose config` passes,
but the local Docker daemon is unavailable, and no in-app browser target is
available for the required browser rerun. Native F Prime remains blocked as
recorded in `docs/gds_satellite_ccsds_fprime_scope_decision_20260721.md`.

## Execution Order

```text
Contract/profile decisions
  -> real transport topology (F-01, F-02)
  -> durable command, sequence, and downlink ledger (F-03, F-04, F-05)
  -> parallel realtime and streaming work (F-06, F-07)
  -> native F Prime decision/deliverable (F-08)
  -> Compose, Playwright, conformance, and release gates
```

## Phase 0: Contract and Prerequisite Gate

Estimated effort: 0.5-1 person-day.

- Freeze one source of truth for APID/VCID, packet type, sequence flags, TC/TM counters, session/generation, event correlation, and RequestKey ownership.
- Define the transport endpoint interface shared by local-SIL in-memory transport and Compose UDP transport.
- Decide whether native F Prime is required for this release. Check the availability of the pinned F Prime source/SDK, compiler, and dictionary-generation toolchain before implementation.
- Add or update contract/golden-vector tests before changing the runtime path.

Exit criteria: profile and wire-contract decisions are recorded, the F-08 environment decision is explicit, and no implementation task depends on an unresolved packet or identity contract.

## Phase 1: Real Transport Boundary

Findings: F-01 and F-02. Estimated effort: 3-5 person-days.

Affected areas: `gds/http_app.py`, `gds/local_sil.py`, `link_sim/__main__.py`, `link_sim/transport.py`, `flight/mission_udp_adapter.py`, `flight/satellite_simulator.py`, and `deploy/docker-compose.yml`.

- Remove direct satellite-payload, journal, and product-completion reads from the HTTP host path. HTTP should depend on a transport-neutral GDS runtime and durable GDS state only.
- Use `MissionUdpAdapter` and the same endpoint abstraction for local-SIL and UDP modes.
- Make the satellite process bind a UDP endpoint, receive validated ingress envelopes, and emit ACK/progress/result/catalog/file TM through the link.
- Make LinkSimulator validate configured peers, direction, session, generation, envelope fields, and frame length. Load the selected fault profile and seed from deployment configuration.
- Replace print-only readiness with socket reachability plus a session handshake and command/TM health exchange.
- Derive catalog, command completion, and product completion from decoded TM and durable GDS state. Do not use direct calls such as `receive_tc_frame()` or `downlink_transfer()` from the HTTP boundary.

Required evidence:

- Unit tests for endpoint, peer, direction, session, and envelope rejection.
- A three-process integration test covering command admission, TC delivery, TM ACK/result, and APID 3 product publication.
- Fault tests for loss, duplication, corruption, latency, blackout, session reset, and file epoch behavior.
- A Playwright test that verifies the wire-correlated lifecycle instead of only polling `/api/state`.

Exit criteria: local-SIL and Compose use the same boundary, and no direct web-to-satellite or web-to-GDS bypass remains.

## Phase 2: Durable Command, Sequence, and Downlink Path

Findings: F-03, F-04, and F-05. Estimated effort: 4-6 person-days.

Affected areas: `gds/outbox.py`, `gds/ledger.py`, `gds/request_keys.py`, `gds/http_app.py`, `gds/sequence.py`, `gds/tm.py`, `gds/ingest.py`, `flight/journal.py`, `flight/file_downlink.py`, and protocol profile/vector files.

- Replace the request-agnostic `claim_next()` usage with an atomic claim-by-key or a single durable outbox dispatcher. The dispatcher must reconcile expired leases at startup, retry due work, and wake on new admission/contact/TM events.
- Keep lease ownership through attempt preparation. Add a `prepare_attempt()` transaction that allocates the packet sequence, encodes the TC with the selected profile, stores the exact bytes and metadata/hash, and increments the attempt atomically.
- Make transport send use the persisted bytes. Advance command state only through correlated TM ACK/result ingestion; link-level acceptance alone is not command completion.
- Resolve packet-type divergence between router, runtime, and golden generator. Persist profile identity and sequence epoch with attempts.
- Persist TM MCFC/VCFC and APID 3 transfer/file sequence counters across restart. Add GDS gap, duplicate, rollover, and stale-generation handling.
- Admit every product downlink through the normal ledger/outbox using a fresh `RequestKey` from `RequestKeyAllocator`. Link it to the originating request and transfer state; remove the high-bit synthetic ID scheme.

Required evidence:

- Concurrent dispatcher/lease race tests with no stranded `DISPATCHING` rows.
- Crash tests after claim, attempt persistence, send, and before ACK, followed by automatic restart recovery.
- Decode persisted TC bytes and assert packet sequence, frame sequence, packet type, profile, and hashes match the stored metadata.
- Rollover vectors for `16382, 16383, 0, 1`, TM counter persistence, APID 3 restart behavior, and GDS gap detection.
- High-bit RequestKey and duplicate/retry downlink tests.

Exit criteria: no nonterminal command remains without a bounded recovery path; runtime bytes, persisted metadata, and golden vectors are identical; every downlink has an independent durable ledger identity.

## Phase 3: Realtime Correctness and Resource Bounds

### F-06: Realtime cursors and resync

Estimated effort: 1-2 person-days. Affected areas: `gds/web/src/api/realtime.ts`, `gds/web/src/state/store.ts`, `gds/realtime.py`, and `gds/http_app.py`.

- Parse fixed-width lowercase hexadecimal U64 cursors with an explicit `0x` prefix for `BigInt` comparisons.
- Verify that initial replay and live queue are separate and that a client is registered at a deterministic cursor boundary.
- Send an explicit `RESYNC_REQUIRED` envelope for stale cursors and slow clients, then close the websocket with a documented recovery code.
- Preserve bounded recent-ID deduplication and consume browser-buffer entries after dispatch.

Required evidence: Vitest and WebSocket E2E tests for event IDs containing `a`-`f`, stale cursors, replay/live boundary duplicates, and slow-client closure.

### F-07: Streaming and memory limits

Estimated effort: 3-5 person-days. Affected areas: `sat_ai/roi.py`, `sat_ai/products.py`, `flight/file_downlink.py`, `gds/file_reassembly.py`, and `gds/product_store.py`.

- Reuse verified ingest/package fingerprints instead of hashing the full source TIFF for every job.
- Stream artifact writes, deterministic TAR generation, downlink reads, reassembly verification, checksum calculation, and safe extraction through bounded buffers.
- Remove large `read_bytes()`/`BytesIO` paths for scene, bundle, artifact, and downlink data while preserving integrity checks and storage reservations.
- Keep binary mask and unknown-nodata rejection fail-closed; add regression tests so this existing protection is not lost during streaming changes.

Required evidence: canonical scene benchmark, near-limit bundle test, RSS delta, p95 latency, logical-read counts, bounded-buffer assertions, and byte-exact product verification. The target is the planned `256 MiB` RSS guard.

## Phase 4: Native F Prime Deliverable

Finding: F-08. Estimated effort: 5-8 person-days after the toolchain is available.

If native F Prime is in scope:

- Pin the F Prime source/SDK version and commit.
- Replace the placeholder `flight/CloudPayload.fpp` with the real typed component, deployment, dictionary, stock APID router/deframer/framer, and build constants.
- Build a native deployment and run the same golden vectors, packet/frame rollover cases, 990-byte file boundary, and completion-gate tests against it.
- Integrate the native satellite endpoint with the Python GDS and LinkSimulator for the three-process E2E.

If native F Prime is intentionally out of scope or the source/toolchain is unavailable, update the simulation plan, `flight/README.md`, tracker, and DoD to mark Phase 2a as conditional. Do not mark F-08 closed based on the Python reference alone.

## Phase 5: Release and Definition of Done

Estimated effort: 2-3 person-days after implementation.

- Run the real Docker Compose topology with selected fault profile, restart/session reset, file transfer, and health/readiness checks.
- Run desktop and mobile Playwright workflows against that topology.
- Run the full Python/frontend/build/compile/round-trip/demo/soak suite and attach reproducible artifacts.
- Produce P6-13 release evidence from a dedicated clean checkout: identical builds under the same `SOURCE_DATE_EPOCH`, pinned dependency/SBOM hashes, and `source_dirty=false`.
- Revalidate and sign P6-15 only after all findings and gates have evidence. Keep CUDA/Jetson status fail-closed when target hardware is unavailable.

## Final Release Gate

Do not mark the simulation plan complete until:

- F-01 through F-05 and F-08 are closed.
- F-06 and F-07 have passing regression/performance evidence.
- Compose and Playwright pass through the real three-process topology.
- Product publication occurs only after complete bundle/artifact verification.
- P6-13 is a clean reproducible release and P6-15 is revalidated.

Approximate total effort: 16-25 person-days, excluding external F Prime/toolchain delays and unavailable CUDA/Jetson hardware evidence.
