# Phase 6 Completion Report

Date: 2026-07-21

Status: P6 implementation complete for the local CPU/local-SIL profile. The official release gate remains conditional because this shared worktree is dirty, CUDA/Jetson hardware is unavailable, and the Compose HTTP endpoint processes do not yet run the full multi-container workflow through UDP.

## Delivered

- A disposable FastAPI local-SIL adapter drives the real CCSDS path: scene catalog, ROI command, TC APID 0, ACK, worker inference, TM FilePacket, reassembly, checksum verification and product publication.
- Run and release manifests are atomic and deterministic. A run can be `OPEN`, `FINAL`, `INCOMPLETE_CRASH` or `INCOMPLETE_STORAGE`; replay bytes are explicit as `PRESENT`, `PINNED` or `EVICTED`.
- Topology and request guards reject public host mode for `host_local_sil`, foreign Host/Origin/peer, oversized body/header, path traversal and rate-limit violations.
- CPU, Compose and Jetson profiles are fail-closed. Compose has a real UDP LinkSimulator bridge and internal-only topology; the verified end-to-end workflow is local-SIL. Jetson remains non-deployable until a target benchmark and TensorRT profile exist.
- The web client reconciles REST snapshots with WebSocket events and tolerates partial product lifecycle events while a product is being verified.
- Playwright covers desktop full workflow and mobile ROI confirmation. The backend E2E covers blackout and next-contact behavior.

## Evidence

| Area | Command or artifact | Result |
|---|---|---|
| Python regression | `python -m pytest -q` | PASS: 216 passed, 19 subtests passed |
| Phase 6 recovery | `tests/test_phase6_hardening.py`, `tests/test_phase6_recovery.py` | PASS |
| Web unit/build | `npm test -- --run`, `npm run build` | PASS: 9 tests; Vite build PASS (578.21 kB initial JS warning) |
| Browser E2E | `npm run test:e2e` | PASS: 2 tests, 2 project-scoped skips; desktop full flow and mobile ROI confirmation |
| CPU batch matrix | `artifacts/benchmarks/phase6-batch-matrix-v1.json` | CPU batch 1/2/4/8 measured; SHA-256 `dfa208a6deb77bdbee6de5e76a2e1f5e5c65ec50e0738c66065911cd6cb0c43d`; CUDA and Jetson explicitly unavailable |
| Soak | `artifacts/soak/phase6_soak_report.json` | 20 iterations; queues bounded; replay within cap; cleanup guards PASS |
| Deployment | `scripts/validate_deploy_profiles.py`; `link_sim/__main__.py` | host loopback, Compose internal bridge/profile and Jetson fail-closed PASS; multi-container HTTP E2E not claimed |
| Demo | `python scripts/demo_scenario.py --timeout 60` | PASS; product `PUBLISHED`, checksum `SHA256_MATCH` |
| Release evidence | `artifacts/release/phase6-release-manifest.json`, `artifacts/release/phase6-sbom.json` | Release `3e8adb71df3dc09225094c0a9895ead3`; manifest SHA-256 `e075c465b8b696052c6357367c15662892481d777eb8af1786133ba8003dbf5b`; SBOM SHA-256 `18f296f00e5f76bdc2cac5f17b09c8c8eed2fd5b3242b27bbf83eac4acd99847`; generated with `--allow-dirty`, not official |

## Known gates

1. Run `python scripts/generate_release_manifest.py` without `--allow-dirty` only after the worktree is clean. The current evidence manifest records `source_dirty=true` by design.
2. CUDA and Jetson targets were not present in this environment. Their profiles remain blocked rather than inheriting CPU SLOs.
3. The in-app Browser integration was unavailable in this environment. Standalone Playwright Chromium was installed and the repository E2E tests were executed successfully.
4. Compose currently validates and starts the real UDP LinkSimulator bridge, but `gds` and `satellite` still use the local endpoint composition in the HTTP demo. A separate multi-container endpoint adapter/E2E is required before claiming the full Compose workflow.

## Reproduction

```text
python -m pytest -q
python scripts/validate_deploy_profiles.py
python scripts/soak_test.py --iterations 100
python scripts/demo_scenario.py --timeout 90
cd gds/web
npm test -- --run
npm run build
npm run test:e2e
```

For an official release, also run the clean release command after committing or otherwise cleanly staging the complete intended change set:

```text
python scripts/generate_release_manifest.py
```
