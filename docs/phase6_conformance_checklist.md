# Phase 6 Conformance Checklist

Date: 2026-07-21

| Contract | Evidence | Status |
|---|---|---|
| APID/descriptor and frame contract | `protocol/golden_vectors/`; existing mission contract suite | PASS |
| Scene to ROI to TC to ACK to inference to TM to product | `tests/test_phase6_hardening.py`; `artifacts/phase6-hardening.xml` | PASS on local-SIL Python E2E |
| Follow-up F-01 through F-07 transport, ledger, TM order, realtime, and streaming regressions | `tests/test_followup_durable_ledger.py`; `tests/test_followup_realtime_streaming.py`; `tests/test_udp_process_boundary.py`; `artifacts/ccsds-core.xml` | PASS for implementation/regression evidence |
| RequestKey trace and idempotent admission | `gds/http_app.py`; phase 4a ledger tests; HTTP E2E | PASS |
| File loss/reorder/retry/cancel | `tests/test_phase6_recovery.py`; `tests/test_phase4b_runtime.py` | PASS |
| Restart and reconciliation | durable reassembler restart test; `flight/journal.py`; phase 2 runtime tests | PASS |
| Deterministic fault/replay retention | `tests/test_phase6_hardening.py`; `tests/test_replay_manager.py` | PASS |
| Blackout and next-contact | HTTP blackout test; `gds/outbox.py` | PASS |
| Security and resource limits | foreign Host/Origin/peer, body/header and topology tests | PASS |
| Queue/watchdog/health SLO | `protocol/slo_profile.yaml`; `/healthz`; `/readyz`; scheduler metrics | PASS on CPU reference |
| CPU batch/resource benchmark | `artifacts/benchmarks/phase6-batch-matrix-v1.json` | PASS |
| CUDA/Jetson benchmark | Target runtime not installed | BLOCKED, fail-closed |
| Native F Prime deployment, dictionary, and vector suite | No pinned source checkout or F Prime/FPP toolchain is available | BLOCKED; F-08 remains open |
| Soak and cleanup | `artifacts/soak/phase6_soak_report.json` | PASS |
| Compose/internal topology and real UDP bridge | `deploy/docker-compose.yml`; `docker compose config`; `link_sim/__main__.py`; `tests/test_udp_process_boundary.py` | PASS for profile/bridge wiring and local three-process UDP; BLOCKED for live multi-container E2E because Docker daemon is unavailable |
| Browser lifecycle rerun | `gds/web/e2e/mission-control.spec.ts`; desktop and mobile workflow | BLOCKED: no in-app browser target is available in this environment |
| Release manifest and SBOM | `gds/release_manifest.py`; `gds/run_manifest.py`; SPDX output | PASS as evidence |
| Clean reproducible official release | Shared worktree has unrelated and active changes | BLOCKED until clean |
| Runbook and repeatable demo | `deploy/local_sil_runbook.md`; `scripts/demo_scenario.py` | PASS on local CPU |

The blocked and conditional rows are intentional release controls. They must not be converted into a READY profile or a clean release by changing metadata only. Native F Prime prerequisites are recorded in `docs/gds_satellite_ccsds_fprime_scope_decision_20260721.md`.
