# Phase 2b expert review response

Date: 2026-07-19

The review was useful, but it mixed real lifecycle gaps with observations from
an older snapshot. This response records the disposition against executable
evidence rather than changing the original review.

| Review finding | Disposition | Remediation/evidence |
|---|---|---|
| Worker IPC incomplete | Valid | Canonical versioned request/result/control/heartbeat messages, OS worker process, deadline/cancel checks, heartbeat watchdog, bounded restart window and process-kill test in `sat_ai/worker_contract.py`, `sat_ai/worker_process.py`, `flight/worker_client.py`. |
| FileDownlinkCoordinator is only a skeleton | Partly valid | The file already generated START/DATA/END, but incorrectly treated frame generation as completion. It now uses one-frame leases, attempt epochs, completion tokens, abort fence, cooldown, completion-wins cancel and late-callback rejection in `flight/file_downlink.py`. |
| Deployment profile is non-deployable | Stale observation | The reviewed tree already had `deployable: true` and a hash-bound artifact. It is now strengthened to measured `local-cpu-pytorch-v2`, CPU thread pinning, exact logical-read and scene-scale/RSS guards. |
| Phase 2b failure matrix is thin | Valid | Added worker process crash, queue saturation, CAS rollback, restart corruption, atomic product failure, link failure, abort/cooldown, cancel race, duplicate/late callback and scheduler flood tests. |
| Golden vectors only contain LUT | Stale but partially useful | `vectors.json` already existed, but lacked TC Type-BD and some file/application coverage. It now includes TC/TM rollover, descriptor/APID, all FilePacket types, 990-byte boundary and checksum vectors with regeneration tests. |
| Native F Prime build absent | Valid limitation | Still explicit. The repository has no F Prime source checkout, so this remains a Python reference deployment plus FPP contract stub; no native-build claim is made. |

Additional defects found during remediation:

- TC Type-BD header now sets the bypass flag and encodes the 10-bit SCID in
  the correct CCSDS bit positions.
- `all_valid` no longer reports fake validity-mask I/O; benchmark logical read
  is `393216` source bytes and `0` validity bytes for a 256x256 RGB uint16 ROI.
- Worker drain waits for the durable result callback, preventing a transient
  `RUNNING` status after the process has already returned success.
- Product publication uses an atomic directory rename; restart reconciliation
  fails STAGING rows and removes abandoned staging directories.

Verification after remediation:

- `90 passed, 19 subtests passed`.
- CPU benchmark v2: p95 `31.091 ms`, p99 `32.595 ms`, scene-scale ratio
  `0.8866 <= 1.25`, RSS guard passed.
- Local process smoke: ROI job `SUCCEEDED`; 526 TM file frames completed;
  durable transfer `SEND_COMPLETED`.
