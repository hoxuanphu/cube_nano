# GDS Satellite CCSDS SIL baseline

Date: 2026-07-19

The repository baseline before this implementation was commit
`6deab67eab18e00c0ff316ff8e4087452859ed75` with 53 existing tests passing.
Existing inference changes and scene/checkpoint files were preserved; the
implementation adds a separate protocol/flight/sat_ai boundary and does not
make the web layer call inference.

## Baseline decisions

- F Prime dictionary and constants remain v4.1.0 / SCID 68 / TM frame 1024.
- Stock APID mapping is TC 0, telemetry 1, event/ACK 2, file 3; VC0 and
  big-endian wire values are fixed in `protocol/mission_profile.yaml`.
- The first runnable profile is loopback-only `host_local_sil` with CPU/PyTorch.
- The released checkpoint is bound to RGB, 256x256, NCHW and
  `uint16 / 65535`; assurance remains `demo_non_validated`.
- Runtime raster input is a single-series, single-level TIFF that passes
  `tifffile.memmap()` and a SHA-bound sidecar. Compressed/JP2/full-decode paths
  fail closed.

## Evidence

- Existing regression suite: `53 passed, 9 subtests passed`.
- Current suite after Phase 0-2 expert-review remediation: `90 passed, 19 subtests passed`.
- Actual local CPU benchmark artifact:
  `artifacts/benchmarks/local-cpu-pytorch-v2.json`.
- Threshold LUT: `protocol/golden_vectors/threshold_lut.bin`.
- Binary vector inventory: `protocol/golden_vectors/vectors.json`.
- Reference smoke through the isolated worker process: a 256x256 ROI produced
  one `SUCCEEDED` job, an atomically published product, 526 completed TM file
  frames and durable transfer state `SEND_COMPLETED`.

The repository has no F Prime source checkout or native deployment tree. The
Phase 2a flight boundary is therefore a Python reference deployment plus an
FPP contract stub; native F Prime compilation remains an explicit follow-up,
not an unverified claim in the tracker.
