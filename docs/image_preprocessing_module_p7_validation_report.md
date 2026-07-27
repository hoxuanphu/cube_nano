# P7 validation report

Date: 2026-07-24

This report records the part of P7 that can be reproduced in the repository.
It does not promote a flight profile or claim Jetson qualification.

## Automated campaign

`tests/test_preprocessing_p7.py` covers:

- identity and affine golden cases;
- LUT/Brown-Conrady are intentionally not promoted: no signed flight
  calibration bundle is available, and the CPU planner remains fail-closed for
  those transform types under blocker B-01;
- `uint8`, `uint16` and `float32` source/output boundaries;
- `YXC`, `CYX` and singleton `YX` layouts;
- target-grid shape, origin, resolution and axis metadata;
- nearest, bilinear, bicubic and support kernels;
- invalid, constant, edge and reflect border policies;
- nearest-even, nearest-away, floor, ceil and truncate rounding;
- clipping, cast overflow and non-finite reject/replace behavior;
- NoData, missing-channel, outside-mapping, border and insufficient-support
  reason bits, including simultaneous source causes;
- trust, calibration, two-tier RAM/disk admission, compressed codec, artifact
  checksum/partial-manifest and engine timeout/reset fault injection;
- georeferencing-invalid decision fallback and the OBC/F' action boundary.

Results:

```text
pytest -q tests/test_preprocessing_p7.py
54 passed

pytest -q
141 passed, 9 subtests passed
```

The adapter now converts expected engine execution faults (`TimeoutError`,
connection/device `OSError` and reset-like `RuntimeError`) to
`RUNTIME_FAULT`/`RETAIN_FOR_GROUND`; programmer errors are still not caught by
the operational failure boundary.

## HIL harness

`scripts/preprocessing_p7_hil_benchmark.py` is the guarded Nano/Orin soak
harness. It requires at least five measured iterations and one warm-up, records
latency, memory and thermal observations when the platform exposes them, and
writes the report atomically. A requested Jetson target is only executed when
`/proc/device-tree/model` identifies that exact board.

Development smoke evidence:

```text
python scripts/preprocessing_p7_hil_benchmark.py --target host \
  --output-json build/p7-host-benchmark.json --iterations 5 --warmup-runs 1 \
  --rows 16 --cols 16
status=COMPLETE, measured_iterations=5

python scripts/preprocessing_p7_hil_benchmark.py --target jetson-nano \
  --output-json build/p7-nano-blocked.json
status=BLOCKED, reason_code=HIL_TARGET_UNAVAILABLE

python scripts/preprocessing_p7_hil_benchmark.py --target jetson-orin-nano \
  --output-json build/p7-orin-blocked.json
status=BLOCKED, reason_code=HIL_TARGET_UNAVAILABLE
```

The host result is a harness smoke test using a synthetic source and
development trust policy. It is not evidence for Nano/Orin RAM, power,
thermal, CUDA, TensorRT or GPU parity.

## Shadow-mode review

The local policy gate is covered by the P7 test: `georef_valid=false` produces
`RETAIN_FOR_GROUND`, and the decision record does not delete or mutate a source
capture. A system shadow-mode review remains blocked until the following
external contracts and evidence exist:

1. signed flight calibration/georeferencing bundle and error-bound acceptance;
2. separate Nano and Orin `ComputeProfile` HIL/soak results;
3. real engine manifest and TensorRT smoke evidence;
4. OBC/F' request, timeout, reset, acknowledgement and source-fingerprint
   authority tests.

Until those gates pass, GPU warp and mission policy remain fail-closed.
