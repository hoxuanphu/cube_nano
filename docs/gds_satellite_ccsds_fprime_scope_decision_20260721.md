# Native F Prime Scope Decision

- Date: 2026-07-21
- Finding: F-08 in `docs/gds_satellite_ccsds_code_review_findings.md`
- Decision: native F Prime is a release-blocking conditional deliverable, not
  evidence supplied by the Python reference implementation.

## Environment Check

The repository contains the generated F Prime v4.1.0 dictionary, a Python
reference flight boundary, and a placeholder FPP contract. It does not contain
the pinned F Prime source checkout or a native deployment tree. The local
environment has CMake and a C++ compiler, but does not provide the `fprime` CLI
or `fpp-to-xml` dictionary-generation tool.

## Release Consequence

Phase 2a reference-contract work remains useful evidence for the Python SIL
path. It must not be reported as a native F Prime build, and F-08 remains open.
P6-15 cannot receive final sign-off on the basis of the reference deployment.

## Prerequisites To Close F-08

1. Add a pinned F Prime source/SDK v4.1.0 checkout and record its immutable
   commit/hash.
2. Add the real typed `CloudPayload` FPP component, native deployment, stock
   router/deframer/framer wiring, build constants, and generated dictionary.
3. Build the native deployment and run golden vectors, packet/frame rollover,
   the 990-byte file boundary, completion-gate tests, and the UDP Compose E2E
   path against it.

Until these inputs are available, the simulation plan and Definition of Done
remain explicitly conditional rather than complete.
