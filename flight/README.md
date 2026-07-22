# Flight boundary

This phase uses a Python reference deployment because the repository contains
the generated F Prime v4.1.0 dictionary but not the pinned F Prime source
checkout. `CloudPayload` and the scheduler keep the same ownership/APID
boundaries that a native FPP component will use later. Native FPP sources are
intentionally not invented here.

Native F Prime is a conditional, release-blocking deliverable. The local
environment currently lacks the F Prime CLI and dictionary-generation tools, so
this directory is not evidence of a native F Prime build and F-08 remains open.
The required source, toolchain, native deployment, and verification gates are
recorded in `docs/gds_satellite_ccsds_fprime_scope_decision_20260721.md`.

The local reference now runs AI inference in an isolated process. File egress
is completion-driven: one frame lease is active at a time, and an attempt slot
is released only after terminal completion or abort-fence cooldown.
