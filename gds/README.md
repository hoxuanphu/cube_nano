# Ground Data System SQLite core

This package implements the Phase 4A storage and command-admission boundary. It
consumes mission contracts from `protocol`; it must never import `sat_ai` or
call inference directly.

## Ownership and durability

- `SQLiteWriter` owns the only read-write SQLite connection for a database.
  Every mutation is a bounded `MutationIntent`; reader connections are
  `query_only` and separately tracked.
- Startup applies checksum-verified, forward-only migrations and fails closed
  when the database is newer than the binary or its migration metadata drifts.
- The local profile is WAL, `synchronous=FULL`, foreign keys on, a 5000 ms busy
  timeout, and a 1000-page auto-checkpoint. WAL warning/throttle thresholds are
  128/256 MiB.
- U64 values are always fixed `BLOB(8)` big-endian in SQLite and fixed 16-digit
  lowercase hexadecimal strings in API cursors.

## Command admission

`AtomicCommandLedger.admit` performs this serialized transaction:

1. Look up the installation-scoped HTTP idempotency key and semantic JCS digest.
2. Return the committed command on a same-body retry, or reject a digest conflict.
3. Validate expiry, contact state, and nonterminal outbox capacity.
4. Allocate a durable `RequestKey` and build the exact mission argument digest.
5. Insert the command and outbox rows, then commit before exposing `202 Accepted`.

The default expiry is materialized only in the first commit. Terminal command
pruning retains a 90-day HTTP idempotency marker. `OutboxService` then provides
durable leases, raw-attempt-before-send records, APID-scoped packet sequences,
ACK timeout/backoff, immediate/next-contact handling and late-ACK audit. The
framework-neutral `GDSApi` maps the command admission/status contract to HTTP
status codes without coupling the package to a web framework.

The remaining link syscall must be wrapped in `OutboxService.send_with_fence`
and the `BindingFence`: the binding read fence stays held through the send, so
a migration can terminalize old-target work without routing a packet to the
new spacecraft instance.

```python
from pathlib import Path

from gds import AtomicCommandLedger, SQLiteWriter
from protocol.schemas import CommandOpcode

with SQLiteWriter(Path("data/ground/gds.sqlite3")) as writer:
    ledger = AtomicCommandLedger(writer)
    accepted = ledger.admit(
        idempotency_key="operator-request-001",
        target_spacecraft_instance_id=1,
        opcode=CommandOpcode.SCENE_REQUEST_CATALOG,
        payload={},
    )
    assert accepted.status_code == 202
```
