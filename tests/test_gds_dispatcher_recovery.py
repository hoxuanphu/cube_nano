"""Regression coverage for the autonomous durable GDS dispatcher."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

from gds.ledger import AtomicCommandLedger
from gds.local_sil import GdsMissionRuntime
from gds.outbox import ContactState, OutboxService
from gds.writer import SQLiteWriter
from protocol.schemas import CommandOpcode, RequestKey


def test_dispatcher_reconciles_lost_ack_and_rediscovers_due_key(tmp_path: Path):
    """A lone SENT command is retried without a new API request arriving."""

    with SQLiteWriter(tmp_path / "gds.sqlite3") as writer:
        ledger = AtomicCommandLedger(writer)
        outbox = OutboxService(writer)
        binding = outbox.register_instance(
            1,
            link_generation=1,
            link_session_id=1,
            contact_state=ContactState.CONTACT_OPEN,
        )
        accepted = ledger.admit(
            idempotency_key="lost-ack-dispatcher-001",
            target_spacecraft_instance_id=1,
            opcode=CommandOpcode.SCENE_REQUEST_CATALOG,
            payload={},
            contact_available=True,
        )
        lease = outbox.claim(accepted.request_key, binding=binding)
        assert lease is not None
        attempt = outbox.persist_attempt(lease, b"tc", apid=7)
        outbox.mark_sent(lease, attempt)
        now_us = time.time_ns() // 1_000
        writer.mutate(
            "expire_test_ack_deadline",
            lambda connection: connection.execute(
                "UPDATE command_outbox SET ack_deadline_at_us=?,available_at_us=? "
                "WHERE ground_instance_id=? AND request_id=?",
                (
                    now_us - 1,
                    now_us,
                    accepted.request_key.ground_instance_id.to_bytes(8, "big"),
                    accepted.request_key.request_id,
                ),
            ),
        )

        dispatched: list[RequestKey] = []
        runtime = object.__new__(GdsMissionRuntime)
        runtime.gds = SimpleNamespace(
            outbox=outbox,
            writer=writer,
            bindings=SimpleNamespace(active_binding=lambda: binding),
        )
        runtime._reconcile_lock = threading.Lock()
        runtime._next_outbox_reconcile_monotonic = 0.0
        runtime._dispatch = lambda request_key: dispatched.append(request_key) or True

        runtime._dispatch_due()
        assert not dispatched
        # The outbox retry backoff is 500ms.  No new command is submitted;
        # the next dispatcher cadence must discover this same request key.
        time.sleep(0.55)
        runtime._dispatch_due()
        assert dispatched == [accepted.request_key]
