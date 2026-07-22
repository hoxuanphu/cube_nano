"""Durable spacecraft binding and in-process read/write migration fences."""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Callable, Iterator

from protocol.canonical import checked_u64, u64_to_json

from .audit import append_audit_in_transaction
from .idempotency import datetime_to_unix_us
from .outbox import (
    BindingGenerationError,
    ContactState,
    LinkBinding,
    OutboxService,
)
from .u64 import decode_sqlite_u64, encode_sqlite_u64
from .writer import MutationPriority, SQLiteWriter


class BindingFenceError(RuntimeError):
    error_code = "BINDING_FENCE_ERROR"


class BindingChangedError(BindingFenceError):
    error_code = "NOT_SENT_REBIND"


class TargetInstanceRetiredError(BindingFenceError):
    error_code = "TARGET_INSTANCE_RETIRED"
    status_code = 410


class BindingFence:
    """Allow link I/O only while the published binding remains unchanged."""

    def __init__(self) -> None:
        self._condition = threading.Condition(threading.RLock())
        self._bindings: dict[int, LinkBinding] = {}
        self._readers = 0
        self._write_active = False

    @property
    def active_readers(self) -> int:
        with self._condition:
            return self._readers

    def publish(self, binding: LinkBinding) -> None:
        with self._condition:
            self._bindings[binding.spacecraft_instance_id] = binding
            self._condition.notify_all()

    def retire(self, spacecraft_instance_id: int) -> None:
        checked_u64(spacecraft_instance_id, "spacecraft_instance_id")
        with self._condition:
            self._bindings.pop(spacecraft_instance_id, None)
            self._condition.notify_all()

    def current(self, spacecraft_instance_id: int) -> LinkBinding | None:
        checked_u64(spacecraft_instance_id, "spacecraft_instance_id")
        with self._condition:
            return self._bindings.get(spacecraft_instance_id)

    @contextmanager
    def read(self, binding: LinkBinding) -> Iterator[LinkBinding]:
        with self._condition:
            while self._write_active:
                self._condition.wait()
            current = self._bindings.get(binding.spacecraft_instance_id)
            if current != binding:
                raise BindingChangedError(
                    "link binding changed before the send read fence was acquired"
                )
            self._readers += 1
        try:
            yield binding
        finally:
            with self._condition:
                self._readers -= 1
                if self._readers == 0:
                    self._condition.notify_all()

    @contextmanager
    def write(self) -> Iterator[None]:
        with self._condition:
            while self._write_active:
                self._condition.wait()
            self._write_active = True
            while self._readers:
                self._condition.wait()
        try:
            yield
        finally:
            with self._condition:
                self._write_active = False
                self._condition.notify_all()


@dataclass(frozen=True)
class BindingMigration:
    previous: LinkBinding | None
    current: LinkBinding
    retired_instance_id: int | None
    terminalized_commands: int


class SpacecraftBindingManager:
    """Fence target replacement while allowing same-instance session rebinds."""

    def __init__(
        self,
        writer: SQLiteWriter,
        outbox: OutboxService | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
        fence: BindingFence | None = None,
    ) -> None:
        self.writer = writer
        self.outbox = outbox
        self._clock = clock or (lambda: datetime.now(UTC))
        self.fence = fence or BindingFence()
        self.refresh()

    def _now_us(self) -> int:
        value = self._clock()
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise ValueError("binding clock must be timezone-aware")
        return datetime_to_unix_us(value.astimezone(UTC))

    def refresh(self) -> LinkBinding | None:
        with self.writer.reader() as connection:
            row = connection.execute(
                "SELECT m.bound_spacecraft_instance_id,m.bound_link_generation,"
                "m.bound_link_session_id,s.contact_state,s.state "
                "FROM gds_metadata m LEFT JOIN spacecraft_instances s ON "
                "s.spacecraft_instance_id=m.bound_spacecraft_instance_id "
                "WHERE m.singleton=1"
            ).fetchone()
        if row is None or row[0] is None or row[1] is None or row[2] is None:
            return None
        if row[4] != "ACTIVE":
            return None
        binding = LinkBinding(
            decode_sqlite_u64(row[0], "bound_spacecraft_instance_id"),
            decode_sqlite_u64(row[1], "bound_link_generation"),
            decode_sqlite_u64(row[2], "bound_link_session_id"),
            ContactState(row[3]),
        )
        self.fence.publish(binding)
        return binding

    def active_binding(self) -> LinkBinding | None:
        binding = self.refresh()
        return binding

    def read_fence(self, binding: LinkBinding) -> Iterator[LinkBinding]:
        return self.fence.read(binding)

    def bind(
        self,
        spacecraft_instance_id: int,
        *,
        link_generation: int,
        link_session_id: int,
        contact_state: ContactState = ContactState.CONTACT_OPEN,
        principal: str = "link-manager",
    ) -> BindingMigration:
        instance = checked_u64(spacecraft_instance_id, "spacecraft_instance_id")
        generation = checked_u64(link_generation, "link_generation")
        session = checked_u64(link_session_id, "link_session_id")
        if not isinstance(contact_state, ContactState):
            contact_state = ContactState(contact_state)
        if not isinstance(principal, str) or not principal:
            raise ValueError("principal must not be empty")
        current = LinkBinding(instance, generation, session, contact_state)

        with self.fence.write():
            migration = self.writer.mutate(
                "migrate_spacecraft_binding",
                lambda connection: self._bind_in_transaction(
                    connection, current, principal
                ),
                priority=MutationPriority.HIGH,
            )
            if migration.retired_instance_id is not None:
                self.fence.retire(migration.retired_instance_id)
            self.fence.publish(current)
            return migration

    def _bind_in_transaction(
        self,
        connection: sqlite3.Connection,
        current: LinkBinding,
        principal: str,
    ) -> BindingMigration:
        now_us = self._now_us()
        metadata = connection.execute(
            "SELECT bound_spacecraft_instance_id,bound_link_generation,"
            "bound_link_session_id FROM gds_metadata WHERE singleton=1"
        ).fetchone()
        if metadata is None:
            raise BindingFenceError("GDS metadata must be initialized before binding")
        metadata_bound = all(item is not None for item in metadata)

        target_row = connection.execute(
            "SELECT link_generation,link_session_id,state FROM spacecraft_instances "
            "WHERE spacecraft_instance_id=?",
            (encode_sqlite_u64(current.spacecraft_instance_id),),
        ).fetchone()
        if target_row is not None and str(target_row[2]) == "RETIRED":
            raise TargetInstanceRetiredError(
                "retired spacecraft instance IDs cannot be reactivated"
            )
        if metadata_bound and target_row is not None and str(target_row[2]) == "ACTIVE":
            old_generation = decode_sqlite_u64(target_row[0], "link_generation")
            old_session = (
                None
                if target_row[1] is None
                else decode_sqlite_u64(target_row[1], "link_session_id")
            )
            if current.link_generation < old_generation or (
                current.link_generation == old_generation
                and old_session not in (None, current.link_session_id)
            ):
                raise BindingGenerationError(
                    "link_generation must be monotonic and session changes require a new generation"
                )

        previous: LinkBinding | None = None
        if metadata_bound:
            previous = LinkBinding(
                decode_sqlite_u64(metadata[0], "bound_spacecraft_instance_id"),
                decode_sqlite_u64(metadata[1], "bound_link_generation"),
                decode_sqlite_u64(metadata[2], "bound_link_session_id"),
                self._read_contact_state(connection, metadata[0]),
            )

        if (
            previous is not None
            and previous.spacecraft_instance_id == current.spacecraft_instance_id
            and previous.link_generation == current.link_generation
            and previous.link_session_id == current.link_session_id
            and previous.contact_state != current.contact_state
        ):
            raise BindingFenceError(
                "contact state changes must use OutboxService.set_contact_state"
            )

        binding_changed = previous is None or previous != current
        retired_instance = None
        terminalized = 0
        if binding_changed and previous is not None:
            old_blob = encode_sqlite_u64(previous.spacecraft_instance_id)
            if previous.spacecraft_instance_id == current.spacecraft_instance_id:
                # A new session/generation is a transport rebind, not a new
                # satellite identity.  Old-session ACKs are no longer valid,
                # so revoke their correlation eligibility and replay the same
                # durable RequestKey through the new binding.
                connection.execute(
                    "UPDATE command_outbox SET state='OUTBOX_PENDING',"
                    "available_at_us=?,lease_owner=NULL,lease_expires_at_us=NULL,"
                    "ack_deadline_at_us=NULL,last_error_code='LINK_REBOUND',"
                    "last_delivery_reason='LINK_REBOUND',updated_at_us=? "
                    "WHERE target_spacecraft_instance_id=? AND state IN "
                    "('HELD_NO_CONTACT','OUTBOX_PENDING','DISPATCHING','SENT')",
                    (now_us, now_us, old_blob),
                )
                connection.execute(
                    "UPDATE command_attempts SET send_result='NOT_SENT:REBIND' "
                    "WHERE target_spacecraft_instance_id=? AND send_result IN "
                    "('PERSISTED_NOT_SENT','SENT')",
                    (old_blob,),
                )
            else:
                retired_instance = previous.spacecraft_instance_id
                outbox_cursor = connection.execute(
                    "UPDATE command_outbox SET state='DELIVERY_FAILED',"
                    "last_error_code='TARGET_INSTANCE_RETIRED',"
                    "last_delivery_reason='TARGET_INSTANCE_RETIRED',"
                    "lease_owner=NULL,lease_expires_at_us=NULL,"
                    "ack_deadline_at_us=NULL,updated_at_us=? "
                    "WHERE target_spacecraft_instance_id=? AND state IN "
                    "('HELD_NO_CONTACT','OUTBOX_PENDING','DISPATCHING','SENT')",
                    (now_us, old_blob),
                )
                terminalized = int(outbox_cursor.rowcount)
                connection.execute(
                    "UPDATE command_attempts SET send_result='NOT_SENT:REBIND' "
                    "WHERE target_spacecraft_instance_id=? AND send_result='PERSISTED_NOT_SENT'",
                    (old_blob,),
                )
                connection.execute(
                    "UPDATE commands SET command_state='FAILED',"
                    "terminal_at_us=MAX(terminal_at_us,?),"
                    "updated_at_us=MAX(updated_at_us,?) "
                    "WHERE target_spacecraft_instance_id=? AND command_state='ADMITTED' "
                    "AND EXISTS (SELECT 1 FROM command_outbox o WHERE "
                    "o.ground_instance_id=commands.ground_instance_id "
                    "AND o.request_id=commands.request_id "
                    "AND o.state='DELIVERY_FAILED' "
                    "AND o.last_error_code='TARGET_INSTANCE_RETIRED')",
                    (now_us, now_us, old_blob),
                )
                connection.execute(
                    "UPDATE spacecraft_instances SET state='RETIRED',"
                    "contact_state='NO_CONTACT',rebaseline_reason=?,"
                    "last_seen_at_us=?,contact_changed_at_us=? "
                    "WHERE spacecraft_instance_id=?",
                    ("TARGET_INSTANCE_RETIRED", now_us, now_us, old_blob),
                )

        new_blob = encode_sqlite_u64(current.spacecraft_instance_id)
        connection.execute(
            "INSERT INTO spacecraft_instances("
            "spacecraft_instance_id,link_generation,link_session_id,state,"
            "first_seen_at_us,last_seen_at_us,rebaseline_reason,contact_state,"
            "contact_changed_at_us) VALUES(?,?,?,'ACTIVE',?,?,NULL,?,?) "
            "ON CONFLICT(spacecraft_instance_id) DO UPDATE SET "
            "link_generation=excluded.link_generation,"
            "link_session_id=excluded.link_session_id,state='ACTIVE',"
            "last_seen_at_us=excluded.last_seen_at_us,rebaseline_reason=NULL,"
            "contact_state=excluded.contact_state,"
            "contact_changed_at_us=excluded.contact_changed_at_us",
            (
                new_blob,
                encode_sqlite_u64(current.link_generation),
                encode_sqlite_u64(current.link_session_id),
                now_us,
                now_us,
                current.contact_state.value,
                now_us,
            ),
        )
        connection.execute(
            "UPDATE gds_metadata SET bound_spacecraft_instance_id=?,"
            "bound_link_generation=?,bound_link_session_id=? WHERE singleton=1",
            (
                new_blob,
                encode_sqlite_u64(current.link_generation),
                encode_sqlite_u64(current.link_session_id),
            ),
        )
        append_audit_in_transaction(
            connection,
            principal=principal,
            action="SPACECRAFT_BINDING_MIGRATED" if binding_changed else "SPACECRAFT_BINDING_REFRESHED",
            target_type="spacecraft_instance",
            target_identity={"spacecraft_instance_id": u64_to_json(current.spacecraft_instance_id)},
            old_value=None if previous is None else {
                "spacecraft_instance_id": u64_to_json(previous.spacecraft_instance_id),
                "link_generation": u64_to_json(previous.link_generation),
                "link_session_id": u64_to_json(previous.link_session_id),
            },
            new_value={
                "spacecraft_instance_id": u64_to_json(current.spacecraft_instance_id),
                "link_generation": u64_to_json(current.link_generation),
                "link_session_id": u64_to_json(current.link_session_id),
                "contact_state": current.contact_state.value,
                "terminalized_commands": terminalized,
            },
            created_at_us=now_us,
        )
        return BindingMigration(previous, current, retired_instance, terminalized)

    @staticmethod
    def _read_contact_state(
        connection: sqlite3.Connection, instance_blob: bytes
    ) -> ContactState:
        row = connection.execute(
            "SELECT contact_state FROM spacecraft_instances "
            "WHERE spacecraft_instance_id=?",
            (instance_blob,),
        ).fetchone()
        return ContactState(row[0]) if row is not None else ContactState.NO_CONTACT
