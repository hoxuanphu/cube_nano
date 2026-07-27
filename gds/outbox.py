"""Durable outbox lease, attempt, retry and ACK lifecycle."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any, Callable, Mapping

from protocol.canonical import checked_u64, u64_to_json
from protocol.ccsds import SpacePacket, TcTypeBdFrame
from protocol.schemas import Command, CommandOpcode, RequestKey, encode_command

from .idempotency import datetime_to_unix_us, unix_us_to_datetime
from .sequence import SequenceAllocation, TcSequenceAllocator
from .u64 import decode_sqlite_u64, encode_sqlite_u64
from .writer import MutationPriority, SQLiteWriter

LEASE_DURATION = timedelta(seconds=10)
ACK_TIMEOUT = timedelta(seconds=5)
BASE_RETRY_BACKOFF = timedelta(milliseconds=500)
MAX_RETRY_BACKOFF = timedelta(seconds=30)
DEFAULT_MAX_ATTEMPTS = 20


class ContactState(str, Enum):
    CONTACT_OPEN = "CONTACT_OPEN"
    NO_CONTACT = "NO_CONTACT"
    BLACKOUT = "BLACKOUT"

    @property
    def is_open(self) -> bool:
        return self is ContactState.CONTACT_OPEN


class OutboxError(RuntimeError):
    error_code = "OUTBOX_ERROR"


class BindingUnavailableError(OutboxError):
    error_code = "NO_CONTACT"


class LeaseLostError(OutboxError):
    error_code = "LEASE_LOST"


class AttemptLimitError(OutboxError):
    error_code = "MAX_ATTEMPTS"


class BindingGenerationError(OutboxError):
    error_code = "LINK_GENERATION_REGRESSION"


@dataclass(frozen=True)
class TcWireProfile:
    """Pinned TC framing values used for one persisted command attempt."""

    profile_id: str
    profile_sha256: str
    spacecraft_id: int
    virtual_channel_id: int
    tc_apid: int
    packet_type: int = 1
    secondary_header_present: bool = False
    sequence_flags: int = 3

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id:
            raise ValueError("TC profile_id must not be empty")
        if (
            not isinstance(self.profile_sha256, str)
            or len(self.profile_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.profile_sha256)
        ):
            raise ValueError("TC profile_sha256 must be lowercase SHA-256 hex")
        if not 0 <= self.spacecraft_id <= 0x3FF:
            raise ValueError("TC spacecraft_id must fit 10 bits")
        if not 0 <= self.virtual_channel_id <= 0x3F:
            raise ValueError("TC virtual_channel_id must fit 6 bits")
        if not 0 <= self.tc_apid <= 0x7FF:
            raise ValueError("TC APID must fit 11 bits")
        if self.packet_type != 1:
            raise ValueError("MVP TC Space Packets use packet_type=1")
        if self.secondary_header_present:
            raise ValueError("MVP TC Space Packets do not use a secondary header")
        if self.sequence_flags != 3:
            raise ValueError("MVP TC Space Packets use unsegmented sequence flags")

    @classmethod
    def from_mission_profile(cls, profile: Any) -> "TcWireProfile":
        return cls(
            profile_id=str(profile.profile_id),
            profile_sha256=str(profile.digest()),
            spacecraft_id=int(profile.spacecraft_id),
            virtual_channel_id=int(profile.tc_virtual_channel),
            tc_apid=int(profile.tc_apid),
        )

    def encode(
        self,
        command: Command,
        *,
        packet_sequence: int,
        frame_sequence: int,
    ) -> bytes:
        if command.target_spacecraft_instance_id < 0:
            raise ValueError("invalid command target")
        return TcTypeBdFrame(
            self.spacecraft_id,
            self.virtual_channel_id,
            frame_sequence,
            SpacePacket(
                self.tc_apid,
                packet_sequence,
                encode_command(command),
                packet_type=self.packet_type,
                secondary_header_present=self.secondary_header_present,
                sequence_flags=self.sequence_flags,
            ),
        ).encode()


@dataclass(frozen=True)
class LinkBinding:
    spacecraft_instance_id: int
    link_generation: int
    link_session_id: int
    contact_state: ContactState

    def __post_init__(self) -> None:
        checked_u64(self.spacecraft_instance_id, "spacecraft_instance_id")
        checked_u64(self.link_generation, "link_generation")
        checked_u64(self.link_session_id, "link_session_id")


@dataclass(frozen=True)
class OutboxPolicy:
    lease_duration: timedelta = LEASE_DURATION
    ack_timeout: timedelta = ACK_TIMEOUT
    base_backoff: timedelta = BASE_RETRY_BACKOFF
    max_backoff: timedelta = MAX_RETRY_BACKOFF
    max_attempts: int = DEFAULT_MAX_ATTEMPTS

    def __post_init__(self) -> None:
        if self.lease_duration <= timedelta(0):
            raise ValueError("lease_duration must be positive")
        if self.ack_timeout <= timedelta(0):
            raise ValueError("ack_timeout must be positive")
        if self.base_backoff <= timedelta(0) or self.max_backoff < self.base_backoff:
            raise ValueError("backoff values are invalid")
        if isinstance(self.max_attempts, bool) or not 1 <= self.max_attempts <= 0xFFFFFFFF:
            raise ValueError("max_attempts must be in [1, 2^32-1]")


@dataclass(frozen=True)
class OutboxLease:
    request_key: RequestKey
    target_spacecraft_instance_id: int
    lease_owner: str
    lease_expires_at: datetime
    effective_expires_at: datetime
    attempt_count: int
    opcode: CommandOpcode
    mission_arguments: bytes
    command: Command
    binding: LinkBinding


@dataclass(frozen=True)
class AttemptRecord:
    attempt_id: int
    request_key: RequestKey
    target_spacecraft_instance_id: int
    attempt_number: int
    apid: int
    packet_sequence: int
    sequence_epoch: int
    rollover: bool
    link_generation: int
    link_session_id: int
    encoded_tc: bytes
    created_at: datetime
    frame_sequence: int = 0
    tc_profile_id: str = "legacy"
    tc_profile_sha256: str = "legacy"
    space_packet_type: int = 1
    space_packet_sequence_flags: int = 3
    encoded_tc_sha256: str = ""


@dataclass(frozen=True)
class AckResult:
    request_key: RequestKey
    state: str
    late: bool
    reason: str | None


@dataclass(frozen=True)
class ReconcileReport:
    recovered_leases: int
    timed_out_sends: int
    expired: int
    failed: int


def _backoff(policy: OutboxPolicy, attempt_count: int) -> timedelta:
    exponent = max(0, attempt_count - 1)
    seconds = min(
        policy.max_backoff.total_seconds(),
        policy.base_backoff.total_seconds() * (2**exponent),
    )
    return timedelta(seconds=seconds)


def _audit_row(
    connection: sqlite3.Connection,
    *,
    principal: str,
    action: str,
    target_type: str,
    target_identity: dict[str, Any],
    old_value: Any,
    new_value: Any,
    created_at_us: int,
) -> None:
    def encode(value: Any) -> str:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=lambda item: {
                "bytes_sha256": hashlib.sha256(bytes(item)).hexdigest(),
                "length": len(item),
            }
            if isinstance(item, (bytes, bytearray, memoryview))
            else str(item),
        )

    connection.execute(
        "INSERT INTO audit_log(principal,action,target_type,target_identity_json,"
        "old_value_json,new_value_json,created_at_us) VALUES(?,?,?,?,?,?,?)",
        (
            principal,
            action,
            target_type,
            encode(target_identity),
            None if old_value is None else encode(old_value),
            None if new_value is None else encode(new_value),
            created_at_us,
        ),
    )


class OutboxService:
    """Own durable delivery state; network I/O remains outside the transaction."""

    def __init__(
        self,
        writer: SQLiteWriter,
        *,
        clock: Callable[[], datetime] | None = None,
        policy: OutboxPolicy | None = None,
        sequence_allocator: TcSequenceAllocator | None = None,
        lease_owner: str = "gds-outbox-0",
        reconcile_on_startup: bool = True,
    ) -> None:
        if not isinstance(lease_owner, str) or not lease_owner:
            raise ValueError("lease_owner must not be empty")
        self.writer = writer
        self._clock = clock or (lambda: datetime.now(UTC))
        self.policy = policy or OutboxPolicy()
        self.sequence_allocator = sequence_allocator or TcSequenceAllocator(
            writer,
            clock_us=lambda: datetime_to_unix_us(self._now()),
        )
        self.lease_owner = lease_owner
        if not isinstance(reconcile_on_startup, bool):
            raise ValueError("reconcile_on_startup must be boolean")
        if reconcile_on_startup:
            # A process crash cannot leave DISPATCHING rows stranded until an
            # unrelated caller happens to poll the outbox.
            self.reconcile()

    def _now(self) -> datetime:
        value = self._clock()
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise ValueError("outbox clock must be timezone-aware")
        return value.astimezone(UTC)

    def register_instance(
        self,
        spacecraft_instance_id: int,
        *,
        link_generation: int = 0,
        link_session_id: int = 0,
        contact_state: ContactState = ContactState.NO_CONTACT,
    ) -> LinkBinding:
        instance = checked_u64(spacecraft_instance_id, "spacecraft_instance_id")
        generation = checked_u64(link_generation, "link_generation")
        session = checked_u64(link_session_id, "link_session_id")
        if not isinstance(contact_state, ContactState):
            contact_state = ContactState(contact_state)

        def mutation(connection: sqlite3.Connection) -> LinkBinding:
            now_us = datetime_to_unix_us(self._now())
            blob = encode_sqlite_u64(instance)
            existing = connection.execute(
                "SELECT link_generation,link_session_id,state FROM spacecraft_instances "
                "WHERE spacecraft_instance_id=?",
                (blob,),
            ).fetchone()
            if existing is not None and str(existing[2]) == "RETIRED":
                raise BindingGenerationError(
                    "retired spacecraft instance IDs cannot be reactivated"
                )
            if existing is not None and str(existing[2]) == "ACTIVE":
                old_generation = decode_sqlite_u64(existing[0], "link_generation")
                old_session = (
                    None
                    if existing[1] is None
                    else decode_sqlite_u64(existing[1], "link_session_id")
                )
                if generation < old_generation or (
                    generation == old_generation and old_session not in (None, session)
                ):
                    raise BindingGenerationError(
                        "link_generation must be monotonic and session changes require a new generation"
                    )
            connection.execute(
                "INSERT INTO spacecraft_instances("
                "spacecraft_instance_id,link_generation,link_session_id,state,"
                "first_seen_at_us,last_seen_at_us,contact_state,contact_changed_at_us) "
                "VALUES(?,?,?,'ACTIVE',?,?,?,?) "
                "ON CONFLICT(spacecraft_instance_id) DO UPDATE SET "
                "link_generation=excluded.link_generation,"
                "link_session_id=excluded.link_session_id,state='ACTIVE',"
                "last_seen_at_us=excluded.last_seen_at_us,"
                "contact_state=excluded.contact_state,"
                "contact_changed_at_us=excluded.contact_changed_at_us",
                (
                    blob,
                    encode_sqlite_u64(generation),
                    encode_sqlite_u64(session),
                    now_us,
                    now_us,
                    contact_state.value,
                    now_us,
                ),
            )
            return LinkBinding(instance, generation, session, contact_state)

        return self.writer.mutate(
            "register_spacecraft_instance",
            mutation,
            priority=MutationPriority.HIGH,
        )

    def binding(self, spacecraft_instance_id: int) -> LinkBinding | None:
        instance = checked_u64(spacecraft_instance_id, "spacecraft_instance_id")
        with self.writer.reader() as connection:
            row = connection.execute(
                "SELECT link_generation,link_session_id,contact_state,state "
                "FROM spacecraft_instances WHERE spacecraft_instance_id=?",
                (encode_sqlite_u64(instance),),
            ).fetchone()
            if row is None or str(row[3]) != "ACTIVE":
                return None
            if row[1] is None:
                return None
            return LinkBinding(
                instance,
                decode_sqlite_u64(row[0], "link_generation"),
                decode_sqlite_u64(row[1], "link_session_id"),
                ContactState(row[2]),
            )

    def set_contact_state(
        self,
        spacecraft_instance_id: int,
        contact_state: ContactState,
    ) -> LinkBinding:
        instance = checked_u64(spacecraft_instance_id, "spacecraft_instance_id")
        if not isinstance(contact_state, ContactState):
            contact_state = ContactState(contact_state)

        def mutation(connection: sqlite3.Connection) -> LinkBinding:
            now = self._now()
            now_us = datetime_to_unix_us(now)
            blob = encode_sqlite_u64(instance)
            row = connection.execute(
                "SELECT link_generation,link_session_id,contact_state,state "
                "FROM spacecraft_instances WHERE spacecraft_instance_id=?",
                (blob,),
            ).fetchone()
            if row is None or str(row[3]) != "ACTIVE" or row[1] is None:
                raise BindingUnavailableError("spacecraft instance is not bound")
            previous = ContactState(row[2])
            connection.execute(
                "UPDATE spacecraft_instances SET contact_state=?,"
                "contact_changed_at_us=?,last_seen_at_us=? "
                "WHERE spacecraft_instance_id=?",
                (contact_state.value, now_us, now_us, blob),
            )
            if previous is not contact_state:
                if contact_state.is_open:
                    connection.execute(
                        "UPDATE command_outbox SET state='OUTBOX_PENDING',"
                        "available_at_us=?,updated_at_us=? "
                        "WHERE target_spacecraft_instance_id=? "
                        "AND state='HELD_NO_CONTACT' AND expires_at_us>? "
                        "AND EXISTS (SELECT 1 FROM commands c WHERE "
                        "c.ground_instance_id=command_outbox.ground_instance_id "
                        "AND c.request_id=command_outbox.request_id "
                        "AND c.delivery_mode='next_contact')",
                        (now_us, now_us, blob, now_us),
                    )
                    connection.execute(
                        "UPDATE command_outbox SET ack_deadline_at_us=? ,updated_at_us=? "
                        "WHERE target_spacecraft_instance_id=? AND state='SENT' "
                        "AND ack_deadline_at_us IS NULL AND expires_at_us>? "
                        "AND EXISTS (SELECT 1 FROM commands c WHERE "
                        "c.ground_instance_id=command_outbox.ground_instance_id "
                        "AND c.request_id=command_outbox.request_id "
                        "AND c.delivery_mode='next_contact')",
                        (
                            now_us + int(self.policy.ack_timeout.total_seconds() * 1_000_000),
                            now_us,
                            blob,
                            now_us,
                        ),
                    )
                else:
                    # next_contact pauses at the contact boundary; immediate delivery
                    # is tied to the current contact episode and fails closed.
                    connection.execute(
                        "UPDATE command_outbox SET state='HELD_NO_CONTACT',"
                        "lease_owner=NULL,lease_expires_at_us=NULL,"
                        "ack_deadline_at_us=NULL,updated_at_us=? "
                        "WHERE target_spacecraft_instance_id=? "
                        "AND state IN ('OUTBOX_PENDING','DISPATCHING') "
                        "AND expires_at_us>? AND EXISTS (SELECT 1 FROM commands c WHERE "
                        "c.ground_instance_id=command_outbox.ground_instance_id "
                        "AND c.request_id=command_outbox.request_id "
                        "AND c.delivery_mode='next_contact')",
                        (now_us, blob, now_us),
                    )
                    connection.execute(
                        "UPDATE command_outbox SET ack_deadline_at_us=NULL,"
                        "updated_at_us=? WHERE target_spacecraft_instance_id=? "
                        "AND state='SENT' AND expires_at_us>? AND EXISTS ("
                        "SELECT 1 FROM commands c WHERE "
                        "c.ground_instance_id=command_outbox.ground_instance_id "
                        "AND c.request_id=command_outbox.request_id "
                        "AND c.delivery_mode='next_contact')",
                        (now_us, blob, now_us),
                    )
                    connection.execute(
                        "UPDATE command_outbox SET state='DELIVERY_FAILED',"
                        "last_error_code='CONTACT_LOST',last_delivery_reason='CONTACT_LOST',"
                        "lease_owner=NULL,lease_expires_at_us=NULL,"
                        "ack_deadline_at_us=NULL,updated_at_us=? "
                        "WHERE target_spacecraft_instance_id=? "
                        "AND state IN ('HELD_NO_CONTACT','OUTBOX_PENDING','DISPATCHING','SENT') "
                        "AND expires_at_us>? AND EXISTS (SELECT 1 FROM commands c WHERE "
                        "c.ground_instance_id=command_outbox.ground_instance_id "
                        "AND c.request_id=command_outbox.request_id "
                        "AND c.delivery_mode='immediate')",
                        (now_us, blob, now_us),
                    )
                    connection.execute(
                        "UPDATE commands SET command_state='FAILED',"
                        "terminal_at_us=MAX(terminal_at_us,?),updated_at_us=MAX(updated_at_us,?) "
                        "WHERE target_spacecraft_instance_id=? AND command_state='ADMITTED' "
                        "AND EXISTS (SELECT 1 FROM command_outbox o WHERE "
                        "o.ground_instance_id=commands.ground_instance_id "
                        "AND o.request_id=commands.request_id "
                        "AND o.state='DELIVERY_FAILED' AND o.last_error_code='CONTACT_LOST')",
                        (now_us, now_us, blob),
                    )
            return LinkBinding(
                instance,
                decode_sqlite_u64(row[0], "link_generation"),
                decode_sqlite_u64(row[1], "link_session_id"),
                contact_state,
            )

        return self.writer.mutate(
            "set_contact_state",
            mutation,
            priority=MutationPriority.HIGH,
        )

    def reconcile(self) -> ReconcileReport:
        def mutation(connection: sqlite3.Connection) -> ReconcileReport:
            now_us = datetime_to_unix_us(self._now())
            recovered = timed_out = expired = failed = 0
            rows = connection.execute(
                "SELECT o.ground_instance_id,o.request_id,o.state,o.attempt_count,"
                "o.lease_expires_at_us,o.ack_deadline_at_us,o.expires_at_us,"
                "o.available_at_us,c.delivery_mode,c.target_spacecraft_instance_id,"
                "s.contact_state "
                "FROM command_outbox o JOIN commands c ON "
                "c.ground_instance_id=o.ground_instance_id AND c.request_id=o.request_id "
                "LEFT JOIN spacecraft_instances s ON "
                "s.spacecraft_instance_id=o.target_spacecraft_instance_id "
                "WHERE o.state IN ('HELD_NO_CONTACT','OUTBOX_PENDING','DISPATCHING','SENT')"
            ).fetchall()
            for row in rows:
                state = str(row[2])
                attempts = int(row[3])
                expires_at_us = int(row[6])
                delivery_mode = str(row[8])
                contact_open = row[10] == ContactState.CONTACT_OPEN.value
                key_params = (row[0], row[1])
                if expires_at_us <= now_us:
                    self._terminalize(
                        connection,
                        key_params,
                        state="EXPIRED",
                        reason="EXPIRED",
                        now_us=now_us,
                    )
                    expired += 1
                    continue
                if state == "DISPATCHING" and row[4] is not None and int(row[4]) <= now_us:
                    if delivery_mode == "next_contact" and not contact_open:
                        next_state = "HELD_NO_CONTACT"
                    elif not contact_open:
                        self._terminalize(
                            connection,
                            key_params,
                            state="DELIVERY_FAILED",
                            reason="CONTACT_LOST",
                            now_us=now_us,
                        )
                        failed += 1
                        continue
                    else:
                        next_state = "OUTBOX_PENDING"
                    connection.execute(
                        "UPDATE command_outbox SET state=?,lease_owner=NULL,"
                        "lease_expires_at_us=NULL,updated_at_us=? WHERE "
                        "ground_instance_id=? AND request_id=? AND state='DISPATCHING'",
                        (next_state, now_us, *key_params),
                    )
                    recovered += 1
                elif state == "SENT" and row[5] is not None and int(row[5]) <= now_us:
                    timed_out += 1
                    if attempts >= self.policy.max_attempts:
                        self._terminalize(
                            connection,
                            key_params,
                            state="DELIVERY_FAILED",
                            reason="MAX_ATTEMPTS",
                            now_us=now_us,
                        )
                        failed += 1
                        continue
                    next_state = (
                        "HELD_NO_CONTACT"
                        if delivery_mode == "next_contact" and not contact_open
                        else "OUTBOX_PENDING"
                    )
                    if delivery_mode == "immediate" and not contact_open:
                        self._terminalize(
                            connection,
                            key_params,
                            state="DELIVERY_FAILED",
                            reason="CONTACT_LOST",
                            now_us=now_us,
                        )
                        failed += 1
                        continue
                    next_at = now_us + int(
                        _backoff(self.policy, attempts).total_seconds() * 1_000_000
                    )
                    connection.execute(
                        "UPDATE command_outbox SET state=?,available_at_us=?,"
                        "lease_owner=NULL,lease_expires_at_us=NULL,"
                        "ack_deadline_at_us=NULL,updated_at_us=? WHERE "
                        "ground_instance_id=? AND request_id=? AND state='SENT'",
                        (next_state, next_at, now_us, *key_params),
                    )
            return ReconcileReport(recovered, timed_out, expired, failed)

        return self.writer.mutate(
            "reconcile_outbox",
            mutation,
            priority=MutationPriority.HIGH,
        )

    def _terminalize(
        self,
        connection: sqlite3.Connection,
        key_params: tuple[object, object],
        *,
        state: str,
        reason: str,
        now_us: int,
    ) -> None:
        connection.execute(
            "UPDATE command_outbox SET state=?,last_error_code=?,"
            "last_delivery_reason=?,lease_owner=NULL,lease_expires_at_us=NULL,"
            "ack_deadline_at_us=NULL,updated_at_us=MAX(updated_at_us,?) "
            "WHERE ground_instance_id=? AND request_id=? AND state NOT IN "
            "('ACKED','EXPIRED','DELIVERY_FAILED','CANCELED')",
            (state, reason, reason, now_us, *key_params),
        )
        command_state = "FAILED" if state != "ACKED" else "ACKED"
        connection.execute(
            "UPDATE commands SET command_state=?,terminal_at_us=MAX(terminal_at_us,?),"
            "updated_at_us=MAX(updated_at_us,?) WHERE ground_instance_id=? "
            "AND request_id=? AND command_state NOT IN "
            "('ACKED','REJECTED','EXECUTED','FAILED','CANCELED')",
            (command_state, now_us, now_us, *key_params),
        )

    def claim(
        self,
        request_key: RequestKey,
        *,
        binding: LinkBinding | None = None,
        lease_owner: str | None = None,
    ) -> OutboxLease | None:
        """Atomically claim one exact request without stealing another worker's work."""

        if not isinstance(request_key, RequestKey):
            raise TypeError("request_key must be a RequestKey")
        return self._claim(
            request_key=request_key,
            binding=binding,
            lease_owner=lease_owner,
        )

    def claim_next(
        self,
        *,
        binding: LinkBinding | None = None,
        lease_owner: str | None = None,
    ) -> OutboxLease | None:
        """Claim the oldest due item for a single durable dispatcher."""

        return self._claim(
            request_key=None,
            binding=binding,
            lease_owner=lease_owner,
        )

    def _claim(
        self,
        *,
        request_key: RequestKey | None,
        binding: LinkBinding | None = None,
        lease_owner: str | None = None,
    ) -> OutboxLease | None:
        self.reconcile()
        owner = lease_owner or self.lease_owner

        def mutation(connection: sqlite3.Connection) -> OutboxLease | None:
            now = self._now()
            now_us = datetime_to_unix_us(now)
            conditions = [
                "o.state='OUTBOX_PENDING'",
                "o.available_at_us<=?",
                "o.expires_at_us>?",
                "s.state='ACTIVE'",
                "s.contact_state='CONTACT_OPEN'",
                "s.link_generation IS NOT NULL",
                "s.link_session_id IS NOT NULL",
            ]
            params: list[object] = [now_us, now_us]
            if request_key is not None:
                conditions.extend(
                    [
                        "o.ground_instance_id=?",
                        "o.request_id=?",
                    ]
                )
                params.extend(
                    [
                        encode_sqlite_u64(request_key.ground_instance_id),
                        request_key.request_id,
                    ]
                )
            if binding is not None:
                conditions.extend(
                    [
                        "o.target_spacecraft_instance_id=?",
                        "s.link_generation=?",
                        "s.link_session_id=?",
                    ]
                )
                params.extend(
                    [
                        encode_sqlite_u64(binding.spacecraft_instance_id),
                        encode_sqlite_u64(binding.link_generation),
                        encode_sqlite_u64(binding.link_session_id),
                    ]
                )
            row = connection.execute(
                "SELECT o.*,c.opcode,c.mission_arguments,c.semantic_body_jcs,"
                "c.effective_expires_at_us,s.link_generation,s.link_session_id,"
                "s.contact_state FROM command_outbox o JOIN commands c ON "
                "c.ground_instance_id=o.ground_instance_id AND c.request_id=o.request_id "
                "JOIN spacecraft_instances s ON "
                "s.spacecraft_instance_id=o.target_spacecraft_instance_id WHERE "
                + " AND ".join(conditions)
                + " ORDER BY o.available_at_us,o.created_at_us LIMIT 1",
                params,
            ).fetchone()
            if row is None:
                return None
            target = decode_sqlite_u64(
                row["target_spacecraft_instance_id"],
                "target_spacecraft_instance_id",
            )
            generation = decode_sqlite_u64(row["link_generation"], "link_generation")
            session = decode_sqlite_u64(row["link_session_id"], "link_session_id")
            lease_expires_us = now_us + int(
                self.policy.lease_duration.total_seconds() * 1_000_000
            )
            cursor = connection.execute(
                "UPDATE command_outbox SET state='DISPATCHING',lease_owner=?,"
                "lease_expires_at_us=?,updated_at_us=? WHERE ground_instance_id=? "
                "AND request_id=? AND state='OUTBOX_PENDING'",
                (owner, lease_expires_us, now_us, row["ground_instance_id"], row["request_id"]),
            )
            if cursor.rowcount != 1:
                raise LeaseLostError("outbox row changed during claim")
            body = json.loads(bytes(row["semantic_body_jcs"]).decode("utf-8"))
            command = Command(
                CommandOpcode(int(row["opcode"])),
                target,
                RequestKey(
                    decode_sqlite_u64(row["ground_instance_id"], "ground_instance_id"),
                    int(row["request_id"]),
                ),
                body["payload"],
            )
            return OutboxLease(
                request_key=command.request_key,
                target_spacecraft_instance_id=target,
                lease_owner=owner,
                lease_expires_at=unix_us_to_datetime(lease_expires_us),
                effective_expires_at=unix_us_to_datetime(int(row["expires_at_us"])),
                attempt_count=int(row["attempt_count"]),
                opcode=command.opcode,
                mission_arguments=bytes(row["mission_arguments"]),
                command=command,
                binding=LinkBinding(
                    target,
                    generation,
                    session,
                    ContactState(row["contact_state"]),
                ),
            )

        return self.writer.mutate(
            "claim_outbox_lease",
            mutation,
            priority=MutationPriority.HIGH,
        )

    def persist_attempt(
        self,
        lease: OutboxLease,
        encoded_tc: bytes,
        *,
        apid: int = 0,
        frame_sequence: int = 0,
        binding: LinkBinding | None = None,
    ) -> AttemptRecord:
        encoded_tc = bytes(encoded_tc)
        if not encoded_tc:
            raise ValueError("encoded_tc must not be empty")
        selected_binding = binding or lease.binding
        if selected_binding != lease.binding:
            raise LeaseLostError("attempt binding differs from the claimed link snapshot")

        def mutation(connection: sqlite3.Connection) -> AttemptRecord:
            now = self._now()
            now_us = datetime_to_unix_us(now)
            key_params = (
                encode_sqlite_u64(lease.request_key.ground_instance_id),
                lease.request_key.request_id,
            )
            row = connection.execute(
                "SELECT o.attempt_count,o.state,o.lease_owner,o.lease_expires_at_us,"
                "o.target_spacecraft_instance_id,c.effective_expires_at_us,"
                "s.state,s.link_generation,s.link_session_id,s.contact_state "
                "FROM command_outbox o JOIN commands c ON "
                "c.ground_instance_id=o.ground_instance_id AND c.request_id=o.request_id "
                "JOIN spacecraft_instances s ON "
                "s.spacecraft_instance_id=o.target_spacecraft_instance_id "
                "WHERE o.ground_instance_id=? AND o.request_id=?",
                key_params,
            ).fetchone()
            if row is None or str(row[1]) != "DISPATCHING" or row[2] != lease.lease_owner:
                raise LeaseLostError("outbox lease is no longer owned")
            if row[3] is None or int(row[3]) <= now_us:
                raise LeaseLostError("outbox lease has expired")
            if int(row[0]) >= self.policy.max_attempts:
                raise AttemptLimitError("maximum outbox attempts reached")
            if decode_sqlite_u64(row[4], "target_spacecraft_instance_id") != selected_binding.spacecraft_instance_id:
                raise LeaseLostError("attempt binding target does not match command target")
            if (
                str(row[6]) != "ACTIVE"
                or decode_sqlite_u64(row[7], "link_generation")
                != selected_binding.link_generation
                or decode_sqlite_u64(row[8], "link_session_id")
                != selected_binding.link_session_id
                or str(row[9]) != ContactState.CONTACT_OPEN.value
            ):
                raise LeaseLostError("link binding is no longer active for this attempt")
            sequence = self.sequence_allocator.allocate_in_transaction(
                connection, selected_binding.spacecraft_instance_id, apid
            )
            attempt_number = int(row[0]) + 1
            cursor = connection.execute(
                "INSERT INTO command_attempts("
                "ground_instance_id,request_id,target_spacecraft_instance_id,attempt_number,"
                "link_generation,link_session_id,apid,packet_sequence,frame_sequence,"
                "encoded_tc,encoded_tc_sha256,send_result,created_at_us,sequence_epoch) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    key_params[0],
                    key_params[1],
                    encode_sqlite_u64(selected_binding.spacecraft_instance_id),
                    attempt_number,
                    encode_sqlite_u64(selected_binding.link_generation),
                    encode_sqlite_u64(selected_binding.link_session_id),
                    apid,
                    sequence.sequence,
                    frame_sequence,
                    encoded_tc,
                    hashlib.sha256(encoded_tc).digest(),
                    "PERSISTED_NOT_SENT",
                    now_us,
                    sequence.sequence_epoch,
                ),
            )
            connection.execute(
                "UPDATE command_outbox SET attempt_count=?,updated_at_us=? "
                "WHERE ground_instance_id=? AND request_id=? AND state='DISPATCHING' "
                "AND lease_owner=?",
                (attempt_number, now_us, *key_params, lease.lease_owner),
            )
            return AttemptRecord(
                int(cursor.lastrowid),
                lease.request_key,
                selected_binding.spacecraft_instance_id,
                attempt_number,
                apid,
                sequence.sequence,
                sequence.sequence_epoch,
                sequence.rollover,
                selected_binding.link_generation,
                selected_binding.link_session_id,
                encoded_tc,
                now,
                frame_sequence=frame_sequence,
                encoded_tc_sha256=hashlib.sha256(encoded_tc).hexdigest(),
            )

        return self.writer.mutate(
            "persist_command_attempt",
            mutation,
            priority=MutationPriority.HIGH,
        )

    def prepare_attempt(
        self,
        lease: OutboxLease,
        *,
        profile: TcWireProfile,
        binding: LinkBinding | None = None,
    ) -> AttemptRecord:
        """Allocate, encode, hash, and persist an exact TC attempt atomically.

        The transport must send ``AttemptRecord.encoded_tc``.  In particular,
        it must not re-encode the command after this transaction because that
        could detach the bytes on the wire from the packet sequence and profile
        identity retained for audit/replay.
        """

        if not isinstance(lease, OutboxLease):
            raise TypeError("lease must be an OutboxLease")
        if not isinstance(profile, TcWireProfile):
            raise TypeError("profile must be a TcWireProfile")
        selected_binding = binding or lease.binding
        if selected_binding != lease.binding:
            raise LeaseLostError("attempt binding differs from the claimed link snapshot")

        def mutation(connection: sqlite3.Connection) -> AttemptRecord:
            now = self._now()
            now_us = datetime_to_unix_us(now)
            key_params = (
                encode_sqlite_u64(lease.request_key.ground_instance_id),
                lease.request_key.request_id,
            )
            row = connection.execute(
                "SELECT o.attempt_count,o.state,o.lease_owner,o.lease_expires_at_us,"
                "o.target_spacecraft_instance_id,s.state,s.link_generation,"
                "s.link_session_id,s.contact_state "
                "FROM command_outbox o JOIN spacecraft_instances s ON "
                "s.spacecraft_instance_id=o.target_spacecraft_instance_id "
                "WHERE o.ground_instance_id=? AND o.request_id=?",
                key_params,
            ).fetchone()
            if row is None or str(row[1]) != "DISPATCHING" or row[2] != lease.lease_owner:
                raise LeaseLostError("outbox lease is no longer owned")
            if row[3] is None or int(row[3]) <= now_us:
                raise LeaseLostError("outbox lease has expired")
            if int(row[0]) >= self.policy.max_attempts:
                raise AttemptLimitError("maximum outbox attempts reached")
            if decode_sqlite_u64(row[4], "target_spacecraft_instance_id") != selected_binding.spacecraft_instance_id:
                raise LeaseLostError("attempt binding target does not match command target")
            if (
                str(row[5]) != "ACTIVE"
                or decode_sqlite_u64(row[6], "link_generation")
                != selected_binding.link_generation
                or decode_sqlite_u64(row[7], "link_session_id")
                != selected_binding.link_session_id
                or str(row[8]) != ContactState.CONTACT_OPEN.value
            ):
                raise LeaseLostError("link binding is no longer active for this attempt")

            sequence = self.sequence_allocator.allocate_in_transaction(
                connection,
                selected_binding.spacecraft_instance_id,
                profile.tc_apid,
            )
            frame_sequence = sequence.sequence & 0xFF
            encoded_tc = profile.encode(
                lease.command,
                packet_sequence=sequence.sequence,
                frame_sequence=frame_sequence,
            )
            encoded_sha256 = hashlib.sha256(encoded_tc).digest()
            attempt_number = int(row[0]) + 1
            cursor = connection.execute(
                "INSERT INTO command_attempts("
                "ground_instance_id,request_id,target_spacecraft_instance_id,attempt_number,"
                "link_generation,link_session_id,apid,packet_sequence,frame_sequence,"
                "encoded_tc,encoded_tc_sha256,send_result,created_at_us,sequence_epoch,"
                "tc_profile_id,tc_profile_sha256,space_packet_type,"
                "space_packet_sequence_flags) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    key_params[0],
                    key_params[1],
                    encode_sqlite_u64(selected_binding.spacecraft_instance_id),
                    attempt_number,
                    encode_sqlite_u64(selected_binding.link_generation),
                    encode_sqlite_u64(selected_binding.link_session_id),
                    profile.tc_apid,
                    sequence.sequence,
                    frame_sequence,
                    encoded_tc,
                    encoded_sha256,
                    "PERSISTED_NOT_SENT",
                    now_us,
                    sequence.sequence_epoch,
                    profile.profile_id,
                    profile.profile_sha256,
                    profile.packet_type,
                    profile.sequence_flags,
                ),
            )
            updated = connection.execute(
                "UPDATE command_outbox SET attempt_count=?,updated_at_us=? "
                "WHERE ground_instance_id=? AND request_id=? AND state='DISPATCHING' "
                "AND lease_owner=? AND lease_expires_at_us>?",
                (
                    attempt_number,
                    now_us,
                    *key_params,
                    lease.lease_owner,
                    now_us,
                ),
            )
            if updated.rowcount != 1:
                raise LeaseLostError("outbox lease was lost while preparing attempt")
            return AttemptRecord(
                attempt_id=int(cursor.lastrowid),
                request_key=lease.request_key,
                target_spacecraft_instance_id=selected_binding.spacecraft_instance_id,
                attempt_number=attempt_number,
                apid=profile.tc_apid,
                packet_sequence=sequence.sequence,
                sequence_epoch=sequence.sequence_epoch,
                rollover=sequence.rollover,
                link_generation=selected_binding.link_generation,
                link_session_id=selected_binding.link_session_id,
                encoded_tc=encoded_tc,
                created_at=now,
                frame_sequence=frame_sequence,
                tc_profile_id=profile.profile_id,
                tc_profile_sha256=profile.profile_sha256,
                space_packet_type=profile.packet_type,
                space_packet_sequence_flags=profile.sequence_flags,
                encoded_tc_sha256=encoded_sha256.hex(),
            )

        return self.writer.mutate(
            "prepare_command_attempt",
            mutation,
            priority=MutationPriority.HIGH,
        )

    def mark_sent(self, lease: OutboxLease, attempt: AttemptRecord) -> None:
        def mutation(connection: sqlite3.Connection) -> None:
            now_us = datetime_to_unix_us(self._now())
            ack_deadline = now_us + int(
                self.policy.ack_timeout.total_seconds() * 1_000_000
            )
            cursor = connection.execute(
                "UPDATE command_outbox SET state='SENT',ack_deadline_at_us=?,"
                "lease_owner=NULL,lease_expires_at_us=NULL,updated_at_us=?,"
                "last_error_code=NULL,last_delivery_reason=NULL "
                "WHERE ground_instance_id=? AND request_id=? AND state='DISPATCHING' "
                "AND lease_owner=? AND lease_expires_at_us>?",
                (
                    ack_deadline,
                    now_us,
                    encode_sqlite_u64(lease.request_key.ground_instance_id),
                    lease.request_key.request_id,
                    lease.lease_owner,
                    now_us,
                ),
            )
            if cursor.rowcount != 1:
                raise LeaseLostError("cannot mark sent after lease was lost")
            attempt_cursor = connection.execute(
                "UPDATE command_attempts SET send_result='SENT',sent_at_us=? "
                "WHERE attempt_id=? AND ground_instance_id=? AND request_id=? "
                "AND attempt_number=? AND send_result='PERSISTED_NOT_SENT'",
                (
                    now_us,
                    attempt.attempt_id,
                    encode_sqlite_u64(attempt.request_key.ground_instance_id),
                    attempt.request_key.request_id,
                    attempt.attempt_number,
                ),
            )
            if attempt_cursor.rowcount != 1:
                raise LeaseLostError("attempt is not the persisted attempt for this lease")

        self.writer.mutate(
            "mark_outbox_sent",
            mutation,
            priority=MutationPriority.HIGH,
        )

    def mark_not_sent(
        self,
        lease: OutboxLease,
        attempt: AttemptRecord,
        *,
        reason: str,
    ) -> None:
        if not reason:
            raise ValueError("not-sent reason must not be empty")

        def mutation(connection: sqlite3.Connection) -> None:
            now_us = datetime_to_unix_us(self._now())
            attempt_cursor = connection.execute(
                "UPDATE command_attempts SET send_result=? WHERE attempt_id=? "
                "AND ground_instance_id=? AND request_id=? "
                "AND attempt_number=? AND send_result='PERSISTED_NOT_SENT'",
                (
                    f"NOT_SENT:{reason}",
                    attempt.attempt_id,
                    encode_sqlite_u64(attempt.request_key.ground_instance_id),
                    attempt.request_key.request_id,
                    attempt.attempt_number,
                ),
            )
            if attempt_cursor.rowcount != 1:
                raise LeaseLostError("attempt is not the persisted attempt for this lease")
            outbox_cursor = connection.execute(
                "UPDATE command_outbox SET state=CASE WHEN expires_at_us<=? "
                "THEN 'EXPIRED' ELSE 'OUTBOX_PENDING' END,"
                "available_at_us=?,lease_owner=NULL,lease_expires_at_us=NULL,"
                "ack_deadline_at_us=NULL,last_error_code=?,last_delivery_reason=?,"
                "updated_at_us=? "
                "WHERE ground_instance_id=? AND request_id=? AND state='DISPATCHING' "
                "AND lease_owner=? AND lease_expires_at_us>?",
                (
                    now_us,
                    now_us,
                    reason,
                    reason,
                    now_us,
                    encode_sqlite_u64(lease.request_key.ground_instance_id),
                    lease.request_key.request_id,
                    lease.lease_owner,
                    now_us,
                ),
            )
            if outbox_cursor.rowcount != 1:
                raise LeaseLostError("cannot complete not-sent attempt after lease loss")
            if self._outbox_state_is_expired(connection, lease.request_key):
                connection.execute(
                    "UPDATE commands SET command_state='FAILED',"
                    "terminal_at_us=MAX(terminal_at_us,?),"
                    "updated_at_us=MAX(updated_at_us,?) "
                    "WHERE ground_instance_id=? AND request_id=? "
                    "AND command_state='ADMITTED'",
                    (
                        now_us,
                        now_us,
                        encode_sqlite_u64(lease.request_key.ground_instance_id),
                        lease.request_key.request_id,
                    ),
                )

        self.writer.mutate(
            "mark_outbox_not_sent",
            mutation,
            priority=MutationPriority.HIGH,
        )

    def rollback_sent_attempt(
        self,
        lease: OutboxLease,
        attempt: AttemptRecord,
        *,
        reason: str,
    ) -> bool:
        """Requeue an armed attempt when the transport syscall reports failure.

        ``send_with_fence`` arms the correlation state before entering an
        asynchronous transport syscall.  A send error is therefore ambiguous:
        the frame may still have reached the peer.  This transition is guarded
        by both the outbox state and this exact attempt, so a concurrent TM ACK
        wins and is never rolled back into a retry.
        """

        if not isinstance(reason, str) or not reason:
            raise ValueError("send rollback reason must not be empty")

        def mutation(connection: sqlite3.Connection) -> bool:
            now_us = datetime_to_unix_us(self._now())
            key_params = (
                encode_sqlite_u64(lease.request_key.ground_instance_id),
                lease.request_key.request_id,
            )
            row = connection.execute(
                "SELECT o.state,o.attempt_count,o.expires_at_us,c.delivery_mode,"
                "s.contact_state FROM command_outbox o JOIN commands c ON "
                "c.ground_instance_id=o.ground_instance_id AND c.request_id=o.request_id "
                "LEFT JOIN spacecraft_instances s ON "
                "s.spacecraft_instance_id=o.target_spacecraft_instance_id "
                "WHERE o.ground_instance_id=? AND o.request_id=?",
                key_params,
            ).fetchone()
            if row is None or str(row["state"]) != "SENT":
                return False
            attempt_row = connection.execute(
                "SELECT 1 FROM command_attempts WHERE attempt_id=? "
                "AND ground_instance_id=? AND request_id=? AND attempt_number=? "
                "AND send_result='SENT'",
                (
                    attempt.attempt_id,
                    *key_params,
                    attempt.attempt_number,
                ),
            ).fetchone()
            if attempt_row is None:
                return False

            attempt_cursor = connection.execute(
                "UPDATE command_attempts SET send_result=? WHERE attempt_id=? "
                "AND ground_instance_id=? AND request_id=? AND attempt_number=? "
                "AND send_result='SENT'",
                (
                    f"NOT_SENT:{reason}",
                    attempt.attempt_id,
                    *key_params,
                    attempt.attempt_number,
                ),
            )
            if attempt_cursor.rowcount != 1:
                return False

            expires_at_us = int(row["expires_at_us"])
            delivery_mode = str(row["delivery_mode"])
            contact_open = row["contact_state"] == ContactState.CONTACT_OPEN.value
            if expires_at_us <= now_us:
                self._terminalize(
                    connection,
                    key_params,
                    state="EXPIRED",
                    reason="EXPIRED",
                    now_us=now_us,
                )
                return True
            if delivery_mode == "immediate" and not contact_open:
                self._terminalize(
                    connection,
                    key_params,
                    state="DELIVERY_FAILED",
                    reason="CONTACT_LOST",
                    now_us=now_us,
                )
                return True

            next_state = (
                "HELD_NO_CONTACT"
                if delivery_mode == "next_contact" and not contact_open
                else "OUTBOX_PENDING"
            )
            next_available_at_us = now_us + int(
                _backoff(self.policy, int(row["attempt_count"])).total_seconds()
                * 1_000_000
            )
            updated = connection.execute(
                "UPDATE command_outbox SET state=?,available_at_us=?,"
                "lease_owner=NULL,lease_expires_at_us=NULL,ack_deadline_at_us=NULL,"
                "last_error_code=?,last_delivery_reason=?,updated_at_us=? "
                "WHERE ground_instance_id=? AND request_id=? AND state='SENT'",
                (
                    next_state,
                    next_available_at_us,
                    reason,
                    reason,
                    now_us,
                    *key_params,
                ),
            )
            return updated.rowcount == 1

        return bool(
            self.writer.mutate(
                "rollback_sent_command_attempt",
                mutation,
                priority=MutationPriority.HIGH,
            )
        )

    def mark_attempt_not_sent(self, attempt: AttemptRecord, *, reason: str) -> bool:
        if not isinstance(reason, str) or not reason:
            raise ValueError("not-sent reason must not be empty")
        return bool(
            self.writer.mutate(
                "mark_attempt_not_sent",
                lambda connection: connection.execute(
                    "UPDATE command_attempts SET send_result=? WHERE attempt_id=? "
                    "AND ground_instance_id=? AND request_id=? "
                    "AND attempt_number=? AND send_result='PERSISTED_NOT_SENT'",
                    (
                        f"NOT_SENT:{reason}",
                        attempt.attempt_id,
                        encode_sqlite_u64(attempt.request_key.ground_instance_id),
                        attempt.request_key.request_id,
                        attempt.attempt_number,
                    ),
                ).rowcount
                == 1,
                priority=MutationPriority.HIGH,
            )
        )

    def send_with_fence(
        self,
        lease: OutboxLease,
        attempt: AttemptRecord,
        *,
        fence: Any,
        send: Callable[[bytes], object],
    ) -> object:
        """Hold the binding read fence through the transport send syscall."""

        if not callable(send):
            raise TypeError("send must be callable")
        try:
            with fence.read(lease.binding):
                # Correlation must be durable before UDP can deliver an ACK on
                # another receiver thread.  A later send failure rolls this
                # exact armed attempt back only if an ACK has not won the race.
                self.mark_sent(lease, attempt)
                try:
                    result = send(attempt.encoded_tc)
                except Exception:
                    self.rollback_sent_attempt(lease, attempt, reason="SEND_ERROR")
                    raise
                return result
        except Exception as exc:
            if getattr(exc, "error_code", None) == "NOT_SENT_REBIND":
                try:
                    self.mark_not_sent(lease, attempt, reason="REBIND")
                except LeaseLostError:
                    self.mark_attempt_not_sent(attempt, reason="REBIND")
            raise

    @staticmethod
    def _outbox_state_is_expired(
        connection: sqlite3.Connection, request_key: RequestKey
    ) -> bool:
        row = connection.execute(
            "SELECT state FROM command_outbox WHERE ground_instance_id=? "
            "AND request_id=?",
            (
                encode_sqlite_u64(request_key.ground_instance_id),
                request_key.request_id,
            ),
        ).fetchone()
        return row is not None and str(row[0]) == "EXPIRED"

    def ingest_ack(
        self,
        request_key: RequestKey,
        *,
        success: bool = True,
        reason: str | None = None,
    ) -> AckResult:
        """Compatibility ingress for an already-correlated command receipt."""

        def mutation(connection: sqlite3.Connection) -> AckResult:
            return self._ingest_ack_in_transaction(
                connection,
                request_key,
                success=success,
                reason=reason,
            )

        return self.writer.mutate(
            "ingest_command_ack",
            mutation,
            priority=MutationPriority.HIGH,
        )

    def ingest_correlated_tm(
        self,
        request_key: RequestKey,
        *,
        source_spacecraft_instance_id: int,
        link_generation: int,
        link_session_id: int,
        success: bool,
        reason: str | None = None,
        result: Mapping[str, Any] | None = None,
    ) -> AckResult:
        """Advance a command only when APID 2 identity matches a sent attempt."""

        if not isinstance(request_key, RequestKey):
            raise TypeError("request_key must be a RequestKey")
        source = checked_u64(source_spacecraft_instance_id, "source_spacecraft_instance_id")
        generation = checked_u64(link_generation, "link_generation")
        session = checked_u64(link_session_id, "link_session_id")
        if not isinstance(success, bool):
            raise TypeError("success must be boolean")
        if result is not None and not isinstance(result, Mapping):
            raise TypeError("result must be a mapping when provided")

        def mutation(connection: sqlite3.Connection) -> AckResult:
            return self._ingest_ack_in_transaction(
                connection,
                request_key,
                success=success,
                reason=reason,
                correlation={
                    "source_spacecraft_instance_id": source,
                    "link_generation": generation,
                    "link_session_id": session,
                    "result": None if result is None else dict(result),
                },
            )

        return self.writer.mutate(
            "ingest_correlated_tm_ack",
            mutation,
            priority=MutationPriority.HIGH,
        )

    def _ingest_ack_in_transaction(
        self,
        connection: sqlite3.Connection,
        request_key: RequestKey,
        *,
        success: bool,
        reason: str | None,
        correlation: Mapping[str, Any] | None = None,
    ) -> AckResult:
        now_us = datetime_to_unix_us(self._now())
        params = (
            encode_sqlite_u64(request_key.ground_instance_id),
            request_key.request_id,
        )
        row = connection.execute(
            "SELECT o.state,o.target_spacecraft_instance_id,o.last_delivery_reason "
            "FROM command_outbox o WHERE o.ground_instance_id=? AND o.request_id=?",
            params,
        ).fetchone()
        if row is None:
            raise KeyError("outbox request does not exist")
        state = str(row[0])
        target = decode_sqlite_u64(row[1], "target_spacecraft_instance_id")
        if state in {"ACKED", "EXPIRED", "DELIVERY_FAILED", "CANCELED"}:
            _audit_row(
                connection,
                principal="link",
                action="LATE_RECEIPT",
                target_type="command",
                target_identity={
                    "request_key": request_key.as_dict(),
                    "target_spacecraft_instance_id": u64_to_json(target),
                },
                old_value={"outbox_state": state},
                new_value={
                    "ack_success": success,
                    "reason": reason,
                    "correlation": None if correlation is None else dict(correlation),
                },
                created_at_us=now_us,
            )
            return AckResult(request_key, state, True, reason)

        if correlation is not None:
            source = int(correlation["source_spacecraft_instance_id"])
            generation = int(correlation["link_generation"])
            session = int(correlation["link_session_id"])
            sent_attempt = connection.execute(
                "SELECT 1 FROM command_attempts WHERE ground_instance_id=? "
                "AND request_id=? AND target_spacecraft_instance_id=? "
                "AND link_generation=? AND link_session_id=? AND send_result='SENT' "
                "LIMIT 1",
                (
                    *params,
                    encode_sqlite_u64(source),
                    encode_sqlite_u64(generation),
                    encode_sqlite_u64(session),
                ),
            ).fetchone()
            if target != source or state != "SENT" or sent_attempt is None:
                mismatch_reason = "UNCORRELATED_TM_ACK"
                _audit_row(
                    connection,
                    principal="link",
                    action="TM_ACK_UNCORRELATED",
                    target_type="command",
                    target_identity={
                        "request_key": request_key.as_dict(),
                        "target_spacecraft_instance_id": u64_to_json(target),
                    },
                    old_value={"outbox_state": state},
                    new_value={
                        "reason": mismatch_reason,
                        "source_spacecraft_instance_id": u64_to_json(source),
                        "link_generation": u64_to_json(generation),
                        "link_session_id": u64_to_json(session),
                        "result": correlation.get("result"),
                    },
                    created_at_us=now_us,
                )
                return AckResult(request_key, state, True, mismatch_reason)

        if not success:
            self._terminalize(
                connection,
                params,
                state="DELIVERY_FAILED",
                reason=reason or "NACK",
                now_us=now_us,
            )
            return AckResult(request_key, "DELIVERY_FAILED", False, reason or "NACK")
        connection.execute(
            "UPDATE command_outbox SET state='ACKED',ack_deadline_at_us=NULL,"
            "lease_owner=NULL,lease_expires_at_us=NULL,last_error_code=NULL,"
            "last_delivery_reason=NULL,"
            "updated_at_us=? WHERE ground_instance_id=? AND request_id=?",
            (now_us, *params),
        )
        connection.execute(
            "UPDATE commands SET command_state='ACKED',terminal_at_us=MAX(terminal_at_us,?),"
            "updated_at_us=MAX(updated_at_us,?) WHERE ground_instance_id=? AND request_id=?",
            (now_us, now_us, *params),
        )
        connection.execute(
            "UPDATE command_attempts SET send_result='ACKED',acked_at_us=? "
            "WHERE ground_instance_id=? AND request_id=? AND send_result='SENT'",
            (now_us, *params),
        )
        return AckResult(request_key, "ACKED", False, None)
