"""Atomic GDS command ledger and transactional outbox admission."""

from __future__ import annotations

import hmac
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Callable, Mapping

from protocol.canonical import checked_u64, u64_to_json
from protocol.schemas import (
    Command,
    CommandOpcode,
    ProductRef,
    RequestKey,
    mission_digest,
)

from .idempotency import (
    IdempotencyValidationError,
    SemanticIdempotency,
    build_semantic_idempotency,
    datetime_to_unix_us,
    format_rfc3339_utc,
    materialize_effective_expiry,
    unix_us_to_datetime,
    validate_idempotency_key,
)
from .audit import append_audit_in_transaction
from .request_keys import RequestKeyAllocator
from .u64 import decode_sqlite_u64, encode_sqlite_u64
from .writer import MutationPriority, SQLiteWriter

DEFAULT_PRINCIPAL = "local-operator"
PRODUCT_DOWNLINK_PRINCIPAL = "gds-product-downlink"
DEFAULT_OUTBOX_CAPACITY = 1_024
HTTP_IDEMPOTENCY_RETENTION = timedelta(days=90)
PENDING_FILE_OBSERVATION_TTL = timedelta(days=1)
_EXPIRY_OMITTED = object()

NONTERMINAL_OUTBOX_STATES = (
    "HELD_NO_CONTACT",
    "OUTBOX_PENDING",
    "DISPATCHING",
    "SENT",
)
TERMINAL_OUTBOX_STATES = ("ACKED", "EXPIRED", "DELIVERY_FAILED", "CANCELED")
TERMINAL_COMMAND_STATES = ("ACKED", "REJECTED", "EXECUTED", "FAILED", "CANCELED")

_PRODUCT_DOWNLINK_STATE_RANK = {
    "ADMITTED": 0,
    "DISPATCHED": 1,
    "RECEIVING": 2,
    "VERIFIED": 3,
    "FAILED": 3,
    "CANCELED": 3,
}


class LedgerError(RuntimeError):
    error_code = "LEDGER_ERROR"
    status_code = 500


class LedgerIntegrityError(LedgerError):
    error_code = "LEDGER_INTEGRITY_ERROR"


class AdmissionError(LedgerError):
    status_code = 409

    def __init__(self, message: str, *, request_key: RequestKey | None = None):
        super().__init__(message)
        self.request_key = request_key


class IdempotencyConflictError(AdmissionError):
    error_code = "IDEMPOTENCY_CONFLICT"


class IdempotencyKeyRetiredError(AdmissionError):
    error_code = "IDEMPOTENCY_KEY_RETIRED"
    status_code = 410


class NoContactError(AdmissionError):
    error_code = "NO_CONTACT"


class TargetRetiredError(AdmissionError):
    error_code = "TARGET_INSTANCE_RETIRED"
    status_code = 410


class OutboxCapacityError(AdmissionError):
    error_code = "QUEUE_FULL"
    status_code = 429
    retry_after_seconds = 1


@dataclass(frozen=True)
class AdmissionResult:
    request_key: RequestKey
    target_spacecraft_instance_id: int
    effective_expires_at: datetime
    command_state: str
    outbox_state: str
    mission_digest_hex: str
    http_digest_hex: str
    accepted_at: datetime
    replayed: bool

    @property
    def status_code(self) -> int:
        return 202

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "accepted",
            "request_key": self.request_key.as_dict(),
            "target_spacecraft_instance_id": u64_to_json(
                self.target_spacecraft_instance_id
            ),
            "effective_expires_at": format_rfc3339_utc(
                self.effective_expires_at
            ),
            "command_state": self.command_state,
            "outbox_state": self.outbox_state,
            "mission_digest": self.mission_digest_hex,
            "http_idempotency_digest": self.http_digest_hex,
            "accepted_at": format_rfc3339_utc(self.accepted_at),
            "replayed": self.replayed,
        }


class AtomicCommandLedger:
    """Serialize idempotency lookup, identity allocation, and outbox admission."""

    def __init__(
        self,
        writer: SQLiteWriter,
        *,
        allocator: RequestKeyAllocator | None = None,
        clock: Callable[[], datetime] | None = None,
        outbox_capacity: int = DEFAULT_OUTBOX_CAPACITY,
        fault_injector: Callable[[str], None] | None = None,
    ) -> None:
        if outbox_capacity <= 0:
            raise ValueError("outbox_capacity must be positive")
        self.writer = writer
        self._clock = clock or (lambda: datetime.now(UTC))
        self.outbox_capacity = outbox_capacity
        self._fault_injector = fault_injector
        self.allocator = allocator or RequestKeyAllocator(
            writer,
            clock_us=lambda: datetime_to_unix_us(self._now()),
        )
        self.allocator.initialize()

    def _now(self) -> datetime:
        value = self._clock()
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise IdempotencyValidationError("ledger clock must be timezone-aware")
        return value.astimezone(UTC)

    @property
    def gds_installation_epoch(self) -> int:
        return self.allocator.state().gds_installation_epoch

    def _validate_principal(self, principal: object) -> str:
        if not isinstance(principal, str) or not 1 <= len(principal) <= 255:
            raise IdempotencyValidationError(
                "principal must contain 1..255 characters"
            )
        if any(ord(char) < 0x20 for char in principal):
            raise IdempotencyValidationError("principal must not contain controls")
        return principal

    def _semantic_request(
        self,
        *,
        target_spacecraft_instance_id: int,
        opcode: CommandOpcode,
        payload: Mapping[str, Any],
        delivery_mode: str,
        expires_at: object,
    ) -> SemanticIdempotency:
        body: dict[str, Any] = {
            "target_spacecraft_instance_id": u64_to_json(
                checked_u64(
                    target_spacecraft_instance_id,
                    "target_spacecraft_instance_id",
                )
            ),
            "opcode": int(opcode),
            "payload": dict(payload),
            "delivery_mode": delivery_mode,
        }
        if expires_at is not _EXPIRY_OMITTED:
            body["expires_at"] = expires_at
        semantic = build_semantic_idempotency(body)
        # Use the frozen JCS round-trip payload for mission validation/admission.
        frozen_payload = semantic.normalized_body["payload"]
        Command(
            opcode,
            target_spacecraft_instance_id,
            RequestKey(0, 0),
            frozen_payload,
        ).argument_bytes()
        return semantic

    def admit(
        self,
        *,
        idempotency_key: str,
        target_spacecraft_instance_id: int,
        opcode: CommandOpcode | int,
        payload: Mapping[str, Any],
        principal: str = DEFAULT_PRINCIPAL,
        delivery_mode: str = "immediate",
        expires_at: object = _EXPIRY_OMITTED,
        contact_available: bool = True,
        pre_admission_check: Callable[[], None] | None = None,
    ) -> AdmissionResult:
        idempotency_key = validate_idempotency_key(idempotency_key)
        principal = self._validate_principal(principal)
        if not isinstance(payload, Mapping):
            raise IdempotencyValidationError("command payload must be an object")
        if not isinstance(contact_available, bool):
            raise IdempotencyValidationError("contact_available must be boolean")
        if isinstance(opcode, bool) or not isinstance(opcode, int):
            raise IdempotencyValidationError("mission command opcode must be an integer")
        try:
            normalized_opcode = CommandOpcode(opcode)
        except (TypeError, ValueError) as exc:
            raise IdempotencyValidationError("unknown mission command opcode") from exc
        semantic = self._semantic_request(
            target_spacecraft_instance_id=target_spacecraft_instance_id,
            opcode=normalized_opcode,
            payload=payload,
            delivery_mode=delivery_mode,
            expires_at=expires_at,
        )
        frozen_payload = semantic.normalized_body["payload"]

        def mutation(connection: sqlite3.Connection) -> AdmissionResult:
            now = self._now()
            now_us = datetime_to_unix_us(now)
            metadata = connection.execute(
                "SELECT gds_installation_epoch FROM gds_metadata WHERE singleton=1"
            ).fetchone()
            if metadata is None:
                raise LedgerIntegrityError("gds_metadata is missing")
            installation_epoch_blob = bytes(metadata[0])
            decode_sqlite_u64(installation_epoch_blob, "gds_installation_epoch")

            existing = connection.execute(
                "SELECT c.*,o.state AS outbox_state "
                "FROM commands AS c LEFT JOIN command_outbox AS o "
                "ON o.ground_instance_id=c.ground_instance_id "
                "AND o.request_id=c.request_id "
                "WHERE c.gds_installation_epoch=? AND c.principal=? "
                "AND c.idempotency_key=?",
                (installation_epoch_blob, principal, idempotency_key),
            ).fetchone()
            if existing is not None:
                request_key = self._request_key_from_row(existing)
                if not hmac.compare_digest(bytes(existing["http_digest"]), semantic.digest):
                    raise IdempotencyConflictError(
                        "Idempotency-Key was already used with another semantic body",
                        request_key=request_key,
                    )
                if existing["outbox_state"] is None:
                    raise LedgerIntegrityError(
                        "admitted command has no transactional outbox row"
                    )
                return self._result_from_row(existing, replayed=True)

            retired = connection.execute(
                "SELECT * FROM http_idempotency_retired "
                "WHERE gds_installation_epoch=? AND principal=? AND idempotency_key=?",
                (installation_epoch_blob, principal, idempotency_key),
            ).fetchone()
            if retired is not None and int(retired["retained_until_us"]) <= now_us:
                connection.execute(
                    "DELETE FROM http_idempotency_retired "
                    "WHERE gds_installation_epoch=? AND principal=? AND idempotency_key=?",
                    (installation_epoch_blob, principal, idempotency_key),
                )
                retired = None
            if retired is not None:
                original = RequestKey(
                    decode_sqlite_u64(
                        retired["original_ground_instance_id"],
                        "original_ground_instance_id",
                    ),
                    int(retired["original_request_id"]),
                )
                if not hmac.compare_digest(bytes(retired["http_digest"]), semantic.digest):
                    raise IdempotencyConflictError(
                        "retained Idempotency-Key has another semantic digest",
                        request_key=original,
                    )
                raise IdempotencyKeyRetiredError(
                    "Idempotency-Key metadata is retained but its command was pruned",
                    request_key=original,
                )

            target_state = connection.execute(
                "SELECT state FROM spacecraft_instances WHERE spacecraft_instance_id=?",
                (encode_sqlite_u64(target_spacecraft_instance_id),),
            ).fetchone()
            if target_state is not None and str(target_state[0]) == "RETIRED":
                raise TargetRetiredError("target spacecraft instance is retired")
            if pre_admission_check is not None:
                if not callable(pre_admission_check):
                    raise IdempotencyValidationError(
                        "pre_admission_check must be callable"
                    )
                pre_admission_check()
            effective_expiry = materialize_effective_expiry(semantic, now)
            effective_expiry_us = datetime_to_unix_us(effective_expiry)
            if delivery_mode == "immediate" and not contact_available:
                raise NoContactError("immediate delivery requires an open contact")
            placeholders = ",".join("?" for _ in NONTERMINAL_OUTBOX_STATES)
            active_count = int(
                connection.execute(
                    f"SELECT count(*) FROM command_outbox WHERE state IN ({placeholders})",
                    NONTERMINAL_OUTBOX_STATES,
                ).fetchone()[0]
            )
            if active_count >= self.outbox_capacity:
                raise OutboxCapacityError(
                    f"outbox capacity {self.outbox_capacity} is exhausted"
                )

            request_key = self.allocator.allocate_in_transaction(connection)
            command = Command(
                normalized_opcode,
                target_spacecraft_instance_id,
                request_key,
                frozen_payload,
            )
            arguments = command.argument_bytes()
            mission_digest_bytes = bytes.fromhex(mission_digest(command))
            ground_blob = encode_sqlite_u64(request_key.ground_instance_id)
            target_blob = encode_sqlite_u64(target_spacecraft_instance_id)
            outbox_state = (
                "HELD_NO_CONTACT"
                if delivery_mode == "next_contact" and not contact_available
                else "OUTBOX_PENDING"
            )
            connection.execute(
                "INSERT INTO commands("
                "ground_instance_id,request_id,target_spacecraft_instance_id,"
                "gds_installation_epoch,principal,idempotency_key,http_digest,"
                "semantic_body_jcs,opcode,mission_arguments,mission_digest,delivery_mode,"
                "effective_expires_at_us,command_state,created_at_us,updated_at_us) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,'ADMITTED',?,?)",
                (
                    ground_blob,
                    request_key.request_id,
                    target_blob,
                    installation_epoch_blob,
                    principal,
                    idempotency_key,
                    semantic.digest,
                    semantic.canonical_jcs,
                    int(normalized_opcode),
                    arguments,
                    mission_digest_bytes,
                    delivery_mode,
                    effective_expiry_us,
                    now_us,
                    now_us,
                ),
            )
            if self._fault_injector is not None:
                self._fault_injector("after_command_insert")
            connection.execute(
                "INSERT INTO command_outbox("
                "ground_instance_id,request_id,target_spacecraft_instance_id,state,"
                "available_at_us,expires_at_us,created_at_us,updated_at_us) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (
                    ground_blob,
                    request_key.request_id,
                    target_blob,
                    outbox_state,
                    now_us,
                    effective_expiry_us,
                    now_us,
                    now_us,
                ),
            )
            if self._fault_injector is not None:
                self._fault_injector("after_outbox_insert")
            append_audit_in_transaction(
                connection,
                principal=principal,
                action="COMMAND_ADMITTED",
                target_type="command",
                target_identity={
                    "ground_instance_id": u64_to_json(request_key.ground_instance_id),
                    "request_id": request_key.request_id,
                },
                old_value=None,
                new_value={
                    "target_spacecraft_instance_id": u64_to_json(
                        target_spacecraft_instance_id
                    ),
                    "delivery_mode": delivery_mode,
                    "outbox_state": outbox_state,
                    "effective_expires_at_us": effective_expiry_us,
                },
                created_at_us=now_us,
            )
            return AdmissionResult(
                request_key=request_key,
                target_spacecraft_instance_id=target_spacecraft_instance_id,
                effective_expires_at=effective_expiry,
                command_state="ADMITTED",
                outbox_state=outbox_state,
                mission_digest_hex=mission_digest_bytes.hex(),
                http_digest_hex=semantic.digest_hex,
                accepted_at=now,
                replayed=False,
            )

        return self.writer.mutate(
            "atomic_command_outbox_admission",
            mutation,
            priority=MutationPriority.HIGH,
        )

    def admit_product_downlink(
        self,
        *,
        origin_request_key: RequestKey,
        product_ref: ProductRef,
        target_spacecraft_instance_id: int,
        delivery_mode: str = "next_contact",
        contact_available: bool = True,
        retry: bool = False,
    ) -> AdmissionResult:
        """Admit a product downlink as an independently durable command.

        Repeated completion TM for one product returns the existing admission.
        A caller may request a new ledger identity after that attempt is
        terminal by setting ``retry=True``; it is never derived from the
        originating request ID.
        """

        if not isinstance(origin_request_key, RequestKey):
            raise TypeError("origin_request_key must be a RequestKey")
        if not isinstance(product_ref, ProductRef):
            raise TypeError("product_ref must be a ProductRef")
        target = checked_u64(target_spacecraft_instance_id, "target_spacecraft_instance_id")
        if product_ref.spacecraft_instance_id != target:
            raise IdempotencyValidationError(
                "product_ref spacecraft_instance_id must match the downlink target"
            )
        if delivery_mode not in {"immediate", "next_contact"}:
            raise IdempotencyValidationError("delivery_mode is invalid")
        if not isinstance(contact_available, bool):
            raise IdempotencyValidationError("contact_available must be boolean")
        if not isinstance(retry, bool):
            raise IdempotencyValidationError("retry must be boolean")

        payload = {
            "origin_request_key": origin_request_key.as_dict(),
            "product_ref": product_ref.as_dict(),
        }
        semantic = self._semantic_request(
            target_spacecraft_instance_id=target,
            opcode=CommandOpcode.PRODUCT_REQUEST_DOWNLINK,
            payload=payload,
            delivery_mode=delivery_mode,
            expires_at=_EXPIRY_OMITTED,
        )
        frozen_payload = semantic.normalized_body["payload"]

        def mutation(connection: sqlite3.Connection) -> AdmissionResult:
            now = self._now()
            now_us = datetime_to_unix_us(now)
            identity_params = (
                encode_sqlite_u64(origin_request_key.ground_instance_id),
                origin_request_key.request_id,
                encode_sqlite_u64(product_ref.spacecraft_instance_id),
                product_ref.origin_boot_id,
                product_ref.product_id,
            )
            existing = connection.execute(
                "SELECT d.*,c.*,o.state AS outbox_state FROM product_downlink_ledger d "
                "JOIN commands c ON c.ground_instance_id=d.downlink_ground_instance_id "
                "AND c.request_id=d.downlink_request_id "
                "JOIN command_outbox o ON o.ground_instance_id=c.ground_instance_id "
                "AND o.request_id=c.request_id "
                "WHERE d.origin_ground_instance_id=? AND d.origin_request_id=? "
                "AND d.product_spacecraft_instance_id=? AND d.origin_boot_id=? "
                "AND d.product_id=? ORDER BY d.admission_ordinal DESC LIMIT 1",
                identity_params,
            ).fetchone()
            if existing is not None:
                current_state = str(existing["outbox_state"])
                if not retry or current_state not in TERMINAL_OUTBOX_STATES:
                    return self._result_from_row(existing, replayed=True)
                ordinal = int(existing["admission_ordinal"]) + 1
            else:
                ordinal = 1

            metadata = connection.execute(
                "SELECT gds_installation_epoch FROM gds_metadata WHERE singleton=1"
            ).fetchone()
            if metadata is None:
                raise LedgerIntegrityError("gds_metadata is missing")
            installation_epoch_blob = bytes(metadata[0])
            decode_sqlite_u64(installation_epoch_blob, "gds_installation_epoch")
            target_state = connection.execute(
                "SELECT state FROM spacecraft_instances WHERE spacecraft_instance_id=?",
                (encode_sqlite_u64(target),),
            ).fetchone()
            if target_state is not None and str(target_state[0]) == "RETIRED":
                raise TargetRetiredError("target spacecraft instance is retired")
            if delivery_mode == "immediate" and not contact_available:
                raise NoContactError("immediate delivery requires an open contact")
            placeholders = ",".join("?" for _ in NONTERMINAL_OUTBOX_STATES)
            active_count = int(
                connection.execute(
                    f"SELECT count(*) FROM command_outbox WHERE state IN ({placeholders})",
                    NONTERMINAL_OUTBOX_STATES,
                ).fetchone()[0]
            )
            if active_count >= self.outbox_capacity:
                raise OutboxCapacityError(
                    f"outbox capacity {self.outbox_capacity} is exhausted"
                )
            effective_expiry = materialize_effective_expiry(semantic, now)
            effective_expiry_us = datetime_to_unix_us(effective_expiry)
            request_key = self.allocator.allocate_in_transaction(connection)
            command = Command(
                CommandOpcode.PRODUCT_REQUEST_DOWNLINK,
                target,
                request_key,
                frozen_payload,
            )
            mission_digest_bytes = bytes.fromhex(mission_digest(command))
            ground_blob = encode_sqlite_u64(request_key.ground_instance_id)
            target_blob = encode_sqlite_u64(target)
            idempotency_key = (
                f"product-downlink:{origin_request_key.ground_instance_id:016x}:"
                f"{origin_request_key.request_id:08x}:{product_ref.origin_boot_id:08x}:"
                f"{product_ref.product_id:08x}:{ordinal}"
            )
            outbox_state = (
                "HELD_NO_CONTACT"
                if delivery_mode == "next_contact" and not contact_available
                else "OUTBOX_PENDING"
            )
            connection.execute(
                "INSERT INTO commands("
                "ground_instance_id,request_id,target_spacecraft_instance_id,"
                "gds_installation_epoch,principal,idempotency_key,http_digest,"
                "semantic_body_jcs,opcode,mission_arguments,mission_digest,delivery_mode,"
                "effective_expires_at_us,command_state,created_at_us,updated_at_us) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,'ADMITTED',?,?)",
                (
                    ground_blob,
                    request_key.request_id,
                    target_blob,
                    installation_epoch_blob,
                    PRODUCT_DOWNLINK_PRINCIPAL,
                    idempotency_key,
                    semantic.digest,
                    semantic.canonical_jcs,
                    int(CommandOpcode.PRODUCT_REQUEST_DOWNLINK),
                    command.argument_bytes(),
                    mission_digest_bytes,
                    delivery_mode,
                    effective_expiry_us,
                    now_us,
                    now_us,
                ),
            )
            connection.execute(
                "INSERT INTO command_outbox("
                "ground_instance_id,request_id,target_spacecraft_instance_id,state,"
                "available_at_us,expires_at_us,created_at_us,updated_at_us) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (
                    ground_blob,
                    request_key.request_id,
                    target_blob,
                    outbox_state,
                    now_us,
                    effective_expiry_us,
                    now_us,
                    now_us,
                ),
            )
            connection.execute(
                "INSERT INTO product_downlink_ledger("
                "downlink_ground_instance_id,downlink_request_id,origin_ground_instance_id,"
                "origin_request_id,product_spacecraft_instance_id,origin_boot_id,product_id,"
                "admission_ordinal,transfer_state,created_at_us,updated_at_us) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    ground_blob,
                    request_key.request_id,
                    encode_sqlite_u64(origin_request_key.ground_instance_id),
                    origin_request_key.request_id,
                    encode_sqlite_u64(product_ref.spacecraft_instance_id),
                    product_ref.origin_boot_id,
                    product_ref.product_id,
                    ordinal,
                    "ADMITTED",
                    now_us,
                    now_us,
                ),
            )
            append_audit_in_transaction(
                connection,
                principal=PRODUCT_DOWNLINK_PRINCIPAL,
                action="PRODUCT_DOWNLINK_ADMITTED",
                target_type="command",
                target_identity={
                    "ground_instance_id": u64_to_json(request_key.ground_instance_id),
                    "request_id": request_key.request_id,
                },
                old_value=None,
                new_value={
                    "origin_request_key": origin_request_key.as_dict(),
                    "product_ref": product_ref.as_dict(),
                    "admission_ordinal": ordinal,
                    "outbox_state": outbox_state,
                },
                created_at_us=now_us,
            )
            return AdmissionResult(
                request_key=request_key,
                target_spacecraft_instance_id=target,
                effective_expires_at=effective_expiry,
                command_state="ADMITTED",
                outbox_state=outbox_state,
                mission_digest_hex=mission_digest_bytes.hex(),
                http_digest_hex=semantic.digest_hex,
                accepted_at=now,
                replayed=False,
            )

        return self.writer.mutate(
            "admit_product_downlink",
            mutation,
            priority=MutationPriority.HIGH,
        )

    def update_product_downlink_transfer(
        self,
        request_key: RequestKey,
        *,
        transfer_id: int | None = None,
        transfer_state: str,
    ) -> bool:
        """Correlate APID 2/APID 3 transfer progress to its downlink command."""

        if not isinstance(request_key, RequestKey):
            raise TypeError("request_key must be a RequestKey")
        if transfer_id is not None and (isinstance(transfer_id, bool) or not 0 <= int(transfer_id) <= 0xFFFFFFFF):
            raise ValueError("transfer_id must fit U32")
        if transfer_state not in {"ADMITTED", "DISPATCHED", "RECEIVING", "VERIFIED", "FAILED", "CANCELED"}:
            raise ValueError("invalid product downlink transfer state")
        now_us = datetime_to_unix_us(self._now())

        def mutation(connection: sqlite3.Connection) -> bool:
            self._reconcile_pending_product_downlink_files_in_transaction(
                connection,
                now_us=now_us,
            )
            row = connection.execute(
                "SELECT transfer_id,transfer_state,product_spacecraft_instance_id,"
                "origin_boot_id,product_id FROM product_downlink_ledger "
                "WHERE downlink_ground_instance_id=? AND downlink_request_id=?",
                (encode_sqlite_u64(request_key.ground_instance_id), request_key.request_id),
            ).fetchone()
            if row is None:
                return False
            existing_transfer = row[0]
            if (
                existing_transfer is not None
                and transfer_id is not None
                and int(existing_transfer) != int(transfer_id)
            ):
                raise LedgerIntegrityError("downlink request was correlated with another transfer_id")
            normalized_transfer_id = (
                int(existing_transfer)
                if transfer_id is None
                else int(transfer_id)
            )
            if transfer_id is not None:
                owner = connection.execute(
                    "SELECT downlink_ground_instance_id,downlink_request_id "
                    "FROM product_downlink_ledger WHERE "
                    "product_spacecraft_instance_id=? AND transfer_id=?",
                    (row["product_spacecraft_instance_id"], int(transfer_id)),
                ).fetchone()
                if owner is not None and (
                    bytes(owner["downlink_ground_instance_id"])
                    != encode_sqlite_u64(request_key.ground_instance_id)
                    or int(owner["downlink_request_id"]) != request_key.request_id
                ):
                    raise LedgerIntegrityError(
                        "transfer_id is already correlated with another downlink request"
                    )
            next_state = self._monotonic_downlink_state(
                str(row["transfer_state"]),
                transfer_state,
            )
            pending_state: str | None = None
            if normalized_transfer_id is not None:
                pending = connection.execute(
                    "SELECT transfer_state FROM product_downlink_pending_files "
                    "WHERE product_spacecraft_instance_id=? AND origin_boot_id=? "
                    "AND product_id=? AND transfer_id=? AND expires_at_us>?",
                    (
                        row["product_spacecraft_instance_id"],
                        int(row["origin_boot_id"]),
                        int(row["product_id"]),
                        normalized_transfer_id,
                        now_us,
                    ),
                ).fetchone()
                if pending is not None:
                    pending_state = str(pending["transfer_state"])
                    next_state = self._monotonic_downlink_state(
                        next_state,
                        pending_state,
                    )
            connection.execute(
                "UPDATE product_downlink_ledger SET transfer_id=COALESCE(transfer_id,?),"
                "transfer_state=?,updated_at_us=? WHERE downlink_ground_instance_id=? "
                "AND downlink_request_id=?",
                (
                    None if transfer_id is None else int(transfer_id),
                    next_state,
                    now_us,
                    encode_sqlite_u64(request_key.ground_instance_id),
                    request_key.request_id,
                ),
            )
            if pending_state is not None:
                connection.execute(
                    "DELETE FROM product_downlink_pending_files WHERE "
                    "product_spacecraft_instance_id=? AND origin_boot_id=? "
                    "AND product_id=? AND transfer_id=?",
                    (
                        row["product_spacecraft_instance_id"],
                        int(row["origin_boot_id"]),
                        int(row["product_id"]),
                        normalized_transfer_id,
                    ),
                )
                append_audit_in_transaction(
                    connection,
                    principal=PRODUCT_DOWNLINK_PRINCIPAL,
                    action="PRODUCT_DOWNLINK_FILE_CORRELATED",
                    target_type="product_transfer",
                    target_identity={
                        "product_spacecraft_instance_id": u64_to_json(
                            decode_sqlite_u64(
                                row["product_spacecraft_instance_id"],
                                "product_spacecraft_instance_id",
                            )
                        ),
                        "origin_boot_id": int(row["origin_boot_id"]),
                        "product_id": int(row["product_id"]),
                        "transfer_id": normalized_transfer_id,
                    },
                    old_value={"pending_transfer_state": pending_state},
                    new_value={
                        "downlink_request_key": request_key.as_dict(),
                        "transfer_state": next_state,
                    },
                    created_at_us=now_us,
                )
            return True

        return bool(
            self.writer.mutate(
                "update_product_downlink_transfer",
                mutation,
                priority=MutationPriority.HIGH,
            )
        )

    @staticmethod
    def _monotonic_downlink_state(current: str, candidate: str) -> str:
        """Never let duplicated/reordered TM regress a durable transfer."""

        if _PRODUCT_DOWNLINK_STATE_RANK[candidate] < _PRODUCT_DOWNLINK_STATE_RANK[current]:
            return current
        if _PRODUCT_DOWNLINK_STATE_RANK[current] == 3 and current != candidate:
            return current
        return candidate

    def update_product_downlink_file_state(
        self,
        product_ref: ProductRef,
        *,
        transfer_id: int,
        transfer_state: str,
    ) -> RequestKey | None:
        """Attach APID 3 only after an exact APID 2 transfer correlation.

        A file observation with no assigned transfer is retained briefly under
        its exact ProductRef and transfer ID.  It is never attached to a newer
        retry whose ``transfer_id`` is still NULL.
        """

        if not isinstance(product_ref, ProductRef):
            raise TypeError("product_ref must be a ProductRef")
        if isinstance(transfer_id, bool) or not 0 <= int(transfer_id) <= 0xFFFFFFFF:
            raise ValueError("transfer_id must fit U32")
        if transfer_state not in {"RECEIVING", "VERIFIED", "FAILED", "CANCELED"}:
            raise ValueError("invalid APID 3 transfer state")
        now_us = datetime_to_unix_us(self._now())

        def mutation(connection: sqlite3.Connection) -> RequestKey | None:
            self._reconcile_pending_product_downlink_files_in_transaction(
                connection,
                now_us=now_us,
            )
            product_blob = encode_sqlite_u64(product_ref.spacecraft_instance_id)
            row = connection.execute(
                "SELECT downlink_ground_instance_id,downlink_request_id,transfer_state "
                "FROM product_downlink_ledger WHERE product_spacecraft_instance_id=? "
                "AND origin_boot_id=? AND product_id=? "
                "AND transfer_id=?",
                (
                    product_blob,
                    product_ref.origin_boot_id,
                    product_ref.product_id,
                    int(transfer_id),
                ),
            ).fetchone()
            if row is None:
                pending = connection.execute(
                    "SELECT transfer_state FROM product_downlink_pending_files "
                    "WHERE product_spacecraft_instance_id=? AND origin_boot_id=? "
                    "AND product_id=? AND transfer_id=?",
                    (
                        product_blob,
                        product_ref.origin_boot_id,
                        product_ref.product_id,
                        int(transfer_id),
                    ),
                ).fetchone()
                if pending is None:
                    connection.execute(
                        "INSERT INTO product_downlink_pending_files("
                        "product_spacecraft_instance_id,origin_boot_id,product_id,"
                        "transfer_id,transfer_state,first_observed_at_us,updated_at_us,"
                        "expires_at_us) VALUES(?,?,?,?,?,?,?,?)",
                        (
                            product_blob,
                            product_ref.origin_boot_id,
                            product_ref.product_id,
                            int(transfer_id),
                            transfer_state,
                            now_us,
                            now_us,
                            now_us
                            + int(
                                PENDING_FILE_OBSERVATION_TTL.total_seconds()
                                * 1_000_000
                            ),
                        ),
                    )
                    append_audit_in_transaction(
                        connection,
                        principal=PRODUCT_DOWNLINK_PRINCIPAL,
                        action="PRODUCT_DOWNLINK_FILE_DEFERRED",
                        target_type="product_transfer",
                        target_identity={
                            "product_ref": product_ref.as_dict(),
                            "transfer_id": int(transfer_id),
                        },
                        old_value=None,
                        new_value={"transfer_state": transfer_state},
                        created_at_us=now_us,
                    )
                else:
                    next_pending_state = self._monotonic_downlink_state(
                        str(pending["transfer_state"]),
                        transfer_state,
                    )
                    if next_pending_state != str(pending["transfer_state"]):
                        connection.execute(
                            "UPDATE product_downlink_pending_files SET "
                            "transfer_state=?,updated_at_us=? "
                            "WHERE product_spacecraft_instance_id=? "
                            "AND origin_boot_id=? AND product_id=? AND transfer_id=?",
                            (
                                next_pending_state,
                                now_us,
                                product_blob,
                                product_ref.origin_boot_id,
                                product_ref.product_id,
                                int(transfer_id),
                            ),
                        )
                return None
            next_state = self._monotonic_downlink_state(
                str(row["transfer_state"]),
                transfer_state,
            )
            connection.execute(
                "UPDATE product_downlink_ledger SET transfer_state=?,updated_at_us=? "
                "WHERE downlink_ground_instance_id=? "
                "AND downlink_request_id=?",
                (
                    next_state,
                    now_us,
                    row[0],
                    row[1],
                ),
            )
            return RequestKey(decode_sqlite_u64(row[0], "downlink_ground_instance_id"), int(row[1]))

        return self.writer.mutate(
            "update_product_downlink_file_state",
            mutation,
            priority=MutationPriority.HIGH,
        )

    def reconcile_pending_product_downlink_files(self) -> int:
        """Expire uncorrelated APID 3 observations through an auditable path."""

        now_us = datetime_to_unix_us(self._now())
        return int(
            self.writer.mutate(
                "reconcile_pending_product_downlink_files",
                lambda connection: self._reconcile_pending_product_downlink_files_in_transaction(
                    connection,
                    now_us=now_us,
                ),
                priority=MutationPriority.HIGH,
            )
        )

    def _reconcile_pending_product_downlink_files_in_transaction(
        self,
        connection: sqlite3.Connection,
        *,
        now_us: int,
    ) -> int:
        rows = connection.execute(
            "SELECT product_spacecraft_instance_id,origin_boot_id,product_id,"
            "transfer_id,transfer_state FROM product_downlink_pending_files "
            "WHERE expires_at_us<=?",
            (now_us,),
        ).fetchall()
        for row in rows:
            connection.execute(
                "DELETE FROM product_downlink_pending_files WHERE "
                "product_spacecraft_instance_id=? AND origin_boot_id=? "
                "AND product_id=? AND transfer_id=?",
                (
                    row["product_spacecraft_instance_id"],
                    int(row["origin_boot_id"]),
                    int(row["product_id"]),
                    int(row["transfer_id"]),
                ),
            )
            append_audit_in_transaction(
                connection,
                principal=PRODUCT_DOWNLINK_PRINCIPAL,
                action="PRODUCT_DOWNLINK_FILE_DEFERRED_EXPIRED",
                target_type="product_transfer",
                target_identity={
                    "product_spacecraft_instance_id": u64_to_json(
                        decode_sqlite_u64(
                            row["product_spacecraft_instance_id"],
                            "product_spacecraft_instance_id",
                        )
                    ),
                    "origin_boot_id": int(row["origin_boot_id"]),
                    "product_id": int(row["product_id"]),
                    "transfer_id": int(row["transfer_id"]),
                },
                old_value={"transfer_state": str(row["transfer_state"])},
                new_value=None,
                created_at_us=now_us,
            )
        return len(rows)

    @staticmethod
    def _request_key_from_row(row: sqlite3.Row) -> RequestKey:
        return RequestKey(
            decode_sqlite_u64(row["ground_instance_id"], "ground_instance_id"),
            int(row["request_id"]),
        )

    def _result_from_row(
        self, row: sqlite3.Row, *, replayed: bool
    ) -> AdmissionResult:
        return AdmissionResult(
            request_key=self._request_key_from_row(row),
            target_spacecraft_instance_id=decode_sqlite_u64(
                row["target_spacecraft_instance_id"],
                "target_spacecraft_instance_id",
            ),
            effective_expires_at=unix_us_to_datetime(
                int(row["effective_expires_at_us"])
            ),
            command_state=str(row["command_state"]),
            outbox_state=str(row["outbox_state"]),
            mission_digest_hex=bytes(row["mission_digest"]).hex(),
            http_digest_hex=bytes(row["http_digest"]).hex(),
            accepted_at=unix_us_to_datetime(int(row["created_at_us"])),
            replayed=replayed,
        )

    def get(self, request_key: RequestKey) -> AdmissionResult | None:
        with self.writer.reader() as connection:
            row = connection.execute(
                "SELECT c.*,o.state AS outbox_state "
                "FROM commands AS c LEFT JOIN command_outbox AS o "
                "ON o.ground_instance_id=c.ground_instance_id "
                "AND o.request_id=c.request_id "
                "WHERE c.ground_instance_id=? AND c.request_id=?",
                (
                    encode_sqlite_u64(request_key.ground_instance_id),
                    request_key.request_id,
                ),
            ).fetchone()
            if row is None:
                return None
            if row["outbox_state"] is None:
                raise LedgerIntegrityError("command has no outbox row")
            return self._result_from_row(row, replayed=True)

    def retire_terminal_command(
        self,
        request_key: RequestKey,
        *,
        retention: timedelta = HTTP_IDEMPOTENCY_RETENTION,
    ) -> datetime:
        if retention <= timedelta(0):
            raise ValueError("idempotency retention must be positive")

        def mutation(connection: sqlite3.Connection) -> datetime:
            row = connection.execute(
                "SELECT c.*,o.state AS outbox_state "
                "FROM commands AS c LEFT JOIN command_outbox AS o "
                "ON o.ground_instance_id=c.ground_instance_id "
                "AND o.request_id=c.request_id "
                "WHERE c.ground_instance_id=? AND c.request_id=?",
                (
                    encode_sqlite_u64(request_key.ground_instance_id),
                    request_key.request_id,
                ),
            ).fetchone()
            if row is None:
                raise KeyError("command does not exist")
            if row["outbox_state"] is None:
                raise LedgerIntegrityError("command has no outbox row")
            if str(row["command_state"]) not in TERMINAL_COMMAND_STATES:
                raise ValueError("only terminal commands may be pruned")
            if str(row["outbox_state"]) not in TERMINAL_OUTBOX_STATES:
                raise ValueError("only terminal outbox rows may be pruned")
            now = self._now()
            retained_until = now + retention
            retained_until_us = datetime_to_unix_us(retained_until)
            connection.execute(
                "INSERT INTO http_idempotency_retired("
                "gds_installation_epoch,principal,idempotency_key,http_digest,"
                "original_ground_instance_id,original_request_id,retained_until_us) "
                "VALUES(?,?,?,?,?,?,?)",
                (
                    row["gds_installation_epoch"],
                    row["principal"],
                    row["idempotency_key"],
                    row["http_digest"],
                    row["ground_instance_id"],
                    row["request_id"],
                    retained_until_us,
                ),
            )
            connection.execute(
                "DELETE FROM commands WHERE ground_instance_id=? AND request_id=?",
                (row["ground_instance_id"], row["request_id"]),
            )
            return retained_until

        return self.writer.mutate(
            "retire_http_idempotency_marker",
            mutation,
            priority=MutationPriority.HIGH,
        )

    def orphan_counts(self) -> tuple[int, int]:
        with self.writer.reader() as connection:
            commands_without_outbox = int(
                connection.execute(
                    "SELECT count(*) FROM commands AS c LEFT JOIN command_outbox AS o "
                    "ON o.ground_instance_id=c.ground_instance_id "
                    "AND o.request_id=c.request_id WHERE o.request_id IS NULL"
                ).fetchone()[0]
            )
            outbox_without_commands = int(
                connection.execute(
                    "SELECT count(*) FROM command_outbox AS o LEFT JOIN commands AS c "
                    "ON c.ground_instance_id=o.ground_instance_id "
                    "AND c.request_id=o.request_id WHERE c.request_id IS NULL"
                ).fetchone()[0]
            )
            return commands_without_outbox, outbox_without_commands
