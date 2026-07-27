"""Durable GDS installation epoch and RequestKey namespace allocator."""

from __future__ import annotations

import secrets
import sqlite3
import time
from dataclasses import dataclass
from typing import Callable

from protocol.canonical import MAX_U32, checked_u64
from protocol.schemas import RequestKey

from .u64 import decode_sqlite_u64, encode_sqlite_u64
from .writer import MutationPriority, SQLiteWriter


class RequestKeyAllocatorError(RuntimeError):
    pass


class RequestNamespaceDrainingError(RequestKeyAllocatorError):
    """The exhausted namespace still owns nonterminal command work."""


@dataclass(frozen=True)
class AllocatorState:
    gds_installation_epoch: int
    ground_instance_id: int
    next_request_id: int
    namespace_state: str


class RequestKeyAllocator:
    """Allocate U32 request IDs inside a durable CSPRNG U64 namespace."""

    def __init__(
        self,
        writer: SQLiteWriter,
        *,
        random_u64: Callable[[], int] | None = None,
        clock_us: Callable[[], int] | None = None,
    ) -> None:
        self.writer = writer
        self._random_u64 = random_u64 or (lambda: secrets.randbits(64))
        self._clock_us = clock_us or (lambda: time.time_ns() // 1_000)

    def initialize(self) -> AllocatorState:
        return self.writer.mutate(
            "initialize_request_key_allocator",
            self._initialize_in_transaction,
            priority=MutationPriority.HIGH,
        )

    def _new_unique_u64(
        self,
        connection: sqlite3.Connection,
        *,
        column: str,
        table: str,
        excluded: set[int] | None = None,
    ) -> int:
        excluded = excluded or set()
        for _ in range(128):
            candidate = checked_u64(self._random_u64(), column)
            if candidate in excluded:
                continue
            row = connection.execute(
                f"SELECT 1 FROM {table} WHERE {column}=? LIMIT 1",
                (encode_sqlite_u64(candidate),),
            ).fetchone()
            if row is None:
                return candidate
        raise RequestKeyAllocatorError(
            f"could not generate a unique {column} after 128 CSPRNG draws"
        )

    def _initialize_in_transaction(
        self, connection: sqlite3.Connection
    ) -> AllocatorState:
        row = connection.execute(
            "SELECT gds_installation_epoch,active_ground_instance_id "
            "FROM gds_metadata WHERE singleton=1"
        ).fetchone()
        if row is None:
            orphan_count = int(
                connection.execute("SELECT count(*) FROM ground_namespaces").fetchone()[0]
            )
            if orphan_count:
                raise RequestKeyAllocatorError(
                    "ground namespace rows exist without gds_metadata"
                )
            epoch = self._new_unique_u64(
                connection,
                column="gds_installation_epoch",
                table="ground_namespaces",
            )
            ground = self._new_unique_u64(
                connection,
                column="ground_instance_id",
                table="ground_namespaces",
                excluded={epoch},
            )
            now_us = int(self._clock_us())
            if now_us < 0:
                raise RequestKeyAllocatorError("clock_us must be non-negative")
            connection.execute(
                "INSERT INTO ground_namespaces"
                "(ground_instance_id,gds_installation_epoch,next_request_id,state,created_at_us) "
                "VALUES(?,?,1,'ACTIVE',?)",
                (encode_sqlite_u64(ground), encode_sqlite_u64(epoch), now_us),
            )
            connection.execute(
                "INSERT INTO gds_metadata"
                "(singleton,gds_installation_epoch,active_ground_instance_id,created_at_us) "
                "VALUES(1,?,?,?)",
                (encode_sqlite_u64(epoch), encode_sqlite_u64(ground), now_us),
            )
            return AllocatorState(epoch, ground, 1, "ACTIVE")
        epoch = decode_sqlite_u64(row[0], "gds_installation_epoch")
        ground = decode_sqlite_u64(row[1], "active_ground_instance_id")
        namespace = connection.execute(
            "SELECT gds_installation_epoch,next_request_id,state "
            "FROM ground_namespaces WHERE ground_instance_id=?",
            (encode_sqlite_u64(ground),),
        ).fetchone()
        if namespace is None:
            raise RequestKeyAllocatorError("active ground namespace is missing")
        namespace_epoch = decode_sqlite_u64(
            namespace[0], "namespace gds_installation_epoch"
        )
        if namespace_epoch != epoch:
            raise RequestKeyAllocatorError(
                "active ground namespace belongs to another installation epoch"
            )
        next_request_id = int(namespace[1])
        state = str(namespace[2])
        if state not in {"ACTIVE", "DRAINING"}:
            raise RequestKeyAllocatorError(
                f"metadata points at non-active namespace state {state!r}"
            )
        if (next_request_id > MAX_U32) != (state == "DRAINING"):
            raise RequestKeyAllocatorError(
                "request allocator wrap marker and namespace state disagree"
            )
        return AllocatorState(epoch, ground, next_request_id, state)

    def state(self) -> AllocatorState:
        with self.writer.reader() as connection:
            row = connection.execute(
                "SELECT m.gds_installation_epoch,m.active_ground_instance_id,"
                "n.next_request_id,n.state "
                "FROM gds_metadata AS m JOIN ground_namespaces AS n "
                "ON n.ground_instance_id=m.active_ground_instance_id "
                "WHERE m.singleton=1"
            ).fetchone()
            if row is None:
                raise RequestKeyAllocatorError("request key allocator is not initialized")
            return AllocatorState(
                decode_sqlite_u64(row[0], "gds_installation_epoch"),
                decode_sqlite_u64(row[1], "ground_instance_id"),
                int(row[2]),
                str(row[3]),
            )

    def allocate(self) -> RequestKey:
        """Allocate outside admission when explicitly needed by an internal caller."""

        return self.writer.mutate(
            "allocate_request_key",
            lambda connection: self.allocate_in_transaction(connection),
            priority=MutationPriority.HIGH,
        )

    def allocate_in_transaction(self, connection: sqlite3.Connection) -> RequestKey:
        state = self._initialize_in_transaction(connection)
        if state.next_request_id > MAX_U32:
            state = self._rotate_if_drained_in_transaction(connection, state)
        request_id = state.next_request_id
        if not 1 <= request_id <= MAX_U32:
            raise RequestKeyAllocatorError("allocator produced an invalid request_id")
        next_request_id = request_id + 1
        next_state = "DRAINING" if next_request_id > MAX_U32 else "ACTIVE"
        cursor = connection.execute(
            "UPDATE ground_namespaces SET next_request_id=?,state=? "
            "WHERE ground_instance_id=? AND next_request_id=? AND state=?",
            (
                next_request_id,
                next_state,
                encode_sqlite_u64(state.ground_instance_id),
                request_id,
                state.namespace_state,
            ),
        )
        if cursor.rowcount != 1:
            raise RequestKeyAllocatorError("request key allocator state changed unexpectedly")
        return RequestKey(state.ground_instance_id, request_id)

    def rotate_if_drained(self) -> AllocatorState:
        return self.writer.mutate(
            "rotate_request_namespace",
            lambda connection: self._rotate_if_drained_in_transaction(
                connection, self._initialize_in_transaction(connection)
            ),
            priority=MutationPriority.HIGH,
        )

    def _rotate_if_drained_in_transaction(
        self, connection: sqlite3.Connection, state: AllocatorState
    ) -> AllocatorState:
        if state.next_request_id <= MAX_U32:
            return state
        old_ground = encode_sqlite_u64(state.ground_instance_id)
        nonterminal_command = connection.execute(
            "SELECT 1 FROM commands WHERE ground_instance_id=? AND command_state "
            "NOT IN ('ACKED','REJECTED','EXECUTED','FAILED','CANCELED') LIMIT 1",
            (old_ground,),
        ).fetchone()
        nonterminal_outbox = connection.execute(
            "SELECT 1 FROM command_outbox WHERE ground_instance_id=? AND state "
            "NOT IN ('ACKED','EXPIRED','DELIVERY_FAILED','CANCELED') LIMIT 1",
            (old_ground,),
        ).fetchone()
        if nonterminal_command is not None or nonterminal_outbox is not None:
            raise RequestNamespaceDrainingError(
                "request_id exhausted; old namespace has nonterminal work"
            )
        new_ground = self._new_unique_u64(
            connection,
            column="ground_instance_id",
            table="ground_namespaces",
            excluded={state.gds_installation_epoch},
        )
        now_us = int(self._clock_us())
        if now_us < 0:
            raise RequestKeyAllocatorError("clock_us must be non-negative")
        retired = connection.execute(
            "UPDATE ground_namespaces SET state='RETIRED',retired_at_us=? "
            "WHERE ground_instance_id=? AND state='DRAINING'",
            (now_us, old_ground),
        )
        if retired.rowcount != 1:
            raise RequestKeyAllocatorError("exhausted namespace is not DRAINING")
        connection.execute(
            "INSERT INTO ground_namespaces"
            "(ground_instance_id,gds_installation_epoch,next_request_id,state,created_at_us) "
            "VALUES(?,?,1,'ACTIVE',?)",
            (
                encode_sqlite_u64(new_ground),
                encode_sqlite_u64(state.gds_installation_epoch),
                now_us,
            ),
        )
        connection.execute(
            "UPDATE gds_metadata SET active_ground_instance_id=? WHERE singleton=1",
            (encode_sqlite_u64(new_ground),),
        )
        return AllocatorState(
            state.gds_installation_epoch, new_ground, 1, "ACTIVE"
        )
