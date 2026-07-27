"""Bounded, priority-aware single-writer service for the GDS SQLite core."""

from __future__ import annotations

import itertools
import queue
import sqlite3
import threading
import time
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Iterator, TypeVar

from .database import (
    SQLiteProfile,
    WalStatus,
    classify_wal,
    open_reader_connection,
    open_writer_connection,
    wal_size_bytes,
)
from .schema import migrate

T = TypeVar("T")

DEFAULT_WRITER_QUEUE_CAPACITY = 4_096
DEFAULT_HIGH_PRIORITY_RESERVE = 256
_WRITER_REGISTRY_LOCK = threading.Lock()
_ACTIVE_WRITER_PATHS: set[Path] = set()


class WriterError(RuntimeError):
    """Base class for writer lifecycle/backpressure errors."""


class WriterClosedError(WriterError):
    pass


class WriterAlreadyRunningError(WriterError):
    pass


class WriterBackpressureError(WriterError):
    status_code = 503
    error_code = "WRITER_BACKPRESSURE"
    retry_after_seconds = 1


class LowPriorityDroppedError(WriterError):
    error_code = "GDS_INGEST_OVERFLOW"


class ActiveReaderError(WriterError):
    pass


class WalRecoveryError(WriterError):
    pass


class MutationPriority(IntEnum):
    HIGH = 0
    LOW = 1


@dataclass(frozen=True)
class MutationIntent:
    name: str
    apply: Callable[[sqlite3.Connection], Any] = field(repr=False, compare=False)
    priority: MutationPriority = MutationPriority.HIGH
    transactional: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("mutation intent name must not be empty")
        if not callable(self.apply):
            raise TypeError("mutation intent apply must be callable")


@dataclass(frozen=True)
class WriterMetrics:
    submitted_high: int
    submitted_low: int
    rejected_high: int
    dropped_low: int
    completed: int
    failed: int
    max_queue_depth: int
    reader_overruns: int


@dataclass
class _MutableMetrics:
    submitted_high: int = 0
    submitted_low: int = 0
    rejected_high: int = 0
    dropped_low: int = 0
    completed: int = 0
    failed: int = 0
    max_queue_depth: int = 0
    reader_overruns: int = 0


@dataclass(frozen=True)
class _QueueItem:
    priority: int
    sequence: int
    intent: MutationIntent | None = field(compare=False)
    future: Future[Any] | None = field(compare=False)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, _QueueItem):
            return NotImplemented
        return (self.priority, self.sequence) < (other.priority, other.sequence)


class SQLiteWriter:
    """Own the only read-write connection and serialize mutation intents."""

    def __init__(
        self,
        path: str | Path,
        *,
        profile: SQLiteProfile | None = None,
        queue_capacity: int = DEFAULT_WRITER_QUEUE_CAPACITY,
        high_priority_reserve: int = DEFAULT_HIGH_PRIORITY_RESERVE,
        startup_timeout: float = 10.0,
    ) -> None:
        if queue_capacity <= 0:
            raise ValueError("queue_capacity must be positive")
        if not 0 <= high_priority_reserve < queue_capacity:
            raise ValueError("high_priority_reserve must be in [0, queue_capacity)")
        self.path = Path(path).resolve()
        with _WRITER_REGISTRY_LOCK:
            if self.path in _ACTIVE_WRITER_PATHS:
                raise WriterAlreadyRunningError(
                    f"a SQLite writer already owns {self.path}"
                )
            _ACTIVE_WRITER_PATHS.add(self.path)
        self._registered = True
        self.profile = profile or SQLiteProfile()
        self.queue_capacity = queue_capacity
        self.high_priority_reserve = high_priority_reserve
        self._low_priority_limit = queue_capacity - high_priority_reserve
        self._queue: queue.PriorityQueue[_QueueItem] = queue.PriorityQueue(
            maxsize=queue_capacity
        )
        self._sequence = itertools.count()
        self._submission_lock = threading.Lock()
        self._metrics_lock = threading.Lock()
        self._metrics = _MutableMetrics()
        self._reader_lock = threading.RLock()
        self._active_readers: dict[int, float] = {}
        self._reader_sequence = itertools.count()
        self._ready = threading.Event()
        self._accepting = True
        self._startup_error: BaseException | None = None
        self._writer_thread_id: int | None = None
        self._thread = threading.Thread(
            target=self._run,
            name=f"gds-sqlite-writer:{self.path.name}",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(startup_timeout):
            self._accepting = False
            self._queue.put_nowait(
                _QueueItem(2, next(self._sequence), None, None)
            )
            raise TimeoutError("SQLite writer did not become ready")
        if self._startup_error is not None:
            self._accepting = False
            self._unregister()
            if isinstance(self._startup_error, Exception):
                raise self._startup_error
            raise WriterError("SQLite writer startup failed") from self._startup_error

    @property
    def writer_thread_id(self) -> int | None:
        return self._writer_thread_id

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def active_reader_count(self) -> int:
        with self._reader_lock:
            return len(self._active_readers)

    def metrics(self) -> WriterMetrics:
        with self._metrics_lock:
            return WriterMetrics(**vars(self._metrics))

    def _record_submission(self, priority: MutationPriority, depth: int) -> None:
        with self._metrics_lock:
            if priority is MutationPriority.HIGH:
                self._metrics.submitted_high += 1
            else:
                self._metrics.submitted_low += 1
            self._metrics.max_queue_depth = max(self._metrics.max_queue_depth, depth)

    def _reject(self, priority: MutationPriority) -> None:
        with self._metrics_lock:
            if priority is MutationPriority.HIGH:
                self._metrics.rejected_high += 1
            else:
                self._metrics.dropped_low += 1

    def submit(self, intent: MutationIntent) -> Future[Any]:
        if not isinstance(intent, MutationIntent):
            raise TypeError("submit requires a MutationIntent")
        with self._submission_lock:
            if not self._accepting:
                raise WriterClosedError("SQLite writer is closed")
            depth = self._queue.qsize()
            if intent.priority is MutationPriority.LOW:
                if not self.wal_status().admit_low_priority or depth >= self._low_priority_limit:
                    self._reject(intent.priority)
                    raise LowPriorityDroppedError(
                        "low-priority mutation rejected by reserve or WAL throttle"
                    )
            elif depth >= self.queue_capacity:
                self._reject(intent.priority)
                raise WriterBackpressureError("all SQLite writer queue slots are occupied")
            future: Future[Any] = Future()
            item = _QueueItem(
                int(intent.priority), next(self._sequence), intent, future
            )
            try:
                self._queue.put_nowait(item)
            except queue.Full as exc:
                self._reject(intent.priority)
                if intent.priority is MutationPriority.HIGH:
                    raise WriterBackpressureError(
                        "all SQLite writer queue slots are occupied"
                    ) from exc
                raise LowPriorityDroppedError(
                    "low-priority mutation rejected by writer queue"
                ) from exc
            self._record_submission(intent.priority, depth + 1)
            return future

    def execute(self, intent: MutationIntent, *, timeout: float | None = None) -> Any:
        return self.submit(intent).result(timeout=timeout)

    def mutate(
        self,
        name: str,
        callback: Callable[[sqlite3.Connection], T],
        *,
        priority: MutationPriority = MutationPriority.HIGH,
        transactional: bool = True,
        timeout: float | None = None,
    ) -> T:
        intent = MutationIntent(name, callback, priority, transactional)
        return self.execute(intent, timeout=timeout)

    def _run(self) -> None:
        connection: sqlite3.Connection | None = None
        try:
            self._writer_thread_id = threading.get_ident()
            connection = open_writer_connection(self.path, self.profile)
            migrate(connection)
        except BaseException as exc:
            self._startup_error = exc
            self._ready.set()
            if connection is not None:
                connection.close()
            self._unregister()
            return
        self._ready.set()
        try:
            while True:
                item = self._queue.get()
                try:
                    if item.intent is None:
                        return
                    assert item.future is not None
                    if not item.future.set_running_or_notify_cancel():
                        continue
                    try:
                        if item.intent.transactional:
                            connection.execute("BEGIN IMMEDIATE")
                        result = item.intent.apply(connection)
                        if item.intent.transactional:
                            connection.commit()
                    except BaseException as exc:
                        if connection.in_transaction:
                            connection.rollback()
                        item.future.set_exception(exc)
                        with self._metrics_lock:
                            self._metrics.failed += 1
                    else:
                        item.future.set_result(result)
                        with self._metrics_lock:
                            self._metrics.completed += 1
                finally:
                    self._queue.task_done()
        finally:
            connection.close()
            self._unregister()

    @contextmanager
    def reader(self) -> Iterator[sqlite3.Connection]:
        with self._reader_lock:
            if not self._accepting:
                raise WriterClosedError("SQLite writer is closed")
            token = next(self._reader_sequence)
            started = time.monotonic()
            self._active_readers[token] = started
        connection: sqlite3.Connection | None = None
        try:
            connection = open_reader_connection(self.path, self.profile)
            yield connection
        finally:
            if connection is not None:
                connection.close()
            duration = time.monotonic() - started
            with self._reader_lock:
                self._active_readers.pop(token, None)
            if duration > self.profile.max_reader_seconds:
                with self._metrics_lock:
                    self._metrics.reader_overruns += 1

    def wal_status(self) -> WalStatus:
        size = wal_size_bytes(self.path)
        now = time.monotonic()
        with self._reader_lock:
            overdue = sum(
                now - started > self.profile.max_reader_seconds
                for started in self._active_readers.values()
            )
            active = len(self._active_readers)
        return WalStatus(size, classify_wal(size, self.profile), active, overdue)

    def checkpoint(self, mode: str = "PASSIVE") -> tuple[int, int, int]:
        normalized = mode.upper()
        if normalized not in {"PASSIVE", "FULL", "RESTART", "TRUNCATE"}:
            raise ValueError("unsupported WAL checkpoint mode")

        def run(connection: sqlite3.Connection) -> tuple[int, int, int]:
            row = connection.execute(f"PRAGMA wal_checkpoint({normalized})").fetchone()
            return int(row[0]), int(row[1]), int(row[2])

        if normalized == "TRUNCATE":
            with self._reader_lock:
                if self._active_readers:
                    raise ActiveReaderError(
                        "TRUNCATE checkpoint requires zero active readers"
                    )
                return self.mutate(
                    "wal_checkpoint_truncate", run, transactional=False
                )
        return self.mutate(
            f"wal_checkpoint_{normalized.lower()}", run, transactional=False
        )

    def recover_wal(self) -> WalStatus:
        self.checkpoint("PASSIVE")
        status = self.wal_status()
        if not status.admit_low_priority and status.active_readers == 0:
            self.checkpoint("TRUNCATE")
            status = self.wal_status()
        return status

    def ensure_wal_ready(self) -> WalStatus:
        status = self.recover_wal()
        if not status.admit_low_priority:
            raise WalRecoveryError(
                "WAL remains above the throttle threshold after checkpoint recovery"
            )
        return status

    def close(self, *, timeout: float = 10.0) -> None:
        with self._submission_lock:
            if not self._accepting:
                return
            self._accepting = False
        sentinel = _QueueItem(2, next(self._sequence), None, None)
        try:
            self._queue.put(sentinel, timeout=timeout)
        except queue.Full as exc:
            raise TimeoutError("could not enqueue SQLite writer shutdown") from exc
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise TimeoutError("SQLite writer did not stop")
        self._unregister()

    def _unregister(self) -> None:
        if not getattr(self, "_registered", False):
            return
        with _WRITER_REGISTRY_LOCK:
            _ACTIVE_WRITER_PATHS.discard(self.path)
        self._registered = False

    def __enter__(self) -> "SQLiteWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
