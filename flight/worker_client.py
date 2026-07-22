"""CloudPayload-side process manager for the serialized AI worker contract."""

from __future__ import annotations

import multiprocessing
import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from sat_ai.worker_contract import (
    WORKER_VERSION,
    WorkerControl,
    WorkerControlAction,
    WorkerHeartbeat,
    WorkerHeartbeatMessage,
    WorkerProtocolError,
    WorkerRequest,
    WorkerResult,
    WorkerResultState,
    decode_worker_message,
)
from sat_ai.worker_process import worker_process_main


logger = logging.getLogger(__name__)


class WorkerQueueFull(RuntimeError):
    code = "QUEUE_FULL"


@dataclass(frozen=True)
class WorkerProcessPolicy:
    max_pending_jobs: int = 4
    heartbeat_interval_ms: int = 1000
    heartbeat_timeout_ms: int = 5000
    startup_timeout_ms: int = 30000
    max_restarts: int = 3
    restart_window_ms: int = 300000
    initial_backoff_ms: int = 250
    cancel_grace_ms: int = 1000

    def __post_init__(self) -> None:
        values = (
            self.max_pending_jobs,
            self.heartbeat_interval_ms,
            self.heartbeat_timeout_ms,
            self.startup_timeout_ms,
            self.restart_window_ms,
            self.cancel_grace_ms,
        )
        if min(values) <= 0 or self.max_restarts < 0 or self.initial_backoff_ms < 0:
            raise ValueError("worker process policy values are invalid")
        if self.heartbeat_interval_ms >= self.heartbeat_timeout_ms:
            raise ValueError("worker heartbeat interval must be below timeout")


class WorkerProcessClient:
    """One process, one active job, bounded pending work and bounded recovery."""

    def __init__(
        self,
        root: str | Path,
        *,
        device: str,
        policy: WorkerProcessPolicy,
        on_result: Callable[[WorkerResult], None],
        on_started: Callable[[WorkerRequest], None] | None = None,
        on_state_change: Callable[[str], None] | None = None,
        process_target: Callable[..., None] = worker_process_main,
    ):
        self.root = str(Path(root).resolve())
        self.device = device
        self.policy = policy
        self.on_result = on_result
        self.on_started = on_started
        self.on_state_change = on_state_change
        self.process_target = process_target
        self._context = multiprocessing.get_context("spawn")
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._pending: deque[WorkerRequest] = deque()
        self._active: WorkerRequest | None = None
        self._callbacks_inflight = 0
        self._deadline_cancel_sent_ns: int | None = None
        self._cancel_sent_ns: int | None = None
        self._closing = False
        self._ready = threading.Event()
        self._monitor: threading.Thread | None = None
        self._restart_times_ns: deque[int] = deque()
        self._restart_count = 0
        self.state = "STARTING"
        self.heartbeat = WorkerHeartbeat(WORKER_VERSION)
        self.process = None
        self.request_queue = None
        self.control_queue = None
        self.result_queue = None
        self.stop_event = None

    def start(self) -> None:
        with self._lock:
            if self.process is not None or self._monitor is not None:
                raise RuntimeError("worker process client already started")
            self._spawn_locked()
            self._monitor = threading.Thread(target=self._monitor_loop, name="cloud-worker-monitor", daemon=True)
            self._monitor.start()
            logger.info("worker_process_started pid=%s device=%s", self.process.pid if self.process else None, self.device)
        if not self._ready.wait(self.policy.startup_timeout_ms / 1000.0):
            logger.error("worker_startup_timeout timeout_ms=%s", self.policy.startup_timeout_ms)
            self.close()
            raise RuntimeError("WORKER_STARTUP_TIMEOUT")

    def _set_state(self, state: str) -> None:
        callback = None
        with self._lock:
            if self.state == state:
                return
            previous = self.state
            self.state = state
            callback = self.on_state_change
            restart_count = self._restart_count
        logger.info("worker_state_changed state=%s previous_state=%s restart_count=%s", state, previous, restart_count)
        if callback is not None:
            callback(state)

    def _spawn_locked(self) -> None:
        self.request_queue = self._context.Queue(maxsize=1)
        self.control_queue = self._context.Queue(maxsize=max(4, self.policy.max_pending_jobs + 2))
        self.result_queue = self._context.Queue(maxsize=64)
        self.stop_event = self._context.Event()
        self.heartbeat = WorkerHeartbeat(WORKER_VERSION)
        self._ready.clear()
        self._deadline_cancel_sent_ns = None
        self._cancel_sent_ns = None
        self.process = self._context.Process(
            target=self.process_target,
            args=(
                self.root,
                self.device,
                self.request_queue,
                self.control_queue,
                self.result_queue,
                self.stop_event,
                self.policy.heartbeat_interval_ms,
            ),
            name="sat-ai-worker",
            daemon=True,
        )
        self.process.start()

    def submit(self, request: WorkerRequest) -> None:
        with self._condition:
            if self.state in {"FAULT", "STOPPED"} or self._closing:
                raise RuntimeError("SERVICE_FAULT")
            pending_limit = self.policy.max_pending_jobs
            if self._ready.is_set() and self._active is None:
                pending_limit += 1
            if len(self._pending) >= pending_limit:
                logger.warning(
                    "worker_queue_full depth=%s capacity=%s active=%s",
                    len(self._pending),
                    pending_limit,
                    self._active.request_key.as_dict() if self._active else None,
                )
                raise WorkerQueueFull("bounded worker queue is full")
            self._pending.append(request)
            self._condition.notify_all()

    def cancel(self, request_key) -> str:
        immediate: WorkerResult | None = None
        with self._condition:
            for request in tuple(self._pending):
                if request.request_key == request_key:
                    self._pending.remove(request)
                    immediate = WorkerResult(
                        request_key,
                        WorkerResultState.CANCELED,
                        {"cancel_stage": "PENDING"},
                        "WORKER_CANCELED",
                    )
                    self._condition.notify_all()
                    break
            if immediate is None and self._active is not None and self._active.request_key == request_key:
                assert self.control_queue is not None
                try:
                    self.control_queue.put_nowait(
                        WorkerControl(WorkerControlAction.CANCEL, request_key, "MISSION_CANCEL").encode()
                    )
                    self._cancel_sent_ns = time.monotonic_ns()
                except queue.Full:
                    # The worker may be blocked or its control mailbox may be
                    # saturated. The watchdog below still owns this request
                    # and will produce a bounded terminal CANCELED result.
                    self._cancel_sent_ns = time.monotonic_ns()
                return "CANCEL_REQUESTED"
        if immediate is not None:
            self._deliver_result(immediate)
            return "CANCELED"
        return "ALREADY_TERMINAL"

    def _monitor_loop(self) -> None:
        while True:
            with self._lock:
                if self._closing:
                    return
                result_queue = self.result_queue
            self._dispatch_next()
            encoded = None
            if result_queue is not None:
                try:
                    encoded = result_queue.get(timeout=0.05)
                except queue.Empty:
                    pass
            if encoded is not None:
                try:
                    message = decode_worker_message(encoded)
                    if isinstance(message, WorkerHeartbeatMessage):
                        self._handle_heartbeat(message)
                    elif isinstance(message, WorkerResult):
                        self._handle_result(message)
                    else:
                        raise WorkerProtocolError("worker emitted an invalid outbound message")
                except Exception:
                    self._handle_loss("WORKER_PROTOCOL_ERROR")
                    continue
            if self._check_deadline():
                continue
            if self._check_cancel_watchdog():
                continue
            with self._lock:
                process = self.process
                ready = self._ready.is_set()
                heartbeat_alive = self.heartbeat.is_alive(self.policy.heartbeat_timeout_ms) if ready else True
            if process is not None and not process.is_alive():
                self._handle_loss("WORKER_LOST")
            elif ready and not heartbeat_alive:
                self._handle_loss("WORKER_LOST")

    def _dispatch_next(self) -> None:
        started = None
        with self._condition:
            if not self._ready.is_set() or self._active is not None or not self._pending:
                return
            request = self._pending.popleft()
            assert self.request_queue is not None
            try:
                self.request_queue.put_nowait(request.encode())
            except queue.Full:
                self._pending.appendleft(request)
                return
            self._active = request
            self._deadline_cancel_sent_ns = None
            self._cancel_sent_ns = None
            started = self.on_started
            self._condition.notify_all()
        if started is not None:
            try:
                started(request)
            except Exception:
                self._handle_loss("WORKER_CALLBACK_ERROR")

    def _handle_heartbeat(self, message: WorkerHeartbeatMessage) -> None:
        try:
            self.heartbeat.touch(message)
        except WorkerProtocolError:
            self._handle_loss("WORKER_PROTOCOL_ERROR")
            return
        if message.worker_state in {"READY", "RUNNING"}:
            self._ready.set()
            if self.state != "READY":
                self._set_state("READY")

    def _handle_result(self, result: WorkerResult) -> None:
        with self._condition:
            if self._active is None or self._active.request_key != result.request_key:
                raise WorkerProtocolError("worker result does not match active request")
        # Keep ownership until the application callback has durably recorded
        # the terminal state. This closes the cancel/result race.
        self._deliver_result(result)
        with self._condition:
            if self._active is not None and self._active.request_key == result.request_key:
                self._active = None
            self._deadline_cancel_sent_ns = None
            self._cancel_sent_ns = None
            self._condition.notify_all()

    def _deliver_result(self, result: WorkerResult) -> None:
        with self._condition:
            self._callbacks_inflight += 1
        try:
            try:
                self.on_result(result)
            except Exception as exc:
                # A malformed product/result callback must not strand the
                # active request. Give the same callback one stable protocol
                # failure record before allowing ownership to be released.
                if result.state != WorkerResultState.FAILED:
                    fallback = WorkerResult(
                        result.request_key,
                        WorkerResultState.FAILED,
                        {"failure_stage": "RESULT_CALLBACK", "exception": type(exc).__name__},
                        "WORKER_CALLBACK_ERROR",
                    )
                    try:
                        self.on_result(fallback)
                    except Exception:
                        pass
        finally:
            with self._condition:
                self._callbacks_inflight -= 1
                self._condition.notify_all()

    def _check_deadline(self) -> bool:
        with self._lock:
            active = self._active
            if active is None or not active.deadline.expired():
                return False
            now = time.monotonic_ns()
            if self._deadline_cancel_sent_ns is None:
                assert self.control_queue is not None
                try:
                    self.control_queue.put_nowait(
                        WorkerControl(
                            WorkerControlAction.CANCEL,
                            active.request_key,
                            "DEADLINE_EXCEEDED",
                        ).encode()
                    )
                except queue.Full:
                    pass
                self._deadline_cancel_sent_ns = now
                return False
            grace_ns = self.policy.cancel_grace_ms * 1_000_000
            if now - self._deadline_cancel_sent_ns < grace_ns:
                return False
        self._handle_loss("DEADLINE_EXCEEDED", result_state=WorkerResultState.TIMEOUT)
        return True

    def _check_cancel_watchdog(self) -> bool:
        with self._lock:
            if self._active is None or self._cancel_sent_ns is None:
                return False
            if time.monotonic_ns() - self._cancel_sent_ns < self.policy.cancel_grace_ms * 1_000_000:
                return False
        self._handle_loss("CANCEL_TIMEOUT", result_state=WorkerResultState.CANCELED)
        return True

    def _handle_loss(self, error_code: str, *, result_state: WorkerResultState = WorkerResultState.FAILED) -> None:
        active_result = None
        pending_failures: list[WorkerResult] = []
        process = None
        with self._condition:
            if self._closing:
                return
            process = self.process
            if self.stop_event is not None:
                self.stop_event.set()
            if self._active is not None:
                active_result = WorkerResult(
                    self._active.request_key,
                    result_state,
                    {"failure_stage": "ACTIVE"},
                    error_code,
                )
                # Keep the active request owned until _deliver_result returns.
                self._deadline_cancel_sent_ns = None
                self._cancel_sent_ns = None
            self._ready.clear()
            now = time.monotonic_ns()
            window_ns = self.policy.restart_window_ms * 1_000_000
            while self._restart_times_ns and now - self._restart_times_ns[0] >= window_ns:
                self._restart_times_ns.popleft()
            if not self._restart_times_ns:
                self._restart_count = 0
            if len(self._restart_times_ns) >= self.policy.max_restarts:
                pending_count = len(self._pending)
                while self._pending:
                    request = self._pending.popleft()
                    pending_failures.append(
                        WorkerResult(
                            request.request_key,
                            WorkerResultState.FAILED,
                            {"failure_stage": "PENDING"},
                            "SERVICE_FAULT",
                        )
                    )
                should_restart = False
            else:
                pending_count = len(self._pending)
                self._restart_times_ns.append(now)
                self._restart_count += 1
                should_restart = True
            self._condition.notify_all()
            active_key = None if active_result is None else active_result.request_key.as_dict()
            restart_count = self._restart_count
        logger.error(
            "worker_loss error_code=%s active_request=%s pending=%s restart_count=%s/%s",
            error_code,
            active_key,
            pending_count,
            restart_count,
            self.policy.max_restarts,
        )
        if process is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=2)
        if active_result is not None:
            self._deliver_result(active_result)
            with self._condition:
                if self._active is not None and self._active.request_key == active_result.request_key:
                    self._active = None
                self._condition.notify_all()
        for result in pending_failures:
            self._deliver_result(result)
        if not should_restart:
            self._set_state("FAULT")
            return
        self._set_state("DEGRADED")
        backoff_ms = self.policy.initial_backoff_ms * (2 ** max(0, self._restart_count - 1))
        logger.warning("worker_restart_scheduled backoff_ms=%s", backoff_ms)
        if backoff_ms:
            time.sleep(min(backoff_ms, 5000) / 1000.0)
        with self._lock:
            if not self._closing:
                self._spawn_locked()

    def wait(self, timeout: float | None = None) -> None:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while self._pending or self._active is not None or self._callbacks_inflight:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("AI worker did not drain before timeout")
                self._condition.wait(timeout=remaining)

    def health(self) -> dict[str, int | str | None]:
        with self._lock:
            age_ms = None
            if self.heartbeat.last_seen_ns:
                age_ms = (time.monotonic_ns() - self.heartbeat.last_seen_ns) // 1_000_000
            return {
                "state": self.state,
                "heartbeat_age_ms": age_ms,
                "queue_depth": len(self._pending),
                "queue_capacity": self.policy.max_pending_jobs,
                "active_request_id": self._active.request_key.request_id if self._active else None,
                "restart_count": self._restart_count,
            }

    def close(self) -> None:
        with self._condition:
            if self._closing:
                return
            self._closing = True
            process = self.process
            if self.control_queue is not None:
                try:
                    self.control_queue.put_nowait(WorkerControl(WorkerControlAction.SHUTDOWN).encode())
                except queue.Full:
                    pass
            if self.stop_event is not None:
                self.stop_event.set()
            self._condition.notify_all()
        if process is not None:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        monitor = self._monitor
        if monitor is not None and monitor is not threading.current_thread():
            monitor.join(timeout=2)
        self._set_state("STOPPED")
        logger.info("worker_process_stopped")
