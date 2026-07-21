"""Bounded worker restart policy used by the reference deployment."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from sat_ai.worker_contract import WorkerHeartbeat


@dataclass(frozen=True)
class RestartPolicy:
    max_restarts: int = 3
    backoff_ms: int = 250
    heartbeat_timeout_ms: int = 2000


class WorkerSupervisor:
    def __init__(self, start_worker: Callable[[], object], stop_worker: Callable[[object], None], policy: RestartPolicy = RestartPolicy()):
        self.start_worker = start_worker
        self.stop_worker = stop_worker
        self.policy = policy
        self.worker = None
        self.restart_count = 0
        self.heartbeat = WorkerHeartbeat("sat-ai-worker-v1")
        self.state = "STOPPED"

    def start(self) -> None:
        self.worker = self.start_worker()
        self.heartbeat.touch()
        self.state = "READY"

    def heartbeat_tick(self) -> None:
        self.heartbeat.touch()
        if self.state == "READY" and not self.heartbeat.is_alive(self.policy.heartbeat_timeout_ms):
            self.state = "WORKER_LOST"

    def check_liveness(self) -> bool:
        alive = self.heartbeat.is_alive(self.policy.heartbeat_timeout_ms)
        if not alive and self.state == "READY":
            self.state = "WORKER_LOST"
        return alive

    def restart(self) -> bool:
        if self.restart_count >= self.policy.max_restarts:
            self.state = "FAULT"
            return False
        if self.worker is not None:
            self.stop_worker(self.worker)
        self.restart_count += 1
        time.sleep(self.policy.backoff_ms / 1000.0)
        self.start()
        self.state = "RESTARTED"
        return True

    def stop(self) -> None:
        if self.worker is not None:
            self.stop_worker(self.worker)
        self.worker = None
        self.state = "STOPPED"
