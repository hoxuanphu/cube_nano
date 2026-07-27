import os
import sqlite3
import json
import tempfile
import time
import unittest
from pathlib import Path

from flight.file_downlink import FileDownlinkCoordinator, TransferState
from flight.journal import SatelliteJournal
from flight.mission_com_scheduler import MissionComScheduler, QueueKind, QueueOverflow
from flight.mission_udp_adapter import MissionUdpAdapter
from flight.state_machine import InvalidTransition, StateMachine
from flight.worker_client import WorkerProcessClient, WorkerProcessPolicy, WorkerQueueFull
from flight.worker_supervisor import RestartPolicy, WorkerSupervisor
from protocol.schemas import ConfigSnapshot, ProductRef, RequestKey
from sat_ai.worker_contract import (
    WORKER_VERSION,
    DeadlineContract,
    WorkerHeartbeatMessage,
    WorkerProtocolError,
    WorkerRequest,
    WorkerResultState,
)


def crashing_worker_target(root, device, request_queue, control_queue, result_queue, stop_event, heartbeat_interval_ms):
    result_queue.put(
        WorkerHeartbeatMessage(WORKER_VERSION, 0, "READY", time.monotonic_ns()).encode()
    )
    request_queue.get(timeout=5)
    os._exit(7)


class Phase2RuntimeTests(unittest.TestCase):
    def test_file_attempt_is_global_and_late_attempt_is_retired(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "bundle.tar"
            bundle.write_bytes(b"payload" * 200)
            product = ProductRef(1, 1, 1)
            coordinator = FileDownlinkCoordinator()
            coordinator.start(7, product, bundle)
            frames = list(coordinator.frames(7))
            self.assertGreater(len(frames), 1)
            self.assertEqual(coordinator.active.state, TransferState.SEND_COMPLETED)
            with self.assertRaisesRegex(RuntimeError, "TRANSFER_RETIRED"):
                coordinator.start(7, product, bundle)

    def test_file_failure_requires_epoch_fence_and_cooldown(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "bundle.tar"
            bundle.write_bytes(b"payload" * 200)
            product = ProductRef(1, 1, 1)
            coordinator = FileDownlinkCoordinator(cooldown_ticks=2)
            coordinator.start(7, product, bundle)
            lease = coordinator.next_frame(7)
            self.assertIsNotNone(lease)
            metadata = json.loads(lease.packet.payload)
            self.assertEqual(len(metadata["source"].encode("ascii")), 23)
            self.assertEqual(len(metadata["destination"].encode("ascii")), 97)
            self.assertTrue(coordinator.complete_frame(lease, "FAILURE"))
            self.assertEqual(coordinator.active.state, TransferState.ABORTING)
            self.assertIsNone(coordinator.next_frame(7))
            with self.assertRaisesRegex(RuntimeError, "TRANSFER_BUSY"):
                coordinator.start(8, product, bundle)
            coordinator.close_abort_fence(7)
            self.assertEqual(coordinator.cooldown_tick(7), TransferState.COOLDOWN)
            self.assertEqual(coordinator.cooldown_tick(7), TransferState.SEND_FAILED)
            coordinator.start(8, product, bundle)
            self.assertFalse(coordinator.complete_frame(lease, "SUCCESS"))
            self.assertEqual(coordinator.active.transfer_id, 8)
            self.assertEqual(coordinator.metrics["late_callbacks"], 1)

    def test_cancel_race_allows_final_data_completion_to_win(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "bundle.tar"
            bundle.write_bytes(b"x" * 100)
            coordinator = FileDownlinkCoordinator(cooldown_ticks=1)
            coordinator.start(1, ProductRef(1, 1, 1), bundle)
            start = coordinator.next_frame(1)
            coordinator.complete_frame(start, "SUCCESS")
            data = coordinator.next_frame(1)
            self.assertEqual(data.packet.packet_type.name, "DATA")
            self.assertEqual(coordinator.cancel(1), "CANCEL_REQUESTED")
            coordinator.complete_frame(data, "SUCCESS")
            end = coordinator.next_frame(1)
            self.assertEqual(end.packet.packet_type.name, "END")
            coordinator.complete_frame(end, "SUCCESS")
            self.assertEqual(coordinator.cooldown_tick(1), TransferState.SEND_COMPLETED)

    def test_scheduler_gate_drives_file_completion_once(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "bundle.tar"
            bundle.write_bytes(b"payload")
            coordinator = FileDownlinkCoordinator()
            coordinator.start(1, ProductRef(1, 1, 1), bundle)
            scheduler = MissionComScheduler()
            coordinator.enqueue_next(1, scheduler)
            adapter = MissionUdpAdapter(scheduler)
            item = adapter.send_next()
            adapter.receive_status("SUCCESS")
            self.assertIsNotNone(coordinator.active.current_lease)
            adapter.receive_return()
            self.assertIsNone(coordinator.active.current_lease)
            self.assertEqual(coordinator.active.next_file_sequence, 1)
            adapter.receive_return()
            adapter.receive_status("SUCCESS")
            self.assertEqual(coordinator.metrics["frames_completed"], 1)

    def test_scheduler_overflow_is_explicit(self):
        scheduler = MissionComScheduler(capacities={QueueKind.ACK: 1, QueueKind.CONTROL: 1, QueueKind.FILE: 1})
        scheduler.enqueue_ack(b"a")
        with self.assertRaises(QueueOverflow):
            scheduler.enqueue_ack(b"b")

    def test_scheduler_flood_obeys_ack_control_file_oracle(self):
        scheduler = MissionComScheduler()
        for index in range(18):
            scheduler.enqueue_ack(f"a{index}".encode())
        for index in range(10):
            scheduler.enqueue_control(f"c{index}".encode())
            scheduler.enqueue_file(f"f{index}".encode())
        order = []
        while any(scheduler.queue_depths().values()) or scheduler.current is not None:
            item = scheduler.poll()
            self.assertIsNotNone(item)
            order.append(item.kind)
            scheduler.mark_upstream_return(item.item_id)
            scheduler.mark_status(item.item_id, "SUCCESS")
        self.assertEqual(order[:10], [QueueKind.ACK] * 8 + [QueueKind.CONTROL, QueueKind.FILE])
        self.assertIn(QueueKind.FILE, order[:18])
        self.assertEqual(order.count(QueueKind.ACK), 18)
        self.assertEqual(order.count(QueueKind.CONTROL), 10)
        self.assertEqual(order.count(QueueKind.FILE), 10)
        self.assertLessEqual(scheduler.metrics["oldest_ack_age_ms"], 1000)
        self.assertLessEqual(scheduler.metrics["oldest_control_age_ms"], 2000)

    def test_state_machine_rejects_implicit_transition(self):
        machine = StateMachine()
        machine.register("job", "job", "QUEUED")
        with self.assertRaises(InvalidTransition):
            machine.transition("job", "SUCCEEDED")
        machine.transition("job", "RUNNING")
        machine.transition("job", "SUCCEEDED")
        self.assertEqual(machine.state("job"), "SUCCEEDED")

    def test_worker_supervisor_has_bounded_restart_count(self):
        supervisor = WorkerSupervisor(lambda: object(), lambda worker: None, RestartPolicy(max_restarts=1, backoff_ms=0))
        supervisor.start()
        self.assertTrue(supervisor.restart())
        self.assertFalse(supervisor.restart())
        self.assertEqual(supervisor.state, "FAULT")
        supervisor.stop()

    def test_worker_contract_round_trip_and_deadline(self):
        deadline = DeadlineContract.after_ms(1000, now_ns=10)
        request = WorkerRequest(RequestKey(2**63, 7), {"snapshot": "immutable"}, deadline)
        self.assertEqual(WorkerRequest.decode(request.encode()), request)
        self.assertFalse(deadline.expired(now_ns=deadline.deadline_monotonic_ns - 1))
        self.assertTrue(deadline.expired(now_ns=deadline.deadline_monotonic_ns))
        malformed = request.encode().replace(b'"api_version":1', b'"api_version":2')
        with self.assertRaises(WorkerProtocolError):
            WorkerRequest.decode(malformed)

    def test_worker_pending_queue_saturation_is_explicit(self):
        client = WorkerProcessClient(
            Path(__file__).resolve().parents[1],
            device="cpu",
            policy=WorkerProcessPolicy(max_pending_jobs=1),
            on_result=lambda result: None,
        )
        try:
            client.submit(WorkerRequest(RequestKey(1, 1), {}, DeadlineContract.after_ms(1000)))
            with self.assertRaises(WorkerQueueFull):
                client.submit(WorkerRequest(RequestKey(1, 2), {}, DeadlineContract.after_ms(1000)))
        finally:
            client.close()

    def test_worker_process_crash_fails_active_job_without_restart_loop(self):
        results = []
        client = WorkerProcessClient(
            Path(__file__).resolve().parents[1],
            device="cpu",
            policy=WorkerProcessPolicy(
                max_pending_jobs=1,
                heartbeat_interval_ms=50,
                heartbeat_timeout_ms=500,
                startup_timeout_ms=30000,
                max_restarts=0,
                restart_window_ms=1000,
                initial_backoff_ms=0,
                cancel_grace_ms=100,
            ),
            on_result=results.append,
            process_target=crashing_worker_target,
        )
        try:
            client.start()
            request = WorkerRequest(RequestKey(1, 1), {}, DeadlineContract.after_ms(2000))
            client.submit(request)
            client.wait(5)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].state, WorkerResultState.FAILED)
            self.assertEqual(results[0].error_code, "WORKER_LOST")
            self.assertEqual(client.state, "FAULT")
        finally:
            client.close()

    def test_restart_reconciliation_fails_staging_and_detects_missing_work(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = SatelliteJournal(Path(directory) / "state.sqlite3", 1)
            broken = RequestKey(1, 1)
            journal.record_command(broken, 1, "digest", {}, "COMMAND_ACCEPTED", {})
            product = ProductRef(1, journal.boot_id, 1)
            journal.create_product(product, RequestKey(1, 2))
            journal.create_transfer(1, product)
            actions = journal.reconcile_after_restart()
            self.assertIn("COMMAND_ACCEPTED_WITHOUT_WORK_ROW", actions)
            self.assertEqual(journal.get_product(product)["state"], "FAILED")
            self.assertEqual(journal.get_transfer(1)["state"], "SEND_FAILED")
            journal.close()

    def test_config_cas_and_cached_command_commit_are_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = SatelliteJournal(Path(directory) / "state.sqlite3", 1)
            key = RequestKey(1, 1)
            snapshot, _ = journal.apply_config_command(key, 1, "digest", {}, 0, 0, 5100, 6100)
            self.assertEqual((snapshot.epoch, snapshot.revision), (0, 1))
            with self.assertRaises(sqlite3.IntegrityError):
                journal.apply_config_command(key, 1, "digest", {}, 0, 1, 5200, 6200)
            current = journal.current_config()
            self.assertEqual((current.epoch, current.revision), (0, 1))
            self.assertEqual((current.model_threshold_bp, current.coverage_limit_bp), (5100, 6100))
            journal.close()

    def test_dispatched_request_cannot_compact_before_related_job_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = SatelliteJournal(Path(directory) / "state.sqlite3", 1)
            key = RequestKey(1, 7)
            product = ProductRef(1, journal.boot_id, 1)
            snapshot = ConfigSnapshot(0, 0, 5000, 6000)
            journal.admit_analysis(
                key,
                0x00010005,
                "digest",
                {},
                {"catalog_epoch": 1, "scene_id": 1, "scene_revision": 1},
                {"x": 0, "y": 0, "width": 256, "height": 256},
                snapshot,
                {"immutable": True},
                product,
            )
            with self.assertRaisesRegex(ValueError, "nonterminal"):
                journal.compact_request(key)
            journal.transition_job(key, {"QUEUED"}, "FAILED", error_code="INJECTED")
            journal.fail_product_for_job(key, "INJECTED")
            journal.compact_request(key)
            self.assertEqual(journal.lookup_request(key, "digest")[0], "RETIRED")
            journal.close()
