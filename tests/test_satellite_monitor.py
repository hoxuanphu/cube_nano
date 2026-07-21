from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path

from flight.journal import SatelliteJournal
from flight.satellite_simulator import _log_health, configure_realtime_logging
from protocol.schemas import RequestKey


def _remove_satellite_handler() -> None:
    flight_logger = logging.getLogger("flight")
    for handler in list(flight_logger.handlers):
        if getattr(handler, "_cube_nano_satellite", False):
            flight_logger.removeHandler(handler)
            handler.close()
    flight_logger.setLevel(logging.NOTSET)
    flight_logger.propagate = True


def test_realtime_status_log_is_human_readable() -> None:
    stream = io.StringIO()
    try:
        configure_realtime_logging("INFO", stream=stream)
        _log_health(
            {
                "state": "READY",
                "worker_state": "READY",
                "worker_heartbeat_age_ms": 3,
                "queue_depth": 1,
                "queue_capacity": 4,
                "active_request_id": 17,
                "worker_restart_count": 0,
                "scheduler_queue_depths": {"ACK": 0, "CONTROL": 1, "FILE": 0},
            }
        )
        output = stream.getvalue()
        assert "[SAT]" in output
        assert "status state=READY worker=READY" in output
        assert "queue=1/4" in output
        assert "scheduler=ACK:0 CONTROL:1 FILE:0" in output
    finally:
        _remove_satellite_handler()


def test_journal_events_are_emitted_to_realtime_log(caplog) -> None:
    with tempfile.TemporaryDirectory() as directory:
        journal = SatelliteJournal(Path(directory) / "state.sqlite3", 1)
        try:
            with caplog.at_level(logging.INFO, logger="flight.journal"):
                journal.append_event(
                    "JOB_STARTED",
                    {"job_key": RequestKey(1, 9).as_dict(), "deadline_monotonic_ns": 123},
                    RequestKey(1, 9),
                )
            assert any("event=JOB_STARTED" in record.message for record in caplog.records)
            assert any("request_key" in record.message for record in caplog.records)
        finally:
            journal.close()
