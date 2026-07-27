"""Real UDP/process regression coverage for the deployed mission boundary."""

from __future__ import annotations

import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from flight.satellite_simulator import SatelliteSimulator, SatelliteUdpService
from gds.outbox import TcWireProfile
from link_sim.__main__ import _probe_health
from link_sim.transport import UdpMissionEndpoint
from protocol.ccsds import decode_tm_frame
from protocol.messages import decode_application_message
from protocol.profile import MissionProfile
from protocol.schemas import Command, CommandOpcode, RequestKey


ROOT = Path(__file__).resolve().parents[1]


def _free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _start_satellite_service(
    simulator: SatelliteSimulator,
    *,
    satellite_port: int,
    link_port: int,
) -> tuple[threading.Thread, dict[str, object]]:
    holder: dict[str, object] = {}

    def run() -> None:
        try:
            holder["service"] = SatelliteUdpService(
                simulator,
                bind_host="127.0.0.1",
                bind_port=satellite_port,
                link_host="127.0.0.1",
                link_port=link_port,
                link_session_id=1,
            )
        except BaseException as exc:  # surfaced by the caller with context
            holder["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, holder


def _wait_for_service(thread: threading.Thread, holder: dict[str, object]) -> SatelliteUdpService:
    thread.join(timeout=12)
    assert not thread.is_alive(), "satellite UDP HELLO did not complete"
    error = holder.get("error")
    assert error is None, f"satellite UDP HELLO failed: {error!r}"
    service = holder.get("service")
    assert isinstance(service, SatelliteUdpService)
    return service


def _wait_for_session(endpoint: UdpMissionEndpoint, *, generation: int, boot_id: int) -> None:
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        endpoint.receive_egress(timeout_ms=100)
        if (
            endpoint.ready
            and getattr(endpoint, "link_generation", None) == generation
            and endpoint.sender_boot_id == boot_id
        ):
            return
    raise AssertionError(
        f"session binding did not converge to generation={generation}, boot={boot_id}; "
        f"actual={endpoint.session_binding}"
    )


def _start_link(
    *,
    link_port: int,
    gds_port: int,
    satellite_port: int,
    profile: MissionProfile,
    state_directory: Path,
) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "link_sim",
            "--serve",
            "--bind-host",
            "127.0.0.1",
            "--bind-port",
            str(link_port),
            "--gds-host",
            "127.0.0.1",
            "--gds-port",
            str(gds_port),
            "--satellite-host",
            "127.0.0.1",
            "--satellite-port",
            str(satellite_port),
            "--spacecraft-instance-id",
            str(profile.spacecraft_instance_id),
            "--fault-profile",
            str(ROOT / "deploy" / "fault_profiles" / "lossless.yaml"),
            "--state-directory",
            str(state_directory),
            "--seed",
            "42",
            "--run-id",
            "42",
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def test_udp_link_process_delivers_tc_tm_and_rebinds_after_satellite_boot(tmp_path: Path):
    link_port = _free_udp_port()
    gds_port = _free_udp_port()
    satellite_port = _free_udp_port()
    profile = MissionProfile.from_file(ROOT / "protocol" / "mission_profile.yaml")
    link_state = tmp_path / "link-state"
    link = _start_link(
        link_port=link_port,
        gds_port=gds_port,
        satellite_port=satellite_port,
        profile=profile,
        state_directory=link_state,
    )
    endpoint: UdpMissionEndpoint | None = None
    first_service: SatelliteUdpService | None = None
    second_service: SatelliteUdpService | None = None
    first_simulator: SatelliteSimulator | None = None
    second_simulator: SatelliteSimulator | None = None
    try:
        first_simulator = SatelliteSimulator(
            ROOT,
            state_directory=tmp_path / "satellite-state",
            product_directory=tmp_path / "satellite-products",
            start_worker=False,
        )
        thread, holder = _start_satellite_service(
            first_simulator,
            satellite_port=satellite_port,
            link_port=link_port,
        )
        endpoint = UdpMissionEndpoint(
            bind_addr=("127.0.0.1", gds_port),
            link_addr=("127.0.0.1", link_port),
            spacecraft_instance_id=profile.spacecraft_instance_id,
            link_session_id=1,
            sender_boot_id=None,
            handshake_role="gds",
        )
        endpoint.establish_session()
        first_service = _wait_for_service(thread, holder)
        assert endpoint.sender_boot_id == first_simulator.payload.journal.boot_id
        assert endpoint.link_session_id == 1
        assert getattr(endpoint, "link_generation", None) == 1

        request_key = RequestKey(0x1234567890ABCDEF, 7)
        tc = TcWireProfile.from_mission_profile(profile).encode(
            Command(
                CommandOpcode.SCENE_REQUEST_CATALOG,
                profile.spacecraft_instance_id,
                request_key,
                {},
            ),
            packet_sequence=1,
            frame_sequence=1,
        )
        endpoint.send_ingress(tc)
        acknowledgements: list[dict] = []
        catalog_seen = False
        last_link_frame_id = 0
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and (not acknowledgements or not catalog_seen):
            first_service.pump(timeout_ms=20)
            frame = endpoint.receive_egress(timeout_ms=30)
            if frame is None:
                continue
            last_link_frame_id = max(last_link_frame_id, frame.envelope.link_frame_id)
            packet = decode_tm_frame(frame.frame_bytes).packet
            if packet.apid != profile.tm_event_apid:
                continue
            message = decode_application_message(packet.payload).body
            if message.get("event_name") == "COMMAND_ACK":
                acknowledgements.append(message)
            elif message.get("event_name") == "CATALOG_SNAPSHOT":
                catalog_seen = isinstance(message.get("catalog_bundle"), bytes)
        assert acknowledgements
        matching_ack = [
            message
            for message in acknowledgements
            if message.get("request_key") == request_key.as_dict()
        ]
        assert matching_ack, acknowledgements
        assert matching_ack[-1]["stage"] == "EXECUTED"
        assert catalog_seen
        health = _probe_health("127.0.0.1", link_port, 1_000)
        assert health["peer_exchange"]["ready"] is True

        # The persisted journal increments boot on a fresh satellite process.
        first_service.close()
        first_service = None
        first_simulator.close()
        first_simulator = None
        second_simulator = SatelliteSimulator(
            ROOT,
            state_directory=tmp_path / "satellite-state",
            product_directory=tmp_path / "satellite-products",
            start_worker=False,
        )
        assert second_simulator.payload.journal.boot_id == 2
        thread, holder = _start_satellite_service(
            second_simulator,
            satellite_port=satellite_port,
            link_port=link_port,
        )
        second_service = _wait_for_service(thread, holder)
        _wait_for_session(endpoint, generation=2, boot_id=2)
        assert second_service.endpoint.link_session_id == 2
        assert getattr(second_service.endpoint, "link_generation", None) == 2

        # A LinkSimulator process restart restores monotonic allocators from
        # its volume and forces a new generation even when satellite boot is
        # unchanged.  This prevents both stale binding acceptance and link
        # frame primary-key reuse in the durable GDS ledger.
        link.terminate()
        link.wait(timeout=5)
        link = _start_link(
            link_port=link_port,
            gds_port=gds_port,
            satellite_port=satellite_port,
            profile=profile,
            state_directory=link_state,
        )
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            second_service.pump(timeout_ms=20)
            endpoint.receive_egress(timeout_ms=20)
            if (
                getattr(endpoint, "link_generation", None) == 3
                and getattr(second_service.endpoint, "link_generation", None) == 3
            ):
                break
        assert endpoint.link_session_id == 3
        assert getattr(endpoint, "link_generation", None) == 3
        assert second_service.endpoint.link_session_id == 3
        assert getattr(second_service.endpoint, "link_generation", None) == 3

        endpoint.send_ingress(
            TcWireProfile.from_mission_profile(profile).encode(
                Command(
                    CommandOpcode.SCENE_REQUEST_CATALOG,
                    profile.spacecraft_instance_id,
                    RequestKey(0x1234567890ABCDEF, 8),
                    {},
                ),
                packet_sequence=2,
                frame_sequence=2,
            )
        )
        post_restart_link_frame_id = 0
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and post_restart_link_frame_id == 0:
            second_service.pump(timeout_ms=20)
            frame = endpoint.receive_egress(timeout_ms=30)
            if frame is not None:
                post_restart_link_frame_id = frame.envelope.link_frame_id
        assert post_restart_link_frame_id > last_link_frame_id
    finally:
        if endpoint is not None:
            endpoint.close()
        if second_service is not None:
            second_service.close()
        if first_service is not None:
            first_service.close()
        if second_simulator is not None:
            second_simulator.close()
        if first_simulator is not None:
            first_simulator.close()
        if link.poll() is None:
            link.terminate()
            try:
                link.wait(timeout=5)
            except subprocess.TimeoutExpired:
                link.kill()
                link.wait(timeout=5)
