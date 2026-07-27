"""Mission-facing bridge that binds schedulers, transport, and LinkSimulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from flight.mission_com_scheduler import MissionComScheduler, QueueKind
from flight.mission_udp_adapter import MissionUdpAdapter

from .control import LinkControlMessage
from .fault_model import FaultProfile
from .link_simulator import LinkSimulator
from .transport import Direction, InMemoryTransport, Transport, TransportFrame
from .virtual_clock import VirtualClock


Receiver = Callable[[TransportFrame], Any]


@dataclass
class _Pending:
    direction: Direction
    receiver: Receiver | None
    result: Any = None
    receiver_error: Exception | None = None
    status: str | None = None


class MissionLink:
    """Synchronous local boundary with the same completion contract as UDP.

    The local SIL uses ``InMemoryTransport`` but still exercises the link
    simulator, scheduler and completion gate. A deployment can replace the
    transport with ``UdpTransport`` and use the same session/envelope contract.
    """

    def __init__(
        self,
        *,
        simulation_run_id: int,
        seed: int,
        spacecraft_instance_id: int,
        sender_boot_id: int,
        uplink_profile: FaultProfile | None = None,
        downlink_profile: FaultProfile | None = None,
        clock: VirtualClock | None = None,
        transport: Transport | None = None,
    ) -> None:
        self.clock = clock or VirtualClock()
        self.transport = transport or InMemoryTransport()
        self.simulator = LinkSimulator(
            simulation_run_id=simulation_run_id,
            seed=seed,
            uplink_profile=uplink_profile or FaultProfile(),
            downlink_profile=downlink_profile or FaultProfile(),
            clock=self.clock,
            transport=self.transport,
        )
        self.spacecraft_instance_id = spacecraft_instance_id
        self.sender_boot_id = sender_boot_id
        self.session_id = self.simulator.create_session(spacecraft_instance_id, sender_boot_id)
        session = self.simulator.session_manager.get_session(self.session_id)
        if session is None:
            raise RuntimeError("LinkSimulator did not create a session")
        self.link_generation = session.generation
        self._next_sender_frame_id = 1
        self._satellite: Any = None
        self._ground: Any = None
        self._pending: _Pending | None = None
        self._scheduler = MissionComScheduler()
        self.adapter = MissionUdpAdapter(self._scheduler, self._send_frame)

    @property
    def control_events(self) -> tuple[LinkControlMessage, ...]:
        return tuple(self.simulator.control_events)

    def attach_satellite(self, satellite: Any) -> None:
        self._satellite = satellite

    def attach_ground(self, ground: Any) -> None:
        self._ground = ground

    def _receiver_for(self, direction: Direction) -> Receiver | None:
        target = self._satellite if direction is Direction.INGRESS else self._ground
        if target is None:
            return None
        method_name = "receive_transport_frame"
        method = getattr(target, method_name, None)
        if method is None:
            raise TypeError(f"link endpoint does not implement {method_name}()")
        return method

    def _send_frame(
        self,
        frame: bytes,
        status_callback: Callable[[str], None],
        return_callback: Callable[[], None],
    ) -> None:
        pending = self._pending
        if pending is None:
            status_callback("FRAME_REJECTED")
            return_callback()
            return

        def on_consumed(message: LinkControlMessage) -> None:
            pending.status = message.status or "FRAME_CONSUMED"
            status_callback(pending.status)
            return_callback()

        def on_delivered(transport_frame: TransportFrame) -> str | None:
            receiver = pending.receiver
            if receiver is None:
                return None
            try:
                pending.result = receiver(transport_frame)
                return None
            except Exception as exc:  # receiver failure is a terminal link result
                pending.receiver_error = exc
                return "FRAME_FAILED"

        envelope = self._ingress_envelope(frame)
        link_id = self.simulator.admit_frame(
            TransportFrame(envelope, frame),
            direction=pending.direction,
            on_consumed=on_consumed,
            on_delivered=on_delivered,
        )
        if link_id is None:
            status_callback("FRAME_REJECTED")
            return_callback()
            return
        self.simulator.run_until_idle()
        if isinstance(self.transport, InMemoryTransport):
            while self.transport.receive() is not None:
                pass

    def _ingress_envelope(self, frame: bytes):
        from .transport import SidebandEnvelope

        sender_frame_id = self._next_sender_frame_id
        self._next_sender_frame_id += 1
        return SidebandEnvelope(
            magic=SidebandEnvelope.MAGIC,
            version=SidebandEnvelope.VERSION,
            direction=Direction.INGRESS,
            reserved=0,
            spacecraft_instance_id=self.spacecraft_instance_id,
            sender_boot_id=0,
            link_session_id=self.session_id,
            sender_frame_id=sender_frame_id,
            link_frame_id=0,
            file_epoch_id=0,
            frame_length=len(frame),
        )

    def transmit(
        self,
        frame: bytes,
        *,
        direction: Direction,
        receiver: Receiver | None = None,
        queue_kind: QueueKind = QueueKind.CONTROL,
    ) -> Any:
        if self._pending is not None or self.adapter.gate is not None or self._scheduler.current is not None:
            raise RuntimeError("FRAME_IN_FLIGHT")
        direction = Direction(direction)
        pending = _Pending(direction, receiver if receiver is not None else self._receiver_for(direction))
        self._pending = pending
        try:
            self._scheduler.enqueue(queue_kind, bytes(frame))
            self.adapter.send_next()
            if self.adapter.gate is not None or self._scheduler.current is not None:
                raise RuntimeError("FRAME_COMPLETION_TIMEOUT")
            if pending.receiver_error is not None:
                raise pending.receiver_error
            if pending.status in {"FRAME_FAILED", "FRAME_REJECTED", "FRAME_LOST", "FILE_EPOCH_REJECTED"}:
                raise RuntimeError(pending.status)
            result = pending.result
        finally:
            self._pending = None
        if direction is Direction.INGRESS:
            self._drain_satellite_tm()
        return result

    def _drain_satellite_tm(self) -> None:
        """Stage flight TM for its endpoint-owned egress pump.

        The GDS endpoint owns when a buffered TM frame crosses the send fence.
        A current ``SatelliteSimulator`` first moves APID 2 telemetry into the
        same scheduler as APID 3, preserving its durable TM channel order.
        """
        if self._satellite is None:
            return
        enqueue = getattr(self._satellite, "enqueue_pending_tm_frames", None)
        if enqueue is not None:
            enqueue()
            return
        producer = getattr(self._satellite, "drain_pending_tm_frames", None)
        if producer is None:
            return
        for frame in producer():
            self.send_downlink(frame)

    def send_uplink(self, frame: bytes, receiver: Receiver | None = None) -> Any:
        return self.transmit(frame, direction=Direction.INGRESS, receiver=receiver)

    def send_downlink(self, frame: bytes, receiver: Receiver | None = None) -> Any:
        return self.transmit(frame, direction=Direction.EGRESS, receiver=receiver, queue_kind=QueueKind.FILE)

    def downlink_transfer(self, transfer_id: int) -> int:
        """Move every satellite TM/file frame through the downlink path."""
        if self._satellite is None:
            raise RuntimeError("satellite endpoint is not attached")
        producer = getattr(self._satellite, "drain_downlink_frames", None)
        if producer is None:
            raise TypeError("satellite endpoint does not implement drain_downlink_frames()")
        count = 0
        for frame in producer(transfer_id):
            self.send_downlink(frame)
            count += 1
        return count

    def reset(self, sender_boot_id: int) -> int:
        self.adapter.reset()
        self.sender_boot_id = sender_boot_id
        self.session_id = self.simulator.reset_session(self.spacecraft_instance_id, sender_boot_id)
        self._next_sender_frame_id = 1
        session = self.simulator.session_manager.get_session(self.session_id)
        self.link_generation = 0 if session is None else session.generation
        return self.session_id

    def health(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "link_generation": self.link_generation,
            "scheduler": {
                "state": self._scheduler.state.value,
                "queue_depths": self._scheduler.queue_depths(),
                "metrics": dict(self._scheduler.metrics),
            },
            "link": self.simulator.get_stats(),
        }

    def close(self) -> None:
        self.adapter.reset()
        self.transport.close()
