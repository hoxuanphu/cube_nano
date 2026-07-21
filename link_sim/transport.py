"""Transport abstraction for Link Simulator.

Implements in-memory and UDP transport with sideband envelope according to
Section 9.1 of the simulation plan.
"""

import json
import socket
import struct
import time
from abc import ABC, abstractmethod
from dataclasses import replace
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable, Optional, Tuple


class Direction(IntEnum):
    """Frame direction in sideband envelope."""
    INGRESS = 0  # Sender -> Link Simulator
    EGRESS = 1   # Link Simulator -> Receiver


# The control plane intentionally uses a separate datagram prefix.  It is not
# an emulated CCSDS frame and therefore cannot be fault-injected or mistaken
# for SidebandEnvelope bytes on the mission data plane.
SESSION_CONTROL_MAGIC = b"CSH2"
SESSION_REPLY_MAGIC = b"CSH2!"


def _bounded_int(value: Any, label: str, maximum: int, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if value < (0 if allow_zero else 1) or value > maximum:
        lower = 0 if allow_zero else 1
        raise ValueError(f"{label} must be in [{lower}, {maximum}]")
    return int(value)


def encode_session_hello(
    *,
    role: str,
    spacecraft_instance_id: int,
    sender_boot_id: int | None,
) -> bytes:
    """Encode a peer-authenticated UDP session HELLO request."""

    if role not in {"gds", "satellite"}:
        raise ValueError("session HELLO role must be gds or satellite")
    instance = _bounded_int(spacecraft_instance_id, "spacecraft_instance_id", 0xFFFFFFFFFFFFFFFF)
    if sender_boot_id is None:
        if role != "gds":
            raise ValueError("satellite HELLO requires sender_boot_id")
        boot: int | None = None
    else:
        boot = _bounded_int(sender_boot_id, "sender_boot_id", 0xFFFFFFFF, allow_zero=True)
    body = {
        "kind": "HELLO",
        "role": role,
        "sender_boot_id": boot,
        "spacecraft_instance_id": instance,
    }
    return SESSION_CONTROL_MAGIC + json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


def encode_session_binding(
    *,
    spacecraft_instance_id: int,
    sender_boot_id: int,
    link_session_id: int,
    link_generation: int,
) -> bytes:
    """Encode LinkSimulator's authoritative session binding response."""

    body = {
        "kind": "SESSION",
        "link_generation": _bounded_int(link_generation, "link_generation", 0xFFFFFFFFFFFFFFFF),
        "link_session_id": _bounded_int(link_session_id, "link_session_id", 0xFFFFFFFFFFFFFFFF),
        "sender_boot_id": _bounded_int(sender_boot_id, "sender_boot_id", 0xFFFFFFFF, allow_zero=True),
        "spacecraft_instance_id": _bounded_int(spacecraft_instance_id, "spacecraft_instance_id", 0xFFFFFFFFFFFFFFFF),
    }
    return SESSION_REPLY_MAGIC + json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


def decode_session_control(datagram: bytes) -> dict[str, Any] | None:
    """Decode a control datagram, returning ``None`` for mission data bytes."""

    data = bytes(datagram)
    if data.startswith(SESSION_REPLY_MAGIC):
        prefix = SESSION_REPLY_MAGIC
        expected_kind = "SESSION"
    elif data.startswith(SESSION_CONTROL_MAGIC):
        prefix = SESSION_CONTROL_MAGIC
        expected_kind = "HELLO"
    else:
        return None
    try:
        body = json.loads(data[len(prefix) :].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("session control payload is not valid JSON") from exc
    if not isinstance(body, dict) or body.get("kind") != expected_kind:
        raise ValueError("session control kind is invalid")
    body["spacecraft_instance_id"] = _bounded_int(
        body.get("spacecraft_instance_id"), "spacecraft_instance_id", 0xFFFFFFFFFFFFFFFF
    )
    if expected_kind == "HELLO":
        role = body.get("role")
        if role not in {"gds", "satellite"}:
            raise ValueError("session HELLO role is invalid")
        boot = body.get("sender_boot_id")
        if role == "satellite" or boot is not None:
            body["sender_boot_id"] = _bounded_int(boot, "sender_boot_id", 0xFFFFFFFF, allow_zero=True)
    else:
        body["sender_boot_id"] = _bounded_int(
            body.get("sender_boot_id"), "sender_boot_id", 0xFFFFFFFF, allow_zero=True
        )
        body["link_session_id"] = _bounded_int(
            body.get("link_session_id"), "link_session_id", 0xFFFFFFFFFFFFFFFF
        )
        body["link_generation"] = _bounded_int(
            body.get("link_generation"), "link_generation", 0xFFFFFFFFFFFFFFFF
        )
    return body


@dataclass(frozen=True)
class SidebandEnvelope:
    """Transport sideband metadata per Section 9.1.

    Ingress (sender -> Link Simulator):
        - direction must be INGRESS (0)
        - spacecraft_instance_id is target
        - sender_boot_id, link_frame_id, file_epoch_id must be 0

    Egress (Link Simulator -> receiver):
        - direction must be EGRESS (1)
        - spacecraft_instance_id is source
        - sender_boot_id is current boot
        - link_frame_id assigned by Link Simulator after ordered ingress
        - file_epoch_id assigned for APID 3 FilePacket only
    """
    magic: int  # 0x43534c31 = "CSL1"
    version: int  # 1
    direction: Direction
    reserved: int  # Must be 0
    spacecraft_instance_id: int  # U64
    sender_boot_id: int  # U32
    link_session_id: int  # U64
    sender_frame_id: int  # U64
    link_frame_id: int  # U64
    file_epoch_id: int  # U64
    frame_length: int  # U16
    # Version 2 carries duplicate identity on the UDP wire.  Keep this at the
    # end with a default so typed version-1 callers remain source-compatible.
    copy_index: int = 0  # U32

    MAGIC = 0x43534c31
    VERSION = 1
    VERSION_WITH_COPY_INDEX = 2
    STRUCT_FORMAT = ">IBBHQIQQQQH"  # Big-endian: I(magic) B(ver) B(dir) H(res) Q(instance) I(boot) Q(session) Q(sender) Q(link) Q(epoch) H(len)
    HEADER_SIZE = struct.calcsize(STRUCT_FORMAT)
    STRUCT_FORMAT_WITH_COPY_INDEX = STRUCT_FORMAT + "I"
    HEADER_SIZE_WITH_COPY_INDEX = struct.calcsize(STRUCT_FORMAT_WITH_COPY_INDEX)

    @classmethod
    def header_size_for_version(cls, version: int) -> int:
        if version == cls.VERSION:
            return cls.HEADER_SIZE
        if version == cls.VERSION_WITH_COPY_INDEX:
            return cls.HEADER_SIZE_WITH_COPY_INDEX
        raise ValueError(f"Unsupported version: {version}")

    @classmethod
    def header_size_from_bytes(cls, data: bytes) -> int:
        if len(data) < 5:
            raise ValueError("Envelope too short to read magic and version")
        magic, version = struct.unpack(">IB", data[:5])
        if magic != cls.MAGIC:
            raise ValueError(f"Invalid magic: {magic:#x} != {cls.MAGIC:#x}")
        return cls.header_size_for_version(version)

    @classmethod
    def from_bytes(cls, data: bytes) -> "SidebandEnvelope":
        """Parse sideband envelope from bytes."""
        header_size = cls.header_size_from_bytes(data)
        if len(data) < header_size:
            raise ValueError(f"Envelope too short: {len(data)} < {header_size}")
        version = int(data[4])
        format_string = (
            cls.STRUCT_FORMAT
            if version == cls.VERSION
            else cls.STRUCT_FORMAT_WITH_COPY_INDEX
        )
        fields = struct.unpack(format_string, data[:header_size])
        magic, version, direction, reserved = fields[:4]
        spacecraft_instance_id, sender_boot_id, link_session_id = fields[4:7]
        sender_frame_id, link_frame_id, file_epoch_id, frame_length = fields[7:11]
        copy_index = 0 if version == cls.VERSION else int(fields[-1])

        return cls(
            magic=magic,
            version=version,
            direction=Direction(direction),
            reserved=reserved,
            spacecraft_instance_id=spacecraft_instance_id,
            sender_boot_id=sender_boot_id,
            link_session_id=link_session_id,
            sender_frame_id=sender_frame_id,
            link_frame_id=link_frame_id,
            file_epoch_id=file_epoch_id,
            frame_length=frame_length,
            copy_index=copy_index,
        )

    def to_bytes(self) -> bytes:
        """Serialize sideband envelope to bytes."""
        if self.version == self.VERSION:
            if self.copy_index != 0:
                raise ValueError("version 1 envelope cannot serialize copy_index")
            return struct.pack(
                self.STRUCT_FORMAT,
                self.magic,
                self.version,
                self.direction.value,
                self.reserved,
                self.spacecraft_instance_id,
                self.sender_boot_id,
                self.link_session_id,
                self.sender_frame_id,
                self.link_frame_id,
                self.file_epoch_id,
                self.frame_length,
            )
        if self.version != self.VERSION_WITH_COPY_INDEX:
            raise ValueError(f"Unsupported version: {self.version}")
        return struct.pack(
            self.STRUCT_FORMAT_WITH_COPY_INDEX,
            self.magic,
            self.version,
            self.direction.value,
            self.reserved,
            self.spacecraft_instance_id,
            self.sender_boot_id,
            self.link_session_id,
            self.sender_frame_id,
            self.link_frame_id,
            self.file_epoch_id,
            self.frame_length,
            self.copy_index,
        )

    def validate_ingress(self) -> None:
        """Validate ingress contract: direction=0, boot/link/epoch IDs must be 0."""
        if self.magic != self.MAGIC:
            raise ValueError(f"Invalid magic: {self.magic:#x} != {self.MAGIC:#x}")
        if self.version not in {self.VERSION, self.VERSION_WITH_COPY_INDEX}:
            raise ValueError(f"Unsupported version: {self.version}")
        if not 0 <= self.copy_index <= 0xFFFFFFFF:
            raise ValueError("copy_index must fit U32")
        if self.version == self.VERSION and self.copy_index != 0:
            raise ValueError("version 1 envelope cannot carry copy_index")
        if self.reserved != 0:
            raise ValueError(f"Reserved field must be 0: {self.reserved}")
        if self.direction != Direction.INGRESS:
            raise ValueError(f"Ingress must have direction=0: {self.direction}")
        if self.sender_boot_id != 0:
            raise ValueError(f"Ingress sender_boot_id must be 0: {self.sender_boot_id}")
        if self.link_frame_id != 0:
            raise ValueError(f"Ingress link_frame_id must be 0: {self.link_frame_id}")
        if self.file_epoch_id != 0:
            raise ValueError(f"Ingress file_epoch_id must be 0: {self.file_epoch_id}")
        if self.spacecraft_instance_id == 0:
            raise ValueError("Ingress spacecraft_instance_id must be > 0")
        if self.link_session_id == 0:
            raise ValueError("Ingress link_session_id must be > 0")
        if self.sender_frame_id == 0:
            raise ValueError("Ingress sender_frame_id must be > 0")
        if not 0 < self.frame_length <= 0xFFFF:
            raise ValueError("Ingress frame_length must be in [1, 65535]")

    def validate_egress(
        self,
        *,
        expected_spacecraft_instance_id: int | None = None,
        expected_sender_boot_id: int | None = None,
        expected_link_session_id: int | None = None,
    ) -> None:
        """Validate egress contract and, when supplied, its binding identity."""
        if self.magic != self.MAGIC:
            raise ValueError(f"Invalid magic: {self.magic:#x} != {self.MAGIC:#x}")
        if self.version not in {self.VERSION, self.VERSION_WITH_COPY_INDEX}:
            raise ValueError(f"Unsupported version: {self.version}")
        if not 0 <= self.copy_index <= 0xFFFFFFFF:
            raise ValueError("copy_index must fit U32")
        if self.version == self.VERSION and self.copy_index != 0:
            raise ValueError("version 1 envelope cannot carry copy_index")
        if self.reserved != 0:
            raise ValueError(f"Reserved field must be 0: {self.reserved}")
        if self.direction != Direction.EGRESS:
            raise ValueError(f"Egress must have direction=1: {self.direction}")
        if self.link_frame_id == 0:
            raise ValueError("Egress link_frame_id must be > 0")
        if self.spacecraft_instance_id == 0:
            raise ValueError("Egress spacecraft_instance_id must be > 0")
        if self.link_session_id == 0:
            raise ValueError("Egress link_session_id must be > 0")
        if self.sender_frame_id == 0:
            raise ValueError("Egress sender_frame_id must be > 0")
        if not 0 < self.frame_length <= 0xFFFF:
            raise ValueError("Egress frame_length must be in [1, 65535]")
        if expected_spacecraft_instance_id is not None and self.spacecraft_instance_id != expected_spacecraft_instance_id:
            raise ValueError("Egress spacecraft_instance_id does not match binding")
        if expected_sender_boot_id is not None and self.sender_boot_id != expected_sender_boot_id:
            raise ValueError("Egress sender_boot_id does not match binding")
        if expected_link_session_id is not None and self.link_session_id != expected_link_session_id:
            raise ValueError("Egress link_session_id does not match binding")


@dataclass(frozen=True)
class TransportFrame:
    """Complete transport frame with sideband and CCSDS frame bytes."""
    envelope: SidebandEnvelope
    frame_bytes: bytes
    copy_index: int = 0

    def __post_init__(self):
        if isinstance(self.copy_index, bool) or not 0 <= self.copy_index <= 0xFFFFFFFF:
            raise ValueError("copy_index must fit U32")
        if len(self.frame_bytes) != self.envelope.frame_length:
            raise ValueError(
                f"Frame length mismatch: {len(self.frame_bytes)} != {self.envelope.frame_length}"
            )


class Transport(ABC):
    """Abstract transport interface."""

    @abstractmethod
    def send(self, envelope: SidebandEnvelope, frame_bytes: bytes) -> None:
        """Send a frame with sideband envelope."""
        pass

    @abstractmethod
    def receive(self, timeout_ms: Optional[int] = None) -> Optional[TransportFrame]:
        """Receive a frame. Returns None on timeout."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the transport."""
        pass


class InMemoryTransport(Transport):
    """In-memory transport for unit/integration tests.

    Uses typed sideband (no serialization) and delivers frames synchronously
    via virtual clock/event queue.
    """

    def __init__(self):
        self._queue: list[TransportFrame] = []

    def send(self, envelope: SidebandEnvelope, frame_bytes: bytes) -> None:
        """Enqueue frame for delivery."""
        self.send_transport_frame(TransportFrame(envelope=envelope, frame_bytes=frame_bytes))

    def send_transport_frame(self, frame: TransportFrame) -> None:
        """Enqueue a complete typed frame, preserving duplicate identity."""
        self._queue.append(frame)

    def receive(self, timeout_ms: Optional[int] = None) -> Optional[TransportFrame]:
        """Dequeue frame. Timeout ignored in synchronous mode."""
        if not self._queue:
            return None
        return self._queue.pop(0)

    def close(self) -> None:
        """Clear queue."""
        self._queue.clear()


class UdpTransport(Transport):
    """UDP transport with serialized sideband envelope.

    Each UDP datagram contains: sideband envelope (big-endian) + CCSDS frame bytes.
    Section 9.1: envelope is not CCSDS wire bytes, not subject to fault injection,
    and does not count toward simulated link bitrate.
    """

    MAX_DATAGRAM_BYTES = 65_507

    def __init__(
        self,
        bind_addr: Tuple[str, int],
        peer_addr: Optional[Tuple[str, int]] = None,
        *,
        expected_direction: Direction | None = None,
    ):
        """Initialize UDP transport.

        Args:
            bind_addr: Local (host, port) to bind
            peer_addr: Remote (host, port) for send; can be set later
        """
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind(bind_addr)
        self._peer_addr = peer_addr
        self._peer_addresses = self._resolve_peer(peer_addr)
        self._expected_direction = None if expected_direction is None else Direction(expected_direction)

    @staticmethod
    def _resolve_peer(peer_addr: Optional[Tuple[str, int]]) -> set[tuple[str, int]]:
        """Resolve a configured peer once so Compose DNS compares by address."""
        if peer_addr is None:
            return set()
        host, port = peer_addr
        try:
            values = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_DGRAM)
        except socket.gaierror as exc:
            raise ValueError(f"unable to resolve UDP peer {host}:{port}") from exc
        return {(str(value[4][0]), int(value[4][1])) for value in values}

    def send(self, envelope: SidebandEnvelope, frame_bytes: bytes) -> None:
        """Send frame as UDP datagram."""
        if self._peer_addr is None:
            raise RuntimeError("Peer address not set")

        if len(frame_bytes) != envelope.frame_length:
            raise ValueError(
                f"Frame length mismatch: {len(frame_bytes)} != {envelope.frame_length}"
            )

        if envelope.direction is Direction.INGRESS:
            envelope.validate_ingress()
        else:
            envelope.validate_egress()

        envelope_bytes = envelope.to_bytes()
        datagram = envelope_bytes + frame_bytes
        if len(datagram) > self.MAX_DATAGRAM_BYTES:
            raise ValueError("UDP datagram exceeds the maximum payload size")
        self._socket.sendto(datagram, self._peer_addr)

    def send_transport_frame(self, frame: TransportFrame) -> None:
        """Serialize a complete frame, preserving a fault duplicate index."""

        envelope = frame.envelope
        if frame.copy_index != envelope.copy_index:
            envelope = replace(
                envelope,
                version=(
                    SidebandEnvelope.VERSION
                    if frame.copy_index == 0 and envelope.version == SidebandEnvelope.VERSION
                    else SidebandEnvelope.VERSION_WITH_COPY_INDEX
                ),
                copy_index=frame.copy_index,
            )
        self.send(envelope, frame.frame_bytes)

    def send_control(self, datagram: bytes) -> None:
        """Send a bounded session-control datagram to the configured peer."""

        if self._peer_addr is None:
            raise RuntimeError("Peer address not set")
        payload = bytes(datagram)
        if not payload or len(payload) > self.MAX_DATAGRAM_BYTES:
            raise ValueError("invalid UDP control datagram length")
        if decode_session_control(payload) is None:
            raise ValueError("UDP control datagram has an unknown prefix")
        self._socket.sendto(payload, self._peer_addr)

    def receive_datagram(self, timeout_ms: Optional[int] = None) -> Optional[bytes]:
        """Receive one peer-validated raw datagram for data/control demux."""

        if timeout_ms is not None:
            self._socket.settimeout(timeout_ms / 1000.0)
        else:
            self._socket.settimeout(None)
        try:
            datagram, peer = self._socket.recvfrom(self.MAX_DATAGRAM_BYTES)
        # Windows reports an ICMP port-unreachable from an earlier UDP HELLO
        # as ECONNRESET on the next recvfrom.  HELLO is retried, so this is a
        # transient reachability result rather than a session/data failure.
        except (socket.timeout, TimeoutError, ConnectionResetError):
            return None
        if self._peer_addr is not None and (str(peer[0]), int(peer[1])) not in self._peer_addresses:
            raise ValueError(f"UDP peer mismatch: received {peer}, expected {self._peer_addr}")
        return bytes(datagram)

    def decode_datagram(self, datagram: bytes) -> TransportFrame:
        """Decode one mission data datagram after peer authentication."""

        if decode_session_control(datagram) is not None:
            raise ValueError("UDP control datagram is not a mission frame")
        if len(datagram) < SidebandEnvelope.HEADER_SIZE:
            raise ValueError("UDP datagram is shorter than its sideband envelope")
        envelope = SidebandEnvelope.from_bytes(datagram)
        header_size = SidebandEnvelope.header_size_for_version(envelope.version)
        frame_bytes = datagram[header_size:]
        if len(frame_bytes) != envelope.frame_length:
            raise ValueError(
                f"Frame length mismatch: {len(frame_bytes)} != {envelope.frame_length}"
            )
        if self._expected_direction is not None and envelope.direction is not self._expected_direction:
            raise ValueError(
                f"UDP direction mismatch: received {envelope.direction.name}, "
                f"expected {self._expected_direction.name}"
            )
        if envelope.direction is Direction.INGRESS:
            envelope.validate_ingress()
        else:
            envelope.validate_egress()
        return TransportFrame(
            envelope=envelope,
            frame_bytes=frame_bytes,
            copy_index=envelope.copy_index,
        )

    def receive(self, timeout_ms: Optional[int] = None) -> Optional[TransportFrame]:
        """Receive frame from UDP datagram."""
        datagram = self.receive_datagram(timeout_ms)
        if datagram is None:
            return None
        return self.decode_datagram(datagram)

    def close(self) -> None:
        """Close UDP socket."""
        self._socket.close()

    def set_peer(self, peer_addr: Tuple[str, int]) -> None:
        """Set peer address for sending."""
        self._peer_addr = peer_addr
        self._peer_addresses = self._resolve_peer(peer_addr)

    @property
    def bound_address(self) -> tuple[str, int]:
        host, port = self._socket.getsockname()
        return str(host), int(port)


class UdpMissionEndpoint:
    """One mission-side UDP endpoint with a strict ingress/egress contract.

    Both GDS and satellite use this adapter.  Outgoing CCSDS frames are always
    wrapped as ingress datagrams for LinkSimulator; incoming datagrams must be
    egress frames from the configured link peer and must match the active
    spacecraft/session/boot binding.
    """

    def __init__(
        self,
        *,
        bind_addr: Tuple[str, int],
        link_addr: Tuple[str, int],
        spacecraft_instance_id: int,
        link_session_id: int,
        sender_boot_id: int | None,
        handshake_role: str | None = None,
        on_session: Callable[[dict[str, int]], None] | None = None,
        handshake_interval_s: float = 1.0,
    ) -> None:
        if spacecraft_instance_id <= 0:
            raise ValueError("spacecraft_instance_id must be positive")
        if link_session_id <= 0:
            raise ValueError("link_session_id must be positive")
        if sender_boot_id is not None and not 0 <= sender_boot_id <= 0xFFFFFFFF:
            raise ValueError("sender_boot_id must fit U32")
        if handshake_role is not None and handshake_role not in {"gds", "satellite"}:
            raise ValueError("handshake_role must be gds, satellite, or None")
        if handshake_role == "satellite" and sender_boot_id is None:
            raise ValueError("satellite handshake requires its durable sender_boot_id")
        if handshake_interval_s <= 0:
            raise ValueError("handshake_interval_s must be positive")
        self.transport = UdpTransport(
            bind_addr,
            link_addr,
            expected_direction=Direction.EGRESS,
        )
        self.spacecraft_instance_id = int(spacecraft_instance_id)
        self.link_session_id = int(link_session_id)
        self.sender_boot_id = None if sender_boot_id is None else int(sender_boot_id)
        self.handshake_role = handshake_role
        self._session_ready = handshake_role is None
        self._on_session = on_session
        self._handshake_interval_s = float(handshake_interval_s)
        self._last_hello_monotonic = 0.0
        self._pending_datagrams: list[bytes] = []
        self._next_sender_frame_id = 1

    @property
    def bound_address(self) -> tuple[str, int]:
        return self.transport.bound_address

    @property
    def ready(self) -> bool:
        try:
            self.transport.bound_address
        except OSError:
            return False
        return self._session_ready

    @property
    def session_binding(self) -> dict[str, int] | None:
        if not self._session_ready or self.sender_boot_id is None:
            return None
        return {
            "spacecraft_instance_id": self.spacecraft_instance_id,
            "sender_boot_id": self.sender_boot_id,
            "link_session_id": self.link_session_id,
            "link_generation": getattr(self, "link_generation", 1),
        }

    def set_session_callback(self, callback: Callable[[dict[str, int]], None] | None) -> None:
        self._on_session = callback

    def _send_hello(self, *, force: bool = False) -> None:
        if self.handshake_role is None:
            return
        now = time.monotonic()
        if not force and now - self._last_hello_monotonic < self._handshake_interval_s:
            return
        boot = self.sender_boot_id if self.handshake_role == "satellite" else None
        self.transport.send_control(
            encode_session_hello(
                role=self.handshake_role,
                spacecraft_instance_id=self.spacecraft_instance_id,
                sender_boot_id=boot,
            )
        )
        self._last_hello_monotonic = now

    def _apply_session_binding(self, control: dict[str, Any]) -> None:
        if control.get("kind") != "SESSION":
            raise ValueError("mission endpoint received a non-session control message")
        if int(control["spacecraft_instance_id"]) != self.spacecraft_instance_id:
            raise ValueError("session control spacecraft instance does not match endpoint")
        next_session = int(control["link_session_id"])
        next_generation = int(control["link_generation"])
        next_boot = int(control["sender_boot_id"])
        current_generation = getattr(self, "link_generation", 0)
        if self._session_ready:
            if next_generation < current_generation:
                raise ValueError("session control generation regressed")
            if next_generation == current_generation and (
                next_session != self.link_session_id or next_boot != self.sender_boot_id
            ):
                raise ValueError("session control changed identity without a generation increment")
        changed = (
            not self._session_ready
            or next_session != self.link_session_id
            or next_generation != current_generation
            or next_boot != self.sender_boot_id
        )
        self.link_session_id = next_session
        self.link_generation = next_generation
        self.sender_boot_id = next_boot
        self._session_ready = True
        if changed and self._on_session is not None:
            self._on_session(dict(self.session_binding or {}))

    def _consume_datagram(self, datagram: bytes) -> TransportFrame | None:
        control = decode_session_control(datagram)
        if control is not None:
            self._apply_session_binding(control)
            return None
        frame = self.transport.decode_datagram(datagram)
        frame.envelope.validate_egress(
            expected_spacecraft_instance_id=self.spacecraft_instance_id,
            expected_sender_boot_id=self.sender_boot_id,
            expected_link_session_id=self.link_session_id,
        )
        return frame

    def establish_session(self, timeout_ms: int = 15_000) -> dict[str, int] | None:
        """Block startup only until LinkSimulator returns an authenticated binding."""

        if self.handshake_role is None:
            return self.session_binding
        if timeout_ms <= 0:
            raise ValueError("session handshake timeout_ms must be positive")
        deadline = time.monotonic() + timeout_ms / 1000.0
        while time.monotonic() < deadline:
            self._send_hello(force=True)
            remaining_ms = max(1, min(250, int((deadline - time.monotonic()) * 1000)))
            datagram = self.transport.receive_datagram(remaining_ms)
            if datagram is None:
                continue
            frame = self._consume_datagram(datagram)
            if frame is not None:
                self._pending_datagrams.append(datagram)
            if self._session_ready:
                return self.session_binding
        raise TimeoutError("UDP session HELLO timed out waiting for LinkSimulator")

    def send_ingress(self, frame_bytes: bytes) -> SidebandEnvelope:
        frame = bytes(frame_bytes)
        if not frame:
            raise ValueError("mission frame must not be empty")
        if not self._session_ready:
            raise RuntimeError("UDP mission endpoint has not completed session HELLO")
        self._send_hello()
        envelope = SidebandEnvelope(
            magic=SidebandEnvelope.MAGIC,
            version=SidebandEnvelope.VERSION,
            direction=Direction.INGRESS,
            reserved=0,
            spacecraft_instance_id=self.spacecraft_instance_id,
            sender_boot_id=0,
            link_session_id=self.link_session_id,
            sender_frame_id=self._next_sender_frame_id,
            link_frame_id=0,
            file_epoch_id=0,
            frame_length=len(frame),
        )
        self._next_sender_frame_id += 1
        envelope.validate_ingress()
        self.transport.send(envelope, frame)
        return envelope

    def receive_egress(self, timeout_ms: Optional[int] = None) -> Optional[TransportFrame]:
        self._send_hello()
        if self._pending_datagrams:
            return self._consume_datagram(self._pending_datagrams.pop(0))
        datagram = self.transport.receive_datagram(timeout_ms)
        if datagram is None:
            return None
        return self._consume_datagram(datagram)

    def close(self) -> None:
        self.transport.close()
