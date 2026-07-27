"""One-process local SIL assembly for the P4b round-trip fixture."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

from flight.mission_com_scheduler import MissionComScheduler, QueueKind
from flight.mission_udp_adapter import MissionUdpAdapter
from flight.satellite_simulator import SatelliteSimulator
from link_sim.mission_link import MissionLink
from link_sim.transport import TransportFrame, UdpMissionEndpoint
from link_sim.virtual_clock import VirtualClock
from protocol.canonical import canonical_json, u64_to_json
from protocol.profile import MissionProfile
from protocol.schemas import CommandOpcode, ProductRef, RequestKey

from .api import GDSApi
from .binding import SpacecraftBindingManager
from .catalog import CatalogReplicaStore
from .events import EventStore
from .file_reassembly import FilePacketReassembler
from .ingest import TmIngestService
from .ledger import AtomicCommandLedger
from .metrics import HealthService, MetricsRegistry
from .outbox import ContactState, OutboxService, TcWireProfile
from .product_store import ProductStore
from .preview import PreviewService
from .realtime import RealtimeHub
from .retention import RetentionManager
from .storage import StorageGuard
from .tm import TMDecoder, ValidatedTransportEnvelope
from .topology import TopologyProfile
from .u64 import decode_sqlite_u64, encode_sqlite_u64
from .writer import SQLiteWriter


class LocalSilRuntime:
    def __init__(
        self,
        root: str | Path = ".",
        *,
        state_directory: str | Path | None = None,
        receive_clock_us: Callable[[], int] | None = None,
    ):
        self.root = Path(root).resolve()
        self._receive_clock_us = receive_clock_us or (lambda: time.time_ns() // 1_000)
        self.state_directory = Path(state_directory or self.root / "data" / "ground").resolve()
        self.state_directory.mkdir(parents=True, exist_ok=True)
        self.profile = MissionProfile.from_file(self.root / "protocol" / "mission_profile.yaml")
        self.topology = TopologyProfile.from_file(self.root / "protocol" / "runtime_profile.yaml")
        self.topology.validate_startup(self.topology.bind_host)
        self.writer = SQLiteWriter(self.state_directory / "gds.sqlite3")
        self.ledger = AtomicCommandLedger(self.writer)
        self.outbox = OutboxService(self.writer)
        self.bindings = SpacecraftBindingManager(self.writer, self.outbox)
        self.bindings.bind(
            self.profile.spacecraft_instance_id,
            link_generation=1,
            link_session_id=1,
            contact_state=ContactState.CONTACT_OPEN,
        )
        self.storage = StorageGuard(
            self.writer,
            self.state_directory,
            cap_bytes=20 * 1024 * 1024 * 1024,
            high_watermark=0.80,
            hard_watermark=0.90,
            usage_provider=self._logical_storage_usage,
        )
        self.catalog = CatalogReplicaStore(self.writer)
        self.product_store = ProductStore(self.writer, self.state_directory / "products")
        self.preview = PreviewService(self.writer, self.catalog, self.product_store)
        self.reassembler = FilePacketReassembler(
            self.state_directory / "reassembly",
            writer=self.writer,
            product_store=self.product_store,
            storage_guard=self.storage,
            clock_us=self._receive_clock_us,
        )
        self.events = EventStore(self.writer)
        self.metrics = MetricsRegistry()
        self.decoder = TMDecoder(
            spacecraft_id=self.profile.spacecraft_id,
            expected_instance_id=self.profile.spacecraft_instance_id,
        )
        self.realtime = RealtimeHub(self.events, self.snapshot_state)
        self.ingest = TmIngestService(
            self.writer,
            self.decoder,
            events=self.events,
            reassembler=self.reassembler,
            metrics=self.metrics,
            realtime=self.realtime,
            outbox=self.outbox,
            ledger=self.ledger,
        )
        self.tm_observer: Callable[[Any], None] | None = None
        self.retention = RetentionManager(self.writer, self.product_store, storage_guard=self.storage)
        self.health = HealthService(writer=self.writer, decoder=self.decoder)
        self.api = GDSApi(
            self.ledger,
            outbox=self.outbox,
            storage_guard=self.storage,
            catalog=self.catalog,
            product_store=self.product_store,
            preview=self.preview,
            retention=self.retention,
            realtime=self.realtime,
            health=self.health,
        )

    def _logical_storage_usage(self) -> int:
        """Count runtime-owned bytes instead of the host volume's global usage."""
        total = 0
        for path in self.state_directory.rglob("*"):
            if path.is_file() and not path.is_symlink():
                try:
                    total += path.stat().st_size
                except FileNotFoundError:
                    continue
        return total

    def snapshot_state(self) -> dict[str, Any]:
        binding = self.outbox.binding(self.profile.spacecraft_instance_id)
        catalog = self.catalog.status(self.profile.spacecraft_instance_id)
        return {
            "spacecraft_instance_id": f"{self.profile.spacecraft_instance_id:016x}",
            "contact_state": None if binding is None else binding.contact_state.value,
            "link_generation": None if binding is None else f"{binding.link_generation:016x}",
            "link_session_id": None if binding is None else f"{binding.link_session_id:016x}",
            "catalog": catalog.as_dict(),
            "metrics": self.metrics.snapshot(),
        }

    def latest_telemetry(self, limit: int = 100) -> tuple[dict[str, Any], ...]:
        return self.ingest.telemetry.latest_for_instance(self.profile.spacecraft_instance_id, limit=limit)

    def ingest_tm(self, envelope: ValidatedTransportEnvelope):
        return self.ingest.ingest(envelope)

    def receive_transport_frame(self, transport_frame):
        """Ingest only a validated egress frame emitted by MissionLink."""
        binding = self.outbox.binding(self.profile.spacecraft_instance_id)
        if binding is None:
            raise RuntimeError("GDS link binding is unavailable")
        transport_frame.envelope.validate_egress(
            expected_spacecraft_instance_id=self.profile.spacecraft_instance_id,
            expected_sender_boot_id=self.decoder.expected_boot_id,
            expected_link_session_id=binding.link_session_id,
        )
        envelope = ValidatedTransportEnvelope.from_sideband(
            transport_frame.envelope,
            transport_frame.frame_bytes,
            received_at_us=self._receive_clock_us(),
            simulation_run_id=getattr(self, "_simulation_run_id", 0),
            copy_index=transport_frame.copy_index,
            link_generation=binding.link_generation,
        )
        result = self.ingest.ingest(envelope)
        if self.tm_observer is not None:
            self.tm_observer(result)
        return result

    def bind_link(self, *, link_session_id: int, link_generation: int, sender_boot_id: int | None = None, simulation_run_id: int = 0) -> None:
        """Atomically move TM decoder and outbox to the active link binding."""
        self.bindings.bind(
            self.profile.spacecraft_instance_id,
            link_generation=link_generation,
            link_session_id=link_session_id,
            contact_state=ContactState.CONTACT_OPEN,
        )
        self.decoder.expected_session_id = link_session_id
        self.decoder.expected_link_generation = link_generation
        self.decoder.expected_boot_id = sender_boot_id
        self._simulation_run_id = simulation_run_id

    def healthz(self) -> dict[str, Any]:
        return self.health.healthz().as_dict()

    def readyz(self) -> dict[str, Any]:
        return self.health.readyz().as_dict()

    def close(self) -> None:
        if getattr(self, "writer", None) is not None:
            self.writer.close()
            self.writer = None

    def __enter__(self) -> "LocalSilRuntime":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


class _LocalGroundEndpoint:
    """In-memory implementation of the same endpoint contract as UDP."""

    def __init__(
        self,
        *,
        root: Path,
        state_root: Path,
        simulation_run_id: int,
    ) -> None:
        self.clock = VirtualClock()
        self.satellite = SatelliteSimulator(
            root,
            state_directory=state_root / "satellite",
            product_directory=state_root / "satellite" / "products",
            device="cpu",
            event_clock=lambda: self.clock.now.ns,
            event_time_base="simulation",
        )
        self.link = MissionLink(
            simulation_run_id=simulation_run_id,
            seed=simulation_run_id,
            spacecraft_instance_id=self.satellite.payload.profile.spacecraft_instance_id,
            sender_boot_id=self.satellite.payload.journal.boot_id,
            clock=self.clock,
        )
        self.link.attach_satellite(self.satellite)
        self.link.attach_ground(self)
        self._inbound: list[TransportFrame] = []
        self._lock = threading.RLock()
        self.frames_sent = 0
        self.frames_received = 0

    @property
    def spacecraft_instance_id(self) -> int:
        return self.satellite.payload.profile.spacecraft_instance_id

    @property
    def sender_boot_id(self) -> int:
        return self.satellite.payload.journal.boot_id

    @property
    def link_session_id(self) -> int:
        return self.link.session_id

    @property
    def link_generation(self) -> int:
        return self.link.link_generation

    @property
    def ready(self) -> bool:
        return True

    def receive_transport_frame(self, frame: TransportFrame) -> str:
        """Buffer TM until the outbox send fence has marked the attempt SENT."""
        with self._lock:
            self._inbound.append(frame)
        return "BUFFERED_FOR_GDS"

    def send_tc(self, frame: bytes) -> Any:
        self.frames_sent += 1
        return self.link.send_uplink(frame)

    def pump(self, receiver: Callable[[TransportFrame], Any], *, timeout_ms: int = 0) -> int:
        del timeout_ms
        # APID 2 and APID 3 both enter the satellite scheduler with the MCFC
        # ordering key allocated at encode time. This prevents ACK priority
        # from overtaking a pending file START in local-SIL.
        self.satellite.enqueue_pending_tm_frames()
        # File packets remain scheduler-owned until the link has consumed the
        # yielded frame.  The iterator resumes only after send_downlink(), so
        # a large product never becomes a resident list of TM frames.
        for frame in self.satellite.iter_scheduled_tm_frames():
            self.link.send_downlink(frame, receiver=self.receive_transport_frame)
        with self._lock:
            pending = self._inbound
            self._inbound = []
        for frame in pending:
            receiver(frame)
            self.frames_received += 1
        return len(pending)

    def health(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "mode": "in_memory",
            "frames_sent": self.frames_sent,
            "frames_received": self.frames_received,
            "link_session_id": self.link_session_id,
            "link_generation": self.link_generation,
            "link": self.link.health(),
        }

    def close(self) -> None:
        self.link.close()
        self.satellite.close()


class _UdpGroundEndpoint:
    """GDS endpoint for the deployed GDS -> UDP -> LinkSimulator topology."""

    def __init__(
        self,
        *,
        bind_host: str,
        bind_port: int,
        link_host: str,
        link_port: int,
        spacecraft_instance_id: int,
        link_session_id: int,
        link_generation: int,
        sender_boot_id: int | None,
    ) -> None:
        self.spacecraft_instance_id = spacecraft_instance_id
        self.sender_boot_id = sender_boot_id
        self.link_session_id = link_session_id
        self.link_generation = link_generation
        self._session_callback: Callable[[dict[str, int]], None] | None = None
        self.endpoint = UdpMissionEndpoint(
            bind_addr=(bind_host, bind_port),
            link_addr=(link_host, link_port),
            spacecraft_instance_id=spacecraft_instance_id,
            link_session_id=link_session_id,
            sender_boot_id=sender_boot_id,
            handshake_role="gds",
            on_session=self._on_session_binding,
        )
        self.endpoint.establish_session()
        self.frames_sent = 0
        self.frames_received = 0

    def _on_session_binding(self, binding: dict[str, int]) -> None:
        self.sender_boot_id = binding["sender_boot_id"]
        self.link_session_id = binding["link_session_id"]
        self.link_generation = binding["link_generation"]
        if self._session_callback is not None:
            self._session_callback(dict(binding))

    def set_session_callback(self, callback: Callable[[dict[str, int]], None] | None) -> None:
        self._session_callback = callback

    @property
    def ready(self) -> bool:
        return self.endpoint.ready

    def send_tc(self, frame: bytes) -> None:
        self.endpoint.send_ingress(frame)
        self.frames_sent += 1

    def pump(self, receiver: Callable[[TransportFrame], Any], *, timeout_ms: int = 0) -> int:
        frame = self.endpoint.receive_egress(timeout_ms)
        if frame is None:
            return 0
        receiver(frame)
        self.frames_received += 1
        return 1

    def health(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "mode": "udp",
            "bound_address": self.endpoint.bound_address,
            "frames_sent": self.frames_sent,
            "frames_received": self.frames_received,
            "link_session_id": self.link_session_id,
            "link_generation": self.link_generation,
            "sender_boot_id": self.sender_boot_id,
        }

    def close(self) -> None:
        self.endpoint.close()


class GdsMissionRuntime:
    """Transport-neutral GDS mission orchestration.

    This runtime owns GDS state, dispatches persisted TC attempts through an
    endpoint, and accepts only decoded TM back from that endpoint.  It never
    reads a flight payload, journal, or product store outside the GDS state.
    """

    _OUTBOX_RECONCILE_INTERVAL_S = 0.25

    def __init__(
        self,
        root: str | Path,
        *,
        state_directory: str | Path | None,
        endpoint: _LocalGroundEndpoint | _UdpGroundEndpoint,
        simulation_run_id: int,
    ) -> None:
        self.root = Path(root).resolve()
        self.endpoint = endpoint
        self.run_id = int(simulation_run_id)
        profile_path = Path(os.environ.get("CUBE_NANO_RUNTIME_PROFILE", str(self.root / "protocol" / "runtime_profile.yaml")))
        self.topology = TopologyProfile.from_file(profile_path)
        self.gds = LocalSilRuntime(
            self.root,
            state_directory=state_directory,
            receive_clock_us=lambda: time.time_ns() // 1_000,
        )
        if self.gds.profile.spacecraft_instance_id != endpoint.spacecraft_instance_id:
            self.gds.close()
            raise ValueError("mission endpoint spacecraft instance does not match the GDS profile")
        self.instance = endpoint.spacecraft_instance_id
        self.session_id = endpoint.link_session_id
        self.gds.bind_link(
            link_session_id=endpoint.link_session_id,
            link_generation=endpoint.link_generation,
            sender_boot_id=endpoint.sender_boot_id,
            simulation_run_id=self.run_id,
        )
        self.tc_profile = TcWireProfile.from_mission_profile(self.gds.profile)
        self.scheduler = MissionComScheduler()
        self.adapter = MissionUdpAdapter(self.scheduler, self._send_endpoint_frame)
        self._dispatch_lock = threading.RLock()
        self._reconcile_lock = threading.Lock()
        self._next_outbox_reconcile_monotonic = 0.0
        self._closed = threading.Event()
        self._tm_health_received = False
        self._last_transport_error: str | None = None
        self.gds.tm_observer = self._on_tm
        self.gds.realtime.state_provider = self.snapshot
        self.gds.health.link = self
        if isinstance(self.endpoint, _UdpGroundEndpoint):
            self.endpoint.set_session_callback(self._on_endpoint_session_binding)
        self._receiver = threading.Thread(
            target=self._run_pump,
            name="gds-transport-pump",
            daemon=True,
        )
        self._receiver.start()
        # Local SIL has an already-created satellite status frame; consume it
        # before issuing the catalog bootstrap request.
        self.pump(timeout_ms=0)
        self._bootstrap_catalog()

    @property
    def ready(self) -> bool:
        return bool(self.endpoint.ready and self._tm_health_received)

    def _run_pump(self) -> None:
        while not self._closed.is_set():
            try:
                self.pump(timeout_ms=50)
                self._dispatch_due()
            except Exception as exc:  # keep the process alive and expose failure in readiness
                self._last_transport_error = type(exc).__name__
            self._closed.wait(0.02)

    def _on_endpoint_session_binding(self, binding: dict[str, int]) -> None:
        """Atomically move the decoder/outbox to a new LinkSimulator session."""

        if self._closed.is_set():
            return
        with self._dispatch_lock:
            current = self.gds.outbox.binding(self.instance)
            if (
                current is not None
                and current.link_session_id == binding["link_session_id"]
                and current.link_generation == binding["link_generation"]
                and self.gds.decoder.expected_boot_id == binding["sender_boot_id"]
            ):
                return
            self.adapter.reset()
            self.scheduler.set_ready()
            self.gds.bind_link(
                link_session_id=binding["link_session_id"],
                link_generation=binding["link_generation"],
                sender_boot_id=binding["sender_boot_id"],
                simulation_run_id=self.run_id,
            )
            self.session_id = binding["link_session_id"]
            self._tm_health_received = False

    def pump(self, *, timeout_ms: int = 0) -> int:
        return self.endpoint.pump(self.gds.receive_transport_frame, timeout_ms=timeout_ms)

    def _send_endpoint_frame(self, frame: bytes, status_callback: Callable[[str], None], return_callback: Callable[[], None]) -> None:
        try:
            self.endpoint.send_tc(frame)
            status_callback("UDP_SENT" if isinstance(self.endpoint, _UdpGroundEndpoint) else "LINK_CONSUMED")
        except Exception:
            status_callback("FRAME_FAILED")
            raise
        finally:
            return_callback()

    def _send_persisted(self, frame: bytes) -> None:
        if self.adapter.gate is not None or self.scheduler.current is not None:
            raise RuntimeError("GDS_TC_FRAME_IN_FLIGHT")
        self.scheduler.enqueue(QueueKind.CONTROL, bytes(frame))
        item = self.adapter.send_next()
        if item is None or self.adapter.gate is not None or self.scheduler.current is not None:
            raise RuntimeError("GDS_TC_COMPLETION_TIMEOUT")

    def _dispatch(self, request_key: RequestKey) -> bool:
        with self._dispatch_lock:
            binding = self.gds.bindings.active_binding()
            if binding is None or not binding.contact_state.is_open:
                return False
            lease = self.gds.outbox.claim(
                request_key,
                binding=binding,
                lease_owner="gds-mission-runtime",
            )
            if lease is None:
                return False
            attempt = self.gds.outbox.prepare_attempt(lease, profile=self.tc_profile)
            self.gds.outbox.send_with_fence(
                lease,
                attempt,
                fence=self.gds.bindings.fence,
                send=self._send_persisted,
            )
        # An in-memory link queues TM while the attempt is transitioning to
        # SENT; UDP delivers asynchronously.  Pump after the fence in both.
        self.pump(timeout_ms=0)
        return True

    def _dispatch_due(self) -> None:
        binding = self.gds.bindings.active_binding()
        if binding is None or not binding.contact_state.is_open:
            return
        # Reconcile expired leases/ACK deadlines on a bounded cadence before
        # selecting due work.  This makes an otherwise solitary SENT command
        # retry after a lost TM receipt without turning the pump loop into a
        # database busy-spin.
        self._reconcile_outbox_if_due()
        now_us = time.time_ns() // 1_000
        # Pick deterministically, then take the only mutable transition via
        # the keyed claim API.  A concurrent state change simply makes
        # ``_dispatch`` return False instead of claiming a different command.
        with self.gds.writer.reader() as connection:
            row = connection.execute(
                "SELECT ground_instance_id,request_id FROM command_outbox "
                "WHERE target_spacecraft_instance_id=? AND state='OUTBOX_PENDING' "
                "AND available_at_us<=? AND expires_at_us>? "
                "ORDER BY available_at_us,created_at_us LIMIT 1",
                (encode_sqlite_u64(binding.spacecraft_instance_id), now_us, now_us),
            ).fetchone()
        if row is None:
            return
        self._dispatch(
            RequestKey(
                decode_sqlite_u64(row["ground_instance_id"], "ground_instance_id"),
                int(row["request_id"]),
            )
        )

    def _reconcile_outbox_if_due(self, *, force: bool = False) -> None:
        now = time.monotonic()
        with self._reconcile_lock:
            if not force and now < self._next_outbox_reconcile_monotonic:
                return
            self.gds.outbox.reconcile()
            self._next_outbox_reconcile_monotonic = (
                time.monotonic() + self._OUTBOX_RECONCILE_INTERVAL_S
            )

    @staticmethod
    def _runtime_key(name: str) -> str:
        return f"mission-runtime:{name}"

    def _set_runtime_state(self, name: str, value: Any) -> None:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
        now_us = time.time_ns() // 1_000
        self.gds.writer.mutate(
            "update_mission_runtime_state",
            lambda connection: connection.execute(
                "INSERT INTO system_state(state_key,value_json,updated_at_us) VALUES(?,?,?) "
                "ON CONFLICT(state_key) DO UPDATE SET value_json=excluded.value_json,updated_at_us=excluded.updated_at_us",
                (self._runtime_key(name), encoded, now_us),
            ),
        )

    def _runtime_state(self, name: str, default: Any) -> Any:
        with self.gds.writer.reader() as connection:
            row = connection.execute(
                "SELECT value_json FROM system_state WHERE state_key=?",
                (self._runtime_key(name),),
            ).fetchone()
        if row is None:
            return default
        try:
            return json.loads(str(row[0]))
        except (TypeError, ValueError, json.JSONDecodeError):
            return default

    def _bootstrap_catalog(self) -> None:
        body = {
            "target_spacecraft_instance_id": u64_to_json(self.instance),
            "opcode": int(CommandOpcode.SCENE_REQUEST_CATALOG),
            "payload": {},
            "delivery_mode": "next_contact",
        }
        response = self.gds.api.post_commands(
            body,
            headers={"Idempotency-Key": f"transport-catalog-{self.session_id:016x}"},
            principal="gds-transport-bootstrap",
        )
        if response.status_code == 202:
            self._dispatch(RequestKey.from_dict(response.body["request_key"]))

    @staticmethod
    def _product_ref_from_event(body: Any) -> ProductRef | None:
        if not isinstance(body, dict):
            return None
        candidates = [body.get("product_ref")]
        product = body.get("product")
        if isinstance(product, dict):
            candidates.append(product.get("product_ref"))
        result = body.get("result")
        if isinstance(result, dict):
            candidates.append(result.get("product_ref"))
            nested = result.get("product")
            if isinstance(nested, dict):
                candidates.append(nested.get("product_ref"))
        for candidate in candidates:
            if candidate is None:
                continue
            try:
                return ProductRef.from_dict(candidate)
            except (TypeError, ValueError):
                continue
        return None

    def _on_tm(self, result: Any) -> None:
        decoded = getattr(result, "decoded", None)
        if decoded is None or decoded.message is None:
            return
        message = decoded.message
        event_name = str(message.get("event_name", ""))
        body = message.get("message", message)
        if event_name == "SATELLITE_STATUS" and isinstance(body, dict):
            self._set_runtime_state("satellite_status", body)
            self._tm_health_received = True
            return
        if event_name == "SATELLITE_SLO" and isinstance(body, dict):
            status = self._runtime_state("satellite_status", {})
            if not isinstance(status, dict):
                status = {}
            status["slo"] = body
            self._set_runtime_state("satellite_status", status)
            return
        if event_name == "CATALOG_SNAPSHOT":
            bundle = message.get("catalog_bundle")
            if isinstance(bundle, (bytes, bytearray)):
                self.gds.catalog.activate(
                    self.instance,
                    bytes(bundle),
                    source_boot_id=decoded.envelope.sender_boot_id,
                    link_session_id=decoded.envelope.link_session_id,
                    received_at_us=decoded.envelope.received_at_us,
                )
                self._tm_health_received = True
            return
        request_key = None
        if message.get("request_key") is not None:
            try:
                request_key = RequestKey.from_dict(message["request_key"])
            except (TypeError, ValueError):
                request_key = None
        if request_key is not None and event_name.startswith("JOB_"):
            job_key = f"{request_key.ground_instance_id:016x}:{request_key.request_id}"
            jobs = self._runtime_state("jobs", {})
            if not isinstance(jobs, dict):
                jobs = {}
            product_ref = self._product_ref_from_event(body)
            state = {
                "JOB_COMPLETED": "SUCCEEDED",
                "JOB_FAILED": "FAILED",
                "JOB_TIMEOUT": "TIMEOUT",
                "JOB_CANCELED": "CANCELED",
            }.get(event_name, event_name.removeprefix("JOB_"))
            jobs[job_key] = {
                "job_key": request_key.as_dict(),
                "spacecraft_instance_id": u64_to_json(self.instance),
                "state": state,
                "progress_bp": 10000 if state == "SUCCEEDED" else 0,
                "stage": event_name,
                "result": body,
                "error_code": body.get("error_code") if isinstance(body, dict) else None,
                "product_ref": None if product_ref is None else product_ref.as_dict(),
                "updated_at": int(time.time() * 1000),
            }
            self._set_runtime_state("jobs", jobs)
            if event_name == "JOB_COMPLETED" and product_ref is not None:
                binding = self.gds.outbox.binding(self.instance)
                try:
                    admission = self.gds.ledger.admit_product_downlink(
                        origin_request_key=request_key,
                        product_ref=product_ref,
                        target_spacecraft_instance_id=self.instance,
                        delivery_mode="next_contact",
                        contact_available=binding is not None and binding.contact_state.is_open,
                    )
                    if admission.outbox_state == "OUTBOX_PENDING":
                        self._dispatch(admission.request_key)
                except Exception as exc:
                    self._last_transport_error = f"PRODUCT_DOWNLINK_{type(exc).__name__}"

    def submit(self, body: dict[str, Any], idempotency_key: str) -> tuple[int, dict[str, Any], dict[str, str]]:
        response = self.gds.api.post_commands(body, headers={"Idempotency-Key": idempotency_key})
        if response.status_code == 202 and not response.body.get("replayed"):
            self._dispatch(RequestKey.from_dict(response.body["request_key"]))
        return response.status_code, response.body, response.headers

    def set_contact(self, state: str) -> None:
        binding = self.gds.outbox.set_contact_state(self.instance, ContactState(state))
        self.gds.bindings.refresh()
        if binding.contact_state.is_open:
            self._dispatch_due()

    def command(self, ground_instance_id: str, request_id: int) -> dict[str, Any] | None:
        response = self.gds.api.get_command(ground_instance_id, request_id)
        if response.status_code != 200:
            return None
        result = dict(response.body)
        jobs = self._runtime_state("jobs", {})
        job = jobs.get(f"{ground_instance_id}:{request_id}") if isinstance(jobs, dict) else None
        if isinstance(job, dict):
            result.update(
                {
                    "job_state": job.get("state"),
                    "product_ref": job.get("product_ref"),
                }
            )
            if job.get("product_ref") is not None:
                try:
                    product = self.gds.product_store.get(ProductRef.from_dict(job["product_ref"]))
                except (TypeError, ValueError):
                    product = None
                if product is not None:
                    result["product_state"] = product.get("state")
        return result

    def _products(self) -> dict[str, dict[str, Any]]:
        products: dict[str, dict[str, Any]] = {}
        with self.gds.writer.reader() as connection:
            rows = connection.execute(
                "SELECT spacecraft_instance_id,origin_boot_id,product_id FROM products "
                "WHERE spacecraft_instance_id=? ORDER BY origin_boot_id,product_id",
                (encode_sqlite_u64(self.instance),),
            ).fetchall()
        for row in rows:
            ref = ProductRef(self.instance, int(row["origin_boot_id"]), int(row["product_id"]))
            value = self.gds.product_store.get(ref)
            if value is not None:
                state = str(value.get("state", "UNKNOWN"))
                verified = state == "PUBLISHED"
                manifest = value.get("manifest")
                artifacts = []
                if isinstance(manifest, dict):
                    for artifact in manifest.get("artifacts", ()):
                        if isinstance(artifact, dict):
                            artifacts.append({**artifact, "verified": verified})
                products[f"{self.instance:016x}:{ref.origin_boot_id}:{ref.product_id}"] = {
                    **value,
                    "artifacts": artifacts,
                    "verified": verified,
                    "transfer_state": "VERIFIED" if verified else state,
                    "progress_bp": 10_000 if verified else 0,
                    "expected_bytes": value.get("bundle_size"),
                    "received_bytes": value.get("bundle_size") if verified else 0,
                    "gap_count": 0 if verified else 1,
                    "checksum_status": "SHA256_MATCH" if verified else "PENDING",
                }
        return products

    def snapshot(self) -> dict[str, Any]:
        binding = self.gds.outbox.binding(self.instance)
        catalog_scenes, _, catalog_status = self.gds.catalog.list_scenes(self.instance)
        status = self._runtime_state("satellite_status", {})
        telemetry = list(self.gds.latest_telemetry(limit=100))
        last_telemetry_at_us = max((int(item["received_at_us"]) for item in telemetry), default=0)
        contact = "UNKNOWN" if binding is None else (
            "CONNECTED" if binding.contact_state.is_open else binding.contact_state.value
        )
        spacecraft = {
            f"{self.instance:016x}": {
                "instance_id": f"{self.instance:016x}",
                "state": str(status.get("state", "DEGRADED")),
                "spacecraft_id": self.gds.profile.spacecraft_id,
                "boot_id": self.endpoint.sender_boot_id,
                "link_session_id": None if binding is None else f"{binding.link_session_id:016x}",
                "link_generation": None if binding is None else f"{binding.link_generation:016x}",
                "queue_depth": 0,
                "queue_capacity": status.get("queue_capacity", 0),
                "contact": contact,
                "last_telemetry_at": None if not telemetry else last_telemetry_at_us // 1_000,
                "tm_stale_after_seconds": 5,
                "model_release_id": status.get("model_release_id"),
                "model_assurance": status.get("assurance_level"),
            }
        }
        commands: dict[str, dict[str, Any]] = {}
        with self.gds.writer.reader() as connection:
            rows = connection.execute(
                "SELECT c.ground_instance_id,c.request_id,c.target_spacecraft_instance_id,c.opcode,"
                "c.command_state,c.created_at_us,c.updated_at_us,o.state AS outbox_state "
                "FROM commands c JOIN command_outbox o ON o.ground_instance_id=c.ground_instance_id "
                "AND o.request_id=c.request_id WHERE c.target_spacecraft_instance_id=?",
                (encode_sqlite_u64(self.instance),),
            ).fetchall()
        for row in rows:
            key = RequestKey(decode_sqlite_u64(row["ground_instance_id"], "ground_instance_id"), int(row["request_id"]))
            value = self.command(f"{key.ground_instance_id:016x}", key.request_id)
            if value is None:
                continue
            value.update({"opcode": int(row["opcode"]), "updated_at": int(row["updated_at_us"]) // 1_000})
            commands[f"{key.ground_instance_id:016x}:{key.request_id}"] = value
        events, _ = self.gds.events.list_events(limit=100)
        last_event = f"{self.gds.events.latest_event_id():016x}"
        scenes = {
            f"{self.instance:016x}:{scene.scene_ref.catalog_epoch}:{scene.scene_ref.scene_id}:{scene.scene_ref.scene_revision}": scene.as_dict()
            for scene in catalog_scenes
        }
        jobs = self._runtime_state("jobs", {})
        return {
            "runtime": {
                "browser_gds": "CONNECTED",
                "gds_satellite": contact,
                "as_of_event_id": last_event,
                "link_session_id": spacecraft[f"{self.instance:016x}"]["link_session_id"],
                "last_updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "fault_profile": {"profile_id": "transport", "seed": f"{self.run_id:016x}", "blackout": contact == "BLACKOUT"},
            },
            "spacecraft": spacecraft,
            "catalogs": {
                f"{self.instance:016x}": {
                    "status": catalog_status.as_dict(),
                    "sceneKeys": list(scenes),
                    "nextCursor": None,
                }
            },
            "scenes": scenes,
            "commands": commands,
            "jobs": jobs if isinstance(jobs, dict) else {},
            "products": self._products(),
            "telemetry": telemetry,
            "events": [event.as_dict() for event in events],
            "configs": {f"{self.instance:016x}": status.get("config", {})},
            "last_event_id": last_event,
            "snapshot_received_at": int(time.time() * 1000),
        }

    def health_payload(self, base: dict[str, Any]) -> dict[str, Any]:
        status = self._runtime_state("satellite_status", {})
        payload = dict(base)
        payload["satellite"] = {"state": status.get("state", "DEGRADED"), **status}
        payload["scheduler"] = {
            "queue_depths": status.get("scheduler_queue_depths", {}),
            "metrics": status.get("scheduler_metrics", {}),
        }
        payload["slo"] = status.get("slo")
        payload["transport"] = {**self.endpoint.health(), "tm_health_received": self._tm_health_received, "last_error": self._last_transport_error}
        return payload

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._receiver.join(timeout=2)
        self.endpoint.close()
        self.gds.close()


class LocalSilMission(GdsMissionRuntime):
    """Local loopback profile using the exact same GDS endpoint contract."""

    def __init__(self, root: str | Path = ".", *, state_directory: str | Path | None = None):
        resolved = Path(root).resolve()
        state_root = Path(state_directory or resolved / ".cube_nano-cache" / "p6-http").resolve()
        state_root.mkdir(parents=True, exist_ok=True)
        run_id = time.time_ns() & ((1 << 64) - 1)
        endpoint = _LocalGroundEndpoint(root=resolved, state_root=state_root, simulation_run_id=run_id)
        self.satellite = endpoint.satellite  # compatibility/diagnostics; HTTP never reads it.
        self.link = endpoint.link
        super().__init__(resolved, state_directory=state_root / "ground", endpoint=endpoint, simulation_run_id=run_id)


class UdpSilMission(GdsMissionRuntime):
    """Compose runtime; only the GDS process and UDP endpoint live here."""

    def __init__(self, root: str | Path = ".", *, state_directory: str | Path | None = None):
        resolved = Path(root).resolve()
        profile = MissionProfile.from_file(resolved / "protocol" / "mission_profile.yaml")
        endpoint = _UdpGroundEndpoint(
            bind_host=os.environ.get("CUBE_NANO_GDS_UDP_BIND_HOST", "0.0.0.0"),
            bind_port=int(os.environ.get("CUBE_NANO_GDS_UDP_PORT", "9001")),
            link_host=os.environ.get("CUBE_NANO_LINK_HOST", "127.0.0.1"),
            link_port=int(os.environ.get("CUBE_NANO_LINK_PORT", "9000")),
            spacecraft_instance_id=profile.spacecraft_instance_id,
            link_session_id=int(os.environ.get("CUBE_NANO_LINK_SESSION_ID", "1")),
            link_generation=int(os.environ.get("CUBE_NANO_LINK_GENERATION", "1")),
            sender_boot_id=(
                None
                if os.environ.get("CUBE_NANO_SENDER_BOOT_ID") is None
                else int(os.environ["CUBE_NANO_SENDER_BOOT_ID"])
            ),
        )
        run_id = int(os.environ.get("CUBE_NANO_SIMULATION_RUN_ID", "1"))
        state_root = Path(state_directory or resolved / ".cube_nano-cache" / "gds-udp").resolve()
        state_root.mkdir(parents=True, exist_ok=True)
        super().__init__(resolved, state_directory=state_root, endpoint=endpoint, simulation_run_id=run_id)


def create_mission_runtime(root: str | Path = ".", *, state_directory: str | Path | None = None) -> GdsMissionRuntime:
    """Select loopback or Compose UDP without exposing flight to the ASGI host."""
    if os.environ.get("CUBE_NANO_LINK_MODE", "local").lower() == "udp":
        return UdpSilMission(root, state_directory=state_directory)
    return LocalSilMission(root, state_directory=state_directory)
