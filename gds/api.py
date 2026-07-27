"""Framework-neutral command admission and status API contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from protocol.canonical import checked_u32, u64_from_json
from protocol.schemas import CommandOpcode, ProductRef, SceneRef, ScopedSceneRef, RequestKey

from .idempotency import IdempotencyValidationError
from .ledger import (
    AdmissionError,
    AtomicCommandLedger,
    IdempotencyConflictError,
    IdempotencyKeyRetiredError,
    NoContactError,
    OutboxCapacityError,
    TargetRetiredError,
)
from .outbox import OutboxService
from .storage import StorageFullError
from .writer import WriterBackpressureError, WriterClosedError

try:
    from .catalog import CatalogError, CatalogReplicaStore
    from .preview import PreviewError, PreviewService
    from .product_store import ProductStore
    from .realtime import RealtimeHub, ResyncRequired
    from .retention import RetentionManager
    from .metrics import HealthService
except ImportError:  # pragma: no cover - keeps the command boundary importable during bootstrap
    CatalogError = PreviewError = ResyncRequired = ValueError
    CatalogReplicaStore = PreviewService = ProductStore = RealtimeHub = RetentionManager = HealthService = Any


@dataclass(frozen=True)
class ApiResponse:
    status_code: int
    body: dict[str, Any]
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.status_code == 202


class GDSApi:
    """Map transport-neutral request objects onto the durable GDS ledger."""

    def __init__(
        self,
        ledger: AtomicCommandLedger,
        *,
        outbox: OutboxService | None = None,
        storage_guard: Any | None = None,
        principal: str = "local-operator",
        catalog: CatalogReplicaStore | None = None,
        product_store: ProductStore | None = None,
        preview: PreviewService | None = None,
        retention: RetentionManager | None = None,
        realtime: RealtimeHub | None = None,
        health: HealthService | None = None,
        state_provider: Any | None = None,
    ) -> None:
        self.ledger = ledger
        self.outbox = outbox
        self.storage_guard = storage_guard
        self.principal = principal
        self.catalog = catalog
        self.product_store = product_store
        self.preview = preview
        self.retention = retention
        self.realtime = realtime
        self.health = health
        self.state_provider = state_provider

    @staticmethod
    def _header(headers: Mapping[str, Any] | None, name: str) -> Any:
        if headers is None:
            return None
        if not isinstance(headers, Mapping):
            raise IdempotencyValidationError("headers must be an object")
        expected = name.lower()
        for key, value in headers.items():
            if isinstance(key, str) and key.lower() == expected:
                return value
        return None

    def _contact_available(self, target: int) -> bool:
        if self.outbox is None:
            return True
        binding = self.outbox.binding(target)
        if binding is not None:
            return binding.contact_state.is_open
        # Retired/no-contact is deliberately left to the serialized ledger
        # transaction so an existing idempotent retry is returned first.
        return False

    def post_commands(
        self,
        body: Mapping[str, Any],
        *,
        headers: Mapping[str, Any] | None = None,
        principal: str | None = None,
    ) -> ApiResponse:
        try:
            if not isinstance(body, Mapping):
                raise IdempotencyValidationError("command body must be an object")
            idempotency_key = self._header(headers, "Idempotency-Key")
            if idempotency_key is None:
                raise IdempotencyValidationError("Idempotency-Key header is required")
            target = u64_from_json(
                body.get("target_spacecraft_instance_id"),
                "target_spacecraft_instance_id",
            )
            opcode_value = body.get("opcode")
            if isinstance(opcode_value, bool) or not isinstance(opcode_value, int):
                raise IdempotencyValidationError("opcode must be an integer")
            opcode = CommandOpcode(opcode_value)
            payload = body.get("payload", {})
            if not isinstance(payload, Mapping):
                raise IdempotencyValidationError("payload must be an object")
            delivery_mode = body.get("delivery_mode", "immediate")
            if not isinstance(delivery_mode, str):
                raise IdempotencyValidationError("delivery_mode must be a string")
            contact_available = self._contact_available(target)
            kwargs: dict[str, Any] = {
                "idempotency_key": idempotency_key,
                "target_spacecraft_instance_id": target,
                "opcode": opcode,
                "payload": payload,
                "principal": self.principal if principal is None else principal,
                "delivery_mode": delivery_mode,
                "contact_available": contact_available,
            }
            if "expires_at" in body:
                kwargs["expires_at"] = body["expires_at"]
            if self.storage_guard is not None:
                kwargs["pre_admission_check"] = self.storage_guard.ensure_admission
            result = self.ledger.admit(**kwargs)
            return ApiResponse(202, result.as_dict())
        except IdempotencyConflictError as exc:
            return self._error(409, exc.error_code, exc)
        except IdempotencyKeyRetiredError as exc:
            return self._error(410, exc.error_code, exc)
        except TargetRetiredError as exc:
            return self._error(410, exc.error_code, exc)
        except NoContactError as exc:
            return self._error(409, exc.error_code, exc)
        except OutboxCapacityError as exc:
            return self._error(
                429,
                exc.error_code,
                exc,
                retry_after=exc.retry_after_seconds,
            )
        except StorageFullError as exc:
            return self._error(507, exc.error_code, exc)
        except WriterBackpressureError as exc:
            return self._error(
                503,
                exc.error_code,
                exc,
                retry_after=exc.retry_after_seconds,
            )
        except WriterClosedError as exc:
            return self._error(503, "WRITER_CLOSED", exc, retry_after=1)
        except AdmissionError as exc:
            return self._error(exc.status_code, exc.error_code, exc)
        except (IdempotencyValidationError, TypeError, ValueError) as exc:
            return self._error(422, "VALIDATION_ERROR", exc)

    post_command = post_commands

    def get_command(
        self,
        ground_instance_id: str,
        request_id: int,
    ) -> ApiResponse:
        try:
            key = RequestKey(
                u64_from_json(ground_instance_id, "ground_instance_id"),
                checked_u32(request_id, "request_id"),
            )
            result = self.ledger.get(key)
            if result is None:
                return self._error(404, "COMMAND_NOT_FOUND", KeyError("command not found"))
            return ApiResponse(200, result.as_dict())
        except (TypeError, ValueError) as exc:
            return self._error(422, "VALIDATION_ERROR", exc)

    get_command_status = get_command

    def get_command_for_instance(
        self,
        spacecraft_instance_id: str,
        ground_instance_id: str,
        request_id: int,
    ) -> ApiResponse:
        """Return command status only when the URL instance matches the command target."""

        try:
            expected_instance = u64_from_json(spacecraft_instance_id, "spacecraft_instance_id")
            response = self.get_command(ground_instance_id, request_id)
            if response.status_code != 200:
                return response
            actual_instance = u64_from_json(
                response.body["target_spacecraft_instance_id"],
                "target_spacecraft_instance_id",
            )
            if actual_instance != expected_instance:
                return self._error(404, "COMMAND_NOT_FOUND", KeyError("command is outside the requested spacecraft instance"))
            return response
        except (TypeError, ValueError) as exc:
            return self._error(422, "VALIDATION_ERROR", exc)

    get_instance_command = get_command_for_instance

    def get_catalog(
        self,
        spacecraft_instance_id: str,
        *,
        limit: int = 100,
        after_scene_id: int | None = None,
    ) -> ApiResponse:
        try:
            if self.catalog is None:
                raise CatalogError("CATALOG_UNAVAILABLE", "catalog replica is not configured")
            instance = u64_from_json(spacecraft_instance_id, "spacecraft_instance_id")
            scenes, cursor, status = self.catalog.list_scenes(instance, limit=limit, after_scene_id=after_scene_id)
            return ApiResponse(
                200,
                {
                    "status": "ok",
                    "catalog": status.as_dict(),
                    "scenes": [scene.as_dict() for scene in scenes],
                    "next_cursor": cursor,
                },
            )
        except CatalogError as exc:
            return self._error(409 if exc.code.endswith("MISMATCH") else 404, exc.code, exc)
        except (TypeError, ValueError) as exc:
            return self._error(422, "VALIDATION_ERROR", exc)

    def get_scene(
        self,
        spacecraft_instance_id: str,
        catalog_epoch: int,
        scene_id: int,
        scene_revision: int,
    ) -> ApiResponse:
        try:
            if self.catalog is None:
                raise CatalogError("CATALOG_UNAVAILABLE", "catalog replica is not configured")
            scoped = ScopedSceneRef(
                u64_from_json(spacecraft_instance_id, "spacecraft_instance_id"),
                SceneRef(catalog_epoch, scene_id, scene_revision),
            )
            scene = self.catalog.get_scene(scoped)
            body = scene.as_dict()
            if self.preview is not None:
                active = self.preview.active_preview(scoped)
                body["active_preview_product_ref"] = None if active is None else active.as_dict()
            return ApiResponse(200, {"status": "ok", "scene": body})
        except CatalogError as exc:
            return self._error(409 if exc.code.endswith("MISMATCH") else 404, exc.code, exc)
        except (TypeError, ValueError) as exc:
            return self._error(422, "VALIDATION_ERROR", exc)

    def get_product(self, spacecraft_instance_id: str, origin_boot_id: int, product_id: int, *, now_us: int = 0) -> ApiResponse:
        try:
            if self.product_store is None:
                raise ValueError("product store is not configured")
            ref = ProductRef(u64_from_json(spacecraft_instance_id, "spacecraft_instance_id"), origin_boot_id, product_id)
            product = self.product_store.get(ref)
            if product is None or product.get("state") == "EVICTED":
                tombstone = None if self.retention is None else self.retention.lookup_tombstone(ref, now_us)
                if tombstone is not None:
                    return ApiResponse(410, tombstone.as_dict())
                return self._error(404, "PRODUCT_NOT_FOUND", KeyError("product not found"))
            return ApiResponse(200, {"status": "ok", "product": product})
        except (TypeError, ValueError) as exc:
            return self._error(422, "VALIDATION_ERROR", exc)

    def download_product(self, spacecraft_instance_id: str, origin_boot_id: int, product_id: int, *, now_us: int = 0) -> ApiResponse:
        response = self.get_product(spacecraft_instance_id, origin_boot_id, product_id, now_us=now_us)
        if response.status_code != 200:
            return response
        product = response.body["product"]
        path = product.get("local_path")
        if product.get("state") != "PUBLISHED" or not path:
            return self._error(409, "PRODUCT_NOT_VERIFIED", RuntimeError("product is not published"))
        try:
            bundle = Path(path).resolve() / "bundle.tar"
            if self.product_store is None or self.product_store.root not in bundle.parents or not bundle.is_file():
                return self._error(410, "PRODUCT_BYTES_UNAVAILABLE", FileNotFoundError("product bytes are unavailable"))
            # The ASGI boundary turns this verified, root-confined file into a
            # FileResponse. Keeping bytes out of the framework-neutral API
            # avoids materializing a near-limit product bundle in GDS memory.
            return ApiResponse(
                200,
                {
                    "status": "ok",
                    "product_ref": product["product_ref"],
                    "path": str(bundle),
                    "content_type": "application/x-tar",
                    "etag": product.get("bundle_sha256"),
                },
            )
        except OSError as exc:
            return self._error(503, "PRODUCT_READ_FAILED", exc)

    def get_tile(self, spacecraft_instance_id: str, origin_boot_id: int, product_id: int, z: int, x: int, y: int) -> ApiResponse:
        try:
            if self.preview is None:
                raise PreviewError("PREVIEW_UNAVAILABLE", "preview service is not configured")
            ref = ProductRef(u64_from_json(spacecraft_instance_id, "spacecraft_instance_id"), origin_boot_id, product_id)
            content, etag = self.preview.tile(ref, z, x, y)
            return ApiResponse(200, {"status": "ok", "product_ref": ref.as_dict(), "content": content, "content_type": "image/webp", "etag": etag})
        except PreviewError as exc:
            return self._error(404 if exc.code.endswith("NOT_FOUND") or exc.code.endswith("AVAILABLE") else 422, exc.code, exc)
        except (TypeError, ValueError) as exc:
            return self._error(422, "VALIDATION_ERROR", exc)

    def get_state(self) -> ApiResponse:
        try:
            if self.realtime is not None:
                return ApiResponse(200, self.realtime.snapshot().as_dict())
            state = self.state_provider() if callable(self.state_provider) else {}
            return ApiResponse(200, {"state": dict(state) if isinstance(state, Mapping) else state})
        except Exception as exc:
            return self._error(503, "STATE_UNAVAILABLE", exc)

    def open_realtime(self, last_event_id: str | None = None) -> ApiResponse:
        try:
            if self.realtime is None:
                raise ResyncRequired("realtime service is not configured")
            snapshot, client, replay = self.realtime.connect(last_event_id)
            return ApiResponse(200, {"snapshot": snapshot.as_dict(), "client_id": client.client_id, "events": list(replay)})
        except ResyncRequired as exc:
            return self._error(409, exc.code, exc)
        except (TypeError, ValueError) as exc:
            return self._error(422, "VALIDATION_ERROR", exc)

    def healthz(self) -> ApiResponse:
        if self.health is None:
            return ApiResponse(200, {"status": "ok"})
        return ApiResponse(200, self.health.healthz().as_dict())

    def readyz(self) -> ApiResponse:
        if self.health is None:
            return ApiResponse(200, {"status": "ok"})
        snapshot = self.health.readyz()
        return ApiResponse(200 if snapshot.status == "ok" else 503, snapshot.as_dict())

    @staticmethod
    def _error(
        status_code: int,
        error_code: str,
        error: BaseException,
        *,
        retry_after: int | None = None,
    ) -> ApiResponse:
        body: dict[str, Any] = {
            "status": "error",
            "error": error_code,
            "message": str(error),
        }
        request_key = getattr(error, "request_key", None)
        if request_key is not None:
            body["request_key"] = request_key.as_dict()
        headers: dict[str, str] = {}
        if retry_after is not None:
            headers["Retry-After"] = str(retry_after)
        return ApiResponse(status_code, body, headers)


CommandApi = GDSApi
