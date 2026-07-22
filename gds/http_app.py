"""ASGI adapter for the transport-neutral GDS mission runtime.

The HTTP process owns browser-facing concerns only.  It does not construct or
inspect a flight simulator, mission link, payload journal, or satellite product
directory; those concerns are behind ``GdsMissionRuntime``'s endpoint contract.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles

from gds.local_sil import GdsMissionRuntime, LocalSilMission, create_mission_runtime
from gds.realtime import RESYNC_CLOSE_CODE, ResyncRequired
from gds.topology import RateLimiter, TopologyError, TopologyProfile


def _u64(value: int) -> str:
    return f"{value:016x}"


def _json_response(response) -> JSONResponse:
    return JSONResponse(response.body, status_code=response.status_code, headers=response.headers)


def create_app(
    root: str | Path = ".",
    *,
    service: GdsMissionRuntime | None = None,
    serve_web: bool = False,
) -> FastAPI:
    """Create the browser adapter around an already-bound GDS runtime."""

    mission = service or create_mission_runtime(root)
    profile = mission.topology
    limiter = RateLimiter(profile.limits.requests_per_minute)
    app = FastAPI(title="Cube Nano GDS", version="p6")
    app.state.mission = mission

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(profile.allowed_origins),
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Accept", "Content-Type", "Idempotency-Key"],
        allow_credentials=True,
    )

    @app.middleware("http")
    async def local_guard(request: Request, call_next):
        host = request.headers.get("host", "")
        peer = request.client.host if request.client else ""
        header_bytes = sum(len(key) + len(value) for key, value in request.headers.items())
        content_length = request.headers.get("content-length")
        try:
            declared_body_bytes = 0
            if content_length is not None:
                if not content_length.isascii() or not content_length.isdecimal():
                    raise TopologyError(
                        "INVALID_CONTENT_LENGTH",
                        "Content-Length must be a non-negative decimal integer",
                        400,
                    )
                declared_body_bytes = int(content_length)
            profile.validate_request(
                host=host,
                origin=request.headers.get("origin"),
                peer=peer,
                body_bytes=declared_body_bytes,
                header_bytes=header_bytes,
                method=request.method,
            )
            if ".." in request.url.path.split("/"):
                raise TopologyError("PATH_TRAVERSAL", "path traversal is forbidden", 400)
            if request.url.path not in {"/healthz", "/readyz"} and not limiter.allow(f"{peer}:{request.url.path}"):
                raise TopologyError("RATE_LIMITED", "request rate limit exceeded", 429)
        except TopologyError as exc:
            return JSONResponse(
                {"status": "error", "error": exc.code, "message": str(exc)},
                status_code=exc.status_code,
            )

        # ``Request.body()`` otherwise buffers a chunked body before the
        # profile limit is checked.  Wrap ASGI receive so both declared and
        # chunked bodies stop at the same hard quota.
        received_body_bytes = 0
        receive = request._receive

        async def limited_receive():
            nonlocal received_body_bytes
            message = await receive()
            if message.get("type") == "http.request":
                received_body_bytes += len(message.get("body", b""))
                if received_body_bytes > profile.limits.request_body_bytes:
                    raise TopologyError(
                        "BODY_TOO_LARGE",
                        "request body exceeds the configured limit",
                        413,
                    )
            return message

        request._receive = limited_receive
        try:
            body = await request.body()
        except TopologyError as exc:
            return JSONResponse(
                {"status": "error", "error": exc.code, "message": str(exc)},
                status_code=exc.status_code,
            )
        request._body = body
        return await call_next(request)

    @app.get("/healthz")
    async def healthz():
        response = mission.gds.api.healthz()
        return JSONResponse(mission.health_payload(response.body), status_code=response.status_code)

    @app.get("/readyz")
    async def readyz():
        response = mission.gds.api.readyz()
        status_code = 200 if response.status_code == 200 and mission.ready else 503
        return JSONResponse(mission.health_payload(response.body), status_code=status_code)

    @app.get("/api/state")
    async def state():
        latest_event_id = mission.gds.events.latest_event_id()
        return {
            "state": mission.snapshot(),
            "as_of_event_id": _u64(latest_event_id),
            "last_event_id": _u64(latest_event_id),
        }

    @app.get("/api/spacecraft/{instance}/scenes")
    async def scenes(instance: str, limit: int = 100, after_scene_id: int | None = None):
        return _json_response(
            mission.gds.api.get_catalog(instance, limit=limit, after_scene_id=after_scene_id)
        )

    @app.get("/api/spacecraft/{instance}/scenes/{epoch}/{scene_id}/{revision}")
    async def scene(instance: str, epoch: int, scene_id: int, revision: int):
        return _json_response(mission.gds.api.get_scene(instance, epoch, scene_id, revision))

    @app.post("/api/commands")
    async def commands(request: Request):
        status_code, body, headers = mission.submit(
            await request.json(),
            request.headers.get("Idempotency-Key", ""),
        )
        return JSONResponse(body, status_code=status_code, headers=headers)

    @app.get("/api/commands/{ground_instance_id}/{request_id}")
    async def command(ground_instance_id: str, request_id: int):
        value = mission.command(ground_instance_id, request_id)
        if value is not None:
            return value
        return _json_response(mission.gds.api.get_command(ground_instance_id, request_id))

    @app.get("/api/products/{instance}/{boot}/{product_id}")
    async def product(instance: str, boot: int, product_id: int):
        return _json_response(mission.gds.api.get_product(instance, boot, product_id))

    @app.get("/api/products/{instance}/{boot}/{product_id}/download")
    async def download(instance: str, boot: int, product_id: int):
        result = mission.gds.api.download_product(instance, boot, product_id)
        if result.status_code != 200:
            return _json_response(result)
        body = result.body
        return FileResponse(
            path=str(body["path"]),
            media_type=str(body.get("content_type", "application/x-tar")),
            filename=f"product-{instance}-{boot:08x}-{product_id:08x}.tar",
            content_disposition_type="attachment",
            headers={"ETag": str(body.get("etag", ""))},
        )

    @app.get("/api/products/{instance}/{boot}/{product_id}/tiles/{z}/{x}/{y}")
    async def tile(instance: str, boot: int, product_id: int, z: int, x: int, y: int):
        result = mission.gds.api.get_tile(instance, boot, product_id, z, x, y)
        if result.status_code != 200:
            return _json_response(result)
        body = result.body
        return Response(
            body["content"],
            media_type=str(body.get("content_type", "image/webp")),
            headers={"ETag": str(body.get("etag", ""))},
        )

    @app.post("/admin/contact/{state}")
    async def contact(state: str):
        try:
            mission.set_contact(state)
        except ValueError as exc:
            return JSONResponse(
                {"status": "error", "error": "VALIDATION_ERROR", "message": str(exc)},
                status_code=422,
            )
        return {"status": "ok", "contact": state}

    @app.websocket("/ws/telemetry")
    async def telemetry(websocket: WebSocket):
        host = websocket.headers.get("host", "")
        peer = websocket.client.host if websocket.client else ""
        origin = websocket.headers.get("origin")
        try:
            profile.validate_request(
                host=host,
                origin=origin,
                peer=peer,
                body_bytes=0,
                header_bytes=sum(len(key) + len(value) for key, value in websocket.headers.items()),
                method="GET",
            )
        except TopologyError as exc:
            await websocket.close(code=1008, reason=exc.code)
            return

        await websocket.accept()
        client = None
        try:
            last_event = websocket.query_params.get("last_event_id")
            try:
                snapshot, client, replay = mission.gds.realtime.connect(last_event)
            except ResyncRequired as exc:
                await websocket.send_json(
                    {"type": "error", "error": ResyncRequired.code, "message": str(exc)}
                )
                await websocket.close(code=RESYNC_CLOSE_CODE, reason=ResyncRequired.code)
                return
            await websocket.send_json({"type": "snapshot", "snapshot": snapshot.as_dict()})
            for event in replay:
                await websocket.send_json({"type": "event", "event": event})
            while True:
                try:
                    await asyncio.wait_for(websocket.receive_text(), timeout=0.25)
                except asyncio.TimeoutError:
                    pass
                terminal = client.take_terminal_envelope()
                if terminal is not None:
                    await websocket.send_json(terminal)
                    await websocket.close(
                        code=client.close_code or RESYNC_CLOSE_CODE,
                        reason=client.close_reason or ResyncRequired.code,
                    )
                    return
                for event in client.drain():
                    await websocket.send_json({"type": "event", "event": event})
        except WebSocketDisconnect:
            pass
        finally:
            if client is not None:
                mission.gds.realtime.disconnect(client.client_id)

    if serve_web:
        static_root = mission.root / "gds" / "web" / "dist"
        if static_root.is_dir():
            app.mount("/", StaticFiles(directory=static_root, html=True), name="web")
    return app


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the Cube Nano GDS HTTP service")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--serve-web", action="store_true")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    args = parser.parse_args()
    topology_path = Path(
        os.environ.get("CUBE_NANO_RUNTIME_PROFILE", str(args.root / "protocol" / "runtime_profile.yaml"))
    )
    topology = TopologyProfile.from_file(topology_path)
    topology.validate_startup(args.host)
    uvicorn.run(
        create_app(args.root, serve_web=args.serve_web),
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
    )


if __name__ == "__main__":
    main()
