"""Run a disposable local-SIL ASGI server for Playwright."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import uvicorn

from gds.http_app import LocalSilMission, create_app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="cube-nano-p6-playwright-") as value:
        mission = LocalSilMission(args.root.resolve(), state_directory=Path(value))
        try:
            uvicorn.run(create_app(args.root.resolve(), service=mission), host=args.host, port=args.port, log_level="warning")
        finally:
            mission.close()


if __name__ == "__main__":
    main()
