"""Regression guards for the F-01 HTTP-to-flight boundary."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_http_adapter_has_no_direct_flight_or_link_sim_imports():
    """The ASGI host must use a transport-neutral GDS runtime only."""

    path = ROOT / "gds" / "http_app.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    forbidden: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".", 1)[0] in {"flight", "link_sim"}:
                    forbidden.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".", 1)[0] in {"flight", "link_sim"}:
                forbidden.append(node.module)
    assert not forbidden, f"HTTP host bypasses the transport boundary: {forbidden}"
