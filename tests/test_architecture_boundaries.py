"""Production import boundaries for legacy inference entry points."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_ROOTS = ("flight", "gds", "sat_ai", "deploy", "scripts")
LEGACY_IMPORT = re.compile(
    r"^\s*(?:from|import)\s+(?:src\.)?"
    r"(?:inference|inference_large_image|inference_large_image_trt|inference_tensorrt)(?:\b|\s|\.)"
)


def test_production_does_not_import_legacy_inference_entry_points():
    violations: list[str] = []
    for root_name in PRODUCTION_ROOTS:
        root = ROOT / root_name
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if LEGACY_IMPORT.search(line):
                    violations.append(f"{path.relative_to(ROOT)}:{line_number}: {line.strip()}")
    assert not violations, "legacy inference import in production path:\n" + "\n".join(violations)
