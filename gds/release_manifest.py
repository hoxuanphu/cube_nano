"""Deterministic release manifest and lightweight SPDX SBOM generation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from protocol.canonical import canonical_json


class ReleaseManifestError(RuntimeError):
    pass


DEFAULT_RELEASE_ARTIFACTS = (
    "protocol/mission_profile.yaml",
    "protocol/runtime_profile.yaml",
    "protocol/slo_profile.yaml",
    "protocol/golden_vectors/vectors.json",
    "protocol/golden_vectors/threshold_lut.bin",
    "sat_ai/model_manifest.yaml",
    "sat_ai/deployment_profile.yaml",
    "artifacts/benchmarks/local-cpu-pytorch-v2.json",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _source_epoch() -> int:
    value = os.environ.get("SOURCE_DATE_EPOCH", "0")
    try:
        epoch = int(value)
    except ValueError as exc:
        raise ReleaseManifestError("SOURCE_DATE_EPOCH must be an integer") from exc
    if epoch < 0:
        raise ReleaseManifestError("SOURCE_DATE_EPOCH must be non-negative")
    return epoch


def _timestamp(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, UTC).isoformat().replace("+00:00", "Z")


def _git(root: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = subprocess.call(
            ["git", "-C", str(root), "diff", "--quiet", "--ignore-submodules", "--"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ) != 0
        untracked = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return commit, dirty or bool(untracked)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ReleaseManifestError("git metadata is unavailable") from exc


def _locked_components(lock_path: Path) -> list[dict[str, Any]]:
    components: list[dict[str, Any]] = []
    for raw_line in lock_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        name, separator, version = line.partition("==")
        if not separator:
            continue
        components.append(
            {
                "type": "library",
                "name": name.strip().lower(),
                "version": version.strip(),
                "scope": "runtime",
            }
        )
    return sorted(components, key=lambda item: (item["name"], item["version"]))


def _has_distribution(name: str) -> bool:
    try:
        importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def generate_sbom(lock_path: str | Path, output_path: str | Path | None = None) -> tuple[dict[str, Any], str]:
    lock = Path(lock_path)
    payload = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": "cube-nano-runtime",
        "documentNamespace": "https://example.invalid/cube-nano/sbom/v1",
        "creationInfo": {"created": _timestamp(_source_epoch()), "creators": ["Tool: cube-nano-release"]},
        "packages": _locked_components(lock),
    }
    encoded = canonical_json(payload) + b"\n"
    digest = sha256_bytes(encoded)
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(encoded)
    return payload, digest


def _runtime_versions() -> dict[str, str | None]:
    names = ("torch", "torchvision", "numpy", "tifffile", "PyYAML", "fastapi", "uvicorn", "websockets")
    return {name: importlib.metadata.version(name) if _has_distribution(name) else None for name in names}


def build_release_manifest(
    root: str | Path,
    *,
    sbom_path: str | Path | None = None,
    artifacts: Iterable[str] = DEFAULT_RELEASE_ARTIFACTS,
    require_clean: bool = False,
    target_id: str = "local-cpu-pytorch",
) -> tuple[dict[str, Any], str]:
    root_path = Path(root).resolve()
    commit, dirty = _git(root_path)
    if require_clean and dirty:
        raise ReleaseManifestError("release build requires a clean git worktree")
    lock_path = root_path / "requirements-lock.txt"
    _, sbom_sha = generate_sbom(lock_path, sbom_path)
    artifact_hashes: dict[str, str] = {}
    for relative in sorted(set(artifacts)):
        path = root_path / relative
        if not path.is_file():
            raise ReleaseManifestError(f"release artifact is missing: {relative}")
        artifact_hashes[relative] = sha256_file(path)
    material = {
        "schema_version": 1,
        "source_commit": commit,
        "source_dirty": dirty,
        "source_date_epoch": _source_epoch(),
        "target_id": target_id,
        "dependency_lock_sha256": sha256_file(lock_path),
        "sbom_sha256": sbom_sha,
        "artifacts": artifact_hashes,
        "runtime": _runtime_versions(),
        "platform": {"python": platform.python_version(), "machine": platform.machine()},
    }
    release_id = sha256_bytes(canonical_json(material))[:32]
    payload = {"schema_version": 1, "release_id": release_id, **material}
    encoded = canonical_json(payload) + b"\n"
    return payload, sha256_bytes(encoded)


def write_release_manifest(path: str | Path, payload: dict[str, Any]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json(payload) + b"\n"
    target.write_bytes(encoded)
    return sha256_bytes(encoded)
