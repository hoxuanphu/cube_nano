"""Immutable memmap scene ingest and startup scrub helpers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sat_ai.roi import SceneContractError, open_memmap_scene

from protocol.canonical import canonical_json
from protocol.schemas import SceneRef


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stat_fingerprint(path: str | Path) -> dict[str, int]:
    value = Path(path).stat()
    return {
        "device": int(getattr(value, "st_dev", 0)),
        "inode": int(getattr(value, "st_ino", 0)),
        "size": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
    }


@dataclass(frozen=True)
class ScenePackage:
    scene_ref: SceneRef
    package_sha256: str
    root: Path
    source_path: Path
    sidecar_path: Path
    source_sha256: str
    sidecar_sha256: str
    source_stat: dict[str, int]
    sidecar_stat: dict[str, int]

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "scene_ref": self.scene_ref.as_dict(),
            "package_sha256": self.package_sha256,
            "source_sha256": self.source_sha256,
            "sidecar_sha256": self.sidecar_sha256,
            "source_stat": self.source_stat,
            "sidecar_stat": self.sidecar_stat,
            "source": self.source_path.name,
            "sidecar": self.sidecar_path.name,
        }


def _copy_fsync(source: Path, destination: Path) -> None:
    with source.open("rb") as source_stream, destination.open("wb") as destination_stream:
        shutil.copyfileobj(source_stream, destination_stream, length=1024 * 1024)
        destination_stream.flush()
        os.fsync(destination_stream.fileno())


def _read_package_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "package.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError) as exc:
        raise SceneContractError("INVALID_SCENE_PACKAGE_MANIFEST") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise SceneContractError("INVALID_SCENE_PACKAGE_MANIFEST")
    if manifest_path.read_bytes() != canonical_json(payload) + b"\n":
        raise SceneContractError("INVALID_SCENE_PACKAGE_MANIFEST")
    return payload


def _package_member(root: Path, name: Any, label: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise SceneContractError(f"INVALID_SCENE_PACKAGE_{label.upper()}")
    candidate = (root / name).resolve()
    if candidate.parent != root.resolve() or candidate.name != name:
        raise SceneContractError(f"INVALID_SCENE_PACKAGE_{label.upper()}")
    return candidate


def _validate_package_artifacts(root: Path, manifest: dict[str, Any]) -> tuple[Path, Path]:
    source = _package_member(root, manifest.get("source"), "source")
    sidecar = _package_member(root, manifest.get("sidecar"), "sidecar")
    if source == sidecar or not source.is_file() or not sidecar.is_file():
        raise SceneContractError("INVALID_SCENE_PACKAGE_ARTIFACTS")
    expected = {"package.json", source.name, sidecar.name}
    try:
        sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SceneContractError("INVALID_SCENE_PACKAGE_SIDECAR") from exc
    validity = sidecar_payload.get("validity", {}) if isinstance(sidecar_payload, dict) else {}
    if isinstance(validity, dict) and validity.get("kind") == "mask":
        mask = _package_member(root, validity.get("relative_path"), "mask")
        if mask in {source, sidecar} or not mask.is_file():
            raise SceneContractError("INVALID_SCENE_PACKAGE_MASK")
        expected.add(mask.relative_to(root).as_posix())
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if any(path.is_symlink() for path in root.rglob("*")) or actual != expected:
        raise SceneContractError("INVALID_SCENE_PACKAGE_ARTIFACTS")
    return source, sidecar


def _package_descriptor(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return the canonical, non-self-referential package identity inputs."""
    return {
        "schema_version": manifest.get("schema_version"),
        "scene_ref": manifest.get("scene_ref"),
        "source": manifest.get("source"),
        "source_sha256": manifest.get("source_sha256"),
        "sidecar": manifest.get("sidecar"),
        "sidecar_sha256": manifest.get("sidecar_sha256"),
        "shape": manifest.get("shape"),
        "input_spec_id": manifest.get("input_spec_id"),
    }


def _expected_package_sha256(manifest: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(_package_descriptor(manifest))).hexdigest()


def _package_from_root(root: Path, scene_ref: SceneRef, expected: dict[str, Any] | None = None) -> ScenePackage:
    manifest = _read_package_manifest(root)
    if expected is not None and any(manifest.get(key) != value for key, value in expected.items()):
        raise SceneContractError("SCENE_PACKAGE_HASH_MISMATCH")
    if manifest.get("scene_ref") != scene_ref.as_dict():
        raise SceneContractError("SCENE_PACKAGE_SCENE_REF_MISMATCH")
    package_sha = str(manifest.get("package_sha256", "")).lower()
    source_sha = str(manifest.get("source_sha256", "")).lower()
    sidecar_sha = str(manifest.get("sidecar_sha256", "")).lower()
    if len(package_sha) != 64 or len(source_sha) != 64 or len(sidecar_sha) != 64:
        raise SceneContractError("INVALID_SCENE_PACKAGE_HASH")
    if package_sha != _expected_package_sha256(manifest):
        raise SceneContractError("SCENE_PACKAGE_HASH_MISMATCH")
    source, sidecar = _validate_package_artifacts(root, manifest)
    package = ScenePackage(
        scene_ref,
        package_sha,
        root,
        source,
        sidecar,
        source_sha,
        sidecar_sha,
        stat_fingerprint(source),
        stat_fingerprint(sidecar),
    )
    scrub_scene_package(package)
    return package


def _quarantine_package(root: Path, package_root: Path) -> None:
    for attempt in range(100):
        target = root / f".quarantine-{package_root.name[:12]}-{os.getpid()}-{attempt}"
        try:
            os.replace(package_root, target)
            return
        except FileExistsError:
            continue
    raise SceneContractError("SCENE_PACKAGE_QUARANTINE_FAILED")


def ingest_scene_package(
    source_path: str | Path,
    sidecar_path: str | Path,
    output_root: str | Path,
    scene_ref: SceneRef,
) -> ScenePackage:
    """Validate and publish a read-only, content-addressed scene package.

    The runtime contract is deliberately strict: only the memmap-compatible
    single-series TIFF path accepted by ``open_memmap_scene`` enters a package.
    """

    source = Path(source_path).resolve()
    sidecar = Path(sidecar_path).resolve()
    if not source.is_file() or not sidecar.is_file():
        raise SceneContractError("scene source and sidecar must be regular files")
    with open_memmap_scene(source, sidecar, verify_source_fingerprint=True) as scene:
        shape = scene.shape
        input_spec_id = scene.input_spec_id
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".scene-build-", dir=root))
    try:
        copied_source = temporary / source.name
        copied_sidecar = temporary / sidecar.name
        _copy_fsync(source, copied_source)
        _copy_fsync(sidecar, copied_sidecar)
        source_sha = sha256_file(copied_source)
        sidecar_payload = json.loads(copied_sidecar.read_text(encoding="utf-8"))
        validity = sidecar_payload.get("validity", {})
        if validity.get("kind") == "mask":
            mask = (sidecar.parent / str(validity.get("relative_path", ""))).resolve()
            if sidecar.parent.resolve() not in mask.parents or not mask.is_file():
                raise SceneContractError("validity mask path escapes the scene package")
            copied_mask = temporary / mask.name
            if copied_mask.name in {copied_source.name, copied_sidecar.name, "package.json"}:
                raise SceneContractError("validity mask filename collides with package artifact")
            _copy_fsync(mask, copied_mask)
            validity["relative_path"] = copied_mask.name
            validity["sha256"] = sha256_file(copied_mask)
            sidecar_payload["validity"] = validity
            copied_sidecar.write_bytes(canonical_json(sidecar_payload) + b"\n")
            with copied_sidecar.open("rb+") as stream:
                stream.flush()
                os.fsync(stream.fileno())

        # The package identity is derived from the bytes that will actually be
        # published, including any normalized sidecar and copied validity mask.
        with open_memmap_scene(copied_source, copied_sidecar, verify_source_fingerprint=True) as copied_scene:
            if copied_scene.shape != shape or copied_scene.input_spec_id != input_spec_id:
                raise SceneContractError("SCENE_PACKAGE_COPY_VALIDATION_FAILED")
        sidecar_sha = sha256_file(copied_sidecar)
        descriptor = {
            "schema_version": 1,
            "scene_ref": scene_ref.as_dict(),
            "source": copied_source.name,
            "source_sha256": source_sha,
            "sidecar": copied_sidecar.name,
            "sidecar_sha256": sidecar_sha,
            "shape": list(shape),
            "input_spec_id": input_spec_id,
        }
        package_sha = hashlib.sha256(canonical_json(descriptor)).hexdigest()
        package_root = root / package_sha
        manifest = {
            **descriptor,
            "package_sha256": package_sha,
        }
        if package_root.exists():
            try:
                existing = _package_from_root(package_root, scene_ref, expected=manifest)
            except SceneContractError:
                _quarantine_package(root, package_root)
            else:
                shutil.rmtree(temporary, ignore_errors=True)
                return existing

        manifest_path = temporary / "package.json"
        manifest_path.write_bytes(canonical_json(manifest) + b"\n")
        with manifest_path.open("rb+") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        for child in temporary.iterdir():
            if child.is_file():
                child.chmod(0o444)
        try:
            descriptor = os.open(temporary, os.O_RDONLY)
        except OSError:
            descriptor = None
        if descriptor is not None:
            try:
                os.fsync(descriptor)
            except OSError:
                pass
            finally:
                os.close(descriptor)
        os.replace(temporary, package_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return _package_from_root(package_root, scene_ref, expected=manifest)


def scrub_scene_package(package: ScenePackage) -> None:
    """Fail closed when source or sidecar changed after ingest."""

    manifest = _read_package_manifest(package.root)
    if manifest.get("package_sha256") != package.package_sha256:
        raise SceneContractError("SCENE_PACKAGE_HASH_MISMATCH")
    if manifest.get("scene_ref") != package.scene_ref.as_dict():
        raise SceneContractError("SCENE_PACKAGE_SCENE_REF_MISMATCH")
    if manifest.get("source") != package.source_path.name or manifest.get("sidecar") != package.sidecar_path.name:
        raise SceneContractError("SCENE_PACKAGE_PATH_MISMATCH")
    if manifest.get("source_sha256") != package.source_sha256 or manifest.get("sidecar_sha256") != package.sidecar_sha256:
        raise SceneContractError("SCENE_PACKAGE_HASH_MISMATCH")
    if manifest.get("package_sha256") != _expected_package_sha256(manifest):
        raise SceneContractError("SCENE_PACKAGE_HASH_MISMATCH")
    _validate_package_artifacts(package.root, manifest)

    if stat_fingerprint(package.source_path) != package.source_stat:
        raise SceneContractError("INVALID_SCENE_SOURCE_STAT")
    if stat_fingerprint(package.sidecar_path) != package.sidecar_stat:
        raise SceneContractError("INVALID_SCENE_SIDECAR_STAT")
    if sha256_file(package.source_path) != package.source_sha256:
        raise SceneContractError("INVALID_SCENE_SOURCE_HASH")
    if sha256_file(package.sidecar_path) != package.sidecar_sha256:
        raise SceneContractError("INVALID_SCENE_SIDECAR_HASH")
    with open_memmap_scene(package.source_path, package.sidecar_path, verify_source_fingerprint=True) as scene:
        if manifest.get("shape") != list(scene.shape) or manifest.get("input_spec_id") != scene.input_spec_id:
            raise SceneContractError("SCENE_PACKAGE_CONTRACT_MISMATCH")
