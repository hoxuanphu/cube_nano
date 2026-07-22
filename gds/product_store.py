"""Generic product manifest validation, safe USTAR extraction and publish."""

from __future__ import annotations

import hashlib
import json
import os
import posixpath
import shutil
import struct
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from protocol.canonical import canonical_json, checked_u32
from protocol.schemas import ProductRef, RequestKey

from .u64 import encode_sqlite_u64
from .writer import MutationPriority, SQLiteWriter
from .audit import append_audit_in_transaction


IO_BUFFER_BYTES = 1024 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024


class ProductVerificationError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ArtifactDescriptor:
    path: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        normalized = normalize_artifact_path(self.path)
        if normalized != self.path:
            raise ProductVerificationError("INVALID_ARTIFACT_PATH", "artifact paths must be normalized POSIX paths")
        if self.size < 0 or not _is_sha256(self.sha256):
            raise ProductVerificationError("INVALID_ARTIFACT_DESCRIPTOR", "artifact size/hash is invalid")

    def as_dict(self) -> dict[str, Any]:
        return {"path": self.path, "size": self.size, "sha256": self.sha256}


@dataclass(frozen=True)
class ProductManifest:
    product_type: str
    product_ref: ProductRef
    origin_request_key: RequestKey
    artifacts: tuple[ArtifactDescriptor, ...]
    metadata: dict[str, Any]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ProductVerificationError("MANIFEST_SCHEMA_ERROR", "manifest schema_version must be 1")
        if not self.product_type or any(ord(char) < 0x20 for char in self.product_type):
            raise ProductVerificationError("MANIFEST_SCHEMA_ERROR", "product_type must be a printable string")
        reserved = {"schema_version", "product_type", "product_ref", "origin_request_key", "artifacts"}
        if reserved.intersection(self.metadata):
            raise ProductVerificationError("MANIFEST_SCHEMA_ERROR", "manifest metadata shadows an envelope field")
        required = {
            "ANALYSIS": {"scene_ref", "source_sha256", "roi", "config_snapshot", "model_release_id", "science_decision"},
            "PREVIEW": {"scene_ref", "source_sha256", "display_profile"},
            "CATALOG": {"catalog_epoch", "catalog_revision", "snapshot_sha256"},
        }.get(self.product_type)
        if self.product_type == "ANALYSIS":
            if self.metadata.get("model_task") == "semantic_cloud_segmentation":
                required = required | {"pixel_cloud_ratio_bp"}
            else:
                required = required | {"cloud_positive_tile_area_ratio_bp"}
        if required is not None and not required.issubset(self.metadata):
            missing = sorted(required - set(self.metadata))
            raise ProductVerificationError("MANIFEST_SCHEMA_ERROR", f"{self.product_type} manifest is missing {missing}")
        paths = [item.path for item in self.artifacts]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ProductVerificationError("MANIFEST_ARTIFACT_SET_ERROR", "artifact paths must be unique and sorted")
        if {"manifest.json", "bundle.tar"}.intersection(paths):
            raise ProductVerificationError("MANIFEST_ARTIFACT_SET_ERROR", "manifest and bundle are reserved product entries")
        folded = [path.casefold() for path in paths]
        if len(folded) != len(set(folded)):
            raise ProductVerificationError("MANIFEST_ARTIFACT_SET_ERROR", "artifact paths collide case-insensitively")

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": self.schema_version,
            "product_type": self.product_type,
            "product_ref": self.product_ref.as_dict(),
            "origin_request_key": self.origin_request_key.as_dict(),
            "artifacts": [artifact.as_dict() for artifact in self.artifacts],
        }
        value.update(self.metadata)
        return value

    def to_bytes(self) -> bytes:
        return canonical_json(self.as_dict()) + b"\n"

    @classmethod
    def from_bytes(cls, data: bytes) -> "ProductManifest":
        raw = bytes(data)
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProductVerificationError("MANIFEST_DECODE_ERROR", "manifest is not UTF-8 JSON") from exc
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise ProductVerificationError("MANIFEST_SCHEMA_ERROR", "manifest schema_version must be 1")
        if canonical_json(value) + b"\n" != raw:
            raise ProductVerificationError("MANIFEST_NON_CANONICAL", "manifest is not canonical JSON")
        raw_artifacts = value.get("artifacts")
        if not isinstance(raw_artifacts, list):
            raise ProductVerificationError("MANIFEST_SCHEMA_ERROR", "manifest artifacts must be an array")
        artifacts = tuple(
            ArtifactDescriptor(
                normalize_artifact_path(str(item.get("path"))),
                int(item.get("size")),
                str(item.get("sha256", "")).lower(),
            )
            for item in raw_artifacts
            if isinstance(item, Mapping)
        )
        if len(artifacts) != len(raw_artifacts):
            raise ProductVerificationError("MANIFEST_SCHEMA_ERROR", "manifest artifact entry must be an object")
        reserved = {"schema_version", "product_type", "product_ref", "origin_request_key", "artifacts"}
        metadata = {key: item for key, item in value.items() if key not in reserved}
        return cls(
            str(value.get("product_type", "")),
            ProductRef.from_dict(value.get("product_ref")),
            RequestKey.from_dict(value.get("origin_request_key")),
            artifacts,
            metadata,
        )


@dataclass(frozen=True)
class VerifiedBundle:
    bundle_path: Path
    bundle_size: int
    manifest: ProductManifest
    extracted_root: Path
    bundle_sha256: str
    file_checksum: int
    entries: tuple[str, ...]


def normalize_artifact_path(path: str) -> str:
    if not isinstance(path, str) or not path:
        raise ProductVerificationError("INVALID_ARTIFACT_PATH", "artifact path must be non-empty ASCII")
    try:
        path.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ProductVerificationError("INVALID_ARTIFACT_PATH", "artifact path must be non-empty ASCII") from exc
    if "\\" in path or path.startswith("/") or path.endswith("/"):
        raise ProductVerificationError("INVALID_ARTIFACT_PATH", "artifact path must be relative POSIX")
    normalized = posixpath.normpath(path)
    if normalized in {"", "."} or normalized != path or any(part in {"", ".", ".."} for part in path.split("/")):
        raise ProductVerificationError("INVALID_ARTIFACT_PATH", "artifact path contains traversal or redundant segments")
    if len(path.encode("ascii")) > 100:
        raise ProductVerificationError("INVALID_ARTIFACT_PATH", "artifact path exceeds USTAR 100-byte limit")
    return path


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(IO_BUFFER_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FileIntegrity:
    size: int
    sha256: str
    cfdp_checksum: int


def stream_file_integrity(
    path: str | Path,
    *,
    max_bytes: int | None = None,
    buffer_bytes: int = IO_BUFFER_BYTES,
) -> FileIntegrity:
    """Hash and calculate the CFDP checksum in one bounded streaming pass."""

    if isinstance(buffer_bytes, bool) or buffer_bytes <= 0:
        raise ValueError("buffer_bytes must be positive")
    if max_bytes is not None and (isinstance(max_bytes, bool) or max_bytes < 0):
        raise ValueError("max_bytes must be non-negative")
    digest = hashlib.sha256()
    checksum = 0
    size = 0
    trailing = b""
    with Path(path).open("rb") as stream:
        while chunk := stream.read(buffer_bytes):
            size += len(chunk)
            if max_bytes is not None and size > max_bytes:
                raise ProductVerificationError("BUNDLE_TOO_LARGE", "bundle exceeds the configured limit")
            digest.update(chunk)
            words = trailing + chunk
            word_bytes = len(words) - (len(words) % 4)
            if word_bytes:
                for (word,) in struct.iter_unpack(">I", words[:word_bytes]):
                    checksum = (checksum + word) & 0xFFFFFFFF
            trailing = words[word_bytes:]
    if trailing:
        checksum = (checksum + int.from_bytes(trailing.ljust(4, b"\0"), "big")) & 0xFFFFFFFF
    return FileIntegrity(size, digest.hexdigest(), checksum)


def _read_bounded(path: Path, *, max_bytes: int, code: str) -> bytes:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ProductVerificationError(code, f"unable to stat {path.name}") from exc
    if size < 0 or size > max_bytes:
        raise ProductVerificationError(code, f"{path.name} exceeds its bounded size")
    result = bytearray()
    with path.open("rb") as stream:
        while chunk := stream.read(min(IO_BUFFER_BYTES, max_bytes + 1 - len(result))):
            result.extend(chunk)
            if len(result) > max_bytes:
                raise ProductVerificationError(code, f"{path.name} exceeds its bounded size")
    return bytes(result)


def _copy_fsync(
    source: Path,
    destination: Path,
    *,
    max_bytes: int | None = None,
    too_large_code: str = "BUNDLE_TOO_LARGE",
) -> int:
    """Copy a regular file without materialising it and return bytes copied."""

    copied = 0
    try:
        with source.open("rb") as input_stream, destination.open("xb") as output_stream:
            while chunk := input_stream.read(IO_BUFFER_BYTES):
                copied += len(chunk)
                if max_bytes is not None and copied > max_bytes:
                    raise ProductVerificationError(too_large_code, f"{source.name} exceeds its bounded size")
                output_stream.write(chunk)
            output_stream.flush()
            os.fsync(output_stream.fileno())
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return copied


def _materialize_bundle(
    bundle: bytes | bytearray | memoryview | str | Path,
    root: Path,
) -> tuple[Path, bool]:
    if isinstance(bundle, (str, Path)):
        path = Path(bundle).resolve()
        if not path.is_file():
            raise ProductVerificationError("BUNDLE_NOT_FOUND", "bundle path must be a regular file")
        return path, False
    if not isinstance(bundle, (bytes, bytearray, memoryview)):
        raise TypeError("bundle must be bytes or a regular file path")
    path = root / ".bundle-input.tar"
    view = memoryview(bundle)
    with path.open("xb") as output:
        for offset in range(0, len(view), IO_BUFFER_BYTES):
            output.write(view[offset : offset + IO_BUFFER_BYTES])
        output.flush()
        os.fsync(output.fileno())
    return path, True


def _safe_member_name(name: str) -> str:
    try:
        name.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ProductVerificationError("UNSAFE_ARCHIVE_PATH", "archive entry must be ASCII") from exc
    return normalize_artifact_path(name)


def safe_extract_ustar(
    bundle: bytes | bytearray | memoryview | str | Path,
    destination: str | Path,
    *,
    max_bundle_bytes: int = 1 << 30,
    max_extract_bytes: int = 2 << 30,
    max_files: int = 256,
) -> tuple[str, ...]:
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    source, temporary_source = _materialize_bundle(bundle, root)
    if source.stat().st_size > max_bundle_bytes:
        if temporary_source:
            source.unlink(missing_ok=True)
        raise ProductVerificationError("BUNDLE_TOO_LARGE", "bundle exceeds the configured limit")
    names: list[str] = []
    folded: set[str] = set()
    extracted_bytes = 0
    try:
        archive = tarfile.open(name=str(source), mode="r:")
    except (tarfile.TarError, OSError) as exc:
        if temporary_source:
            source.unlink(missing_ok=True)
        raise ProductVerificationError("INVALID_ARCHIVE", "bundle is not a readable USTAR archive") from exc
    try:
        with archive:
            for member in archive:
                if len(names) >= max_files:
                    raise ProductVerificationError("EXTRACT_FILE_LIMIT", "bundle contains too many entries")
                if not member.isreg() or member.issym() or member.islnk() or member.pax_headers:
                    raise ProductVerificationError("UNSAFE_ARCHIVE_ENTRY", "only regular USTAR entries are supported")
                name = _safe_member_name(member.name)
                if name == "manifest.json" and name in names:
                    raise ProductVerificationError("DUPLICATE_ARCHIVE_ENTRY", "manifest.json appears twice")
                if name.casefold() in folded:
                    raise ProductVerificationError("DUPLICATE_ARCHIVE_ENTRY", "archive entries collide case-insensitively")
                folded.add(name.casefold())
                if member.size < 0:
                    raise ProductVerificationError("INVALID_ARCHIVE_ENTRY", "negative archive member size")
                extracted_bytes += int(member.size)
                if extracted_bytes > max_extract_bytes:
                    raise ProductVerificationError("EXTRACT_SIZE_LIMIT", "bundle exceeds extraction limit")
                target = (root / name).resolve()
                if root not in target.parents:
                    raise ProductVerificationError("UNSAFE_ARCHIVE_PATH", "archive entry escapes product root")
                target.parent.mkdir(parents=True, exist_ok=True)
                stream = archive.extractfile(member)
                if stream is None:
                    raise ProductVerificationError("INVALID_ARCHIVE_ENTRY", "archive member cannot be read")
                remaining = int(member.size)
                with stream, target.open("xb") as output:
                    while remaining:
                        chunk = stream.read(min(IO_BUFFER_BYTES, remaining))
                        if not chunk:
                            raise ProductVerificationError("INVALID_ARCHIVE_ENTRY", "archive member is truncated")
                        output.write(chunk)
                        remaining -= len(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                if target.stat().st_size != member.size:
                    raise ProductVerificationError("INVALID_ARCHIVE_ENTRY", "extracted size differs from archive header")
                names.append(name)
        return tuple(sorted(names))
    finally:
        if temporary_source:
            source.unlink(missing_ok=True)


def verify_bundle(
    bundle: bytes | bytearray | memoryview | str | Path,
    *,
    expected_bundle_sha256: str,
    expected_file_checksum: int,
    expected_product_ref: ProductRef | None = None,
    temporary_root: str | Path | None = None,
    max_bundle_bytes: int = 1 << 30,
    max_extract_bytes: int = 2 << 30,
    max_files: int = 256,
) -> VerifiedBundle:
    expected_bundle_sha256 = str(expected_bundle_sha256).lower()
    if not _is_sha256(expected_bundle_sha256):
        raise ProductVerificationError("INVALID_BUNDLE_HASH", "expected bundle SHA-256 is invalid")
    checked_u32(expected_file_checksum, "expected_file_checksum")
    root = (Path(temporary_root) if temporary_root is not None else Path(tempfile.mkdtemp(prefix="gds-product-"))).resolve()
    root.mkdir(parents=True, exist_ok=True)
    try:
        # Freeze an on-disk input before hashing and extracting it.  A caller
        # may hand us a path controlled by the downlink writer; keeping the
        # verified snapshot separate prevents a check/use race without using
        # a whole-bundle in-memory buffer.
        source_root = root / ".source"
        source_root.mkdir(exist_ok=True)
        source, materialized = _materialize_bundle(bundle, source_root)
        if source.stat().st_size > max_bundle_bytes:
            raise ProductVerificationError("BUNDLE_TOO_LARGE", "bundle exceeds the configured limit")
        if not materialized:
            snapshot = (source_root / ".bundle-input.tar").resolve()
            if source != snapshot:
                _copy_fsync(source, snapshot, max_bytes=max_bundle_bytes)
            source = snapshot
        integrity = stream_file_integrity(source, max_bytes=max_bundle_bytes)
        if integrity.sha256 != expected_bundle_sha256:
            raise ProductVerificationError("BUNDLE_SHA_MISMATCH", "bundle SHA-256 mismatch")
        if integrity.cfdp_checksum != expected_file_checksum:
            raise ProductVerificationError("FILE_CHECKSUM_MISMATCH", "F Prime file checksum mismatch")
        names = safe_extract_ustar(
            source,
            root / "extracted",
            max_bundle_bytes=max_bundle_bytes,
            max_extract_bytes=max_extract_bytes,
            max_files=max_files,
        )
        if "manifest.json" not in names:
            raise ProductVerificationError("MANIFEST_MISSING", "bundle does not contain manifest.json")
        manifest = ProductManifest.from_bytes(
            _read_bounded(root / "extracted" / "manifest.json", max_bytes=MAX_MANIFEST_BYTES, code="MANIFEST_TOO_LARGE")
        )
        if expected_product_ref is not None and manifest.product_ref != expected_product_ref:
            raise ProductVerificationError("PRODUCT_REF_MISMATCH", "manifest ProductRef differs from transfer identity")
        expected_names = {"manifest.json", *(item.path for item in manifest.artifacts)}
        if set(names) != expected_names:
            raise ProductVerificationError("MANIFEST_ENTRY_SET_MISMATCH", "archive entry set differs from manifest")
        for artifact in manifest.artifacts:
            path = root / "extracted" / artifact.path
            if path.stat().st_size != artifact.size or _sha256(path) != artifact.sha256:
                raise ProductVerificationError("ARTIFACT_HASH_MISMATCH", f"artifact verification failed: {artifact.path}")
        return VerifiedBundle(
            source,
            integrity.size,
            manifest,
            root / "extracted",
            integrity.sha256,
            integrity.cfdp_checksum,
            names,
        )
    except Exception:
        if temporary_root is None:
            shutil.rmtree(root, ignore_errors=True)
        raise


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


class ProductStore:
    """Durable ground product lifecycle and crash-safe final-directory publish."""

    def __init__(self, writer: SQLiteWriter, root: str | Path, *, clock: Callable[[], int] | None = None):
        self.writer = writer
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._clock = clock or (lambda: time.time_ns() // 1_000)

    def _now_us(self) -> int:
        value = self._clock()
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("product clock must return a non-negative integer microsecond timestamp")
        return value

    @staticmethod
    def _validate_final_directory(final: Path, verified: VerifiedBundle) -> None:
        if not final.is_dir():
            raise ProductVerificationError("PRODUCT_FINAL_CONFLICT", "final product path is not a directory")
        manifest_path = final / "manifest.json"
        bundle_path = final / "bundle.tar"
        if not manifest_path.is_file() or not bundle_path.is_file():
            raise ProductVerificationError("PRODUCT_FINAL_CONFLICT", "final product is missing manifest or bundle")
        if _read_bounded(manifest_path, max_bytes=MAX_MANIFEST_BYTES, code="PRODUCT_FINAL_CONFLICT") != verified.manifest.to_bytes():
            raise ProductVerificationError("PRODUCT_FINAL_CONFLICT", "final manifest differs from the verified bundle")
        if _sha256(bundle_path) != verified.bundle_sha256 or bundle_path.stat().st_size != verified.bundle_size:
            raise ProductVerificationError("PRODUCT_FINAL_CONFLICT", "final bundle differs from the verified bundle")
        expected_names = {"manifest.json", "bundle.tar", *(item.path for item in verified.manifest.artifacts)}
        expected_directories: set[str] = set()
        for name in expected_names:
            parent = Path(name).parent
            while parent.as_posix() != ".":
                expected_directories.add(parent.as_posix())
                parent = parent.parent
        actual_names = {
            path.relative_to(final).as_posix()
            for path in final.rglob("*")
            if path.is_file() and not path.is_symlink()
        }
        actual_directories = {
            path.relative_to(final).as_posix()
            for path in final.rglob("*")
            if path.is_dir() and not path.is_symlink()
        }
        if (
            any(path.is_symlink() for path in final.rglob("*"))
            or actual_names != expected_names
            or actual_directories != expected_directories
        ):
            raise ProductVerificationError("PRODUCT_FINAL_ARTIFACT_SET_MISMATCH", "final product artifact set is not exact")
        for artifact in verified.manifest.artifacts:
            path = final / artifact.path
            if not path.is_file() or path.stat().st_size != artifact.size or _sha256(path) != artifact.sha256:
                raise ProductVerificationError("PRODUCT_FINAL_ARTIFACT_HASH_MISMATCH", artifact.path)

    def product_directory(self, product_ref: ProductRef) -> Path:
        return self.root / f"{product_ref.spacecraft_instance_id:016x}" / f"{product_ref.origin_boot_id:08x}" / f"{product_ref.product_id:08x}"

    def ensure_staging(
        self,
        product_ref: ProductRef,
        *,
        origin_request_key: RequestKey | None = None,
        expected_size: int | None = None,
        expected_bundle_sha256: str | None = None,
        created_at_us: int = 0,
        retention_until_us: int = 0,
    ) -> None:
        if expected_size is not None and expected_size < 0:
            raise ValueError("expected_size must be non-negative")
        if expected_bundle_sha256 is not None and not _is_sha256(expected_bundle_sha256.lower()):
            raise ValueError("expected_bundle_sha256 must be SHA-256")
        origin = origin_request_key or RequestKey(0, 0)
        self.writer.mutate(
            "ensure_ground_product_staging",
            lambda connection: connection.execute(
                "INSERT INTO products(spacecraft_instance_id,origin_boot_id,product_id,origin_ground_instance_id,origin_request_id,product_type,state,bundle_size,bundle_sha256,origin_request_key_json,created_at_us,retention_until_us) VALUES(?,?,?,?,?,'UNKNOWN','RECEIVING',?,?,?, ?,?) ON CONFLICT(spacecraft_instance_id,origin_boot_id,product_id) DO UPDATE SET bundle_size=COALESCE(products.bundle_size,excluded.bundle_size),bundle_sha256=COALESCE(products.bundle_sha256,excluded.bundle_sha256),origin_request_key_json=COALESCE(products.origin_request_key_json,excluded.origin_request_key_json)",
                (
                    encode_sqlite_u64(product_ref.spacecraft_instance_id),
                    product_ref.origin_boot_id,
                    product_ref.product_id,
                    encode_sqlite_u64(origin.ground_instance_id),
                    origin.request_id,
                    expected_size,
                    None if expected_bundle_sha256 is None else bytes.fromhex(expected_bundle_sha256.lower()),
                    json.dumps(origin.as_dict(), sort_keys=True, separators=(",", ":")),
                    created_at_us,
                    retention_until_us,
                ),
            ),
            priority=MutationPriority.HIGH,
        )

    def publish(self, verified: VerifiedBundle, *, retention_until_us: int | None = None) -> dict[str, Any]:
        verified_at_us = self._now_us()
        published_at_us = self._now_us()
        if retention_until_us is None:
            retention_until_us = published_at_us + 30 * 86_400_000_000
        if isinstance(retention_until_us, bool) or not isinstance(retention_until_us, int) or retention_until_us < 0:
            raise ValueError("retention_until_us must be a non-negative integer")
        if retention_until_us < published_at_us:
            raise ValueError("retention_until_us cannot precede published_at_us")
        product_ref = verified.manifest.product_ref
        final = self.product_directory(product_ref)
        final.parent.mkdir(parents=True, exist_ok=True)
        if final.exists():
            self._validate_final_directory(final, verified)
            return self._summary(final, verified.manifest, verified)
        self.ensure_staging(product_ref, expected_size=verified.bundle_size, expected_bundle_sha256=verified.bundle_sha256, retention_until_us=retention_until_us)
        staging = Path(tempfile.mkdtemp(prefix=f".staging-{product_ref.product_id:08x}-", dir=final.parent))
        try:
            manifest_bytes = verified.manifest.to_bytes()
            manifest_destination = staging / "manifest.json"
            _copy_fsync(verified.extracted_root / "manifest.json", manifest_destination)
            if _read_bounded(manifest_destination, max_bytes=MAX_MANIFEST_BYTES, code="MANIFEST_COPY_MISMATCH") != manifest_bytes:
                raise ProductVerificationError("MANIFEST_COPY_MISMATCH", "manifest changed after bundle verification")
            for artifact in verified.manifest.artifacts:
                source = verified.extracted_root / artifact.path
                destination = staging / artifact.path
                destination.parent.mkdir(parents=True, exist_ok=True)
                _copy_fsync(source, destination, max_bytes=artifact.size, too_large_code="ARTIFACT_COPY_MISMATCH")
                staged_artifact = stream_file_integrity(destination, max_bytes=artifact.size)
                if staged_artifact.size != artifact.size or staged_artifact.sha256 != artifact.sha256:
                    raise ProductVerificationError("ARTIFACT_COPY_MISMATCH", f"artifact changed after bundle verification: {artifact.path}")
            staged_bundle = staging / "bundle.tar"
            _copy_fsync(verified.bundle_path, staged_bundle)
            staged_integrity = stream_file_integrity(staged_bundle, max_bytes=verified.bundle_size)
            if (
                staged_integrity.size != verified.bundle_size
                or staged_integrity.sha256 != verified.bundle_sha256
                or staged_integrity.cfdp_checksum != verified.file_checksum
            ):
                raise ProductVerificationError("BUNDLE_COPY_MISMATCH", "bundle changed while being published")
            _fsync_directory(staging)
            self.writer.mutate(
                "mark_product_verified",
                lambda connection: connection.execute(
                    "UPDATE products SET state='VERIFIED',product_type=?,manifest_json=?,manifest_sha256=?,bundle_size=?,bundle_sha256=?,file_checksum=?,verified_at_us=?,retention_until_us=? WHERE spacecraft_instance_id=? AND origin_boot_id=? AND product_id=? AND state IN ('RECEIVING','VERIFIED','PUBLISHING')",
                    (
                        verified.manifest.product_type,
                        manifest_bytes.decode("utf-8"),
                        bytes.fromhex(hashlib.sha256(manifest_bytes).hexdigest()),
                        verified.bundle_size,
                        bytes.fromhex(verified.bundle_sha256),
                        verified.file_checksum,
                        verified_at_us,
                        retention_until_us,
                        encode_sqlite_u64(product_ref.spacecraft_instance_id),
                        product_ref.origin_boot_id,
                        product_ref.product_id,
                    ),
                ),
                priority=MutationPriority.HIGH,
            )
            os.replace(staging, final)
            _fsync_directory(final.parent)
            self.writer.mutate(
                "mark_product_published",
                lambda connection: connection.execute(
                    "UPDATE products SET state='PUBLISHED',local_path=?,published_at_us=? WHERE spacecraft_instance_id=? AND origin_boot_id=? AND product_id=? AND state='VERIFIED'",
                    (str(final), published_at_us, encode_sqlite_u64(product_ref.spacecraft_instance_id), product_ref.origin_boot_id, product_ref.product_id),
                ),
                priority=MutationPriority.HIGH,
            )
            self._record_artifacts(product_ref, verified.manifest, final, retention_until_us)
            return self._summary(final, verified.manifest, verified)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            raise

    def _record_artifacts(self, product_ref: ProductRef, manifest: ProductManifest, root: Path, at_us: int) -> None:
        rows = tuple((encode_sqlite_u64(product_ref.spacecraft_instance_id), product_ref.origin_boot_id, product_ref.product_id, artifact.path, artifact.size, bytes.fromhex(artifact.sha256), "VERIFIED") for artifact in manifest.artifacts)
        if not rows:
            return
        self.writer.mutate(
            "record_product_artifacts",
            lambda connection: connection.executemany(
                "INSERT OR REPLACE INTO product_artifacts(spacecraft_instance_id,origin_boot_id,product_id,artifact_path,artifact_size,artifact_sha256,state) VALUES(?,?,?,?,?,?,?)",
                rows,
            ),
            priority=MutationPriority.HIGH,
        )

    @staticmethod
    def _summary(root: Path, manifest: ProductManifest, verified: VerifiedBundle) -> dict[str, Any]:
        return {
            "product_ref": manifest.product_ref.as_dict(),
            "product_directory": str(root),
            "manifest_sha256": hashlib.sha256(manifest.to_bytes()).hexdigest(),
            "bundle_sha256": verified.bundle_sha256,
            "bundle_size": verified.bundle_size,
            "file_checksum": verified.file_checksum,
            "artifacts": [artifact.as_dict() for artifact in manifest.artifacts],
        }

    def get(self, product_ref: ProductRef) -> dict[str, Any] | None:
        with self.writer.reader() as connection:
            row = connection.execute(
                "SELECT state,product_type,bundle_size,bundle_sha256,manifest_json,manifest_sha256,local_path,created_at_us,verified_at_us,published_at_us,evicted_at_us,retention_until_us,pinned,eviction_reason,file_checksum FROM products WHERE spacecraft_instance_id=? AND origin_boot_id=? AND product_id=?",
                (encode_sqlite_u64(product_ref.spacecraft_instance_id), product_ref.origin_boot_id, product_ref.product_id),
            ).fetchone()
        if row is None:
            return None
        return {
            "product_ref": product_ref.as_dict(),
            "state": str(row[0]),
            "product_type": str(row[1]),
            "bundle_size": row[2],
            "bundle_sha256": None if row[3] is None else bytes(row[3]).hex(),
            "manifest": None if row[4] is None else json.loads(str(row[4])),
            "manifest_sha256": None if row[5] is None else bytes(row[5]).hex(),
            "local_path": row[6],
            "created_at_us": int(row[7]),
            "verified_at_us": row[8],
            "published_at_us": row[9],
            "evicted_at_us": row[10],
            "retention_until_us": row[11],
            "pinned": bool(row[12]),
            "eviction_reason": row[13],
            "file_checksum": row[14],
        }

    def reconcile(self) -> tuple[str, ...]:
        """Repair DB rows after a crash between rename and DB commit."""

        repaired: list[str] = []
        for bundle_path in self.root.glob("*/[0-9a-f]{8}/[0-9a-f]{8}/bundle.tar"):
            # Path.glob does not support regex braces; this branch is kept for
            # clarity and the recursive scan below handles actual directories.
            _ = bundle_path
        for bundle_path in self.root.glob("**/bundle.tar"):
            final = bundle_path.parent
            if final.name.startswith(".staging-") or not (final / "manifest.json").is_file():
                continue
            try:
                manifest_bytes = _read_bounded(
                    final / "manifest.json",
                    max_bytes=MAX_MANIFEST_BYTES,
                    code="MANIFEST_TOO_LARGE",
                )
                manifest = ProductManifest.from_bytes(manifest_bytes)
                integrity = stream_file_integrity(bundle_path)
                with tempfile.TemporaryDirectory(prefix=".reconcile-", dir=str(self.root)) as temporary:
                    verified = verify_bundle(
                        bundle_path,
                        expected_bundle_sha256=integrity.sha256,
                        expected_file_checksum=integrity.cfdp_checksum,
                        expected_product_ref=manifest.product_ref,
                        temporary_root=Path(temporary),
                    )
                    if verified.manifest.to_bytes() != manifest_bytes:
                        raise ProductVerificationError("MANIFEST_BUNDLE_MISMATCH", "final manifest differs from bundle manifest")
                    self._validate_final_directory(final, verified)
                product = manifest.product_ref
                existing = self.get(product)
                if existing is None or existing.get("state") != "PUBLISHED":
                    verified_at_us = self._now_us()
                    published_at_us = self._now_us()
                    retention_until_us = published_at_us + 30 * 86_400_000_000
                    self.ensure_staging(
                        product,
                        origin_request_key=manifest.origin_request_key,
                        expected_size=integrity.size,
                        expected_bundle_sha256=integrity.sha256,
                        created_at_us=verified_at_us,
                        retention_until_us=retention_until_us,
                    )
                    self.writer.mutate(
                        "reconcile_verified_product",
                        lambda connection: connection.execute(
                            "UPDATE products SET state='VERIFIED',bundle_size=?,bundle_sha256=?,manifest_json=?,manifest_sha256=?,file_checksum=?,verified_at_us=?,retention_until_us=? WHERE spacecraft_instance_id=? AND origin_boot_id=? AND product_id=? AND state IN ('RECEIVING','VERIFIED','PUBLISHING')",
                            (integrity.size, bytes.fromhex(integrity.sha256), manifest.to_bytes().decode("utf-8"), bytes.fromhex(hashlib.sha256(manifest.to_bytes()).hexdigest()), integrity.cfdp_checksum, verified_at_us, retention_until_us, encode_sqlite_u64(product.spacecraft_instance_id), product.origin_boot_id, product.product_id),
                        ),
                        priority=MutationPriority.HIGH,
                    )
                    self.writer.mutate(
                        "reconcile_published_product",
                        lambda connection: connection.execute(
                            "UPDATE products SET state='PUBLISHED',local_path=?,published_at_us=? WHERE spacecraft_instance_id=? AND origin_boot_id=? AND product_id=? AND state='VERIFIED'",
                            (str(final), published_at_us, encode_sqlite_u64(product.spacecraft_instance_id), product.origin_boot_id, product.product_id),
                        ),
                        priority=MutationPriority.HIGH,
                    )
                    self._record_artifacts(product, manifest, final, published_at_us)
                    repaired.append(str(final))
            except (OSError, ProductVerificationError, ValueError) as exc:
                self._record_reconciliation_failure(final, type(exc).__name__, str(exc))
                continue
        return tuple(repaired)

    def _record_reconciliation_failure(self, path: Path, code: str, message: str) -> None:
        self.writer.mutate(
            "audit_product_reconciliation_failure",
            lambda connection: append_audit_in_transaction(
                connection,
                principal="gds-product-reconciler",
                action="PRODUCT_RECONCILIATION_FAILED",
                target_type="product_directory",
                target_identity={"path": str(path)},
                old_value=None,
                new_value={"error_code": code, "message": message},
                created_at_us=self._now_us(),
            ),
            priority=MutationPriority.HIGH,
        )
