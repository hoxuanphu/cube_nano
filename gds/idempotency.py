"""RFC 8785 semantic HTTP idempotency and expiry materialization."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Mapping

from protocol.canonical import jcs_canonical_json
from protocol.schemas import normalize_http_idempotency_body

DEFAULT_EXPIRY_SENTINEL = "DEFAULT"
DEFAULT_IMMEDIATE_TTL = timedelta(minutes=5)
DEFAULT_NEXT_CONTACT_TTL = timedelta(hours=1)
MIN_COMMAND_TTL = timedelta(seconds=1)
MAX_COMMAND_TTL = timedelta(hours=24)

_RFC3339 = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})T"
    r"(?P<time>\d{2}:\d{2}:\d{2})"
    r"(?P<fraction>\.\d{1,6})?"
    r"(?P<zone>Z|[+-]\d{2}:\d{2})$"
)


class IdempotencyValidationError(ValueError):
    pass


@dataclass(frozen=True)
class SemanticIdempotency:
    normalized_body: dict[str, Any]
    canonical_jcs: bytes
    digest: bytes

    @property
    def digest_hex(self) -> str:
        return self.digest.hex()


def validate_idempotency_key(value: object) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 255:
        raise IdempotencyValidationError(
            "Idempotency-Key must contain 1..255 characters"
        )
    if value.strip() != value or any(ord(char) < 0x21 or ord(char) > 0x7E for char in value):
        raise IdempotencyValidationError(
            "Idempotency-Key must use visible ASCII without surrounding whitespace"
        )
    return value


def parse_rfc3339_utc(value: object) -> datetime:
    if not isinstance(value, str) or _RFC3339.fullmatch(value) is None:
        raise IdempotencyValidationError(
            "expires_at must be an RFC 3339 timestamp with an explicit timezone"
        )
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise IdempotencyValidationError("expires_at is not a valid timestamp") from exc
    return parsed.astimezone(UTC)


def format_rfc3339_utc(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise IdempotencyValidationError("timestamp must be timezone-aware")
    utc = value.astimezone(UTC)
    base = utc.strftime("%Y-%m-%dT%H:%M:%S")
    if utc.microsecond:
        fraction = f"{utc.microsecond:06d}".rstrip("0")
        return f"{base}.{fraction}Z"
    return f"{base}Z"


def datetime_to_unix_us(value: datetime) -> int:
    if value.tzinfo is None:
        raise IdempotencyValidationError("timestamp must be timezone-aware")
    delta = value.astimezone(UTC) - datetime(1970, 1, 1, tzinfo=UTC)
    result = (
        delta.days * 86_400_000_000
        + delta.seconds * 1_000_000
        + delta.microseconds
    )
    if result < 0:
        raise IdempotencyValidationError("timestamp must not precede Unix epoch")
    return result


def unix_us_to_datetime(value: int) -> datetime:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise IdempotencyValidationError("Unix microseconds must be non-negative")
    return datetime(1970, 1, 1, tzinfo=UTC) + timedelta(microseconds=value)


def jcs_canonicalize(value: Any) -> bytes:
    """Canonicalize the validated no-float mission API subset of RFC 8785."""

    try:
        return jcs_canonical_json(value)
    except (TypeError, ValueError) as exc:
        raise IdempotencyValidationError(str(exc)) from exc


def normalize_semantic_body(body: Mapping[str, Any]) -> dict[str, Any]:
    try:
        normalized = normalize_http_idempotency_body(body)
    except (TypeError, ValueError) as exc:
        raise IdempotencyValidationError(str(exc)) from exc
    # Canonicalization performs the recursive JSON/I-JSON validation.
    jcs_canonicalize(normalized)
    return normalized


def build_semantic_idempotency(body: Mapping[str, Any]) -> SemanticIdempotency:
    normalized = normalize_semantic_body(body)
    canonical = jcs_canonicalize(normalized)
    frozen_normalized = json.loads(canonical.decode("utf-8"))
    return SemanticIdempotency(
        normalized_body=frozen_normalized,
        canonical_jcs=canonical,
        digest=hashlib.sha256(canonical).digest(),
    )


def materialize_effective_expiry(
    semantic: SemanticIdempotency, now: datetime
) -> datetime:
    if not isinstance(now, datetime) or now.tzinfo is None:
        raise IdempotencyValidationError("server clock must be timezone-aware")
    now = now.astimezone(UTC)
    delivery_mode = semantic.normalized_body["delivery_mode"]
    expires_at = semantic.normalized_body["expires_at"]
    if expires_at == DEFAULT_EXPIRY_SENTINEL:
        ttl = (
            DEFAULT_IMMEDIATE_TTL
            if delivery_mode == "immediate"
            else DEFAULT_NEXT_CONTACT_TTL
        )
        effective = now + ttl
    else:
        effective = parse_rfc3339_utc(expires_at)
        ttl = effective - now
    if ttl < MIN_COMMAND_TTL or ttl > MAX_COMMAND_TTL:
        raise IdempotencyValidationError(
            "effective command TTL must be between 1 second and 24 hours"
        )
    return effective
