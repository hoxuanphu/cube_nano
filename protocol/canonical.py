"""Canonical scalar, JSON and deterministic-CBOR helpers.

The mission profile deliberately keeps unsigned 64-bit values out of JSON
numbers. This module is the one place where the representation is defined so
Python, the binary protocol and future TypeScript clients cannot drift apart.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping
from typing import Any

MAX_U8 = (1 << 8) - 1
MAX_U16 = (1 << 16) - 1
MAX_U32 = (1 << 32) - 1
MAX_U64 = (1 << 64) - 1
MAX_SAFE_JSON_INTEGER = (1 << 53) - 1


def checked_uint(value: int, bits: int, label: str = "value") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    maximum = (1 << bits) - 1
    if value < 0 or value > maximum:
        raise ValueError(f"{label} must be in [0, {maximum}]")
    return value


def checked_u16(value: int, label: str = "value") -> int:
    return checked_uint(value, 16, label)


def checked_u32(value: int, label: str = "value") -> int:
    return checked_uint(value, 32, label)


def checked_u64(value: int, label: str = "value") -> int:
    return checked_uint(value, 64, label)


def checked_add_u64(left: int, right: int, label: str = "sum") -> int:
    result = checked_u64(left, "left") + checked_u64(right, "right")
    return checked_u64(result, label)


def checked_mul_u64(left: int, right: int, label: str = "product") -> int:
    result = checked_u64(left, "left") * checked_u64(right, "right")
    return checked_u64(result, label)


def u64_to_bytes(value: int) -> bytes:
    return checked_u64(value, "U64").to_bytes(8, "big")


def u64_from_bytes(value: bytes, label: str = "U64") -> int:
    if not isinstance(value, (bytes, bytearray, memoryview)) or len(value) != 8:
        raise ValueError(f"{label} must contain exactly 8 bytes")
    return int.from_bytes(value, "big")


def u64_to_json(value: int) -> str:
    return f"{checked_u64(value, 'U64'):016x}"


def u64_from_json(value: Any, label: str = "U64") -> int:
    if not isinstance(value, str) or len(value) != 16:
        raise ValueError(f"{label} must be a 16-character lowercase hex string")
    if value != value.lower() or value.strip() != value:
        raise ValueError(f"{label} must use lowercase hex without whitespace")
    if any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be lowercase hexadecimal")
    return checked_u64(int(value, 16), label)


def canonical_json(value: Any) -> bytes:
    """Return the UTF-8 JSON form used by the MVP JCS boundary."""

    def validate(item: Any) -> None:
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("canonical JSON does not allow non-finite numbers")
            raise TypeError("canonical JSON mission values must not contain floats")
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise TypeError("canonical JSON object keys must be strings")
            for key, child in item.items():
                validate(key)
                validate(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                validate(child)
        elif not isinstance(item, (str, int, bool, type(None), bytes)):
            raise TypeError(f"unsupported canonical JSON value: {type(item).__name__}")
        elif isinstance(item, bytes):
            raise TypeError("canonical JSON does not encode raw bytes")

    validate(value)
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _jcs_utf16_sort_key(value: str) -> bytes:
    try:
        return value.encode("utf-16-be")
    except UnicodeEncodeError as exc:
        raise ValueError(
            "JCS strings must not contain unpaired UTF-16 surrogates"
        ) from exc


def _jcs_string(value: str) -> str:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(
            "JCS strings must contain valid Unicode scalar values"
        ) from exc
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _jcs_text(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return _jcs_string(value)
    if isinstance(value, int) and not isinstance(value, bool):
        if abs(value) > MAX_SAFE_JSON_INTEGER:
            raise ValueError(
                "JCS integers outside the exact IEEE-754 range must be strings"
            )
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JCS does not allow non-finite numbers")
        raise TypeError(
            "validated mission HTTP bodies must not contain floating-point values"
        )
    if isinstance(value, list):
        return "[" + ",".join(_jcs_text(item) for item in value) + "]"
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("JCS object keys must be strings")
        keys = sorted(value, key=_jcs_utf16_sort_key)
        return "{" + ",".join(
            _jcs_string(key) + ":" + _jcs_text(value[key]) for key in keys
        ) + "}"
    raise TypeError(f"unsupported JSON value type {type(value).__name__}")


def jcs_canonical_json(value: Any) -> bytes:
    """Canonicalize the no-float I-JSON subset used by mission HTTP bodies."""

    return _jcs_text(value).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _cbor_head(major: int, value: int) -> bytes:
    checked_u64(value, "CBOR value")
    if value < 24:
        return bytes([(major << 5) | value])
    if value <= MAX_U8:
        return bytes([(major << 5) | 24, value])
    if value <= MAX_U16:
        return bytes([(major << 5) | 25]) + struct.pack(">H", value)
    if value <= MAX_U32:
        return bytes([(major << 5) | 26]) + struct.pack(">I", value)
    return bytes([(major << 5) | 27]) + struct.pack(">Q", value)


def deterministic_cbor_encode(value: Any) -> bytes:
    """Encode the definite-length, no-float subset used by replay artifacts."""

    if value is None:
        return b"\xf6"
    if value is False:
        return b"\xf4"
    if value is True:
        return b"\xf5"
    if isinstance(value, int) and not isinstance(value, bool):
        if value >= 0:
            return _cbor_head(0, checked_u64(value, "CBOR unsigned integer"))
        return _cbor_head(1, checked_u64(-1 - value, "CBOR negative integer"))
    if isinstance(value, bytes):
        return _cbor_head(2, len(value)) + value
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        return _cbor_head(3, len(encoded)) + encoded
    if isinstance(value, (list, tuple)):
        return _cbor_head(4, len(value)) + b"".join(
            deterministic_cbor_encode(item) for item in value
        )
    if isinstance(value, dict):
        entries = []
        for key, item in value.items():
            key_bytes = deterministic_cbor_encode(key)
            entries.append((key_bytes, deterministic_cbor_encode(item)))
        entries.sort(key=lambda pair: (len(pair[0]), pair[0]))
        return _cbor_head(5, len(entries)) + b"".join(
            key_bytes + item_bytes for key_bytes, item_bytes in entries
        )
    raise TypeError(f"unsupported deterministic CBOR type: {type(value).__name__}")


class _CborReader:
    def __init__(self, data: bytes):
        self.data = bytes(data)
        self.offset = 0

    def _read(self, size: int) -> bytes:
        end = self.offset + size
        if end > len(self.data):
            raise ValueError("truncated deterministic CBOR")
        result = self.data[self.offset:end]
        self.offset = end
        return result

    def _length(self, additional: int) -> int:
        if additional < 24:
            return additional
        if additional == 24:
            value = self._read(1)[0]
            if value < 24:
                raise ValueError("non-canonical CBOR length")
            return value
        if additional == 25:
            value = struct.unpack(">H", self._read(2))[0]
            if value <= MAX_U8:
                raise ValueError("non-canonical CBOR length")
            return value
        if additional == 26:
            value = struct.unpack(">I", self._read(4))[0]
            if value <= MAX_U16:
                raise ValueError("non-canonical CBOR length")
            return value
        if additional == 27:
            value = struct.unpack(">Q", self._read(8))[0]
            if value <= MAX_U32:
                raise ValueError("non-canonical CBOR length")
            return value
        raise ValueError("indefinite-length or reserved CBOR item is not supported")

    def value(self) -> Any:
        initial = self._read(1)[0]
        major, additional = initial >> 5, initial & 0x1F
        if major in (0, 1):
            number = self._length(additional)
            return number if major == 0 else -1 - number
        if major in (2, 3):
            raw = self._read(self._length(additional))
            if major == 2:
                return raw
            try:
                return raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError("invalid UTF-8 in deterministic CBOR") from exc
        if major == 4:
            return [self.value() for _ in range(self._length(additional))]
        if major == 5:
            result = {}
            encoded_keys = set()
            for _ in range(self._length(additional)):
                key_start = self.offset
                key = self.value()
                key_bytes = self.data[key_start:self.offset]
                if key_bytes in encoded_keys or key in result:
                    raise ValueError("duplicate deterministic CBOR map key")
                encoded_keys.add(key_bytes)
                result[key] = self.value()
            return result
        if major == 7 and additional in (20, 21, 22):
            return {20: False, 21: True, 22: None}[additional]
        raise ValueError("floats, tags and reserved CBOR values are not supported")


def deterministic_cbor_decode(data: bytes) -> Any:
    reader = _CborReader(data)
    value = reader.value()
    if reader.offset != len(reader.data):
        raise ValueError("trailing bytes after deterministic CBOR value")
    return value
