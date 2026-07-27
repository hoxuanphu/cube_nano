"""Explicit LinkControl messages for session and transfer ownership.

LinkControl is intentionally separate from CCSDS application packets. It is a
small deterministic control plane used by the link boundary to make session
opening, frame acceptance/consumption, reset, and file-epoch abort observable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping

from protocol.canonical import canonical_json, checked_u32, checked_u64


class LinkControlType(StrEnum):
    OPEN_SESSION = "OPEN_SESSION"
    SESSION_READY = "SESSION_READY"
    FRAME_ACCEPTED = "FRAME_ACCEPTED"
    FRAME_CONSUMED = "FRAME_CONSUMED"
    ABORT_FILE_EPOCH = "ABORT_FILE_EPOCH"
    SESSION_RESET = "SESSION_RESET"


@dataclass(frozen=True)
class LinkControlMessage:
    message_type: LinkControlType
    spacecraft_instance_id: int
    sender_boot_id: int
    link_session_id: int = 0
    link_generation: int = 0
    link_frame_id: int = 0
    file_epoch_id: int = 0
    copy_index: int = 0
    status: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "message_type", LinkControlType(self.message_type))
        checked_u64(self.spacecraft_instance_id, "spacecraft_instance_id")
        checked_u32(self.sender_boot_id, "sender_boot_id")
        for name in ("link_session_id", "link_generation", "link_frame_id", "file_epoch_id"):
            checked_u64(getattr(self, name), name)
        checked_u32(self.copy_index, "copy_index")
        if self.message_type in {LinkControlType.FRAME_ACCEPTED, LinkControlType.FRAME_CONSUMED} and not self.link_frame_id:
            raise ValueError("frame lifecycle controls require link_frame_id")
        if self.message_type is LinkControlType.ABORT_FILE_EPOCH and not self.file_epoch_id:
            raise ValueError("ABORT_FILE_EPOCH requires file_epoch_id")

    def as_dict(self) -> dict[str, Any]:
        return {
            "message_type": self.message_type.value,
            "spacecraft_instance_id": self.spacecraft_instance_id,
            "sender_boot_id": self.sender_boot_id,
            "link_session_id": self.link_session_id,
            "link_generation": self.link_generation,
            "link_frame_id": self.link_frame_id,
            "file_epoch_id": self.file_epoch_id,
            "copy_index": self.copy_index,
            "status": self.status,
            "reason": self.reason,
        }

    def encode(self) -> bytes:
        return canonical_json(self.as_dict())

    @classmethod
    def decode(cls, data: bytes | bytearray | Mapping[str, Any]) -> "LinkControlMessage":
        if isinstance(data, Mapping):
            value = dict(data)
        else:
            raw = bytes(data)
            try:
                value = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("invalid LinkControl payload") from exc
        if not isinstance(value, dict):
            raise ValueError("LinkControl payload must be an object")
        if not isinstance(data, Mapping) and canonical_json(value) != raw:
            raise ValueError("invalid LinkControl payload")
        try:
            return cls(
                message_type=LinkControlType(value["message_type"]),
                spacecraft_instance_id=value["spacecraft_instance_id"],
                sender_boot_id=value["sender_boot_id"],
                link_session_id=value.get("link_session_id", 0),
                link_generation=value.get("link_generation", 0),
                link_frame_id=value.get("link_frame_id", 0),
                file_epoch_id=value.get("file_epoch_id", 0),
                copy_index=value.get("copy_index", 0),
                status=None if value.get("status") is None else str(value["status"]),
                reason=None if value.get("reason") is None else str(value["reason"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid LinkControl fields") from exc


# Short aliases make the control plane easy to consume from tests and adapters.
ControlType = LinkControlType
LinkControl = LinkControlMessage
