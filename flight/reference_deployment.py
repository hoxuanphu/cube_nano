"""Build-time reference deployment checks for profile/dictionary parity."""

from __future__ import annotations

from pathlib import Path

from protocol.profile import MissionProfile, load_fprime_constants

from .cloud_payload import CloudPayload
from .deployment import SatelliteDeployment


def build_reference_deployment(root: str | Path = ".", *, state_directory: str | Path | None = None) -> tuple[SatelliteDeployment, CloudPayload]:
    root = Path(root)
    profile = MissionProfile.from_file(root / "protocol" / "mission_profile.yaml")
    constants = load_fprime_constants(root / "fprime_dictionary.json")
    if constants["ComCfg.SpacecraftId"] != profile.spacecraft_id or constants["ComCfg.TmFrameFixedSize"] != profile.tm_frame_size:
        raise RuntimeError("profile/dictionary build constants diverge")
    deployment = SatelliteDeployment(root, state_directory=state_directory)
    payload = CloudPayload(deployment)
    return deployment, payload
