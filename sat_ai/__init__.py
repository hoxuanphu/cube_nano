"""Satellite-side model and ROI adapter package."""

from .manifest import InputSpec, ModelManifest, load_model_manifest
from .contracts import (
    AcceptanceProfile,
    DecisionSpec,
    ModelOutputSpec,
    PostprocessSpec,
    ProductSpec,
    TargetDeploymentSpec,
)
from .roi import ROI, SceneWindow, VerifiedFileFingerprint, open_memmap_scene, verify_file_fingerprint

__all__ = [
    "InputSpec",
    "ModelManifest",
    "ModelOutputSpec",
    "PostprocessSpec",
    "ProductSpec",
    "DecisionSpec",
    "AcceptanceProfile",
    "TargetDeploymentSpec",
    "ROI",
    "SceneWindow",
    "VerifiedFileFingerprint",
    "load_model_manifest",
    "open_memmap_scene",
    "verify_file_fingerprint",
]
