"""Satellite-side model and ROI adapter package."""

from .manifest import InputSpec, ModelManifest, load_model_manifest
from .roi import ROI, SceneWindow, VerifiedFileFingerprint, open_memmap_scene, verify_file_fingerprint

__all__ = [
    "InputSpec",
    "ModelManifest",
    "ROI",
    "SceneWindow",
    "VerifiedFileFingerprint",
    "load_model_manifest",
    "open_memmap_scene",
    "verify_file_fingerprint",
]
