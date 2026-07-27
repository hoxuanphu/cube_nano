"""Public preprocessing facade.

Importing this package defines contracts only.  It does not initialize an
image decoder, GPU runtime, TensorRT engine, or filesystem writer.
"""

from ._version import __version__
from .admission import (
    AdmissionReceipt,
    ResourceAdmission,
    ResourceEstimate,
    ResourceSnapshot,
    RuntimeAdmission,
    StaticResourceProbe,
    SystemResourceProbe,
)
from .api import (
    ArtifactManifest,
    ArtifactOpenRequest,
    ModelGridReader,
    PreprocessArtifact,
    PreprocessRequest,
    open_preprocessed_artifact,
    preprocess_capture,
)
from .contracts import (
    PREPROCESS_ARTIFACT_SCHEMA_VERSION,
    PREPROCESSING_API_VERSION,
    SCHEMA_VERSION,
    CalibrationBundle,
    CalibrationSelector,
    CaptureManifest,
    ClippingPolicy,
    ComputeProfile,
    Halo,
    MaskEncoding,
    ModelCompatibilityProfile,
    PreprocessingProfile,
    ReasonEncoding,
    SourceDescriptor,
    SourceROI,
    SourceSchema,
    TargetGrid,
    TrustMetadata,
    TrustPolicy,
    attach_trust,
)
from .errors import FailureReason, PreprocessError, PreprocessFailure, RunState, SafeAction, StateMachine
from .resolver import ContractResolver, ResolvedContracts, resolve_contracts, sha256_file, verify_trust


__all__ = [
    "__version__",
    "ArtifactManifest",
    "ArtifactOpenRequest",
    "CalibrationBundle",
    "CalibrationSelector",
    "CaptureManifest",
    "ClippingPolicy",
    "ComputeProfile",
    "FailureReason",
    "Halo",
    "MaskEncoding",
    "ModelCompatibilityProfile",
    "ModelGridReader",
    "PREPROCESS_ARTIFACT_SCHEMA_VERSION",
    "PREPROCESSING_API_VERSION",
    "PreprocessArtifact",
    "PreprocessError",
    "PreprocessFailure",
    "PreprocessRequest",
    "PreprocessingProfile",
    "ReasonEncoding",
    "RunState",
    "SCHEMA_VERSION",
    "SafeAction",
    "SourceDescriptor",
    "SourceROI",
    "SourceSchema",
    "TargetGrid",
    "TrustMetadata",
    "TrustPolicy",
    "attach_trust",
    "open_preprocessed_artifact",
    "preprocess_capture",
]
