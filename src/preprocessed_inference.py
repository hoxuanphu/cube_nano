"""Inference consumer for verified preprocessing artifacts.

The consumer deliberately has no raw-image entry point.  Orchestration must
first call the public ``preprocessing`` facade and then pass the resulting
``PreprocessArtifact`` to :class:`PreprocessedInference`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import preprocessing as preprocessing_api
from inference_adapter import InferenceAdapter, InferenceResult
from preprocessing import (
    ArtifactOpenRequest,
    PreprocessArtifact,
    PreprocessFailure,
    PreprocessRequest,
)


@dataclass(frozen=True)
class InferenceProcessRequest:
    """Explicitly select raw or already-materialized artifact mode."""

    preprocessing: PreprocessRequest | None = None
    artifact: ArtifactOpenRequest | None = None

    def __post_init__(self) -> None:
        if (self.preprocessing is None) == (self.artifact is None):
            raise ValueError("exactly one of preprocessing or artifact must be provided")


class PreprocessedInference:
    """Run model inference using only an already verified artifact."""

    def __init__(self, adapter: InferenceAdapter) -> None:
        if not hasattr(adapter, "run") or not callable(adapter.run):
            raise TypeError("adapter must expose run(artifact)")
        self.adapter = adapter

    def run(self, artifact: PreprocessArtifact) -> InferenceResult | PreprocessFailure:
        if isinstance(artifact, PreprocessFailure):
            return artifact
        if not isinstance(artifact, PreprocessArtifact):
            raise TypeError("run() accepts only a PreprocessArtifact")
        return self.adapter.run(artifact)


def process_capture(
    request: InferenceProcessRequest | None = None,
    *,
    preprocessing_request: PreprocessRequest | None = None,
    artifact_request: ArtifactOpenRequest | None = None,
    inference: PreprocessedInference,
) -> InferenceResult | PreprocessFailure:
    """Resolve one explicit mode through the public facade, then run inference.

    The facade call is made exactly once.  In artifact mode no source reader,
    calibration, or warp backend is touched; the artifact manifest is the
    authority for all preprocessing metadata.
    """

    if request is not None:
        if preprocessing_request is not None or artifact_request is not None:
            raise TypeError("request cannot be combined with explicit mode arguments")
        preprocessing_request = request.preprocessing
        artifact_request = request.artifact
    if not isinstance(inference, PreprocessedInference):
        raise TypeError("inference must be a PreprocessedInference")
    if (preprocessing_request is None) == (artifact_request is None):
        raise ValueError("exactly one of preprocessing_request or artifact_request is required")

    if preprocessing_request is not None:
        artifact = preprocessing_api.preprocess_capture(preprocessing_request)
    else:
        artifact = preprocessing_api.open_preprocessed_artifact(artifact_request)
    if isinstance(artifact, PreprocessFailure):
        return artifact
    return inference.run(artifact)


__all__ = [
    "PreprocessedInference",
    "InferenceProcessRequest",
    "process_capture",
]
