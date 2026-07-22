"""Reference satellite deployment readiness and lifecycle."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import yaml

from protocol.profile import MissionProfile, load_fprime_constants
from protocol.slo import SloProfile
from sat_ai.contracts import AcceptanceProfile, TargetDeploymentSpec
from sat_ai.manifest import ModelManifest, load_model_manifest
from sat_ai.products import cleanup_staging_products
from sat_ai.threshold_lut import ThresholdLUT

from .catalog import SceneCatalog
from .journal import SatelliteJournal


logger = logging.getLogger(__name__)


def validate_segmentation_release_contracts(
    manifest: ModelManifest,
    deployment_profile: Mapping[str, Any],
    acceptance_profile: AcceptanceProfile | None,
    target_deployment_spec: TargetDeploymentSpec | None,
) -> None:
    """Prevent a pilot SegFormer contract from being promoted by profile edits alone."""

    if manifest.model_task != "semantic_cloud_segmentation":
        return
    if acceptance_profile is None:
        raise ValueError("SegFormer deployment requires an acceptance profile")
    if acceptance_profile.profile_id != manifest.acceptance_profile_id:
        raise ValueError("SegFormer acceptance profile does not match model manifest")
    if target_deployment_spec is None:
        raise ValueError("SegFormer deployment requires a target deployment spec")
    if target_deployment_spec.target_id != deployment_profile.get("target_id"):
        raise ValueError("SegFormer target deployment spec does not match deployment profile")
    if target_deployment_spec.batch_size != int(deployment_profile.get("batch_size", 0)):
        raise ValueError("SegFormer target batch size does not match deployment profile")
    if not deployment_profile.get("deployable"):
        return
    if deployment_profile.get("ready") is not True:
        raise ValueError("deployable SegFormer profile must be explicitly ready")
    if manifest.assurance_level != "validated":
        raise ValueError("deployable SegFormer manifest must be validated")
    if acceptance_profile.status != "approved":
        raise ValueError("deployable SegFormer acceptance profile must be approved")


class DeploymentState(str, Enum):
    STARTING = "STARTING"
    READY = "READY"
    DEGRADED = "DEGRADED"
    FAULT = "FAULT"
    STOPPED = "STOPPED"


@dataclass
class DeploymentReadiness:
    state: DeploymentState
    reason: str | None = None


class SatelliteDeployment:
    def __init__(
        self,
        root: str | Path,
        *,
        state_directory: str | Path | None = None,
        product_directory: str | Path | None = None,
        deployment_profile_path: str | Path | None = None,
    ):
        self.root = Path(root).resolve()
        self.state_directory = Path(state_directory or self.root / "data" / "satellite" / "state").resolve()
        self.product_directory = Path(product_directory or self.root / "data" / "satellite" / "products").resolve()
        self.deployment_profile_path = Path(
            deployment_profile_path or self.root / "sat_ai" / "deployment_profile.yaml"
        ).resolve()
        self.state_directory.mkdir(parents=True, exist_ok=True)
        self.product_directory.mkdir(parents=True, exist_ok=True)
        self.readiness = DeploymentReadiness(DeploymentState.STARTING)
        self.profile: MissionProfile | None = None
        self.manifest: ModelManifest | None = None
        self.lut: ThresholdLUT | None = None
        self.catalog: SceneCatalog | None = None
        self.journal: SatelliteJournal | None = None
        self.deployment_profile: dict[str, Any] | None = None
        self.benchmark_artifact: dict[str, Any] | None = None
        self.slo_profile: SloProfile | None = None
        self.acceptance_profile: AcceptanceProfile | None = None
        self.target_deployment_spec: TargetDeploymentSpec | None = None
        logger.info(
            "deployment_start root=%s state_directory=%s product_directory=%s",
            self.root,
            self.state_directory,
            self.product_directory,
        )
        self._validate()

    def _validate(self) -> None:
        try:
            self.profile = MissionProfile.from_file(self.root / "protocol" / "mission_profile.yaml")
            load_fprime_constants(self.root / "fprime_dictionary.json")
            self.deployment_profile = yaml.safe_load(self.deployment_profile_path.read_text(encoding="utf-8"))
            manifest_path = self.root / str(self.deployment_profile.get("model_manifest_path", "sat_ai/model_manifest.yaml"))
            checkpoint_path = self.root / str(self.deployment_profile.get("checkpoint_path", "checkpoints/best_model.pth"))
            self.manifest = load_model_manifest(manifest_path, checkpoint_path)
            if self.manifest.model_task != str(self.deployment_profile.get("model_task", "patch_classification")):
                raise ValueError("deployment profile model_task does not match model manifest")
            acceptance_path = self.deployment_profile.get("acceptance_profile_path")
            if acceptance_path:
                self.acceptance_profile = AcceptanceProfile.from_file(self.root / str(acceptance_path))
            target_path = self.deployment_profile.get("target_deployment_spec_path")
            if target_path:
                self.target_deployment_spec = TargetDeploymentSpec.from_file(self.root / str(target_path))
            validate_segmentation_release_contracts(
                self.manifest,
                self.deployment_profile,
                self.acceptance_profile,
                self.target_deployment_spec,
            )
            self.lut = ThresholdLUT.from_file(
                self.root / "protocol" / "golden_vectors" / "threshold_lut.bin",
                self.manifest.threshold_lut_sha256 or None,
            )
            self._validate_benchmark()
            self.slo_profile = SloProfile.from_file(self.root / "protocol" / "slo_profile.yaml")
            assert self.benchmark_artifact is not None
            self.slo_profile.validate_benchmark(
                self.benchmark_artifact,
                hashlib.sha256(
                    (self.root / "artifacts" / "benchmarks" / f"{self.slo_profile.benchmark_artifact_id}.json").read_bytes()
                ).hexdigest(),
            )
            self.catalog = SceneCatalog.from_file(self.root / "data" / "satellite" / "scenes" / "catalog.json")
            self.journal = SatelliteJournal(self.state_directory / "satellite.sqlite3", self.profile.spacecraft_instance_id)
            reconciliation = self.journal.reconcile_after_restart(self.state_directory / "staging")
            cleanup_staging_products(self.product_directory)
            if reconciliation:
                raise RuntimeError(";".join(reconciliation))
            self.readiness = DeploymentReadiness(DeploymentState.READY)
            logger.info(
                "deployment_ready spacecraft_instance_id=%016x catalog=%s/%s model=%s",
                self.profile.spacecraft_instance_id,
                self.catalog.epoch,
                self.catalog.revision,
                self.manifest.model_release_id,
            )
        except Exception as exc:
            self.readiness = DeploymentReadiness(DeploymentState.FAULT, str(exc))
            logger.exception("deployment_fault reason=%s", exc)

    def _validate_benchmark(self) -> None:
        assert self.deployment_profile is not None
        if self.deployment_profile.get("schema_version") != 1:
            raise ValueError("deployment profile schema_version must be 1")
        if not self.deployment_profile.get("deployable"):
            raise ValueError("deployment profile is not deployable")
        benchmark_id = self.deployment_profile.get("benchmark_artifact_id")
        benchmark_sha = str(self.deployment_profile.get("benchmark_artifact_sha256", ""))
        if not benchmark_id or len(benchmark_sha) != 64 or set(benchmark_sha) == {"0"}:
            raise ValueError("deployment profile requires a non-null benchmark artifact")
        path = self.root / "artifacts" / "benchmarks" / f"{benchmark_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"benchmark artifact not found: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != benchmark_sha:
            raise ValueError("benchmark artifact SHA-256 mismatch")
        artifact = json.loads(path.read_text(encoding="utf-8"))
        if artifact.get("schema_version") != 1 or artifact.get("artifact_id") != benchmark_id:
            raise ValueError("benchmark artifact identity mismatch")
        if artifact.get("target_id") != self.deployment_profile.get("target_id"):
            raise ValueError("benchmark target does not match deployment profile")
        if artifact.get("runtime") != self.deployment_profile.get("runtime"):
            raise ValueError("benchmark runtime does not match deployment profile")
        if int(self.deployment_profile.get("cpu_threads", 0)) <= 0:
            raise ValueError("deployment cpu_threads must be positive")
        if int(artifact.get("cpu_threads", 0)) != int(self.deployment_profile["cpu_threads"]):
            raise ValueError("benchmark CPU thread count does not match deployment profile")
        batch_size = int(self.deployment_profile.get("batch_size", 0))
        max_batch_size = int(self.deployment_profile.get("max_batch_size", 0))
        if batch_size <= 0 or batch_size > max_batch_size or batch_size not in artifact.get("batch_sizes", []):
            raise ValueError("deployment batch size is not covered by benchmark")
        measurements = artifact.get("measurements", {})
        if float(measurements.get("p95_latency_ms", 0)) <= 0:
            raise ValueError("benchmark p95 latency is missing")
        if float(measurements.get("p99_latency_ms", 0)) <= 0:
            raise ValueError("benchmark p99 latency is missing")
        if int(measurements.get("logical_bytes_read", 0)) <= 0:
            raise ValueError("benchmark logical-read measurement is missing")
        if int(measurements.get("rss_delta_bytes", -1)) > int(self.deployment_profile.get("max_window_rss_delta_bytes", 0)):
            raise ValueError("benchmark exceeds deployment RSS guard")
        if float(measurements.get("scene_scale_p95_ratio", 999)) > float(self.deployment_profile.get("scene_scale_p95_ratio", 0)):
            raise ValueError("benchmark exceeds deployment scene-scale guard")
        guards = artifact.get("guards", {})
        if guards.get("rss_pass") is not True or guards.get("scene_scale_pass") is not True:
            raise ValueError("benchmark guard result is not deployable")
        self.benchmark_artifact = artifact

    def set_worker_state(self, worker_state: str) -> None:
        previous_state = self.state.value
        if worker_state == "FAULT":
            self.readiness = DeploymentReadiness(DeploymentState.FAULT, "AI worker restart limit exceeded")
        elif worker_state == "DEGRADED" and self.state != DeploymentState.FAULT:
            self.readiness = DeploymentReadiness(DeploymentState.DEGRADED, "AI worker unavailable")
        elif worker_state == "READY" and self.state == DeploymentState.DEGRADED:
            self.readiness = DeploymentReadiness(DeploymentState.READY)
        if self.state.value != previous_state:
            logger.info(
                "deployment_state_changed worker=%s state=%s previous_state=%s reason=%s",
                worker_state,
                self.state.value,
                previous_state,
                self.readiness.reason,
            )

    @property
    def state(self) -> DeploymentState:
        return self.readiness.state

    @property
    def ready(self) -> bool:
        return self.state == DeploymentState.READY

    def health(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "reason": self.readiness.reason,
            "spacecraft_instance_id": f"{self.profile.spacecraft_instance_id:016x}" if self.profile else None,
            "sender_boot_id": self.journal.boot_id if self.journal else None,
            "model_release_id": self.manifest.model_release_id if self.manifest else None,
            "model_task": self.manifest.model_task if self.manifest else None,
            "model_contracts": None if self.manifest is None else {
                "input_spec_id": self.manifest.input_spec.input_spec_id,
                "decision_spec_id": None if self.manifest.decision_spec is None else self.manifest.decision_spec.decision_spec_id,
                "postprocess_id": None if self.manifest.postprocess_spec is None else self.manifest.postprocess_spec.postprocess_id,
                "product_spec_id": None if self.manifest.product_spec is None else self.manifest.product_spec.product_spec_id,
            },
            "model_assurance": self.manifest.assurance_level if self.manifest else None,
            "slo_revision": self.slo_profile.config_revision if self.slo_profile else None,
            "catalog_epoch": self.catalog.epoch if self.catalog else None,
            "catalog_revision": self.catalog.revision if self.catalog else None,
        }

    def close(self) -> None:
        if self.journal is not None:
            self.journal.close()
        self.readiness = DeploymentReadiness(DeploymentState.STOPPED)
        logger.info("deployment_stopped")
