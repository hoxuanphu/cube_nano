"""Isolated AI worker process for the local satellite reference profile."""

from __future__ import annotations

import queue
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import yaml

from protocol.schemas import ConfigSnapshot, ProductRef, ROI, RequestKey

from .inference import InferenceConfig, InsufficientValidData, SingletonModelRuntime, configure_cpu_runtime, infer_region
from .manifest import load_model_manifest
from .products import build_products
from .roi import open_memmap_scene
from .threshold_lut import ThresholdLUT
from .worker_contract import (
    WORKER_VERSION,
    WorkerControl,
    WorkerControlAction,
    WorkerHeartbeatMessage,
    WorkerRequest,
    WorkerResult,
    WorkerResultState,
    map_worker_error,
)


class DeadlineExceeded(RuntimeError):
    pass


class WorkerCanceled(RuntimeError):
    pass


def _key(request_key: RequestKey) -> tuple[int, int]:
    return request_key.ground_instance_id, request_key.request_id


def _remove_product_directory(summary: dict[str, Any] | None) -> None:
    if summary is None:
        return
    value = summary.get("product_directory")
    if not value:
        return
    path = Path(str(value))
    if path.is_dir():
        shutil.rmtree(path)


def _execute_request(
    request: WorkerRequest,
    runtime: SingletonModelRuntime,
    lut: ThresholdLUT,
    canceled: set[tuple[int, int]],
    cancel_lock: threading.Lock,
) -> WorkerResult:
    snapshot = request.job_snapshot
    product_summary: dict[str, Any] | None = None

    def guard(*_args: object) -> None:
        with cancel_lock:
            is_canceled = _key(request.request_key) in canceled
        if is_canceled:
            raise WorkerCanceled("worker job was canceled")
        if request.deadline.expired():
            raise DeadlineExceeded("worker job deadline exceeded")

    try:
        guard()
        required = {
            "scene",
            "scene_ref",
            "roi",
            "config_snapshot",
            "product_ref",
            "origin_request_key",
            "product_root",
        }
        if required - set(snapshot):
            raise ValueError("worker job_snapshot is missing required fields")
        scene_value = snapshot["scene"]
        if not isinstance(scene_value, dict):
            raise ValueError("worker scene snapshot must be an object")
        config_value = snapshot["config_snapshot"]
        if not isinstance(config_value, dict):
            raise ValueError("worker config snapshot must be an object")
        config = ConfigSnapshot(
            int(config_value["config_epoch"]),
            int(config_value["config_revision"]),
            int(config_value["model_threshold_bp"]),
            int(config_value["coverage_limit_bp"]),
        )
        if runtime.manifest.model_task == "semantic_cloud_segmentation":
            decision = runtime.manifest.decision_spec
            if decision is None:
                raise ValueError("SegFormer manifest is missing DecisionSpec")
            decision.require_config(config.model_threshold_bp, config.coverage_limit_bp)
        roi = ROI.from_dict(snapshot["roi"])
        product_ref = ProductRef.from_dict(snapshot["product_ref"])
        origin = RequestKey.from_dict(snapshot["origin_request_key"])
        if origin != request.request_key:
            raise ValueError("worker origin_request_key mismatch")
        domain = scene_value.get("domain")
        domain_status = runtime.manifest.domain_status(domain if isinstance(domain, dict) else None)
        if domain_status == "DOMAIN_MISMATCH":
            raise ValueError("DOMAIN_MISMATCH")
        expected_contract = runtime.manifest.input_contract()
        if snapshot.get("model_task") != runtime.manifest.model_task:
            raise ValueError("MODEL_TASK_SNAPSHOT_MISMATCH")
        if snapshot.get("model_release_id") != runtime.manifest.model_release_id:
            raise ValueError("MODEL_RELEASE_SNAPSHOT_MISMATCH")
        if snapshot.get("input_spec_id") != expected_contract["input_spec_id"]:
            raise ValueError("INPUT_SPEC_SNAPSHOT_MISMATCH")
        if snapshot.get("input_contract") != expected_contract:
            raise ValueError("INPUT_CONTRACT_SNAPSHOT_MISMATCH")
        with open_memmap_scene(scene_value["path"], scene_value["sidecar_path"]) as scene:
            if scene.input_spec_id != expected_contract["input_spec_id"]:
                raise ValueError("INPUT_SPEC_SCENE_MISMATCH")
            if scene.source_dtype != expected_contract["source_dtype"]:
                raise ValueError("SOURCE_DTYPE_MISMATCH")
            if scene.normalization_id != expected_contract["normalization_id"]:
                raise ValueError("NORMALIZATION_MISMATCH")
            if scene.band_order != tuple(expected_contract["band_order"]):
                raise ValueError("BAND_ORDER_MISMATCH")
            result = infer_region(
                scene,
                roi,
                runtime,
                InferenceConfig(config, lut),
                progress_callback=guard,
                domain=domain if isinstance(domain, dict) else None,
            )
            guard()
            for provenance_key in (
                "acceptance_profile_id",
                "target_id",
                "deployment_profile_id",
            ):
                if provenance_key in snapshot:
                    result[provenance_key] = snapshot[provenance_key]
            result["scene_ref"] = snapshot["scene_ref"]
            result["product_ref"] = product_ref.as_dict()
            product_summary = build_products(
                result,
                scene,
                snapshot["product_root"],
                product_ref,
                origin,
                source_sha256=str(scene_value["source_sha256"]),
            )
        guard()
        result_summary = {
            name: value
            for name, value in result.items()
            if name not in {"cloud_mask", "validity_mask"}
        }
        result_summary["product"] = product_summary
        return WorkerResult(request.request_key, WorkerResultState.SUCCEEDED, result_summary)
    except InsufficientValidData:
        return WorkerResult(
            request.request_key,
            WorkerResultState.REJECTED,
            {"science_decision": "REJECTED", "error_code": "INSUFFICIENT_VALID_DATA"},
            "INSUFFICIENT_VALID_DATA",
        )
    except WorkerCanceled as exc:
        _remove_product_directory(product_summary)
        return WorkerResult(
            request.request_key,
            WorkerResultState.CANCELED,
            {"detail": str(exc)},
            "WORKER_CANCELED",
        )
    except DeadlineExceeded as exc:
        _remove_product_directory(product_summary)
        return WorkerResult(
            request.request_key,
            WorkerResultState.TIMEOUT,
            {"detail": str(exc)},
            "DEADLINE_EXCEEDED",
        )
    except BaseException as exc:
        _remove_product_directory(product_summary)
        return WorkerResult(
            request.request_key,
            WorkerResultState.FAILED,
            {"detail": str(exc)[:512], "exception_type": type(exc).__name__},
            map_worker_error(exc),
        )
    finally:
        with cancel_lock:
            canceled.discard(_key(request.request_key))


def worker_process_main(
    root: str,
    device: str,
    request_queue: Any,
    control_queue: Any,
    result_queue: Any,
    stop_event: Any,
    heartbeat_interval_ms: int,
    deployment_profile_path: str | None = None,
) -> None:
    """Multiprocessing target. All queue payloads are canonical JSON bytes."""

    root_path = Path(root)
    active: list[RequestKey | None] = [None]
    worker_state = ["STARTING"]
    state_lock = threading.Lock()
    cancel_lock = threading.Lock()
    canceled: set[tuple[int, int]] = set()
    heartbeat_sequence = [0]

    def emit_heartbeats() -> None:
        while not stop_event.is_set():
            with state_lock:
                state = worker_state[0]
                active_key = active[0]
                sequence = heartbeat_sequence[0]
                heartbeat_sequence[0] = (sequence + 1) & 0xFFFFFFFF
            message = WorkerHeartbeatMessage(
                WORKER_VERSION,
                sequence,
                state,
                time.monotonic_ns(),
                active_key,
            )
            try:
                result_queue.put(message.encode(), timeout=max(0.05, heartbeat_interval_ms / 1000.0))
            except queue.Full:
                pass
            stop_event.wait(heartbeat_interval_ms / 1000.0)

    def consume_control() -> None:
        while not stop_event.is_set():
            try:
                encoded = control_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                control = WorkerControl.decode(encoded)
            except Exception:
                continue
            if control.action == WorkerControlAction.SHUTDOWN:
                with state_lock:
                    worker_state[0] = "STOPPING"
                stop_event.set()
                return
            assert control.request_key is not None
            with cancel_lock:
                canceled.add(_key(control.request_key))

    heartbeat_thread = threading.Thread(target=emit_heartbeats, name="sat-ai-heartbeat", daemon=True)
    control_thread = threading.Thread(target=consume_control, name="sat-ai-control", daemon=True)
    heartbeat_thread.start()
    control_thread.start()
    runtime: SingletonModelRuntime | None = None
    try:
        profile_path = (
            root_path / "sat_ai" / "deployment_profile.yaml"
            if deployment_profile_path is None
            else Path(deployment_profile_path)
        )
        if not profile_path.is_absolute():
            profile_path = root_path / profile_path
        deployment_profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
        model_task = str(deployment_profile.get("model_task", "patch_classification"))
        manifest_path = root_path / str(
            deployment_profile.get(
                "model_manifest_path",
                "sat_ai/model_manifest.yaml",
            )
        )
        checkpoint_path = root_path / str(
            deployment_profile.get("checkpoint_path", "checkpoints/best_model.pth")
        )
        if device == "cpu":
            configure_cpu_runtime(int(deployment_profile["cpu_threads"]))
        manifest = load_model_manifest(
            manifest_path,
            checkpoint_path,
        )
        if manifest.model_task != model_task:
            raise ValueError("deployment profile model_task does not match model manifest")
        lut = ThresholdLUT.from_file(
            root_path / "protocol" / "golden_vectors" / "threshold_lut.bin",
            manifest.threshold_lut_sha256,
        )
        runtime = SingletonModelRuntime(
            manifest,
            str(checkpoint_path),
            device=device,
        )
        with state_lock:
            worker_state[0] = "READY"
        while not stop_event.is_set():
            try:
                encoded = request_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                request = WorkerRequest.decode(encoded)
            except Exception:
                continue
            with state_lock:
                active[0] = request.request_key
                worker_state[0] = "RUNNING"
            result = _execute_request(request, runtime, lut, canceled, cancel_lock)
            result_queue.put(result.encode())
            with state_lock:
                active[0] = None
                worker_state[0] = "READY"
    finally:
        with state_lock:
            worker_state[0] = "STOPPING"
        stop_event.set()
        if runtime is not None:
            runtime.close()
        heartbeat_thread.join(timeout=1)
        control_thread.join(timeout=1)
