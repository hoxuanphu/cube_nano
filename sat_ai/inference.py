"""Mission inference adapter with singleton model ownership and ROI logic."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from protocol.schemas import ConfigSnapshot, ROI

from .manifest import ModelManifest
from .model_runtime import load_model
from .roi import ProgressEmitter, SceneWindow, build_padded_patch, iter_patch_windows
from .threshold_lut import ThresholdLUT, coverage_accepted, coverage_ratio_bp


class InferenceQueueFull(RuntimeError):
    code = "QUEUE_FULL"


class WorkerLost(RuntimeError):
    code = "WORKER_LOST"


def configure_cpu_runtime(cpu_threads: int) -> None:
    if isinstance(cpu_threads, bool) or not isinstance(cpu_threads, int) or cpu_threads <= 0:
        raise ValueError("cpu_threads must be a positive integer")
    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch only allows setting interop threads before parallel work starts.
        pass


@dataclass(frozen=True)
class InferenceConfig:
    config: ConfigSnapshot
    lut: ThresholdLUT
    tiling_algorithm_id: str = "scene-anchored-grid-v1"
    coverage_algorithm_id: str = "tile-area-intersection-v1"
    validity_algorithm_id: str = "strict-full-patch-10000bp-v1"
    padding_algorithm_id: str = "scene-edge-constant-raw-v1"


class SingletonModelRuntime:
    """Load one checkpoint per worker lifetime and expose logits only."""

    def __init__(self, manifest: ModelManifest, checkpoint_path: str, *, device: str = "cpu"):
        manifest.verify_checkpoint(checkpoint_path)
        self.manifest = manifest
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        self.model = load_model(
            checkpoint_path,
            channels=manifest.input_spec.channels,
            device=self.device,
            allow_untrained=False,
        )
        self.model.eval()
        self.load_count = 1
        self._closed = False

    def infer_logits(self, batch: np.ndarray) -> np.ndarray:
        if self._closed:
            raise WorkerLost("model runtime is closed")
        batch = np.asarray(batch, dtype=np.float32)
        expected = (self.manifest.input_spec.channels, self.manifest.input_spec.patch_size, self.manifest.input_spec.patch_size)
        if batch.ndim != 4 or tuple(batch.shape[1:]) != expected:
            raise ValueError(f"expected batch shape (N, {expected[0]}, {expected[1]}, {expected[2]}), got {batch.shape}")
        with torch.inference_mode():
            tensor = torch.from_numpy(np.ascontiguousarray(batch)).to(self.device)
            logits = self.model(tensor).reshape(-1).detach().cpu().numpy()
        if not np.isfinite(logits).all():
            raise ValueError("model returned a non-finite logit")
        return logits.astype(np.float32, copy=False)

    def close(self) -> None:
        self._closed = True
        self.model = None


class BoundedInferenceQueue:
    """Bounded async hook used by the worker; queue overflow is observable."""

    def __init__(self, runtime: SingletonModelRuntime, capacity: int = 8):
        if capacity <= 0:
            raise ValueError("inference queue capacity must be positive")
        self.runtime = runtime
        self.queue: queue.Queue[tuple[np.ndarray, Callable[[np.ndarray], None] | None]] = queue.Queue(maxsize=capacity)
        self.capacity = capacity
        self.accepted = 0
        self.rejected = 0
        self.completed = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="sat-ai-worker", daemon=True)
        self._thread.start()

    def submit(self, batch: np.ndarray, callback: Callable[[np.ndarray], None] | None = None) -> None:
        try:
            self.queue.put_nowait((np.asarray(batch), callback))
        except queue.Full as exc:
            self.rejected += 1
            raise InferenceQueueFull("bounded inference queue is full") from exc
        self.accepted += 1

    def _run(self) -> None:
        while not self._stop.is_set() or not self.queue.empty():
            try:
                batch, callback = self.queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                logits = self.runtime.infer_logits(batch)
                if callback is not None:
                    callback(logits)
                self.completed += 1
            finally:
                self.queue.task_done()

    def join(self, timeout: float | None = None) -> None:
        self.queue.join()
        if timeout is not None:
            self._thread.join(timeout)

    def shutdown(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


class InsufficientValidData(RuntimeError):
    code = "INSUFFICIENT_VALID_DATA"


def infer_region(
    scene: SceneWindow,
    roi: ROI,
    runtime: SingletonModelRuntime,
    config: InferenceConfig,
    *,
    progress_callback: Callable[[int, int, int], None] | None = None,
    batch_size: int = 1,
    domain: dict[str, str] | None = None,
) -> dict[str, Any]:
    metrics_before = scene.metrics.as_dict()
    if roi.width < runtime.manifest.input_spec.patch_size or roi.height < runtime.manifest.input_spec.patch_size:
        raise ValueError("ROI width and height must be at least patch_size")
    windows = list(iter_patch_windows(scene.shape, roi, runtime.manifest.input_spec.patch_size))
    if not windows:
        raise ValueError("ROI does not intersect the scene")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    domain_status = runtime.manifest.domain_status(domain)
    if domain_status == "DOMAIN_MISMATCH":
        raise ValueError("DOMAIN_MISMATCH")
    emitter = ProgressEmitter(progress_callback)
    started = time.perf_counter()
    total = len(windows)
    emitter.emit(0, total, 0, force=True)
    cloud_flags: list[bool] = []
    processed = 0
    for start in range(0, len(windows), batch_size):
        batch_windows = windows[start : start + batch_size]
        patches = []
        for window in batch_windows:
            patch, valid = build_padded_patch(scene, window)
            if not np.all(valid[: window.scene_height, : window.scene_width]):
                raise InsufficientValidData("strict full-patch validity policy failed")
            patches.append(
                np.transpose(runtime.manifest.input_spec.normalize(patch), (2, 0, 1))
            )
        logits = runtime.infer_logits(np.stack(patches))
        cloud_flags.extend(
            config.lut.classify(float(logit), config.config.model_threshold_bp)
            for logit in logits
        )
        processed = min(total, start + len(batch_windows))
        elapsed = int((time.perf_counter() - started) * 1000)
        emitter.emit(processed, total, elapsed)
    cloud_positive_area = sum(window.roi_weight for window, is_cloud in zip(windows, cloud_flags) if is_cloud)
    analyzed_area = sum(window.roi_weight for window in windows)
    ratio_bp = coverage_ratio_bp(cloud_positive_area, analyzed_area)
    accepted = coverage_accepted(cloud_positive_area, analyzed_area, config.config.coverage_limit_bp)
    mask = np.zeros((roi.height, roi.width), dtype=np.uint8)
    for window, is_cloud in zip(windows, cloud_flags):
        if not is_cloud:
            continue
        x0 = max(window.x, roi.x) - roi.x
        y0 = max(window.y, roi.y) - roi.y
        x1 = min(window.x + window.scene_width, roi.x_end) - roi.x
        y1 = min(window.y + window.scene_height, roi.y_end) - roi.y
        mask[y0:y1, x0:x1] = 255
    elapsed = int((time.perf_counter() - started) * 1000)
    emitter.emit(total, total, elapsed, force=True)
    metrics_after = scene.metrics.as_dict()
    reader_metrics = {
        key: metrics_after[key] - metrics_before[key]
        for key in (
            "source_read_calls",
            "validity_read_calls",
            "logical_source_bytes_read",
            "logical_validity_bytes_read",
            "logical_bytes_read",
        )
    }
    return {
        "roi": roi.as_dict(),
        "patch_count": total,
        "analyzed_area": analyzed_area,
        "cloud_positive_area": cloud_positive_area,
        "cloud_positive_tile_area_ratio_bp": ratio_bp,
        "coverage_limit_bp": config.config.coverage_limit_bp,
        "model_threshold_bp": config.config.model_threshold_bp,
        "science_decision": "ACCEPTED" if accepted else "REJECTED",
        "science_status": "DOMAIN_UNVERIFIED" if domain_status == "DOMAIN_UNVERIFIED" else "demo_non_validated",
        "model_release_id": runtime.manifest.model_release_id,
        "model_sha256": runtime.manifest.checkpoint_sha256,
        "input_spec_id": runtime.manifest.input_spec.input_spec_id,
        "threshold_mapping_id": config.lut.lut_id,
        "threshold_lut_sha256": config.lut.sha256,
        "config_snapshot": config.config.as_dict(),
        "tiling_algorithm_id": config.tiling_algorithm_id,
        "coverage_algorithm_id": config.coverage_algorithm_id,
        "validity_algorithm_id": config.validity_algorithm_id,
        "padding_algorithm_id": config.padding_algorithm_id,
        "latency_ms": elapsed,
        "reader_metrics": reader_metrics,
        "cloud_mask": mask,
    }
