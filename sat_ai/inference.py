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
from .segmentation import postprocess_segmentation_logits
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
            model_task=manifest.model_task,
        )
        self.model.eval()
        self.load_count = 1
        self._closed = False

    def infer_outputs(self, batch: np.ndarray) -> np.ndarray:
        if self._closed:
            raise WorkerLost("model runtime is closed")
        batch = np.asarray(batch, dtype=np.float32)
        expected = (self.manifest.input_spec.channels, self.manifest.input_spec.patch_size, self.manifest.input_spec.patch_size)
        if batch.ndim != 4 or tuple(batch.shape[1:]) != expected:
            raise ValueError(f"expected batch shape (N, {expected[0]}, {expected[1]}, {expected[2]}), got {batch.shape}")
        with torch.inference_mode():
            tensor = torch.from_numpy(np.ascontiguousarray(batch)).to(self.device)
            outputs = self.model(tensor).detach().cpu().numpy()
        if not np.isfinite(outputs).all():
            raise ValueError("model returned non-finite output")
        outputs = outputs.astype(np.float32, copy=False)
        if self.manifest.model_task == "patch_classification":
            if outputs.size != batch.shape[0]:
                raise ValueError(f"classifier returned unexpected output shape {outputs.shape}")
            return outputs.reshape(-1)
        expected_spatial = self.manifest.input_spec.patch_size // self.manifest.output_spec.output_stride
        if outputs.shape != (batch.shape[0], 2, expected_spatial, expected_spatial):
            raise ValueError(f"SegFormer returned unexpected output shape {outputs.shape}")
        return outputs

    def infer_logits(self, batch: np.ndarray) -> np.ndarray:
        """Backward-compatible classifier name; segmentation returns raw logits."""

        return self.infer_outputs(batch)

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


def _reader_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {
        key: after[key] - before[key]
        for key in (
            "source_read_calls",
            "validity_read_calls",
            "logical_source_bytes_read",
            "logical_validity_bytes_read",
            "logical_bytes_read",
        )
    }


def _infer_region_classification(
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
    reader_metrics = _reader_delta(metrics_before, metrics_after)
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
        "model_task": runtime.manifest.model_task,
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


def _infer_region_segmentation(
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
    patch_size = runtime.manifest.input_spec.patch_size
    if batch_size != 1:
        raise ValueError("SegFormer MVP runtime requires batch_size=1")
    windows = list(iter_patch_windows(scene.shape, roi, patch_size))
    if not windows:
        raise ValueError("ROI does not intersect the scene")
    domain_status = runtime.manifest.domain_status(domain)
    if domain_status == "DOMAIN_MISMATCH":
        raise ValueError("DOMAIN_MISMATCH")
    decision = runtime.manifest.decision_spec
    postprocess = runtime.manifest.postprocess_spec
    product = runtime.manifest.product_spec
    output_spec = runtime.manifest.output_spec
    if decision is None or postprocess is None or product is None or output_spec is None:
        raise ValueError("SegFormer manifest is missing its immutable contracts")
    decision.require_config(
        config.config.model_threshold_bp,
        config.config.coverage_limit_bp,
    )
    threshold_bp = decision.pixel_cloud_probability_threshold_bp
    coverage_limit_bp = decision.coverage_limit_bp
    min_valid_ratio_bp = decision.min_valid_pixel_ratio_bp
    emitter = ProgressEmitter(progress_callback)
    started = time.perf_counter()
    total = len(windows)
    emitter.emit(0, total, 0, force=True)
    cloud_mask = np.zeros((roi.height, roi.width), dtype=np.uint8)
    validity_mask = np.zeros((roi.height, roi.width), dtype=np.uint8)
    processed = 0
    for window in windows:
        patch, valid = build_padded_patch(scene, window)
        tensor = np.transpose(runtime.manifest.input_spec.normalize(patch), (2, 0, 1))[None, ...]
        if hasattr(runtime, "infer_outputs"):
            raw_logits = runtime.infer_outputs(tensor)
        else:
            raw_logits = runtime.infer_logits(tensor)
        tile = postprocess_segmentation_logits(
            raw_logits,
            valid[None, ...],
            target_size=(patch_size, patch_size),
            threshold_bp=threshold_bp,
            output_spec=output_spec,
            postprocess_spec=postprocess,
            product_spec=product,
        )
        roi_x0 = max(window.x, roi.x) - roi.x
        roi_y0 = max(window.y, roi.y) - roi.y
        roi_x1 = min(window.x + window.scene_width, roi.x_end) - roi.x
        roi_y1 = min(window.y + window.scene_height, roi.y_end) - roi.y
        tile_x0 = max(window.x, roi.x) - window.x
        tile_y0 = max(window.y, roi.y) - window.y
        tile_x1 = tile_x0 + max(0, roi_x1 - roi_x0)
        tile_y1 = tile_y0 + max(0, roi_y1 - roi_y0)
        cloud_mask[roi_y0:roi_y1, roi_x0:roi_x1] = tile.cloud_mask[0, tile_y0:tile_y1, tile_x0:tile_x1]
        validity_mask[roi_y0:roi_y1, roi_x0:roi_x1] = tile.validity_mask[0, tile_y0:tile_y1, tile_x0:tile_x1]
        processed += 1
        elapsed = int((time.perf_counter() - started) * 1000)
        emitter.emit(processed, total, elapsed)

    valid_pixels = int(np.count_nonzero(validity_mask))
    roi_area = roi.width * roi.height
    if valid_pixels * 10000 < min_valid_ratio_bp * roi_area:
        raise InsufficientValidData("valid pixel ratio is below the segmentation contract")
    cloud_pixels = int(np.count_nonzero((cloud_mask == product.cloud_value) & (validity_mask == product.valid_value)))
    pixel_ratio_bp = coverage_ratio_bp(cloud_pixels, valid_pixels)
    accepted = coverage_accepted(cloud_pixels, valid_pixels, coverage_limit_bp)
    elapsed = int((time.perf_counter() - started) * 1000)
    emitter.emit(total, total, elapsed, force=True)
    metrics_after = scene.metrics.as_dict()
    decision_status = "DOMAIN_UNVERIFIED" if domain_status == "DOMAIN_UNVERIFIED" else runtime.manifest.assurance_level
    return {
        "roi": roi.as_dict(),
        "patch_count": total,
        "analyzed_area": valid_pixels,
        "cloud_positive_area": cloud_pixels,
        "cloud_positive_tile_area_ratio_bp": None,
        "pixel_cloud_ratio_bp": pixel_ratio_bp,
        "valid_pixel_ratio_bp": (valid_pixels * 10000) // roi_area,
        "coverage_limit_bp": coverage_limit_bp,
        "model_threshold_bp": threshold_bp,
        "science_decision": "ACCEPTED" if accepted else "REJECTED",
        "science_status": decision_status,
        "model_task": runtime.manifest.model_task,
        "model_release_id": runtime.manifest.model_release_id,
        "model_sha256": runtime.manifest.checkpoint_sha256,
        "input_spec_id": runtime.manifest.input_spec.input_spec_id,
        "threshold_mapping_id": "probability-bp-v1",
        "threshold_lut_sha256": config.lut.sha256,
        "decision_spec_id": decision.decision_spec_id,
        "postprocess_id": postprocess.postprocess_id,
        "product_spec_id": product.product_spec_id,
        "config_snapshot": config.config.as_dict(),
        "tiling_algorithm_id": config.tiling_algorithm_id,
        "coverage_algorithm_id": "valid-pixel-ratio-v1",
        "validity_algorithm_id": "pixel-validity-mask-v1",
        "padding_algorithm_id": config.padding_algorithm_id,
        "latency_ms": elapsed,
        "reader_metrics": _reader_delta(metrics_before, metrics_after),
        "cloud_mask": cloud_mask,
        "validity_mask": validity_mask,
    }


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
    """Dispatch one immutable task selected by the worker deployment profile."""

    if runtime.manifest.model_task == "semantic_cloud_segmentation":
        return _infer_region_segmentation(
            scene,
            roi,
            runtime,
            config,
            progress_callback=progress_callback,
            batch_size=batch_size,
            domain=domain,
        )
    return _infer_region_classification(
        scene,
        roi,
        runtime,
        config,
        progress_callback=progress_callback,
        batch_size=batch_size,
        domain=domain,
    )
