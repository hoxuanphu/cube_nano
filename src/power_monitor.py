"""Optional power sampling for Linux hardware power sensors."""

from __future__ import annotations

import threading
import time
from pathlib import Path


POWER_UNITS = {"uw": 1_000_000.0, "mw": 1_000.0, "w": 1.0}


def _read_watts(sensor_path: Path, unit: str) -> float:
    try:
        raw_value = float(sensor_path.read_text(encoding="ascii").strip())
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Could not read power sensor: {sensor_path}") from exc
    try:
        divisor = POWER_UNITS[unit]
    except KeyError as exc:
        raise ValueError(f"Unsupported power unit: {unit}") from exc
    return raw_value / divisor


class PowerSampler:
    """Sample one power sensor and integrate power over elapsed time."""

    def __init__(
        self,
        sensor_path: str | Path | None,
        *,
        sample_interval_seconds: float = 0.1,
        unit: str = "uw",
    ) -> None:
        if sample_interval_seconds <= 0:
            raise ValueError("sample_interval_seconds must be positive")
        if unit not in POWER_UNITS:
            raise ValueError(f"unit must be one of: {', '.join(POWER_UNITS)}")
        self.sensor_path = Path(sensor_path) if sensor_path is not None else None
        self.sample_interval_seconds = sample_interval_seconds
        self.unit = unit
        self._samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        if self.sensor_path is not None:
            self._samples.append((time.monotonic(), _read_watts(self.sensor_path, self.unit)))

    def _run(self) -> None:
        while not self._stop.wait(self.sample_interval_seconds):
            self._sample()

    def start(self) -> None:
        if self.sensor_path is None:
            return
        if self._thread is not None:
            raise RuntimeError("PowerSampler has already started")
        self._sample()
        self._thread = threading.Thread(target=self._run, name="power-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, object] | None:
        if self.sensor_path is None:
            return None
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._sample()
        if not self._samples:
            return None

        energy_joules = 0.0
        for (previous_time, previous_power), (current_time, current_power) in zip(
            self._samples,
            self._samples[1:],
        ):
            energy_joules += (current_time - previous_time) * (
                previous_power + current_power
            ) / 2.0
        duration = self._samples[-1][0] - self._samples[0][0]
        average_power = energy_joules / duration if duration > 0 else self._samples[0][1]
        return {
            "sensor": str(self.sensor_path),
            "unit": self.unit,
            "sample_count": len(self._samples),
            "duration_seconds": round(duration, 6),
            "average_power_watts": round(average_power, 6),
            "energy_joules": round(energy_joules, 6),
            "energy_watt_hours": round(energy_joules / 3600.0, 9),
        }
