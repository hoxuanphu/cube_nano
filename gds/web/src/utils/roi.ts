import type { ROI } from "../types";

export interface SceneBounds {
  width: number;
  height: number;
  minPatchSize: number;
}

export interface RoiValidation {
  valid: boolean;
  errors: Partial<Record<keyof ROI, string>> & { scene?: string };
}

function integer(value: number): boolean {
  return Number.isInteger(value) && Number.isFinite(value);
}

export function validateRoi(roi: ROI, bounds: SceneBounds): RoiValidation {
  const errors: RoiValidation["errors"] = {};
  if (!integer(roi.x) || roi.x < 0) errors.x = "x must be a non-negative integer";
  if (!integer(roi.y) || roi.y < 0) errors.y = "y must be a non-negative integer";
  if (!integer(roi.width) || roi.width < bounds.minPatchSize) {
    errors.width = `minimum ${bounds.minPatchSize}px`;
  }
  if (!integer(roi.height) || roi.height < bounds.minPatchSize) {
    errors.height = `minimum ${bounds.minPatchSize}px`;
  }
  if (integer(roi.x) && integer(roi.width) && roi.x + roi.width > bounds.width) {
    errors.scene = "ROI exceeds scene width";
  }
  if (integer(roi.y) && integer(roi.height) && roi.y + roi.height > bounds.height) {
    errors.scene = "ROI exceeds scene height";
  }
  return { valid: Object.keys(errors).length === 0, errors };
}

export function clampRoi(roi: ROI, bounds: SceneBounds): ROI {
  const width = Math.max(bounds.minPatchSize, Math.min(bounds.width, Math.round(roi.width)));
  const height = Math.max(bounds.minPatchSize, Math.min(bounds.height, Math.round(roi.height)));
  const x = Math.max(0, Math.min(bounds.width - width, Math.round(roi.x)));
  const y = Math.max(0, Math.min(bounds.height - height, Math.round(roi.y)));
  return { x, y, width, height };
}

export function roiFromDrag(
  start: { x: number; y: number },
  end: { x: number; y: number },
  bounds: SceneBounds,
): ROI {
  const x0 = Math.min(start.x, end.x);
  const y0 = Math.min(start.y, end.y);
  const x1 = Math.max(start.x, end.x);
  const y1 = Math.max(start.y, end.y);
  const raw = {
    x: Math.floor(x0),
    y: Math.floor(y0),
    width: Math.ceil(x1) - Math.floor(x0),
    height: Math.ceil(y1) - Math.floor(y0),
  };
  return clampRoi(raw, bounds);
}

export function normalizedRoi(roi: ROI, bounds: SceneBounds): { x: number; y: number; width: number; height: number } {
  return {
    x: roi.x / bounds.width,
    y: roi.y / bounds.height,
    width: roi.width / bounds.width,
    height: roi.height / bounds.height,
  };
}

export function roiFromNormalized(
  normalized: { x: number; y: number; width: number; height: number },
  bounds: SceneBounds,
): ROI {
  return clampRoi(
    {
      x: Math.floor(normalized.x * bounds.width),
      y: Math.floor(normalized.y * bounds.height),
      width: Math.ceil(normalized.width * bounds.width),
      height: Math.ceil(normalized.height * bounds.height),
    },
    bounds,
  );
}
