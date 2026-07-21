import { describe, expect, it } from "vitest";
import { clampRoi, normalizedRoi, roiFromDrag, roiFromNormalized, validateRoi } from "./roi";

const bounds = { width: 10980, height: 10980, minPatchSize: 256 };

describe("ROI contract", () => {
  it("uses floor/ceil half-open pixels for a drag", () => {
    expect(roiFromDrag({ x: 10.2, y: 20.8 }, { x: 266.1, y: 281.2 }, bounds)).toEqual({
      x: 10,
      y: 20,
      width: 257,
      height: 262,
    });
  });

  it("clamps dragged windows to the scene and minimum patch", () => {
    expect(roiFromDrag({ x: -30, y: -5 }, { x: 3, y: 12 }, bounds)).toEqual({
      x: 0,
      y: 0,
      width: 256,
      height: 256,
    });
    expect(clampRoi({ x: 10_000, y: 10_000, width: 2_000, height: 2_000 }, bounds)).toEqual({
      x: 8_980,
      y: 8_980,
      width: 2_000,
      height: 2_000,
    });
  });

  it("rejects numeric values outside authoritative bounds", () => {
    expect(validateRoi({ x: 1, y: 2, width: 128, height: 256 }, bounds).valid).toBe(false);
    expect(validateRoi({ x: 10_000, y: 0, width: 2_000, height: 256 }, bounds).errors.scene).toBe("ROI exceeds scene width");
    expect(validateRoi({ x: 10, y: 10, width: 256, height: 256 }, bounds).valid).toBe(true);
  });

  it("round-trips normalized view state without changing authoritative integers", () => {
    const source = { x: 4096, y: 3072, width: 2048, height: 2048 };
    expect(roiFromNormalized(normalizedRoi(source, bounds), bounds)).toEqual(source);
  });
});
