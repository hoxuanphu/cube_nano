import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from input_contract import SUPPORTED_BAND_ROLES, sha256_file


def _select_index(label, requested, count):
    if requested is None:
        if count != 1:
            raise ValueError(f"TIFF contains {count} {label} entries; select one explicitly")
        return 0
    if requested < 0 or requested >= count:
        raise ValueError(f"Requested {label} index {requested} is out of range")
    return requested


def _parse_band_order(value):
    roles = tuple(role.strip().lower() for role in value.split(",") if role.strip())
    if len(roles) not in (3, 4):
        raise ValueError("band_order must contain exactly three or four roles")
    if len(set(roles)) != len(roles):
        raise ValueError("band_order must not contain duplicate roles")
    unsupported = set(roles) - set(SUPPORTED_BAND_ROLES)
    if unsupported:
        raise ValueError(f"Unsupported band roles: {sorted(unsupported)}")
    return roles


def build_sidecar_payload(
    tiff_path,
    band_order,
    input_spec_id,
    normalization,
    *,
    series_index=None,
    level_index=None,
):
    import tifffile

    tiff_path = Path(tiff_path)
    if not isinstance(input_spec_id, str) or not input_spec_id:
        raise ValueError("input_spec_id must be a non-empty string")
    if not isinstance(normalization, str) or not normalization:
        raise ValueError("normalization must be a non-empty ID")
    roles = _parse_band_order(band_order) if isinstance(band_order, str) else tuple(band_order)
    with tifffile.TiffFile(tiff_path) as tiff:
        selected_series_index = _select_index("series", series_index, len(tiff.series))
        base_series = tiff.series[selected_series_index]
        levels = tuple(base_series.levels)
        selected_level_index = _select_index("pyramid level", level_index, len(levels))
        series = levels[selected_level_index]
        if len(series.pages) != 1:
            raise ValueError("Selected TIFF series/level spans multiple pages")

        axes = str(series.axes)
        shape = tuple(int(value) for value in series.shape)
        channel_positions = [index for index, axis in enumerate(axes) if axis in {"C", "S"}]
        if len(channel_positions) != 1:
            raise ValueError(f"TIFF axes must contain exactly one channel/sample axis, got {axes!r}")
        channels = shape[channel_positions[0]]
        if channels != len(roles):
            raise ValueError(
                f"band_order contains {len(roles)} roles but TIFF contains {channels} channels"
            )
        if axes.count("Y") != 1 or axes.count("X") != 1:
            raise ValueError(f"TIFF axes must contain exactly one Y and X axis, got {axes!r}")
        used_positions = {axes.index("Y"), axes.index("X"), channel_positions[0]}
        if any(size != 1 for index, size in enumerate(shape) if index not in used_positions):
            raise ValueError(f"TIFF contains unsupported non-singleton axes: {axes!r} {shape}")

        page = series.pages[0]
        if int(page.samplesperpixel) != channels:
            raise ValueError("SamplesPerPixel does not match the TIFF channel axis")
        if int(page.planarconfig) not in {1, 2}:
            raise ValueError(f"Unsupported PlanarConfiguration {page.planarconfig!s}")
        if int(page.photometric) not in {1, 2}:
            raise ValueError(f"Unsupported PhotometricInterpretation {page.photometric!s}")
        if int(page.photometric) == 2 and roles[:3] != ("red", "green", "blue"):
            raise ValueError("band_order conflicts with TIFF RGB photometric semantics")
        extras = tuple(int(value) for value in page.extrasamples)
        if any(value in {1, 2} for value in extras):
            raise ValueError("TIFF alpha samples cannot be packaged as RGBNIR")
        if any(value != 0 for value in extras):
            raise ValueError(f"Unsupported TIFF ExtraSamples values {page.extrasamples!r}")
        if len(roles) == 4 and roles[3] != "nir":
            raise ValueError("The fourth production band must be NIR")
        if set(roles) != set(("red", "green", "blue")) and set(roles) != set(
            ("red", "green", "blue", "nir")
        ):
            raise ValueError("band_order must define RGB or RGBNIR exactly")

        return {
            "schema_version": 1,
            "source_fingerprint": {
                "algorithm": "sha256",
                "digest": sha256_file(tiff_path),
            },
            "axes": axes,
            "shape": list(shape),
            "band_order": list(roles),
            "dtype": str(np.dtype(series.dtype)),
            "input_spec_id": input_spec_id,
            "normalization": normalization,
            "selection": {
                "series": selected_series_index,
                "level": selected_level_index,
            },
        }


def write_sidecar(path, payload, force=False):
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(f"Sidecar already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def build_parser():
    parser = argparse.ArgumentParser(description="Create a validated TIFF input sidecar")
    parser.add_argument("--tiff", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--band_order", required=True)
    parser.add_argument("--input_spec_id", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--series", type=int)
    parser.add_argument("--level", type=int)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    payload = build_sidecar_payload(
        args.tiff,
        args.band_order,
        args.input_spec_id,
        args.normalization,
        series_index=args.series,
        level_index=args.level,
    )
    write_sidecar(args.output, payload, force=args.force)


if __name__ == "__main__":
    main()
