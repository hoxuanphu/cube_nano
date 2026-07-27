"""Create a small analytic scene for the CPU container profile."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tifffile

from protocol.canonical import canonical_json


def create(root: Path) -> None:
    scene_dir = root / "data" / "satellite" / "scenes"
    scene_dir.mkdir(parents=True, exist_ok=True)
    source = scene_dir / "fixture.tif"
    sidecar = scene_dir / "fixture.sidecar.json"
    image = np.zeros((512, 512, 3), dtype=np.uint16)
    image[64:320, 80:336] = 32768
    tifffile.imwrite(source, image, metadata={"axes": "YXC"}, compression=None)
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    sidecar_value = {
        "schema_version": 1,
        "source_fingerprint": {"algorithm": "sha256", "digest": source_sha},
        "axes": "YXC",
        "shape": [512, 512, 3],
        "band_order": ["red", "green", "blue"],
        "dtype": "uint16",
        "input_spec_id": "rgb-uint16-linear-v1",
        "validity": {"kind": "all_valid"},
    }
    sidecar.write_bytes(canonical_json(sidecar_value) + b"\n")
    scene = {
        "scene_ref": {"catalog_epoch": 1, "scene_id": 1, "scene_revision": 1},
        "path": "fixture.tif",
        "sidecar_path": "fixture.sidecar.json",
        "source_sha256": source_sha,
        "sidecar_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
        "shape": [512, 512, 3],
        "capability": "VERIFIED",
        "domain": {"profile": "cpu-fixture"},
    }
    unsigned = {"catalog_epoch": 1, "catalog_revision": 1, "scenes": [scene]}
    catalog = {
        "schema_version": 1,
        **unsigned,
        "snapshot_sha256": hashlib.sha256(canonical_json(unsigned)).hexdigest(),
    }
    (scene_dir / "catalog.json").write_bytes(canonical_json(catalog) + b"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    create(parser.parse_args().root.resolve())
