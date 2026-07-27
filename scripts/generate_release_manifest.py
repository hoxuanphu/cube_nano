"""Generate a deterministic release manifest and SPDX SBOM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gds.release_manifest import ReleaseManifestError, build_release_manifest, write_release_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("artifacts/release/release_manifest.json"))
    parser.add_argument("--sbom", type=Path, default=Path("artifacts/release/sbom.spdx.json"))
    parser.add_argument("--target-id", default="local-cpu-pytorch")
    parser.add_argument("--allow-dirty", action="store_true", help="produce evidence only; never an official release")
    args = parser.parse_args()
    try:
        payload, digest = build_release_manifest(
            args.root,
            sbom_path=args.sbom,
            require_clean=not args.allow_dirty,
            target_id=args.target_id,
        )
    except ReleaseManifestError as exc:
        parser.error(str(exc))
    manifest_sha = write_release_manifest(args.output, payload)
    print(json.dumps({"release_id": payload["release_id"], "manifest_sha256": manifest_sha, "payload_sha256": digest, "dirty": payload["source_dirty"]}, sort_keys=True))


if __name__ == "__main__":
    main()
