"""``python -m tal_calibration.data verify``: check that every file listed in
``data/manifest.json`` is in place.
``build-ego4d-json``: rebuild the Ego4D-MQ ``ego4d.json`` from the official
moments files (see ``ego4d_annotations``)."""
from __future__ import annotations

import argparse
import json
import sys

from .paths import DATA_ROOT, resolve

MANIFEST = DATA_ROOT / "manifest.json"


def _load_manifest() -> dict:
    if not MANIFEST.is_file():
        raise SystemExit(f"{MANIFEST} not found")
    return json.load(open(MANIFEST))


def verify() -> int:
    inputs = _load_manifest()["inputs"]
    missing = [rel for rel in sorted(inputs) if not resolve(rel).is_file()]
    print(f"data verify against {MANIFEST.name}")
    print(f"  present : {len(inputs) - len(missing)}/{len(inputs)}")
    print(f"  missing : {len(missing)}")
    for rel in missing:
        print(f"    MISSING: {rel}")
    if missing:
        print("VERIFY FAILED")
        return 1
    print("VERIFY OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tal_calibration.data")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify", help="check that every manifest file is in data/raw")
    p_ego = sub.add_parser(
        "build-ego4d-json",
        help="rebuild data/raw/annotations/ego4d/ego4d.json from moments_{train,val}.json",
    )
    p_ego.add_argument("--out", default=None,
                       help="output path (default: the configured ego4d.json)")
    args = parser.parse_args(argv)
    if args.command == "verify":
        return verify()
    if args.command == "build-ego4d-json":
        from .ego4d_annotations import write_ego4d_json
        return write_ego4d_json(out=args.out)
    return 2


if __name__ == "__main__":
    sys.exit(main())
