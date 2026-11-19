"""``python -m tal_calibration info``: print the version and resolved paths."""
from __future__ import annotations

import argparse
import sys

from . import __version__, paths


def _cmd_info(_args: argparse.Namespace) -> int:
    print(f"tal_calibration {__version__}")
    print(f"FRAMEWORK_ROOT : {paths.FRAMEWORK_ROOT}")
    print(f"DATA_ROOT      : {paths.DATA_ROOT}")
    print(f"RAW_DATA_ROOT  : {paths.RAW_DATA_ROOT}")
    print(f"CONFIG_ROOT    : {paths.CONFIG_ROOT}")
    print(f"RESULTS_ROOT   : {paths.RESULTS_ROOT}")
    print(f"RUNS_ROOT      : {paths.RUNS_ROOT}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tal_calibration")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_info = sub.add_parser("info", help="report resolved framework paths")
    p_info.set_defaults(func=_cmd_info)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
