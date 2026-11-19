"""The frozen results under ``results/<chain>/`` and the tool that guards them.

Each chain directory holds the result files the paper is built from plus one
``MANIFEST.json``::

    chain        the chain name (= the directory name)
    command      the command that produces the results, run from the framework root
    config       {config file: sha256 of its canonical content}
    params       (optional) protocol constants recorded with the results (annotation_ego4d)
    inputs       [input file, ...] -- the files under data/raw/ the chain reads
    results      {result file: sha256} -- exactly the files in the directory

Commands (exit code 0 = pass, 1 = fail; the last line is ``RESULT: ...``)::

    python -m tal_calibration.results compare CHAIN RUN_DIR [--tolerance FILE[:COLUMN]=ABS ...]
    python -m tal_calibration.results check [CHAIN ...]
    python -m tal_calibration.results manifest CHAIN
    python -m tal_calibration.results reproduce CHAIN --work DIR

``compare`` decides whether a run reproduces the frozen results. A file whose
bytes are identical passes at once. Otherwise CSVs are compared cell by cell
in file order (header and row count must match; numeric cells are parsed with
Python's correctly rounded ``float``, never with a lossy parser), ``.npz``
array by array (names, dtype, shape, bytes) and JSON leaf by leaf. A file
passes when no text cell differs and every numeric |delta| is 0, or within a
tolerance the caller declared explicitly with ``--tolerance``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .paths import FRAMEWORK_ROOT, RESULTS_ROOT, RUNS_ROOT

MANIFEST = "MANIFEST.json"
REQUIRED_KEYS = ("chain", "command", "config", "inputs", "results")
OPTIONAL_KEYS = ("params",)
_HEX64 = re.compile(r"[0-9a-f]{64}")
_NUMBER = re.compile(
    r"[+-]?((\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?|nan|inf|infinity)", re.IGNORECASE)
_SHOW = 10  # differing cells / leaves listed per file


# --------------------------------------------------------------------------
# hashing
# --------------------------------------------------------------------------
def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def config_sha256(path) -> str:
    """sha256 of a YAML file's parsed content (comments and layout excluded)."""
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        content = yaml.safe_load(f)
    text = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _rel(path) -> str:
    return Path(path).resolve().relative_to(FRAMEWORK_ROOT.resolve()).as_posix()


# --------------------------------------------------------------------------
# chain registry
# --------------------------------------------------------------------------
QU_CONFIGS = "src/tal_calibration/protocols/quality_uncertainty/configs"
DATASETS_YAML = "configs/datasets.yaml"


def _benchmark_inputs():
    from .datasets import load_benchmark_pairs
    from .protocols.benchmark import benchmark_tasks
    pairs = [p for p in load_benchmark_pairs()
             if (p["dataset"], p["model"]) in benchmark_tasks.BENCHMARK_PAIRS]
    return {_rel(p[k]) for p in pairs for k in ("pred", "anno")}


def _fragmentation_inputs():
    from .datasets import load_benchmark_pairs
    from .protocols.benchmark import protocol
    pairs = [p for p in load_benchmark_pairs()
             if (p["dataset"], p["model"]) in protocol.TABLE1_PAIRS]
    return {_rel(p[k]) for p in pairs for k in ("pred", "anno")}


def _benchmark_quality_inputs():
    from .datasets import load_benchmark_pairs
    from .protocols.benchmark import table1_quality_estimation as T1
    pairs = [p for p in load_benchmark_pairs()
             if (p["dataset"], p["model"]) in T1.TABLE1_PAIRS]
    return {_rel(p[k]) for p in pairs for k in ("pred", "anno")}


def _annotation_inputs():
    from .protocols.annotation import lib as L
    return {_rel(L.PRED_PATH), _rel(L.ANNO_PATH)}


def _annotation_ego4d_inputs():
    from .datasets import load_annotator_sources, load_datasets
    from .protocols.annotation import ego4d as E
    sources = load_annotator_sources()[E.DATASET]
    return ({_rel(p) for p in sources.values()}
            | {_rel(load_datasets([E.DATASET])[E.DATASET]["pred"])})


def _annotation_ego4d_params():
    from .protocols.annotation import ego4d as E
    from .protocols.annotation import lib as L
    return {"audit_tau": E.AUDIT_TAU, "audit_taus": list(E.AUDIT_TAUS),
            "lrp_anchor_tau": E.LRP_ANCHOR_TAU, "n_worlds": E.N_WORLDS,
            "world_seeds": list(E.WORLD_SEEDS), "n_boot": L.N_BOOT}


def _quality_uncertainty_inputs(config):
    def inputs():
        import yaml
        from .datasets import load_datasets
        with open(FRAMEWORK_ROOT / config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        names = sorted({d for exp in cfg["experiments"].values()
                        for d in exp["datasets"]})
        out = set()
        for entry in load_datasets(names).values():
            out |= {_rel(entry[k]) for k in ("anno", "pred", "pre_nms") if k in entry}
        return out
    return inputs


@dataclass(frozen=True)
class Chain:
    #: entry script and arguments; ``{out}`` is the run's output root
    argv: tuple
    #: "flat": ``{out}`` is the run directory; "subdir": the run writes
    #: ``{out}/<run_id>/`` and the newest such directory is the run
    layout: str
    config: tuple
    inputs: Callable[[], set]
    params: Callable[[], dict] | None = None

    def command(self, out: str) -> list:
        return ["python"] + [a.replace("{out}", out) for a in self.argv]


CHAINS = {
    "benchmark": Chain(
        ("experiments/benchmark/run_benchmark.py", "--out-dir", "{out}"),
        "flat", (DATASETS_YAML,), _benchmark_inputs),
    "fragmentation": Chain(
        ("experiments/fragmentation/run_fragmentation.py", "--out-dir", "{out}"),
        "flat", (DATASETS_YAML,), _fragmentation_inputs),
    "benchmark_quality": Chain(
        ("experiments/benchmark/run_benchmark_quality.py", "--runs-root", "{out}"),
        "subdir", (DATASETS_YAML,), _benchmark_quality_inputs),
    "annotation": Chain(
        ("experiments/annotation/run_annotation.py", "--out-dir", "{out}"),
        "flat", (DATASETS_YAML,), _annotation_inputs),
    "annotation_ego4d": Chain(
        ("experiments/annotation_ego4d/run_annotation_ego4d.py", "--out-dir", "{out}"),
        "flat", (DATASETS_YAML,), _annotation_ego4d_inputs,
        params=_annotation_ego4d_params),
    "quality_uncertainty": Chain(
        ("experiments/quality_uncertainty/run_quality_uncertainty.py",
         "--config", f"{QU_CONFIGS}/formal_all_datasets.yaml", "--runs-root", "{out}"),
        "subdir", (f"{QU_CONFIGS}/formal_all_datasets.yaml", DATASETS_YAML),
        _quality_uncertainty_inputs(f"{QU_CONFIGS}/formal_all_datasets.yaml")),
    "quality_uncertainty_multidetector": Chain(
        ("experiments/quality_uncertainty/run_quality_uncertainty.py",
         "--config", f"{QU_CONFIGS}/formal_multi_detector.yaml", "--runs-root", "{out}"),
        "subdir", (f"{QU_CONFIGS}/formal_multi_detector.yaml", DATASETS_YAML),
        _quality_uncertainty_inputs(f"{QU_CONFIGS}/formal_multi_detector.yaml")),
}


def manifest_command(name: str) -> str:
    out = (RUNS_ROOT / name).relative_to(FRAMEWORK_ROOT).as_posix()
    return shlex.join(CHAINS[name].command(out))


def _result_files(chain_dir: Path) -> list:
    return sorted(p.relative_to(chain_dir).as_posix()
                  for p in chain_dir.rglob("*") if p.is_file() and p.name != MANIFEST)


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------
def build_manifest(name: str, results_root: Path = RESULTS_ROOT) -> dict:
    chain = CHAINS[name]
    chain_dir = results_root / name
    man = {"chain": name, "command": manifest_command(name),
           "config": {p: config_sha256(FRAMEWORK_ROOT / p) for p in chain.config}}
    if chain.params is not None:
        man["params"] = chain.params()
    man["inputs"] = sorted(chain.inputs())
    man["results"] = {rel: sha256_file(chain_dir / rel) for rel in _result_files(chain_dir)}
    if not man["results"]:
        raise SystemExit(f"{name}: no result files under {chain_dir}")
    return man


def write_manifest(name: str, results_root: Path = RESULTS_ROOT) -> Path:
    man = build_manifest(name, results_root)
    path = results_root / name / MANIFEST
    path.write_text(json.dumps(man, indent=1) + "\n", encoding="utf-8")
    return path


def load_manifest(name: str, results_root: Path = RESULTS_ROOT) -> dict:
    with open(results_root / name / MANIFEST, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------
def check_chain(name: str, results_root: Path = RESULTS_ROOT, log=print) -> bool:
    """Every check on one chain directory; logs one line per problem."""
    problems = []
    chain = CHAINS.get(name)
    chain_dir = results_root / name
    if chain is None:
        problems.append("not a registered chain")
    elif not (chain_dir / MANIFEST).is_file():
        problems.append(f"{MANIFEST} missing")
    if problems:
        for p in problems:
            log(f"  FAIL {name}: {p}")
        return False
    man = load_manifest(name, results_root)

    keys = set(man)
    missing = [k for k in REQUIRED_KEYS if k not in man]
    extra = sorted(keys - set(REQUIRED_KEYS) - set(OPTIONAL_KEYS))
    if missing or extra:
        problems.append(f"manifest keys: missing {missing}, unexpected {extra}")
    if man.get("chain") != name:
        problems.append(f"chain {man.get('chain')!r} != directory {name!r}")
    if man.get("command") != manifest_command(name):
        problems.append(f"command {man.get('command')!r} != registered "
                        f"{manifest_command(name)!r}")

    config = man.get("config", {})
    if list(config) != list(chain.config):
        problems.append(f"config files {list(config)} != registered {list(chain.config)}")
    for rel, sha in config.items():
        if not _HEX64.fullmatch(str(sha)):
            problems.append(f"config {rel}: {sha!r} is not a sha256")
        path = FRAMEWORK_ROOT / rel
        now = config_sha256(path) if path.is_file() else None
        if now != sha:
            problems.append(f"config {rel}: content sha {now} != manifest {sha}")

    want_params = chain.params() if chain.params is not None else None
    if man.get("params") != want_params:
        problems.append(f"params {man.get('params')} != code {want_params}")

    inputs = man.get("inputs", [])
    registered = sorted(chain.inputs())
    if inputs != registered:
        problems.append(f"inputs: missing {sorted(set(registered) - set(inputs))}, "
                        f"unregistered {sorted(set(inputs) - set(registered))}")

    results = man.get("results", {})
    on_disk = _result_files(chain_dir)
    if sorted(results) != on_disk:
        problems.append(f"result files: missing {sorted(set(results) - set(on_disk))}, "
                        f"not in manifest {sorted(set(on_disk) - set(results))}")
    for rel, sha in results.items():
        if not _HEX64.fullmatch(str(sha)):
            problems.append(f"result {rel}: {sha!r} is not a sha256")
        elif (chain_dir / rel).is_file() and sha256_file(chain_dir / rel) != sha:
            problems.append(f"result {rel}: file sha256 != manifest")

    for p in problems:
        log(f"  FAIL {name}: {p}")
    if not problems:
        log(f"  ok   {name}: {len(results)} result files, {len(inputs)} inputs, "
            f"{len(config)} config")
    return not problems


def check(names=None, results_root: Path = RESULTS_ROOT, log=print) -> bool:
    """With no names: every registered chain and every directory under
    ``results_root`` (an unregistered directory fails)."""
    on_disk = {p.name for p in results_root.iterdir() if p.is_dir()}
    names = list(names) if names else sorted(on_disk | set(CHAINS))
    ok = True
    for name in names:
        ok = check_chain(name, results_root, log) and ok
    log(f"RESULT: {'PASS' if ok else 'FAIL'} check {' '.join(names)}")
    return ok


# --------------------------------------------------------------------------
# compare
# --------------------------------------------------------------------------
def parse_tolerances(specs) -> dict:
    """``FILE[:COLUMN]=ABS`` -> {(file, column | None): abs}. COLUMN is a CSV
    column, an npz array name or a JSON leaf path."""
    out = {}
    for spec in specs or ():
        target, sep, value = spec.rpartition("=")
        if not sep or not target:
            raise SystemExit(f"--tolerance {spec!r}: expected FILE[:COLUMN]=ABS")
        tol = float(value)
        if not (tol >= 0 and math.isfinite(tol)):
            raise SystemExit(f"--tolerance {spec!r}: ABS must be finite and >= 0")
        file, _, column = target.partition(":")
        out[(file, column or None)] = tol
    return out


class _Tol:
    def __init__(self, tolerances: dict, file: str):
        self.by_col = {c: t for (f, c), t in tolerances.items() if f == file}

    def __call__(self, column) -> float:
        return self.by_col.get(column, self.by_col.get(None, 0.0))

    def unknown(self, columns) -> list:
        return sorted(c for c in self.by_col if c is not None and c not in columns)


def _delta(a: str, b: str):
    """|a - b| for two numeric texts, or None if either is not a number."""
    if not (_NUMBER.fullmatch(a) and _NUMBER.fullmatch(b)):
        return None
    x, y = float(a), float(b)
    if x == y or (math.isnan(x) and math.isnan(y)):
        return 0.0
    if math.isnan(x) or math.isnan(y):
        return math.inf
    return abs(x - y)


@dataclass
class FileResult:
    rel: str
    ok: bool
    summary: str
    details: list = field(default_factory=list)
    cells: int = 0
    max_delta: float = 0.0


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    return (rows[0], rows[1:]) if rows else ([], [])


def compare_csv(rel, ref_path, run_path, tol: _Tol) -> FileResult:
    ref_head, ref_rows = _read_csv(ref_path)
    run_head, run_rows = _read_csv(run_path)
    unknown = tol.unknown(ref_head)
    if unknown:
        return FileResult(rel, False, f"tolerance names unknown column(s) {unknown}")
    if ref_head != run_head:
        if set(ref_head) == set(run_head):
            msg = "same columns, different order"
        else:
            msg = (f"columns differ: results-only {sorted(set(ref_head) - set(run_head))}, "
                   f"run-only {sorted(set(run_head) - set(ref_head))}")
        return FileResult(rel, False, msg)
    if len(ref_rows) != len(run_rows):
        return FileResult(rel, False, f"row count {len(run_rows)} != {len(ref_rows)}")

    cells = numeric = text_bad = fmt_only = 0
    max_delta = 0.0
    bad = []
    for i, (r, c) in enumerate(zip(ref_rows, run_rows), 1):
        if len(r) != len(c):
            text_bad += 1
            bad.append(f"row {i}: {len(c)} fields != {len(r)}")
            continue
        for col, a, b in zip(ref_head, r, c):
            cells += 1
            if a == b:
                if _NUMBER.fullmatch(a):
                    numeric += 1
                continue
            d = _delta(a, b)
            if d is None:
                text_bad += 1
                bad.append(f"row {i} {col}: {a!r} -> {b!r}")
                continue
            numeric += 1
            if d == 0.0:
                fmt_only += 1
                continue
            max_delta = max(max_delta, d)
            if d > tol(col):
                bad.append(f"row {i} {col}: {a} -> {b}  |delta| = {d:.3e}")
    ok = not bad
    summary = (f"{len(ref_rows)} rows x {len(ref_head)} cols, {numeric} numeric cells, "
               f"max|delta| = {max_delta:.3e}")
    if text_bad:
        summary += f", {text_bad} text cells differ"
    if fmt_only:
        summary += f", {fmt_only} cells differ in format only"
    if ok and max_delta > 0:
        summary += " (within declared tolerance)"
    details = bad[:_SHOW] + ([f"... {len(bad) - _SHOW} more"] if len(bad) > _SHOW else [])
    if not ok and sorted(map(tuple, ref_rows)) == sorted(map(tuple, run_rows)):
        details = ["same row set, only the row order differs"]
    return FileResult(rel, ok, summary, details, cells, max_delta)


def compare_npz(rel, ref_path, run_path, tol: _Tol) -> FileResult:
    import numpy as np
    with np.load(ref_path, allow_pickle=False) as a, np.load(run_path, allow_pickle=False) as b:
        names_a, names_b = set(a.files), set(b.files)
        unknown = tol.unknown(names_a)
        if unknown:
            return FileResult(rel, False, f"tolerance names unknown array(s) {unknown}")
        if names_a != names_b:
            return FileResult(rel, False,
                              f"arrays differ: results-only {sorted(names_a - names_b)}, "
                              f"run-only {sorted(names_b - names_a)}")
        bad, cells, max_delta = [], 0, 0.0
        for name in sorted(names_a):
            x, y = a[name], b[name]
            cells += x.size
            if x.dtype != y.dtype or x.shape != y.shape:
                bad.append(f"{name}: {y.dtype}{y.shape} != {x.dtype}{x.shape}")
                continue
            if x.tobytes() == y.tobytes():
                continue
            if x.dtype.kind in "fiub":
                xf, yf = x.astype(np.float64), y.astype(np.float64)
                both_nan = np.isnan(xf) & np.isnan(yf)
                d = np.where(both_nan, 0.0, np.abs(xf - yf))
                d = np.where(np.isnan(d), np.inf, d)
                m = float(d.max()) if d.size else 0.0
                max_delta = max(max_delta, m)
                if m > tol(name):
                    bad.append(f"{name}: max|delta| = {m:.3e}")
            else:
                bad.append(f"{name}: contents differ")
    summary = (f"{len(names_a)} arrays, {cells} elements, max|delta| = {max_delta:.3e}"
               + (" (within declared tolerance)" if not bad and max_delta > 0 else ""))
    return FileResult(rel, not bad, summary, bad[:_SHOW], cells, max_delta)


def _json_leaves(obj, path="$"):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _json_leaves(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _json_leaves(v, f"{path}[{i}]")
    else:
        yield path, obj


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def compare_json(rel, ref_path, run_path, tol: _Tol) -> FileResult:
    with open(ref_path, "r", encoding="utf-8") as f:
        ref = json.load(f)
    with open(run_path, "r", encoding="utf-8") as f:
        run = json.load(f)
    ref_leaves = dict(_json_leaves(ref))
    run_leaves = dict(_json_leaves(run))
    unknown = tol.unknown(ref_leaves)
    if unknown:
        return FileResult(rel, False, f"tolerance names unknown leaf path(s) {unknown}")
    bad, max_delta = [], 0.0
    for path in sorted(set(ref_leaves) | set(run_leaves)):
        if path not in run_leaves:
            bad.append(f"{path}: missing in run")
            continue
        if path not in ref_leaves:
            bad.append(f"{path}: not in results")
            continue
        a, b = ref_leaves[path], run_leaves[path]
        if _is_number(a) and _is_number(b):
            if isinstance(a, int) and isinstance(b, int):
                d = float(abs(a - b))
            else:
                d = _delta(repr(float(a)), repr(float(b)))
            max_delta = max(max_delta, d)
            if d > tol(path):
                bad.append(f"{path}: {a} -> {b}  |delta| = {d:.3e}")
        elif a != b or type(a) is not type(b):
            bad.append(f"{path}: {a!r} -> {b!r}")
    summary = (f"{len(ref_leaves)} leaves, max|delta| = {max_delta:.3e}"
               + (" (within declared tolerance)" if not bad and max_delta > 0 else ""))
    details = bad[:_SHOW] + ([f"... {len(bad) - _SHOW} more"] if len(bad) > _SHOW else [])
    return FileResult(rel, not bad, summary, details, len(ref_leaves), max_delta)


_COMPARERS = {".csv": compare_csv, ".npz": compare_npz, ".json": compare_json}


def compare_file(rel, ref_path: Path, run_path: Path, tolerances: dict) -> FileResult:
    tol = _Tol(tolerances, rel)
    fast = sha256_file(ref_path) == sha256_file(run_path)
    comparer = _COMPARERS.get(ref_path.suffix)
    if fast and not tol.by_col:
        return FileResult(rel, True, "bytes identical")
    if comparer is None:
        return FileResult(rel, fast, "bytes identical" if fast else "bytes differ")
    res = comparer(rel, ref_path, run_path, tol)
    if fast and res.ok:
        res.summary = "bytes identical"
    return res


def compare(name: str, run_dir, results_root: Path = RESULTS_ROOT,
            tolerances: dict | None = None, log=print) -> bool:
    run_dir = Path(run_dir)
    tolerances = tolerances or {}
    man = load_manifest(name, results_root)
    expected = sorted(man["results"])
    log(f"compare {name}")
    log(f"  results {results_root / name}")
    log(f"  run     {run_dir}")
    stray = sorted({f for f, _ in tolerances} - set(expected))
    if stray:
        log(f"  FAIL tolerance names file(s) not in the results: {stray}")
        log(f"RESULT: FAIL {name}")
        return False
    for (f, c), t in sorted(tolerances.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")):
        log(f"  tolerance (declared): {f}{':' + c if c else ''} |delta| <= {t:g}")
    if not run_dir.is_dir():
        log(f"  FAIL run directory {run_dir} does not exist")
        log(f"RESULT: FAIL {name}")
        return False

    results = []
    for rel in expected:
        run_path = run_dir / rel
        if not run_path.is_file():
            results.append(FileResult(rel, False, "missing in the run"))
            continue
        results.append(compare_file(rel, results_root / name / rel, run_path, tolerances))

    width = max(len(r.rel) for r in results)
    for r in results:
        log(f"  {'ok  ' if r.ok else 'FAIL'} {r.rel:<{width}}  {r.summary}")
        for d in r.details:
            log(f"         {d}")
    extra = sorted(p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*")
                   if p.is_file() and p.relative_to(run_dir).as_posix() not in expected)
    if extra:
        log(f"  note: {len(extra)} run file(s) not among the results (ignored): "
            + ", ".join(extra[:8]) + (" ..." if len(extra) > 8 else ""))
    n_bytes = sum(r.summary == "bytes identical" for r in results)
    overall = max((r.max_delta for r in results), default=0.0)
    ok = all(r.ok for r in results)
    log(f"  files: {len(results)} ({n_bytes} bytes identical); cells compared: "
        f"{sum(r.cells for r in results)}; OVERALL max|delta| = {overall:.6e}")
    log(f"RESULT: {'PASS' if ok else 'FAIL'} {name}")
    return ok


# --------------------------------------------------------------------------
# reproduce
# --------------------------------------------------------------------------
def run_dir_of(name: str, out: Path) -> Path:
    if CHAINS[name].layout == "flat":
        return out
    runs = sorted(p for p in out.iterdir()
                  if p.is_dir() and not p.name.startswith(".") and p.name != "failed")
    if not runs:
        raise SystemExit(f"{name}: no completed run under {out}")
    return runs[-1]


def reproduce(name: str, work, results_root: Path = RESULTS_ROOT, log=print) -> bool:
    """Run the chain's registered command into ``work/<chain>`` and compare."""
    out = Path(work).resolve() / name
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    cmd = CHAINS[name].command(str(out))
    log(f"reproduce {name}: {shlex.join(cmd)}")
    proc = subprocess.run([sys.executable, *cmd[1:]], cwd=FRAMEWORK_ROOT)
    if proc.returncode != 0:
        log(f"RESULT: FAIL {name} (run exited {proc.returncode})")
        return False
    return compare(name, run_dir_of(name, out), results_root, log=log)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tal_calibration.results",
                                 description=__doc__.split("\n\n")[0])
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--results-root", type=Path, default=RESULTS_ROOT,
                        help="frozen results root (default: results/)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("compare", parents=[common],
                       help="compare a run directory with results/<chain>/")
    p.add_argument("chain", choices=sorted(CHAINS))
    p.add_argument("run_dir", type=Path)
    p.add_argument("--tolerance", action="append", default=[], metavar="FILE[:COLUMN]=ABS",
                   help="accept |delta| <= ABS in FILE (or in one of its columns); "
                        "default tolerance is 0")
    p = sub.add_parser("check", parents=[common],
                       help="check results/<chain>/ against its MANIFEST.json")
    p.add_argument("chains", nargs="*", metavar="CHAIN")
    p = sub.add_parser("manifest", parents=[common],
                       help="write results/<chain>/MANIFEST.json")
    p.add_argument("chain", choices=sorted(CHAINS))
    p = sub.add_parser("reproduce", parents=[common],
                       help="run a chain into WORK/<chain> and compare")
    p.add_argument("chain", choices=sorted(CHAINS))
    p.add_argument("--work", type=Path, required=True)
    args = ap.parse_args(argv)
    root = args.results_root.resolve()

    if args.cmd == "compare":
        ok = compare(args.chain, args.run_dir, root, parse_tolerances(args.tolerance))
    elif args.cmd == "check":
        ok = check(args.chains, root)
    elif args.cmd == "manifest":
        print(f"wrote {write_manifest(args.chain, root)}")
        ok = True
    else:
        ok = reproduce(args.chain, args.work, root)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
