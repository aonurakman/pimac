"""Run multiple study manifests in sequence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from study import build_dry_run, run_manifest
from utils import OPTUNA_RESULTS_ROOT, write_json


def _load_library(path: str | Path) -> list[str]:
    resolved = Path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        manifests = data.get("manifests", [])
    else:
        manifests = data
    if not isinstance(manifests, list):
        raise ValueError("Sweep library must be a JSON list or an object with a 'manifests' list.")
    return [str(item) for item in manifests]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a sequence of Optuna study manifests.")
    parser.add_argument("--suite-id", type=str, default="full_sweep_01")
    parser.add_argument("--manifest", action="append", default=[], help="One study manifest path. Can be repeated.")
    parser.add_argument("--library", type=str, default=None, help="Optional JSON file listing manifest paths.")
    parser.add_argument("--parallel-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--show-child-output", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest_paths = list(args.manifest)
    if args.library:
        manifest_paths.extend(_load_library(args.library))
    if not manifest_paths:
        raise SystemExit("No manifests were provided.")

    if args.dry_run:
        data = {
            "suite_id": args.suite_id,
            "results_root": str(OPTUNA_RESULTS_ROOT),
            "studies": [
                build_dry_run(manifest_path, suite_id=args.suite_id)
                for manifest_path in manifest_paths
            ],
        }
        print(json.dumps(data, indent=2, sort_keys=True))
        return 0

    results = []
    for manifest_path in manifest_paths:
        results.append(
            run_manifest(
                manifest_path=manifest_path,
                suite_id=args.suite_id,
                parallel_jobs=int(args.parallel_jobs),
                seed=args.seed,
                show_child_output=bool(args.show_child_output),
            )
        )

    summary_path = OPTUNA_RESULTS_ROOT / args.suite_id / "suite_summary.json"
    write_json(summary_path, {"suite_id": args.suite_id, "results": results})
    print(json.dumps({"suite_id": args.suite_id, "summary_path": str(summary_path)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
