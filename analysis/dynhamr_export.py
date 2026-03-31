#!/usr/bin/env python3
import argparse
from pathlib import Path

import FILENAME as CONFIG
from dynhamr_io import export_latest_dynhamr_runs


def _default_logs_root():
    return CONFIG.OUTPUTS_ROOT / "dynhamr" / "logs" / "video-custom"


def _default_output_root():
    return CONFIG.OUTPUTS_ROOT / "dynhamr"


def main():
    parser = argparse.ArgumentParser(description="Normalize DynHAMR runs into repo-style per-frame outputs.")
    parser.add_argument("--logs-root", type=Path, default=_default_logs_root())
    parser.add_argument("--output-root", type=Path, default=_default_output_root())
    parser.add_argument(
        "--clip",
        action="append",
        default=None,
        help="Optional repeatable clip filter. When omitted, export all complete clips.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing normalized clip folders.")
    args = parser.parse_args()

    summaries = export_latest_dynhamr_runs(
        logs_root=args.logs_root,
        output_root=args.output_root,
        clip_ids=args.clip,
        overwrite=args.overwrite,
    )

    if not summaries:
        print(f"No complete DynHAMR runs found under: {args.logs_root}")
        return 1

    print(f"Exported {len(summaries)} DynHAMR clip(s) into: {args.output_root}")
    for summary in summaries:
        print(
            f"  {summary['clip_id']}: "
            f"records={summary['records_exported']} "
            f"phase={summary['phase']} "
            f"run={summary['run_name']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
