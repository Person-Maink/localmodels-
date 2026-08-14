from __future__ import annotations

import argparse
from pathlib import Path

from _path_setup import PROJECT_ROOT  # noqa: F401
import pandas as pd

from dataset_tremor_common import FAMILY_CHOICES, default_custom_output_dir, load_metrics_cache
from dataset_tremor_ground_truth import attach_ground_truth, frequency_accuracy_summary, load_ground_truth


def _parse_args():
    parser = argparse.ArgumentParser(description="Score dataset model caches against annotated motion frequency.")
    parser.add_argument("--cache-root", type=Path, default=None, help="Root containing wilor/ and stride/ cache folders.")
    parser.add_argument("--ground-truth-csv", type=Path, default=None, help="Annotated ground-truth CSV path.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for accuracy CSV and Markdown exports.")
    return parser.parse_args()


def main():
    args = _parse_args()
    cache_root = (args.cache_root or default_custom_output_dir(FAMILY_CHOICES[0]).parent).expanduser().resolve()
    output_dir = (args.output_dir or cache_root).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    truth = load_ground_truth(args.ground_truth_csv)
    summaries = []
    for family in FAMILY_CHOICES:
        frame, _ = load_metrics_cache(cache_root / family)
        merged = attach_ground_truth(frame, truth)
        summary = frequency_accuracy_summary(merged, family)
        summaries.append(summary)
        print(f"{family}: MAE = {summary['mae_hz']:.4f} Hz ({summary['matched_clips']} annotated clips)")

    summary_df = pd.DataFrame(summaries)
    csv_path = output_dir / "ground_truth_frequency_accuracy.csv"
    markdown_path = output_dir / "ground_truth_frequency_accuracy.md"
    summary_df.to_csv(csv_path, index=False)
    columns = list(summary_df.columns)
    markdown_rows = [
        "# Ground-Truth Frequency Accuracy",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in summary_df.itertuples(index=False, name=None):
        markdown_rows.append(
            "| " + " | ".join(f"{value:.4f}" if isinstance(value, float) else str(value) for value in row) + " |"
        )
    markdown_path.write_text("\n".join(markdown_rows) + "\n", encoding="utf-8")
    print(f"Saved accuracy summary: {csv_path}")


if __name__ == "__main__":
    main()
