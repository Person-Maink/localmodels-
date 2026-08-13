"""Ground-truth frequency helpers for the new tremor dataset."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


GROUND_TRUTH_REQUIRED_COLUMNS = (
    "video_filename",
    "frequency_hz",
    "frequency_ci95_lower_hz",
    "frequency_ci95_upper_hz",
)
GROUND_TRUTH_COLUMNS = (
    "ground_truth_hz",
    "ground_truth_ci95_lower_hz",
    "ground_truth_ci95_upper_hz",
)


def default_ground_truth_csv() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "new_dataset" / "ground_truth.csv"


def load_ground_truth(csv_path: Path | str | None = None) -> pd.DataFrame:
    """Load annotations indexed by dataset clip id, rejecting ambiguous input."""
    path = Path(csv_path or default_ground_truth_csv()).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Ground-truth CSV does not exist: {path}")
    frame = pd.read_csv(path)
    missing = [column for column in GROUND_TRUTH_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Ground-truth CSV is missing required column(s): {', '.join(missing)}")

    truth = frame.loc[:, list(GROUND_TRUTH_REQUIRED_COLUMNS)].copy()
    truth["clip_id"] = truth["video_filename"].astype(str).map(lambda value: Path(value).stem)
    if truth["clip_id"].duplicated().any():
        duplicate_ids = sorted(truth.loc[truth["clip_id"].duplicated(keep=False), "clip_id"].unique())
        raise ValueError(f"Ground-truth CSV contains duplicate clip ids: {', '.join(duplicate_ids[:10])}")
    for source, target in (
        ("frequency_hz", "ground_truth_hz"),
        ("frequency_ci95_lower_hz", "ground_truth_ci95_lower_hz"),
        ("frequency_ci95_upper_hz", "ground_truth_ci95_upper_hz"),
    ):
        truth[target] = pd.to_numeric(truth[source], errors="coerce")
    if truth[list(GROUND_TRUTH_COLUMNS)].isna().any().any():
        bad_ids = truth.loc[truth[list(GROUND_TRUTH_COLUMNS)].isna().any(axis=1), "clip_id"].tolist()
        raise ValueError(f"Ground-truth CSV has non-numeric frequency values for: {', '.join(bad_ids[:10])}")
    if (truth["ground_truth_ci95_lower_hz"] > truth["ground_truth_ci95_upper_hz"]).any():
        raise ValueError("Ground-truth CSV contains inverted 95% confidence intervals.")
    return truth[["clip_id", *GROUND_TRUTH_COLUMNS]].sort_values("clip_id").reset_index(drop=True)


def attach_ground_truth(metrics_df: pd.DataFrame, truth_df: pd.DataFrame, require_all_truth: bool = True) -> pd.DataFrame:
    """Attach annotations to metric rows and calculate annotation-relative error."""
    if "clip_id" not in metrics_df.columns or "dominant_hz" not in metrics_df.columns:
        raise ValueError("Metric frame must include clip_id and dominant_hz columns.")
    if metrics_df["clip_id"].duplicated().any():
        raise ValueError("Metric frame contains duplicate clip_id values.")
    metric_ids = set(metrics_df["clip_id"].astype(str))
    truth_ids = set(truth_df["clip_id"].astype(str))
    if require_all_truth:
        missing = sorted(truth_ids - metric_ids)
        if missing:
            raise ValueError(f"Metric cache is missing annotated clip(s): {', '.join(missing[:10])}")

    merged = metrics_df.merge(truth_df, on="clip_id", how="left", validate="one_to_one")
    merged["ground_truth_freq_error_hz"] = np.abs(
        pd.to_numeric(merged["dominant_hz"], errors="coerce") - merged["ground_truth_hz"]
    )
    return merged


def mean_ci95(values: pd.Series | np.ndarray) -> tuple[float, float, float]:
    """Return mean and normal-approximation 95% CI of finite video-level values."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean = float(np.mean(array))
    if array.size == 1:
        return (mean, mean, mean)
    margin = 1.96 * float(np.std(array, ddof=1)) / float(np.sqrt(array.size))
    return (mean, mean - margin, mean + margin)


def frequency_accuracy_summary(merged_df: pd.DataFrame, family: str) -> dict[str, float | int | str]:
    annotated = merged_df.dropna(subset=["ground_truth_hz", "dominant_hz"]).copy()
    if annotated.empty:
        raise ValueError(f"No annotated rows available for family '{family}'.")
    errors = annotated["ground_truth_freq_error_hz"].to_numpy(dtype=float)
    signed_errors = annotated["dominant_hz"].to_numpy(dtype=float) - annotated["ground_truth_hz"].to_numpy(dtype=float)
    return {
        "family": family,
        "matched_clips": int(len(annotated)),
        "mae_hz": float(np.mean(errors)),
        "rmse_hz": float(np.sqrt(np.mean(np.square(errors)))),
        "mean_signed_error_hz": float(np.mean(signed_errors)),
        "median_absolute_error_hz": float(np.median(errors)),
    }
