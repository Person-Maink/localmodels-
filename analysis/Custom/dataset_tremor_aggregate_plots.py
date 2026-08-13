from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

from _path_setup import PROJECT_ROOT  # noqa: F401  # ensures root imports work

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "analysis_matplotlib_cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from dataset_tremor_common import (
    ANGLE_ORDER,
    DEFAULT_METRIC_COLUMNS,
    FAMILY_CHOICES,
    default_custom_output_dir,
    ensure_dir,
    load_metrics_cache,
    metric_display_name,
)
from dataset_tremor_ground_truth import attach_ground_truth, load_ground_truth, mean_ci95


CLEAN_ACTIVITY_ORDER = ("slow_finger", "fast_finger", "slow_wrist", "fast_wrist")
SPEED_ORDER = ("slow", "fast")
EDGE_CASE_ORDER = ("finger", "wrist", "still", "leaving")
GRAPH4A_ACTIVITY_ORDER = ("finger", "wrist")
GRAPH4A_SPEED_ORDER = ("slow", "fast")
GRAPH4B_ACTIVITY_ORDER = ("big_spiral", "small_spiral", "line")
GRAPH4B_STATE_ORDER = ("normal", "tremor")
GRAPH6_FAMILIES = {
    "finger_vs_still": ("fast_finger", "slow_finger", "still"),
    "wrist_vs_still": ("fast_wrist", "slow_wrist", "still"),
    "big_spiral_state": ("big_spiral_normal", "big_spiral_tremor"),
    "small_spiral_state": ("small_spiral_normal", "small_spiral_tremor"),
    "line_state": ("line_normal", "line_tremor"),
}
COHORT_ORDER = ("clean", "tremor_simulation", "edge_cases")
COHORT_DISPLAY = {
    "clean": "Clean",
    "tremor_simulation": "Tremor Simulation",
    "edge_cases": "Edge Cases",
}
MIXED_VIEW_NAME = "mixed"
FAMILY_DISPLAY = {
    "wilor": "WiLoR",
    "stride": "Stride",
}
FAMILY_STYLES = {
    "wilor": {
        "linestyle": "-",
        "marker": "o",
        "hatch": "",
        "alpha": 0.72,
        "participant_alpha": 0.18,
        "edgecolor": "#202020",
    },
    "stride": {
        "linestyle": "--",
        "marker": "s",
        "hatch": "////",
        "alpha": 0.44,
        "participant_alpha": 0.13,
        "edgecolor": "#202020",
    },
}
GRAPH2_ACTIVITY_COLORS = {"finger": "#1f77b4", "wrist": "#ff7f0e"}
MODEL_ALPHA = 0.55
FIGURE_WIDTH_SCALE = 0.82


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render the dataset-level aggregate figures (graphs 2-6) for WiLoR, Stride, "
            "and a mixed comparison view from analysis_images/custom/."
        )
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Root directory containing the family cache folders. Defaults to analysis_images/custom/.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Optional subset of metric columns to plot. Defaults to the full dataset metric set.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Root directory where the wilor/, stride/, and mixed/ plot folders will be written. "
            "Defaults to analysis_images/custom/."
        ),
    )
    parser.add_argument(
        "--ground-truth-csv",
        type=Path,
        default=None,
        help="Optional annotated-frequency CSV. When supplied (or omitted), ground-truth companion plots are rendered.",
    )
    parser.add_argument(
        "--skip-ground-truth",
        action="store_true",
        help="Render only the existing metric plots.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    cache_root = (args.cache_root or default_custom_output_dir(FAMILY_CHOICES[0]).parent).expanduser().resolve()
    output_root = ensure_dir(args.output_dir or cache_root)

    metrics_by_family = _load_family_frames(cache_root)
    truth = None
    if not args.skip_ground_truth:
        truth = load_ground_truth(args.ground_truth_csv)
        metrics_by_family = {
            family: attach_ground_truth(frame, truth)
            for family, frame in metrics_by_family.items()
        }
    mixed_metrics = pd.concat([metrics_by_family[family] for family in FAMILY_CHOICES], ignore_index=True)
    if mixed_metrics.empty:
        raise RuntimeError(f"No metric rows were found under cache root: {cache_root}")

    metric_names = tuple(args.metrics or DEFAULT_METRIC_COLUMNS)
    unknown_metrics = [metric for metric in metric_names if metric not in mixed_metrics.columns]
    if unknown_metrics:
        unknown_text = ", ".join(unknown_metrics)
        raise ValueError(f"Unknown metric column(s) requested: {unknown_text}")

    saved_grand_total = 0
    for family in FAMILY_CHOICES:
        family_output_dir = ensure_dir(output_root / family)
        saved_count = _render_all_graphs(family_output_dir, metrics_by_family[family], metric_names, mixed=False)
        saved_grand_total += saved_count
        print(f"Saved aggregate dataset plots for {FAMILY_DISPLAY.get(family, family)}: {saved_count} figure(s) under {family_output_dir}")

    mixed_output_dir = ensure_dir(output_root / MIXED_VIEW_NAME)
    mixed_saved_count = _render_all_graphs(mixed_output_dir, mixed_metrics, metric_names, mixed=True)
    saved_grand_total += mixed_saved_count
    print(f"Saved aggregate dataset plots for WiLoR vs Stride: {mixed_saved_count} figure(s) under {mixed_output_dir}")
    if not args.skip_ground_truth:
        truth_saved = 0
        for family in FAMILY_CHOICES:
            truth_saved += _render_ground_truth_companions(output_root / family / "ground_truth", metrics_by_family[family], mixed=False)
        truth_saved += _render_ground_truth_companions(output_root / MIXED_VIEW_NAME / "ground_truth", mixed_metrics, mixed=True)
        saved_grand_total += truth_saved
        print(f"Saved ground-truth companion plots: {truth_saved} figure(s) under {output_root}")
    print(f"Saved aggregate dataset plots total: {saved_grand_total} figure(s) under {output_root}")


def _load_family_frames(cache_root: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for family in FAMILY_CHOICES:
        cache_dir = cache_root / family
        frame, _ = load_metrics_cache(cache_dir)
        if frame.empty:
            raise RuntimeError(f"No metric rows were found in cache: {cache_dir}")
        frame = frame.copy()
        frame["family"] = family
        frames[family] = frame
    return frames


def _wider(width: float, height: float) -> tuple[float, float]:
    return (float(width) * FIGURE_WIDTH_SCALE, float(height))


def _render_all_graphs(output_dir: Path, metrics_df: pd.DataFrame, metric_names: tuple[str, ...], mixed: bool) -> int:
    saved_total = 0
    saved_total += _render_graph2(output_dir, metrics_df, metric_names, mixed=mixed)
    saved_total += _render_graph3(output_dir, metrics_df, metric_names, mixed=mixed)
    saved_total += _render_graph4(output_dir, metrics_df, metric_names, mixed=mixed)
    saved_total += _render_graph5(output_dir, metrics_df, metric_names, mixed=mixed)
    saved_total += _render_graph6(output_dir, metrics_df, metric_names, mixed=mixed)
    return saved_total


def _resolved_error_metric(metric_name: str, metrics_df: pd.DataFrame) -> tuple[str, str]:
    if metric_name == "peak_freq_error_hz" and "ground_truth_freq_error_hz" in metrics_df.columns:
        return "ground_truth_freq_error_hz", "Frequency Error vs Ground Truth (Hz)"
    return metric_name, metric_display_name(metric_name)


GROUND_TRUTH_METRICS = (
    ("dominant_hz", "Dominant Frequency (Hz)"),
    ("ground_truth_freq_error_hz", "Frequency Error vs Ground Truth (Hz)"),
)


def _render_ground_truth_companions(output_dir: Path, metrics_df: pd.DataFrame, mixed: bool) -> int:
    """Render only views for which the annotation CSV supplies a meaningful overlay."""
    output_dir = ensure_dir(output_dir)
    clean = metrics_df[metrics_df["cohort"] == "clean"].dropna(subset=["ground_truth_hz"]).copy()
    if clean.empty:
        return 0
    saved = 0
    for metric_name, ylabel in GROUND_TRUTH_METRICS:
        # Remove pre-split companions on rerun so consumers only see the requested layout.
        (output_dir / "graph2_angle_vs_metric" / f"graph2__{metric_name}__ground_truth.svg").unlink(missing_ok=True)
        (output_dir / "graph3_clean_grouped" / f"graph3__{metric_name}__ground_truth.svg").unlink(missing_ok=True)
        saved += _render_ground_truth_angle_view(
            output_dir / "graph2_angle_vs_metric", clean, metric_name, ylabel, mixed, None,
        )
        for speed in SPEED_ORDER:
            speed_clean = clean[clean["speed_label"] == speed].copy()
            saved += _render_ground_truth_angle_view(
                output_dir / "graph2_angle_vs_metric",
                speed_clean,
                metric_name,
                ylabel,
                mixed,
                speed,
            )
        graph3 = clean.copy()
        graph3["condition"] = graph3["speed_label"].astype(str) + "_" + graph3["activity_family"].astype(str)
        # Graph 3's error chart is already an error-to-truth measure; an additional
        # truth overlay would be visually redundant.
        if metric_name == "dominant_hz":
            for speed in SPEED_ORDER:
                speed_graph3 = graph3[graph3["speed_label"] == speed].copy()
                saved += _render_ground_truth_grouped_view(
                    output_dir / "graph3_clean_grouped",
                    speed_graph3,
                    metric_name,
                    ylabel,
                    primary_col="angle",
                    primary_order=ANGLE_ORDER,
                    condition_col="condition",
                    condition_order=tuple(f"{speed}_{activity}" for activity in GRAPH4A_ACTIVITY_ORDER),
                    filename=f"graph3__{metric_name}__{speed}__ground_truth.svg",
                    title=f"Graph 3: {speed.title()} Clean Activities vs Angle for {ylabel} (with Ground Truth)",
                    mixed=mixed,
                )
        if metric_name == "dominant_hz":
            saved += _render_ground_truth_grouped_view(
                output_dir / "graph4a_clean_activity", clean, metric_name, ylabel,
                primary_col="activity_family", primary_order=GRAPH4A_ACTIVITY_ORDER,
                condition_col="speed_label", condition_order=GRAPH4A_SPEED_ORDER,
                filename=f"graph4a__{metric_name}__ground_truth.svg",
                title=f"Graph 4a: Clean Activity Summary for {ylabel} (with Ground Truth)", mixed=mixed,
            )
            for speed in SPEED_ORDER:
                speed_clean = clean[clean["speed_label"] == speed].copy()
                saved += _render_ground_truth_grouped_view(
                    output_dir / "graph4a_clean_activity", speed_clean, metric_name, ylabel,
                    primary_col="activity_family", primary_order=GRAPH4A_ACTIVITY_ORDER,
                    condition_col="speed_label", condition_order=(speed,),
                    filename=f"graph4a__{metric_name}__{speed}__ground_truth.svg",
                    title=f"Graph 4a: {speed.title()} Clean Activity Summary for {ylabel} (with Ground Truth)", mixed=mixed,
                )
            for activity in ("finger", "wrist"):
                subset = clean[clean["activity_family"] == activity].copy()
                subset["condition"] = subset["speed_label"].astype(str) + "_" + subset["activity_family"].astype(str)
                saved += _render_ground_truth_grouped_view(
                    output_dir / "graph6_angle_vs_control", subset, metric_name, ylabel,
                    primary_col="angle", primary_order=ANGLE_ORDER, condition_col="condition",
                    condition_order=tuple(f"{speed}_{activity}" for speed in GRAPH4A_SPEED_ORDER),
                    filename=f"graph6__{activity}_vs_still__{metric_name}__ground_truth.svg",
                    title=f"Graph 6: {activity.title()} vs Still for {ylabel} (with Ground Truth)", mixed=mixed,
                )
                for speed in SPEED_ORDER:
                    speed_subset = subset[subset["speed_label"] == speed].copy()
                    speed_condition = f"{speed}_{activity}"
                    saved += _render_ground_truth_grouped_view(
                        output_dir / "graph6_angle_vs_control", speed_subset, metric_name, ylabel,
                        primary_col="angle", primary_order=ANGLE_ORDER, condition_col="condition",
                        condition_order=(speed_condition,),
                        filename=f"graph6__{activity}_vs_still__{metric_name}__{speed}__ground_truth.svg",
                        title=f"Graph 6: {speed.title()} {activity.title()} vs Still for {ylabel} (with Ground Truth)", mixed=mixed,
                    )
        else:
            (output_dir / "graph4a_clean_activity" / f"graph4a__{metric_name}__ground_truth.svg").unlink(missing_ok=True)
            for activity in ("finger", "wrist"):
                (output_dir / "graph6_angle_vs_control" / f"graph6__{activity}_vs_still__{metric_name}__ground_truth.svg").unlink(missing_ok=True)
    return saved


def _truth_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    # Model family duplicates must not change a ground-truth summary in mixed plots.
    deduplicated = df.drop_duplicates(subset=["clip_id"])
    for keys, group in deduplicated.groupby(group_cols, sort=True, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        mean, lower, upper = mean_ci95(group["ground_truth_hz"])
        rows.append(dict(zip(group_cols, keys), ground_truth_mean=mean, ground_truth_lower=lower, ground_truth_upper=upper))
    return pd.DataFrame(rows)


def _model_summary(df: pd.DataFrame, metric_name: str, group_cols: list[str], mixed: bool) -> pd.DataFrame:
    family_cols = ["family"] if mixed else []
    rows = []
    for keys, group in df.dropna(subset=[metric_name]).groupby(family_cols + group_cols + ["person"], sort=True, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows.append(dict(zip(family_cols + group_cols, keys[:-1]), participant_value=float(group[metric_name].mean())))
    participants = pd.DataFrame(rows)
    if participants.empty:
        return participants
    grouping = family_cols + group_cols
    return participants.groupby(grouping, sort=True, dropna=False)["participant_value"].agg(mean="mean", sd="std").reset_index()


def _render_ground_truth_angle_view(
    output_dir: Path,
    clean: pd.DataFrame,
    metric_name: str,
    ylabel: str,
    mixed: bool,
    speed: str | None,
    output_filename: str | None = None,
) -> int:
    output_dir = ensure_dir(output_dir)
    working = clean.copy()
    working["condition"] = working["speed_label"].astype(str) + "_" + working["activity_family"].astype(str)
    model = _model_summary(working, metric_name, ["angle", "condition"], mixed)
    truth = _truth_summary(working, ["angle", "condition"])
    if model.empty or truth.empty:
        return 0
    fig, ax = plt.subplots(figsize=_wider(13.5, 6))
    condition_order = (
        tuple(f"{speed}_{activity}" for activity in GRAPH4A_ACTIVITY_ORDER)
        if speed is not None
        else CLEAN_ACTIVITY_ORDER
    )
    palette = {condition: GRAPH2_ACTIVITY_COLORS[condition.rsplit("_", 1)[1]] for condition in condition_order}
    for condition in condition_order:
        color = palette[condition]
        truth_part = truth[truth["condition"] == condition].copy()
        truth_part["angle"] = pd.Categorical(truth_part["angle"], categories=ANGLE_ORDER, ordered=True)
        truth_part = truth_part.sort_values("angle")
        if not truth_part.empty:
            x = np.asarray([ANGLE_ORDER.index(str(value)) for value in truth_part["angle"].astype(str)])
            y = truth_part["ground_truth_mean"].to_numpy()
            ax.errorbar(x, y, yerr=np.vstack((y - truth_part["ground_truth_lower"], truth_part["ground_truth_upper"] - y)), color=color, marker="o", markerfacecolor=color, markeredgecolor=color, linestyle="none", markersize=6, capsize=3, label=f"{_display_label(condition)} Ground Truth")
        model_part = model[model["condition"] == condition]
        families = _observed_family_order(model_part) if mixed else (None,)
        for family in families:
            subset = model_part if family is None else model_part[model_part["family"] == family]
            subset = subset.copy()
            subset["angle"] = pd.Categorical(subset["angle"], categories=ANGLE_ORDER, ordered=True)
            subset = subset.sort_values("angle")
            if subset.empty:
                continue
            x = np.asarray([ANGLE_ORDER.index(str(value)) for value in subset["angle"].astype(str)])
            label = _display_label(condition) if family is None else f"{_display_label(condition)} {_display_family(family)}"
            ax.errorbar(x, subset["mean"], yerr=subset["sd"].fillna(0), color=color, marker="o", markerfacecolor="none", markeredgecolor=color, linestyle="none", markersize=6, capsize=3, label=label, alpha=MODEL_ALPHA)
    speed_title = f"{speed.title()} " if speed is not None else ""
    ax.set_title(_with_mixed_suffix(f"Graph 2: {speed_title}Clean Activities vs Angle for {ylabel} (with Ground Truth)", mixed))
    ax.set_ylabel(ylabel); ax.set_xlabel("Camera Angle"); ax.set_xticks(range(len(ANGLE_ORDER)), ANGLE_ORDER); ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize="x-small")
    filename = output_filename or (
        f"graph2__{metric_name}__{speed}__ground_truth.svg" if speed is not None else f"graph2__{metric_name}__ground_truth.svg"
    )
    fig.tight_layout(); fig.savefig(output_dir / filename, dpi=100, bbox_inches="tight"); plt.close(fig)
    return 1


def _render_ground_truth_grouped_view(output_dir: Path, df: pd.DataFrame, metric_name: str, ylabel: str, primary_col: str, primary_order: tuple[str, ...], condition_col: str, condition_order: tuple[str, ...], filename: str, title: str, mixed: bool) -> int:
    output_dir = ensure_dir(output_dir)
    model = _model_summary(df, metric_name, [primary_col, condition_col], mixed)
    truth = _truth_summary(df, [primary_col, condition_col])
    primary_order = tuple(value for value in primary_order if value in set(df[primary_col].astype(str)))
    condition_order = tuple(value for value in condition_order if value in set(df[condition_col].astype(str)))
    if model.empty or truth.empty or not primary_order or not condition_order:
        return 0
    fig, ax = plt.subplots(figsize=_wider(11.5, 6))
    positions = np.arange(len(primary_order), dtype=float)
    families = _observed_family_order(model) if mixed else (None,)
    point_spacing = 0.18 if len(condition_order) * len(families) > 1 else 0.0
    offsets = np.linspace(-point_spacing * (len(condition_order) * len(families) - 1) / 2, point_spacing * (len(condition_order) * len(families) - 1) / 2, len(condition_order) * len(families))
    palette = {condition: f"C{index}" for index, condition in enumerate(condition_order)}
    slot = 0
    truth_label_added = False
    for condition in condition_order:
        for family in families:
            offset = offsets[slot]; slot += 1
            subset = model[model[condition_col].astype(str) == condition]
            if family is not None: subset = subset[subset["family"] == family]
            means = [float(subset.loc[subset[primary_col].astype(str) == primary, "mean"].iloc[0]) if not subset.loc[subset[primary_col].astype(str) == primary].empty else np.nan for primary in primary_order]
            sds = [float(subset.loc[subset[primary_col].astype(str) == primary, "sd"].fillna(0).iloc[0]) if not subset.loc[subset[primary_col].astype(str) == primary].empty else 0.0 for primary in primary_order]
            label = _display_label(condition) if family is None else f"{_display_label(condition)} {_display_family(family)}"
            point_centers = positions + offset
            ax.errorbar(point_centers, means, yerr=sds, color=palette[condition], marker="o", markerfacecolor="none", markeredgecolor=palette[condition], linestyle="none", markersize=6, capsize=3, label=label, alpha=MODEL_ALPHA)
            truth_part = truth[truth[condition_col].astype(str) == condition]
            truth_values = []
            for primary in primary_order:
                row = truth_part[truth_part[primary_col].astype(str) == primary]
                truth_values.append(None if row.empty else tuple(row[["ground_truth_mean", "ground_truth_lower", "ground_truth_upper"]].iloc[0]))
            valid = np.asarray([value is not None for value in truth_values])
            if valid.any():
                values = np.asarray([value for value in truth_values if value is not None], dtype=float)
                ax.errorbar(point_centers[valid], values[:, 0], yerr=np.vstack((values[:, 0] - values[:, 1], values[:, 2] - values[:, 0])), color=palette[condition], marker="o", markerfacecolor=palette[condition], markeredgecolor=palette[condition], linestyle="none", markersize=6, capsize=4, zorder=5, label="Ground Truth (95% CI)" if not truth_label_added else None)
                truth_label_added = True
    ax.set_title(_with_mixed_suffix(title, mixed)); ax.set_ylabel(ylabel); ax.set_xticks(positions, [_display_label(value) for value in primary_order]); ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize="x-small")
    fig.tight_layout(); fig.savefig(output_dir / filename, dpi=100, bbox_inches="tight"); plt.close(fig)
    return 1


def _render_graph2(output_dir: Path, metrics_df: pd.DataFrame, metric_names: tuple[str, ...], mixed: bool) -> int:
    graph_dir = ensure_dir(output_dir / "graph2_angle_vs_metric")
    clean = metrics_df[metrics_df["cohort"] == "clean"].copy()
    saved = 0
    for metric_name in metric_names:
        if metric_name == "dominant_hz" and "ground_truth_hz" in clean.columns:
            saved += _render_ground_truth_angle_view(
                graph_dir, clean, metric_name, metric_display_name(metric_name), mixed, None,
                output_filename=f"graph2__{metric_name}.svg",
            )
        else:
            per_person, summary = _graph2_activity_frame(clean, "clean", metric_name, include_family=mixed)
            if not per_person.empty and not summary.empty:
                fig, ax = plt.subplots(figsize=_wider(10.5, 5.2))
                _plot_graph2_cohort(ax, "clean", per_person, summary, metric_name, mixed=mixed)
                ax.set_xlabel("Camera Angle")
                fig.suptitle(_with_mixed_suffix(f"Graph 2: Clean Activities vs {metric_display_name(metric_name)}", mixed), y=0.995)
                fig.tight_layout(); fig.savefig(graph_dir / f"graph2__{metric_name}.svg", dpi=100, bbox_inches="tight"); plt.close(fig)
                saved += 1
        for speed in SPEED_ORDER:
            speed_clean = clean[clean["speed_label"] == speed].copy()
            if metric_name == "dominant_hz" and "ground_truth_hz" in speed_clean.columns:
                saved += _render_ground_truth_angle_view(
                    graph_dir, speed_clean, metric_name, metric_display_name(metric_name), mixed, speed,
                    output_filename=f"graph2__{metric_name}__{speed}.svg",
                )
                continue
            per_person, summary = _graph2_activity_frame(speed_clean, "clean", metric_name, include_family=mixed)
            if per_person.empty or summary.empty:
                continue
            fig, ax = plt.subplots(figsize=_wider(13.5, 5.2))
            _plot_graph2_cohort(ax, "clean", per_person, summary, metric_name, mixed=mixed)
            ax.set_xlabel("Camera Angle")
            fig.suptitle(
                _with_mixed_suffix(f"Graph 2: {speed.title()} Clean Activities vs {metric_display_name(metric_name)}", mixed),
                y=0.995,
            )
            fig.tight_layout()
            path = graph_dir / f"graph2__{metric_name}__{speed}.svg"
            fig.savefig(path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            saved += 1
    return saved


def _render_graph3(output_dir: Path, metrics_df: pd.DataFrame, metric_names: tuple[str, ...], mixed: bool) -> int:
    graph_dir = ensure_dir(output_dir / "graph3_clean_grouped")
    source = metrics_df[metrics_df["cohort"] == "clean"].copy()
    if source.empty:
        return 0
    source["graph3_condition"] = source["speed_label"].astype(str) + "_" + source["activity_family"].astype(str)

    saved = 0
    for metric_name in metric_names:
        (graph_dir / f"graph3__{metric_name}.svg").unlink(missing_ok=True)
        plot_metric_name, ylabel = _resolved_error_metric(metric_name, source)
        for speed in SPEED_ORDER:
            speed_source = source[source["speed_label"] == speed].copy()
            summary = _participant_summary(speed_source, plot_metric_name, ["angle", "graph3_condition"], include_family=mixed)
            if summary.empty:
                continue
            condition_order = tuple(f"{speed}_{activity}" for activity in GRAPH4A_ACTIVITY_ORDER)
            if metric_name == "dominant_hz" and "ground_truth_hz" in speed_source.columns:
                saved += _render_ground_truth_grouped_view(
                    graph_dir, speed_source, metric_name, ylabel,
                    primary_col="angle", primary_order=ANGLE_ORDER,
                    condition_col="graph3_condition", condition_order=condition_order,
                    filename=f"graph3__{metric_name}__{speed}.svg",
                    title=f"Graph 3: {speed.title()} Clean Activities vs Angle for {ylabel} (with Ground Truth)",
                    mixed=mixed,
                )
                continue
            fig, ax = plt.subplots(figsize=_wider(13.5, 6))
            title = _with_mixed_suffix(
                f"Graph 3: {speed.title()} Clean Activities vs Angle for {ylabel}", mixed
            )
            if mixed:
                _grouped_bar_with_family_points(
                    ax, summary, ANGLE_ORDER, condition_order, "angle", "graph3_condition", title, ylabel,
                    show_individual_points=False,
                )
            else:
                _grouped_bar_with_points(
                    ax, summary, ANGLE_ORDER, condition_order, "angle", "graph3_condition", title, ylabel,
                    show_individual_points=False,
                )
            path = graph_dir / f"graph3__{metric_name}__{speed}.svg"
            fig.tight_layout(); fig.savefig(path, dpi=100, bbox_inches="tight"); plt.close(fig)
            saved += 1
    return saved


def _render_graph4(output_dir: Path, metrics_df: pd.DataFrame, metric_names: tuple[str, ...], mixed: bool) -> int:
    graph4a_dir = ensure_dir(output_dir / "graph4a_clean_activity")
    graph4b_dir = ensure_dir(output_dir / "graph4b_tremor_activity")
    saved = 0

    clean = metrics_df[metrics_df["cohort"] == "clean"].copy()
    tremor = metrics_df[metrics_df["cohort"] == "tremor_simulation"].copy()
    if not clean.empty:
        clean = clean[clean["activity_family"].isin(GRAPH4A_ACTIVITY_ORDER)]
    if not tremor.empty:
        tremor = tremor[tremor["activity_family"].isin(GRAPH4B_ACTIVITY_ORDER)]

    for metric_name in metric_names:
        plot_metric_name, ylabel = _resolved_error_metric(metric_name, metrics_df)
        clean_summary = _participant_summary(clean, plot_metric_name, ["activity_family", "speed_label"], include_family=mixed)
        if not clean_summary.empty:
            if metric_name == "dominant_hz" and "ground_truth_hz" in clean.columns:
                saved += _render_ground_truth_grouped_view(
                    graph4a_dir, clean, metric_name, ylabel,
                    primary_col="activity_family", primary_order=GRAPH4A_ACTIVITY_ORDER,
                    condition_col="speed_label", condition_order=GRAPH4A_SPEED_ORDER,
                    filename=f"graph4a__{metric_name}.svg",
                    title=f"Graph 4a: Clean Activity Summary for {ylabel} (with Ground Truth)", mixed=mixed,
                )
            else:
                fig, ax = plt.subplots(figsize=_wider(9.5, 6))
                if mixed:
                    _grouped_bar_with_family_points(
                        ax, clean_summary, primary_order=GRAPH4A_ACTIVITY_ORDER, secondary_order=GRAPH4A_SPEED_ORDER,
                        primary_col="activity_family", secondary_col="speed_label",
                        title=_with_mixed_suffix(f"Graph 4a: Clean Activity Summary for {ylabel}", mixed), ylabel=ylabel,
                        show_individual_points=False,
                    )
                else:
                    _grouped_bar_with_points(
                        ax, clean_summary, primary_order=GRAPH4A_ACTIVITY_ORDER, secondary_order=GRAPH4A_SPEED_ORDER,
                        primary_col="activity_family", secondary_col="speed_label",
                        title=f"Graph 4a: Clean Activity Summary for {ylabel}", ylabel=ylabel, show_individual_points=False,
                    )
                fig.tight_layout(); fig.savefig(graph4a_dir / f"graph4a__{metric_name}.svg", dpi=100, bbox_inches="tight"); plt.close(fig)
                saved += 1
            for speed in SPEED_ORDER:
                speed_clean = clean[clean["speed_label"] == speed].copy()
                speed_summary = _participant_summary(speed_clean, plot_metric_name, ["activity_family", "speed_label"], include_family=mixed)
                if speed_summary.empty:
                    continue
                split_path = graph4a_dir / f"graph4a__{metric_name}__{speed}.svg"
                if metric_name == "dominant_hz" and "ground_truth_hz" in speed_clean.columns:
                    saved += _render_ground_truth_grouped_view(
                        graph4a_dir, speed_clean, metric_name, ylabel,
                        primary_col="activity_family", primary_order=GRAPH4A_ACTIVITY_ORDER,
                        condition_col="speed_label", condition_order=(speed,),
                        filename=split_path.name,
                        title=f"Graph 4a: {speed.title()} Clean Activity Summary for {ylabel} (with Ground Truth)", mixed=mixed,
                    )
                    continue
                fig, ax = plt.subplots(figsize=_wider(7.5, 6))
                title = _with_mixed_suffix(f"Graph 4a: {speed.title()} Clean Activity Summary for {ylabel}", mixed)
                if mixed:
                    _grouped_bar_with_family_points(ax, speed_summary, GRAPH4A_ACTIVITY_ORDER, (speed,), "activity_family", "speed_label", title, ylabel, show_individual_points=False)
                else:
                    _grouped_bar_with_points(ax, speed_summary, GRAPH4A_ACTIVITY_ORDER, (speed,), "activity_family", "speed_label", title, ylabel, show_individual_points=False)
                fig.tight_layout(); fig.savefig(split_path, dpi=100, bbox_inches="tight"); plt.close(fig)
                saved += 1

        tremor_summary = _participant_summary(tremor, metric_name, ["activity_family", "activity_state"], include_family=mixed)
        if not tremor_summary.empty:
            fig, ax = plt.subplots(figsize=_wider(11.5, 6))
            if mixed:
                _grouped_bar_with_family_points(
                    ax,
                    tremor_summary,
                    primary_order=GRAPH4B_ACTIVITY_ORDER,
                    secondary_order=GRAPH4B_STATE_ORDER,
                    primary_col="activity_family",
                    secondary_col="activity_state",
                    title=_with_mixed_suffix(f"Graph 4b: Tremor-Simulation Summary for {metric_display_name(metric_name)}", mixed),
                    ylabel=metric_display_name(metric_name),
                    show_individual_points=False,
                )
            else:
                _grouped_bar_with_points(
                    ax,
                    tremor_summary,
                    primary_order=GRAPH4B_ACTIVITY_ORDER,
                    secondary_order=GRAPH4B_STATE_ORDER,
                    primary_col="activity_family",
                    secondary_col="activity_state",
                    title=f"Graph 4b: Tremor-Simulation Summary for {metric_display_name(metric_name)}",
                    ylabel=metric_display_name(metric_name),
                    show_individual_points=False,
                )
            path = graph4b_dir / f"graph4b__{metric_name}.svg"
            fig.tight_layout()
            fig.savefig(path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            saved += 1
    return saved


def _render_graph5(output_dir: Path, metrics_df: pd.DataFrame, metric_names: tuple[str, ...], mixed: bool) -> int:
    graph_dir = ensure_dir(output_dir / "graph5_edge_cases")
    edge = metrics_df[metrics_df["cohort"] == "edge_cases"].copy()
    if edge.empty:
        return 0

    saved = 0
    for metric_name in metric_names:
        summary = _participant_summary(edge, metric_name, ["activity_family"], include_family=mixed)
        if summary.empty:
            continue
        summary = summary[summary["activity_family"].isin(EDGE_CASE_ORDER)]
        if summary.empty:
            continue
        fig, ax = plt.subplots(figsize=_wider(9.5, 6))
        if mixed:
            _single_bar_with_family_points(
                ax,
                summary,
                category_order=EDGE_CASE_ORDER,
                category_col="activity_family",
                title=_with_mixed_suffix(f"Graph 5: Edge-Case Summary for {metric_display_name(metric_name)}", mixed),
                ylabel=metric_display_name(metric_name),
            )
        else:
            _single_bar_with_points(
                ax,
                summary,
                category_order=EDGE_CASE_ORDER,
                category_col="activity_family",
                title=f"Graph 5: Edge-Case Summary for {metric_display_name(metric_name)}",
                ylabel=metric_display_name(metric_name),
            )
        path = graph_dir / f"graph5__{metric_name}.svg"
        fig.tight_layout()
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        saved += 1
    return saved


def _render_graph6(output_dir: Path, metrics_df: pd.DataFrame, metric_names: tuple[str, ...], mixed: bool) -> int:
    graph_dir = ensure_dir(output_dir / "graph6_angle_vs_control")
    working = metrics_df.copy()
    working["graph6_condition"] = working.apply(_graph6_condition_label, axis=1)

    saved = 0
    for metric_name in metric_names:
        plot_metric_name, ylabel = _resolved_error_metric(metric_name, working)
        summary = _participant_summary(working, plot_metric_name, ["angle", "graph6_condition"], include_family=mixed)
        if summary.empty:
            continue
        for family_key, condition_order in GRAPH6_FAMILIES.items():
            family_summary = summary[summary["graph6_condition"].isin(condition_order)].copy()
            if family_summary.empty:
                continue
            if metric_name == "dominant_hz" and "ground_truth_hz" in working.columns:
                truth_source = working[working["graph6_condition"].isin(condition_order)].copy()
                if truth_source["ground_truth_hz"].notna().any():
                    saved += _render_ground_truth_grouped_view(
                        graph_dir, truth_source, metric_name, ylabel,
                        primary_col="angle", primary_order=ANGLE_ORDER,
                        condition_col="graph6_condition", condition_order=condition_order,
                        filename=f"graph6__{family_key}__{metric_name}.svg",
                        title=f"Graph 6: {family_key.replace('_', ' ').title()} for {ylabel} (with Ground Truth)",
                        mixed=mixed,
                    )
                    if family_key in {"finger_vs_still", "wrist_vs_still"}:
                        activity = family_key.split("_vs_", 1)[0]
                        for speed in SPEED_ORDER:
                            speed_condition = f"{speed}_{activity}"
                            split_conditions = (speed_condition, "still")
                            split_truth = working[working["graph6_condition"].isin(split_conditions)].copy()
                            if split_truth["ground_truth_hz"].notna().any():
                                saved += _render_ground_truth_grouped_view(
                                    graph_dir, split_truth, metric_name, ylabel,
                                    primary_col="angle", primary_order=ANGLE_ORDER,
                                    condition_col="graph6_condition", condition_order=split_conditions,
                                    filename=f"graph6__{family_key}__{metric_name}__{speed}.svg",
                                    title=f"Graph 6: {speed.title()} {activity.title()} vs Still for {ylabel} (with Ground Truth)", mixed=mixed,
                                )
                    continue
            fig, ax = plt.subplots(figsize=_wider(12.5, 6))
            if mixed:
                _grouped_bar_with_family_points(
                    ax,
                    family_summary,
                    primary_order=ANGLE_ORDER,
                    secondary_order=condition_order,
                    primary_col="angle",
                    secondary_col="graph6_condition",
                    title=_with_mixed_suffix(
                        f"Graph 6: {family_key.replace('_', ' ').title()} for {ylabel}",
                        mixed,
                    ),
                    ylabel=ylabel,
                    show_individual_points=False,
                )
            else:
                _grouped_bar_with_points(
                    ax,
                    family_summary,
                    primary_order=ANGLE_ORDER,
                    secondary_order=condition_order,
                    primary_col="angle",
                    secondary_col="graph6_condition",
                    title=f"Graph 6: {family_key.replace('_', ' ').title()} for {ylabel}",
                    ylabel=ylabel,
                    show_individual_points=False,
                )
            path = graph_dir / f"graph6__{family_key}__{metric_name}.svg"
            fig.tight_layout()
            fig.savefig(path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            saved += 1
            if family_key not in {"finger_vs_still", "wrist_vs_still"}:
                continue
            activity = family_key.split("_vs_", 1)[0]
            for speed in SPEED_ORDER:
                speed_condition = f"{speed}_{activity}"
                split_conditions = (speed_condition, "still")
                split_summary = summary[summary["graph6_condition"].isin(split_conditions)].copy()
                if split_summary.empty:
                    continue
                split_path = graph_dir / f"graph6__{family_key}__{metric_name}__{speed}.svg"
                if metric_name == "dominant_hz" and "ground_truth_hz" in working.columns:
                    split_truth = working[working["graph6_condition"].isin(split_conditions)].copy()
                    if split_truth["ground_truth_hz"].notna().any():
                        saved += _render_ground_truth_grouped_view(
                            graph_dir, split_truth, metric_name, ylabel,
                            primary_col="angle", primary_order=ANGLE_ORDER,
                            condition_col="graph6_condition", condition_order=split_conditions,
                            filename=split_path.name,
                            title=f"Graph 6: {speed.title()} {activity.title()} vs Still for {ylabel} (with Ground Truth)", mixed=mixed,
                        )
                        continue
                fig, ax = plt.subplots(figsize=_wider(8.5, 6))
                title = _with_mixed_suffix(f"Graph 6: {speed.title()} {activity.title()} vs Still for {ylabel}", mixed)
                if mixed:
                    _grouped_bar_with_family_points(ax, split_summary, ANGLE_ORDER, split_conditions, "angle", "graph6_condition", title, ylabel, show_individual_points=False)
                else:
                    _grouped_bar_with_points(ax, split_summary, ANGLE_ORDER, split_conditions, "angle", "graph6_condition", title, ylabel, show_individual_points=False)
                fig.tight_layout(); fig.savefig(split_path, dpi=100, bbox_inches="tight"); plt.close(fig)
                saved += 1
    return saved


def _graph2_activity_frame(
    metrics_df: pd.DataFrame,
    cohort: str,
    metric_name: str,
    include_family: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cohort_df = metrics_df[metrics_df["cohort"] == cohort].copy()
    if cohort_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    if cohort == "clean":
        cohort_df["graph2_condition"] = cohort_df["speed_label"].astype(str) + "_" + cohort_df["activity_family"].astype(str)
    elif cohort == "tremor_simulation":
        cohort_df["graph2_condition"] = cohort_df["activity_family"].astype(str) + "_" + cohort_df["activity_state"].astype(str)
    else:
        cohort_df["graph2_condition"] = cohort_df["activity_family"].astype(str)

    if metric_name not in cohort_df.columns:
        return pd.DataFrame(), pd.DataFrame()
    cohort_df = cohort_df.dropna(subset=[metric_name]).copy()
    if cohort_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    family_cols = ["family"] if include_family else []
    per_person_group_cols = family_cols + ["angle", "graph2_condition", "person"]
    summary_group_cols = family_cols + ["angle", "graph2_condition"]

    per_person = (
        cohort_df.groupby(per_person_group_cols, sort=True, dropna=False)[metric_name]
        .mean()
        .reset_index(name="participant_value")
    )
    summary = (
        per_person.groupby(summary_group_cols, sort=True, dropna=False)["participant_value"]
        .agg(
            mean="mean",
            sd=lambda series: float(series.std(ddof=1)) if len(series) > 1 else np.nan,
            n="count",
        )
        .reset_index()
    )
    return per_person, summary


def _plot_graph2_cohort(
    ax,
    cohort: str,
    per_person: pd.DataFrame,
    summary: pd.DataFrame,
    metric_name: str,
    mixed: bool,
) -> None:
    condition_order = _condition_order_for_cohort(cohort, summary["graph2_condition"].unique().tolist())
    angle_positions = np.arange(len(ANGLE_ORDER), dtype=np.float32)
    palette = {
        condition: GRAPH2_ACTIVITY_COLORS.get(str(condition).rsplit("_", 1)[-1], f"C{index % 10}")
        for index, condition in enumerate(condition_order)
    }
    family_order = _observed_family_order(summary) if mixed else ()

    if mixed:
        for condition in condition_order:
            for family in family_order:
                condition_frame = summary[
                    (summary["graph2_condition"] == condition) & (summary["family"].astype(str) == family)
                ].copy()
                if condition_frame.empty:
                    continue
                condition_frame["angle"] = pd.Categorical(condition_frame["angle"], categories=ANGLE_ORDER, ordered=True)
                condition_frame = condition_frame.sort_values("angle")
                x_values = np.asarray(
                    [ANGLE_ORDER.index(str(angle)) for angle in condition_frame["angle"].astype(str)],
                    dtype=np.float32,
                )
                style = FAMILY_STYLES[family]
                ax.errorbar(
                    x_values,
                    condition_frame["mean"].to_numpy(dtype=np.float32),
                    yerr=condition_frame["sd"].fillna(0.0).to_numpy(dtype=np.float32),
                    color=palette[condition],
                    marker="o",
                    markerfacecolor="none",
                    markeredgecolor=palette[condition],
                    markersize=6,
                    linewidth=1.4,
                    linestyle="none",
                    capsize=3,
                    zorder=3,
                    alpha=MODEL_ALPHA,
                )

        _add_graph2_mixed_legends(ax, condition_order, palette, family_order)
    else:
        for condition in condition_order:
            condition_frame = summary[summary["graph2_condition"] == condition].copy()
            if condition_frame.empty:
                continue
            condition_frame["angle"] = pd.Categorical(condition_frame["angle"], categories=ANGLE_ORDER, ordered=True)
            condition_frame = condition_frame.sort_values("angle")
            x_values = np.asarray(
                [ANGLE_ORDER.index(str(angle)) for angle in condition_frame["angle"].astype(str)],
                dtype=np.float32,
            )
            ax.errorbar(
                x_values,
                condition_frame["mean"].to_numpy(dtype=np.float32),
                yerr=condition_frame["sd"].fillna(0.0).to_numpy(dtype=np.float32),
                color=palette[condition],
                marker="o",
                markerfacecolor="none",
                markeredgecolor=palette[condition],
                markersize=6,
                linewidth=1.4,
                linestyle="none",
                capsize=3,
                label=_display_label(condition),
                zorder=3,
                alpha=MODEL_ALPHA,
            )
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize="small")

    ax.set_title(COHORT_DISPLAY[cohort])
    ax.set_ylabel(metric_display_name(metric_name))
    ax.set_xticks(angle_positions.tolist(), ANGLE_ORDER)
    ax.grid(True, axis="y", alpha=0.25)
    if metric_name == "dominant_hz" and cohort == "clean":
        ax.axhline(4.0, color="#4a4a4a", linestyle="--", linewidth=1.0, alpha=0.45)
        ax.axhline(5.0, color="#4a4a4a", linestyle=":", linewidth=1.0, alpha=0.45)


def _grouped_bar_with_points(
    ax,
    summary: pd.DataFrame,
    primary_order: tuple[str, ...],
    secondary_order: tuple[str, ...],
    primary_col: str,
    secondary_col: str,
    title: str,
    ylabel: str,
    show_individual_points: bool = True,
) -> None:
    primary_order = tuple(item for item in primary_order if item in set(summary[primary_col].astype(str)))
    secondary_order = tuple(item for item in secondary_order if item in set(summary[secondary_col].astype(str)))
    if not primary_order or not secondary_order:
        return

    x_positions = np.arange(len(primary_order), dtype=np.float32)
    offsets = np.linspace(-0.12 * (len(secondary_order) - 1) / 2, 0.12 * (len(secondary_order) - 1) / 2, num=len(secondary_order))
    palette = {condition: f"C{index % 10}" for index, condition in enumerate(secondary_order)}

    for offset, secondary in zip(offsets.tolist(), secondary_order):
        subset = summary[summary[secondary_col] == secondary].copy()
        if subset.empty:
            continue
        means = []
        sds = []
        per_primary_groups = []
        for primary in primary_order:
            row = subset[subset[primary_col] == primary]
            means.append(float(row["mean"].iloc[0]) if not row.empty else np.nan)
            sds.append(float(row["sd"].fillna(0.0).iloc[0]) if not row.empty else 0.0)
            per_primary_groups.append(row)

        ax.errorbar(x_positions + offset, means, yerr=sds, color=palette[secondary], marker="o", markerfacecolor="none", markeredgecolor=palette[secondary], linestyle="none", markersize=6, capsize=3, label=_display_label(secondary), zorder=2, alpha=MODEL_ALPHA)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions.tolist(), [_display_label(label) for label in primary_order])
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize="small")


def _grouped_bar_with_family_points(
    ax,
    summary: pd.DataFrame,
    primary_order: tuple[str, ...],
    secondary_order: tuple[str, ...],
    primary_col: str,
    secondary_col: str,
    title: str,
    ylabel: str,
    show_individual_points: bool = True,
) -> None:
    primary_order = tuple(item for item in primary_order if item in set(summary[primary_col].astype(str)))
    secondary_order = tuple(item for item in secondary_order if item in set(summary[secondary_col].astype(str)))
    family_order = _observed_family_order(summary)
    if not primary_order or not secondary_order or not family_order:
        return

    x_positions = np.arange(len(primary_order), dtype=np.float32)
    slot_count = len(secondary_order) * len(family_order)
    offsets = np.linspace(-0.10 * (slot_count - 1) / 2, 0.10 * (slot_count - 1) / 2, num=slot_count)
    palette = {condition: f"C{index % 10}" for index, condition in enumerate(secondary_order)}

    slot_index = 0
    for secondary in secondary_order:
        for family in family_order:
            offset = offsets[slot_index]
            slot_index += 1
            subset = summary[
                (summary[secondary_col].astype(str) == secondary) & (summary["family"].astype(str) == family)
            ].copy()
            if subset.empty:
                continue
            means = []
            sds = []
            for primary in primary_order:
                row = subset[subset[primary_col].astype(str) == primary]
                means.append(float(row["mean"].iloc[0]) if not row.empty else np.nan)
                sds.append(float(row["sd"].fillna(0.0).iloc[0]) if not row.empty else 0.0)

            style = FAMILY_STYLES[family]
            point_centers = x_positions + float(offset)
            ax.errorbar(point_centers, means, yerr=sds, color=palette[secondary], marker=style["marker"], markerfacecolor="none", markeredgecolor=palette[secondary], linestyle="none", markersize=6, capsize=3, label=f"{_display_label(secondary)} {_display_family(family)}", zorder=2, alpha=MODEL_ALPHA)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions.tolist(), [_display_label(label) for label in primary_order])
    ax.grid(True, axis="y", alpha=0.25)
    _add_grouped_bar_mixed_legends(ax, secondary_order, palette, family_order)


def _single_bar_with_points(
    ax,
    summary: pd.DataFrame,
    category_order: tuple[str, ...],
    category_col: str,
    title: str,
    ylabel: str,
) -> None:
    category_order = tuple(item for item in category_order if item in set(summary[category_col].astype(str)))
    if not category_order:
        return

    x_positions = np.arange(len(category_order), dtype=np.float32)
    palette = {condition: f"C{index % 10}" for index, condition in enumerate(category_order)}
    means = []
    sds = []
    for condition in category_order:
        row = summary[summary[category_col] == condition]
        means.append(float(row["mean"].iloc[0]) if not row.empty else np.nan)
        sds.append(float(row["sd"].fillna(0.0).iloc[0]) if not row.empty else 0.0)
    for index, condition in enumerate(category_order):
        ax.errorbar(x_positions[index], means[index], yerr=sds[index], color=palette[condition], marker="o", markerfacecolor="none", markeredgecolor=palette[condition], linestyle="none", markersize=6, capsize=3, label=_display_label(condition), zorder=2, alpha=MODEL_ALPHA)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions.tolist(), [_display_label(label) for label in category_order])
    ax.grid(True, axis="y", alpha=0.25)


def _single_bar_with_family_points(
    ax,
    summary: pd.DataFrame,
    category_order: tuple[str, ...],
    category_col: str,
    title: str,
    ylabel: str,
) -> None:
    category_order = tuple(item for item in category_order if item in set(summary[category_col].astype(str)))
    family_order = _observed_family_order(summary)
    if not category_order or not family_order:
        return

    x_positions = np.arange(len(category_order), dtype=np.float32)
    offsets = np.linspace(-0.10 * (len(family_order) - 1) / 2, 0.10 * (len(family_order) - 1) / 2, num=len(family_order))
    palette = {condition: f"C{index % 10}" for index, condition in enumerate(category_order)}

    for offset, family in zip(offsets.tolist(), family_order):
        subset = summary[summary["family"].astype(str) == family].copy()
        if subset.empty:
            continue
        means = []
        sds = []
        for condition in category_order:
            row = subset[subset[category_col].astype(str) == condition]
            means.append(float(row["mean"].iloc[0]) if not row.empty else np.nan)
            sds.append(float(row["sd"].fillna(0.0).iloc[0]) if not row.empty else 0.0)

        style = FAMILY_STYLES[family]
        for index, condition in enumerate(category_order):
            ax.errorbar(x_positions[index] + offset, means[index], yerr=sds[index], color=palette[condition], marker=style["marker"], markerfacecolor="none", markeredgecolor=palette[condition], linestyle="none", markersize=6, capsize=3, label=f"{_display_label(condition)} {_display_family(family)}", zorder=2, alpha=MODEL_ALPHA)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions.tolist(), [_display_label(label) for label in category_order])
    ax.grid(True, axis="y", alpha=0.25)
    _add_family_only_legend(ax, family_order)


def _participant_summary(
    df: pd.DataFrame,
    metric_name: str,
    group_cols: list[str],
    include_family: bool,
) -> pd.DataFrame:
    if df.empty or metric_name not in df.columns:
        return pd.DataFrame()

    value_df = df.dropna(subset=[metric_name]).copy()
    if value_df.empty:
        return pd.DataFrame()

    family_cols = ["family"] if include_family else []
    grouping_cols = family_cols + list(group_cols)

    per_person = (
        value_df.groupby(grouping_cols + ["person"], sort=True, dropna=False)[metric_name]
        .mean()
        .reset_index(name="participant_value")
    )
    summary = (
        per_person.groupby(grouping_cols, sort=True, dropna=False)["participant_value"]
        .agg(
            mean="mean",
            sd=lambda series: float(series.std(ddof=1)) if len(series) > 1 else np.nan,
            n="count",
            participant_value=lambda series: list(map(float, series.tolist())),
        )
        .reset_index()
    )
    return summary


def _condition_order_for_cohort(cohort: str, observed_conditions: list[str]) -> tuple[str, ...]:
    observed = set(map(str, observed_conditions))
    if cohort == "clean":
        return tuple(condition for condition in CLEAN_ACTIVITY_ORDER if condition in observed)
    if cohort == "tremor_simulation":
        ordered = []
        for activity_family in GRAPH4B_ACTIVITY_ORDER:
            for state in GRAPH4B_STATE_ORDER:
                condition = f"{activity_family}_{state}"
                if condition in observed:
                    ordered.append(condition)
        return tuple(ordered)
    return tuple(condition for condition in EDGE_CASE_ORDER if condition in observed)


def _graph6_condition_label(row: pd.Series) -> str | None:
    cohort = str(row["cohort"])
    if cohort == "clean":
        return f"{row['speed_label']}_{row['activity_family']}"
    if cohort == "edge_cases" and str(row["activity_family"]) == "still":
        return "still"
    if cohort == "tremor_simulation":
        return f"{row['activity_family']}_{row['activity_state']}"
    return None


def _observed_family_order(df: pd.DataFrame) -> tuple[str, ...]:
    if "family" not in df.columns:
        return ()
    observed = set(df["family"].astype(str))
    return tuple(family for family in FAMILY_CHOICES if family in observed)


def _add_graph2_mixed_legends(ax, condition_order: tuple[str, ...], palette: dict[str, str], family_order: tuple[str, ...]) -> None:
    condition_handles = [
        Line2D([0], [0], color=palette[condition], linewidth=2.0, label=_display_label(condition))
        for condition in condition_order
    ]
    family_handles = [
        Line2D(
            [0],
            [0],
            color="#2a2a2a",
            linewidth=1.8,
            linestyle=FAMILY_STYLES[family]["linestyle"],
            marker=FAMILY_STYLES[family]["marker"],
            markersize=5,
            label=_display_family(family),
        )
        for family in family_order
    ]
    condition_legend = ax.legend(
        handles=condition_handles,
        title="Condition",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize="small",
        title_fontsize="small",
    )
    ax.add_artist(condition_legend)
    ax.legend(
        handles=family_handles,
        title="Family",
        loc="upper left",
        bbox_to_anchor=(1.01, 0.43),
        fontsize="small",
        title_fontsize="small",
    )


def _add_grouped_bar_mixed_legends(
    ax,
    secondary_order: tuple[str, ...],
    palette: dict[str, str],
    family_order: tuple[str, ...],
) -> None:
    condition_handles = [
        Patch(facecolor=palette[condition], alpha=0.72, label=_display_label(condition))
        for condition in secondary_order
    ]
    family_handles = [
        Patch(
            facecolor="#bfbfbf",
            alpha=FAMILY_STYLES[family]["alpha"],
            hatch=FAMILY_STYLES[family]["hatch"],
            edgecolor=FAMILY_STYLES[family]["edgecolor"],
            linewidth=0.9,
            label=_display_family(family),
        )
        for family in family_order
    ]
    condition_legend = ax.legend(
        handles=condition_handles,
        title="Condition",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize="small",
        title_fontsize="small",
    )
    ax.add_artist(condition_legend)
    ax.legend(
        handles=family_handles,
        title="Family",
        loc="upper left",
        bbox_to_anchor=(1.01, 0.43),
        fontsize="small",
        title_fontsize="small",
    )


def _add_family_only_legend(ax, family_order: tuple[str, ...]) -> None:
    family_handles = [
        Patch(
            facecolor="#bfbfbf",
            alpha=FAMILY_STYLES[family]["alpha"],
            hatch=FAMILY_STYLES[family]["hatch"],
            edgecolor=FAMILY_STYLES[family]["edgecolor"],
            linewidth=0.9,
            label=_display_family(family),
        )
        for family in family_order
    ]
    ax.legend(
        handles=family_handles,
        title="Family",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize="small",
        title_fontsize="small",
    )


def _with_mixed_suffix(title: str, mixed: bool) -> str:
    if not mixed:
        return title
    return f"{title} (WiLoR vs Stride)"


def _display_family(value: str) -> str:
    return FAMILY_DISPLAY.get(str(value), str(value).title())


def _display_label(value: str) -> str:
    text = str(value).replace("_", " ")
    return " ".join(token.capitalize() if token != "BE" else token for token in text.split())


if __name__ == "__main__":
    main()
