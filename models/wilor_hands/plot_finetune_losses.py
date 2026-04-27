from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


METRIC_NAME = "loss_total"
RUN_LINE_COLOR = "#2a6f97"
OTHER_RUN_COLORS = [
    "#4c78a8",
    "#72b7b2",
    "#54a24b",
    "#e45756",
    "#b279a2",
    "#ff9da6",
    "#9d755d",
    "#bab0ac",
]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EXPERIMENTS_ROOT = SCRIPT_DIR / "experiments"
DEFAULT_RUNS_ROOT = SCRIPT_DIR.parent.parent / "outputs" / "wilor_finetune"
DEFAULT_OUTPUT_ROOT = (
    SCRIPT_DIR.parent.parent / "analysis" / "analysis_images" / "wilor_finetune_plots"
)


def _import_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    return plt


@dataclass(frozen=True)
class StageDefinition:
    suite: str
    suite_slug: str
    stage: str
    config_path: Path
    run_names: list[str]


@dataclass(frozen=True)
class RunMetrics:
    suite: str
    suite_slug: str
    stage: str
    name: str
    run_dir: Path
    metrics_path: Path
    resolved_experiment_path: Path | None
    validation_steps: list[int]
    validation_losses: list[float]
    config_summary: str

    @property
    def has_validation(self) -> bool:
        return bool(self.validation_steps)

    @property
    def best_validation(self) -> tuple[int, float] | None:
        if not self.validation_losses:
            return None
        best_index = min(
            range(len(self.validation_losses)),
            key=lambda idx: (self.validation_losses[idx], self.validation_steps[idx]),
        )
        return self.validation_steps[best_index], self.validation_losses[best_index]


@dataclass(frozen=True)
class StageResult:
    definition: StageDefinition
    runs: list[RunMetrics]

    @property
    def validation_runs(self) -> list[RunMetrics]:
        return [run for run in self.runs if run.has_validation]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate validation-loss figures for WiLoR finetune runs. "
            "By default, this discovers all stage configs, creates one plot per run, "
            "and creates one comparison plot per stage."
        )
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
        help="Directory containing local finetune run folders.",
    )
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=DEFAULT_EXPERIMENTS_ROOT,
        help="Directory containing hparam_stage_*.yaml files.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        help="Optional stage filter such as 'a' or 'B'. Defaults to all stages.",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default=None,
        help=(
            "Optional suite filter such as 'main', 'lora', or 'frozen wilor'. "
            "Defaults to every suite under experiments_root."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where generated plot files should be written.",
    )
    return parser.parse_args(argv)


def _stage_from_config_name(config_path: Path) -> str:
    stem = config_path.stem
    if not stem.startswith("hparam_stage_"):
        raise ValueError(f"Unexpected stage config name: {config_path.name}")
    return stem.removeprefix("hparam_stage_").split("_", 1)[0].lower()


def _slugify_suite_name(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "main"


def _suite_from_config_path(experiments_root: Path, config_path: Path) -> tuple[str, str]:
    relative_parent = config_path.parent.relative_to(experiments_root)
    if relative_parent == Path("."):
        return "main", "main"
    suite = str(relative_parent).replace("\\", "/")
    return suite, _slugify_suite_name(suite)


def _matches_suite_filter(suite: str, suite_slug: str, suite_filter: str | None) -> bool:
    if suite_filter is None:
        return True
    normalized = suite_filter.strip().lower()
    return normalized in {suite.lower(), suite_slug.lower()}


def _format_suite_label(suite: str) -> str:
    if suite == "main":
        return "Main"
    words = []
    for token in suite.replace("/", " / ").split():
        lowered = token.lower()
        if lowered == "lora":
            words.append("LoRA")
        elif lowered == "wilor":
            words.append("WiLoR")
        else:
            words.append(token.title())
    return " ".join(words)


def _format_stage_label(stage: str, suite: str) -> str:
    suite_label = _format_suite_label(suite)
    if suite == "main":
        return f"Stage {stage.upper()}"
    return f"{suite_label} Stage {stage.upper()}"


def _normalize_stage_experiments(payload: dict[str, Any]) -> list[dict[str, Any]]:
    experiments = payload.get("experiments", [])
    if isinstance(experiments, dict):
        normalized = []
        for name, value in experiments.items():
            entry = dict(value or {})
            entry.setdefault("name", name)
            normalized.append(entry)
        return normalized
    if not isinstance(experiments, list):
        raise ValueError("'experiments' must be a list or mapping.")
    normalized = []
    for entry in experiments:
        if not isinstance(entry, dict):
            raise ValueError("Each experiment entry must be a mapping.")
        if not entry.get("name"):
            raise ValueError("Each experiment entry must define a 'name'.")
        normalized.append(dict(entry))
    return normalized


def discover_stage_definitions(
    experiments_root: Path,
    stage_filter: str | None = None,
    suite_filter: str | None = None,
) -> list[StageDefinition]:
    normalized_filter = stage_filter.lower() if stage_filter else None
    stage_definitions: list[StageDefinition] = []
    for config_path in sorted(experiments_root.rglob("hparam_stage_*.yaml")):
        stage = _stage_from_config_name(config_path)
        if normalized_filter and stage != normalized_filter:
            continue
        suite, suite_slug = _suite_from_config_path(experiments_root, config_path)
        if not _matches_suite_filter(suite, suite_slug, suite_filter):
            continue
        with open(config_path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Stage config must be a mapping: {config_path}")
        experiments = _normalize_stage_experiments(payload)
        run_names = [str(entry["name"]) for entry in experiments]
        stage_definitions.append(
            StageDefinition(
                suite=suite,
                suite_slug=suite_slug,
                stage=stage,
                config_path=config_path,
                run_names=run_names,
            )
        )
    return sorted(
        stage_definitions,
        key=lambda definition: (
            definition.suite_slug,
            definition.stage,
            definition.config_path.name,
        ),
    )


def load_metrics_rows(metrics_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(metrics_path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Metrics row must decode to a mapping: {metrics_path}:{line_number}"
                )
            rows.append(payload)
    return rows


def extract_validation_series(rows: list[dict[str, Any]]) -> tuple[list[int], list[float]]:
    validation_by_step: dict[int, float] = {}
    for row in rows:
        if row.get("split") != "val":
            continue
        if "step" not in row or METRIC_NAME not in row:
            continue
        validation_by_step[int(row["step"])] = float(row[METRIC_NAME])
    sorted_points = sorted(validation_by_step.items())
    steps = [step for step, _ in sorted_points]
    losses = [loss for _, loss in sorted_points]
    return steps, losses


def _format_number(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric == 0:
        return "0"
    if abs(numeric) < 1.0e-3 or abs(numeric) >= 1.0e3:
        return f"{numeric:.1e}"
    text = f"{numeric:.6f}".rstrip("0").rstrip(".")
    return text


def build_config_summary(
    stage: str,
    suite: str,
    resolved_experiment_path: Path | None,
) -> str:
    parts = [_format_stage_label(stage, suite)]
    if resolved_experiment_path is None or not resolved_experiment_path.exists():
        return parts[0]

    with open(resolved_experiment_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        return parts[0]

    temporal_cfg = payload.get("temporal") or {}
    optimizer_cfg = payload.get("optimizer") or {}
    if temporal_cfg.get("window_size") is not None:
        parts.append(f"ws={temporal_cfg['window_size']}")
    if temporal_cfg.get("window_stride") is not None:
        parts.append(f"stride={temporal_cfg['window_stride']}")
    if payload.get("batch_size") is not None:
        parts.append(f"batch={payload['batch_size']}")
    if optimizer_cfg.get("lr") is not None:
        parts.append(f"lr={_format_number(optimizer_cfg['lr'])}")
    if optimizer_cfg.get("weight_decay") is not None:
        parts.append(f"wd={_format_number(optimizer_cfg['weight_decay'])}")
    if payload.get("train_scope"):
        parts.append(f"scope={payload['train_scope']}")
    lora_cfg = payload.get("lora") or {}
    if lora_cfg.get("enabled"):
        parts.append(
            "lora="
            f"{','.join(lora_cfg.get('target_modules') or ['?'])}:"
            f"r{lora_cfg.get('rank', '?')}"
            f"@{lora_cfg.get('block_start', '?')}-{lora_cfg.get('block_end', '?')}"
        )
    losses_cfg = payload.get("losses") or {}
    vipe_cfg = losses_cfg.get("vipe_camera") or {}
    if vipe_cfg.get("enabled") and vipe_cfg.get("weight") is not None:
        parts.append(f"vipe={_format_number(vipe_cfg['weight'])}")
    temporal_vipe_cfg = losses_cfg.get("temporal_vipe_camera") or {}
    if temporal_vipe_cfg.get("enabled"):
        parts.append(
            "tvc="
            f"{_format_number(temporal_vipe_cfg.get('weight', 0.0))}/"
            f"{_format_number(temporal_vipe_cfg.get('smoothness_weight', 0.0))}/"
            f"{_format_number(temporal_vipe_cfg.get('anchor_weight', 0.0))}"
        )
    return " | ".join(parts)


def load_run_metrics(
    suite: str,
    suite_slug: str,
    stage: str,
    run_name: str,
    runs_root: Path,
) -> RunMetrics | None:
    run_dir = runs_root / run_name
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return None
    resolved_experiment_path = run_dir / "resolved_experiment.yaml"
    if not resolved_experiment_path.exists():
        resolved_experiment_path = None

    steps, losses = extract_validation_series(load_metrics_rows(metrics_path))
    return RunMetrics(
        suite=suite,
        suite_slug=suite_slug,
        stage=stage,
        name=run_name,
        run_dir=run_dir,
        metrics_path=metrics_path,
        resolved_experiment_path=resolved_experiment_path,
        validation_steps=steps,
        validation_losses=losses,
        config_summary=build_config_summary(stage, suite, resolved_experiment_path),
    )


def build_stage_results(
    experiments_root: Path,
    runs_root: Path,
    stage_filter: str | None = None,
    suite_filter: str | None = None,
) -> list[StageResult]:
    results: list[StageResult] = []
    for definition in discover_stage_definitions(experiments_root, stage_filter, suite_filter):
        runs = [
            run
            for run_name in definition.run_names
            if (
                run := load_run_metrics(
                    definition.suite,
                    definition.suite_slug,
                    definition.stage,
                    run_name,
                    runs_root,
                )
            )
            is not None
        ]
        results.append(StageResult(definition=definition, runs=runs))
    return results


def select_best_run(stage_result: StageResult) -> RunMetrics | None:
    validation_runs = stage_result.validation_runs
    if not validation_runs:
        return None
    order = {name: index for index, name in enumerate(stage_result.definition.run_names)}
    return min(
        validation_runs,
        key=lambda run: (
            run.best_validation[1],
            run.best_validation[0],
            order.get(run.name, len(order)),
        ),
    )


def plot_run_validation(run: RunMetrics, output_root: Path) -> Path:
    plt = _import_pyplot()
    output_path = (
        output_root / "runs" / run.suite_slug / run.name / "validation_loss_vs_step.svg"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.suptitle(
        f"{run.name} Validation Loss vs Step",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_title(run.config_summary, fontsize=10, color="#555555", pad=10)
    ax.set_xlabel("Step")
    ax.set_ylabel(f"Validation {METRIC_NAME}")
    ax.grid(True, alpha=0.3)

    if run.has_validation:
        ax.plot(
            run.validation_steps,
            run.validation_losses,
            color=RUN_LINE_COLOR,
            linewidth=2.2,
            label="validation",
        )
        best_step, best_loss = run.best_validation
        ax.scatter(
            [best_step],
            [best_loss],
            color=RUN_LINE_COLOR,
            s=60,
            zorder=3,
            label=f"best {best_loss:.4f} @ step {best_step}",
        )
        ax.legend()
    else:
        ax.text(
            0.5,
            0.5,
            "No validation metrics recorded",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color="#666666",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_stage_comparison(stage_result: StageResult, output_root: Path) -> Path:
    plt = _import_pyplot()
    output_path = (
        output_root
        / "stages"
        / stage_result.definition.suite_slug
        / f"stage_{stage_result.definition.stage}_validation_comparison.svg"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    stage_label = _format_stage_label(
        stage_result.definition.stage,
        stage_result.definition.suite,
    )
    best_run = select_best_run(stage_result)
    validation_runs = stage_result.validation_runs

    ax.set_xlabel("Step")
    ax.set_ylabel(f"Validation {METRIC_NAME}")
    ax.grid(True, alpha=0.3)

    if validation_runs:
        for index, run in enumerate(validation_runs):
            is_best = best_run is not None and run.name == best_run.name
            color = OTHER_RUN_COLORS[index % len(OTHER_RUN_COLORS)]
            ax.plot(
                run.validation_steps,
                run.validation_losses,
                label=f"{run.name} (best)" if is_best else run.name,
                color=color,
                linewidth=1.8,
                alpha=0.9,
            )
        if best_run is not None:
            best_step, best_loss = best_run.best_validation
            subtitle = (
                f"Best validation run: {best_run.name} | best val {best_loss:.4f} @ step {best_step}"
            )
        else:
            subtitle = f"Color-coded validation curves for {len(validation_runs)} run(s)"
        ax.legend()
    else:
        subtitle = "No comparable validation curves were available"
        ax.text(
            0.5,
            0.5,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color="#666666",
        )

    fig.suptitle(
        f"{stage_label} Validation Loss Comparison",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_title(subtitle, fontsize=10, color="#555555", pad=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_all_figures(
    experiments_root: Path,
    runs_root: Path,
    output_root: Path,
    stage_filter: str | None = None,
    suite_filter: str | None = None,
) -> list[tuple[StageResult, Path, list[Path]]]:
    stage_results = build_stage_results(
        experiments_root,
        runs_root,
        stage_filter,
        suite_filter,
    )
    if not stage_results:
        raise ValueError(
            f"No stage configs were discovered in {experiments_root}"
            + (
                f" for stage '{stage_filter}' and suite '{suite_filter}'."
                if stage_filter and suite_filter
                else f" for stage '{stage_filter}'."
                if stage_filter
                else f" for suite '{suite_filter}'."
                if suite_filter
                else "."
            )
        )

    generated_outputs: list[tuple[StageResult, Path, list[Path]]] = []
    for stage_result in stage_results:
        run_outputs = [plot_run_validation(run, output_root) for run in stage_result.runs]
        stage_output = plot_stage_comparison(stage_result, output_root)
        generated_outputs.append((stage_result, stage_output, run_outputs))
    return generated_outputs


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    outputs = generate_all_figures(
        experiments_root=args.experiments_root.expanduser().resolve(),
        runs_root=args.runs_root.expanduser().resolve(),
        output_root=args.output_root.expanduser().resolve(),
        stage_filter=args.stage,
        suite_filter=args.suite,
    )

    for stage_result, stage_output, run_outputs in outputs:
        best_run = select_best_run(stage_result)
        best_text = ""
        if best_run is not None:
            best_step, best_loss = best_run.best_validation
            best_text = f" best={best_run.name} best_val={best_loss:.4f} step={best_step}"
        print(
            f"{_format_stage_label(stage_result.definition.stage, stage_result.definition.suite)}: "
            f"runs={len(stage_result.runs)} per_run_plots={len(run_outputs)} "
            f"report={stage_output}{best_text}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
