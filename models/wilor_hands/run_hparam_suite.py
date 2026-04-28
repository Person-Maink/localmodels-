from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from experiment_config import list_experiment_names, resolve_experiment_config
from launch_temporal_sweep import submit_experiments
from plot_finetune_losses import StageDefinition, StageResult, load_run_metrics, select_best_run


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EXPERIMENTS_ROOT = SCRIPT_DIR / "experiments"
DEFAULT_OUTPUTS_ROOT = SCRIPT_DIR.parent.parent / "outputs" / "wilor_finetune"
STATE_FILENAME = "sweep_state.json"
ACTIVE_SLURM_STATES = {
    "BOOT_FAIL",
    "CANCELLED+",
    "COMPLETING",
    "CONFIGURING",
    "PENDING",
    "PREEMPTED",
    "REQUEUED",
    "RESIZING",
    "RUNNING",
    "SIGNALING",
    "SUSPENDED",
    "STAGE_OUT",
    "STOPPED",
}
FAILURE_SLURM_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
}


@dataclass(frozen=True)
class SetupDefinition:
    name: str
    relative_path: Path
    setup_dir: Path
    run_dir: Path
    output_root: Path


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = json.loads(json.dumps(base))
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = json.loads(json.dumps(value))
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping YAML at {path}")
    return payload


def _dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _stage_from_path(path: Path) -> str:
    stem = path.stem
    if not stem.startswith("hparam_stage_"):
        raise ValueError(f"Unexpected stage config name: {path.name}")
    return stem.removeprefix("hparam_stage_").split("_", 1)[0].lower()


def _slugify(value: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "main"


def _is_run_workspace_path(experiments_root: Path, candidate: Path) -> bool:
    try:
        relative = candidate.relative_to(experiments_root)
    except ValueError:
        return False
    return "run" in relative.parts


def discover_setups(experiments_root: Path = DEFAULT_EXPERIMENTS_ROOT) -> list[SetupDefinition]:
    setup_dirs: set[Path] = set()
    for config_path in experiments_root.rglob("hparam_stage_*.yaml"):
        if _is_run_workspace_path(experiments_root, config_path):
            continue
        setup_dirs.add(config_path.parent)

    setups: list[SetupDefinition] = []
    for setup_dir in sorted(setup_dirs):
        relative_path = setup_dir.relative_to(experiments_root)
        name = "main" if relative_path == Path(".") else relative_path.as_posix()
        slug = _slugify(name)
        setups.append(
            SetupDefinition(
                name=name,
                relative_path=relative_path,
                setup_dir=setup_dir,
                run_dir=setup_dir / "run",
                output_root=DEFAULT_OUTPUTS_ROOT / slug,
            )
        )
    return setups


def find_setup(setup_name: str, experiments_root: Path = DEFAULT_EXPERIMENTS_ROOT) -> SetupDefinition:
    normalized = setup_name.strip().lower()
    for setup in discover_setups(experiments_root):
        if normalized in {setup.name.lower(), setup.relative_path.as_posix().lower(), _slugify(setup.name)}:
            return setup
    available = ", ".join(setup.name for setup in discover_setups(experiments_root))
    raise ValueError(f"Unknown setup '{setup_name}'. Available setups: {available}")


def list_stage_configs(setup_dir: Path) -> list[Path]:
    return sorted(
        setup_dir.glob("hparam_stage_*.yaml"),
        key=lambda path: (_stage_from_path(path), path.name),
    )


def initialize_run_workspace(setup: SetupDefinition, *, start_over: bool = False) -> list[Path]:
    if start_over and setup.run_dir.exists():
        shutil.rmtree(setup.run_dir)
    setup.run_dir.mkdir(parents=True, exist_ok=True)

    source_stage_paths = list_stage_configs(setup.setup_dir)
    if not source_stage_paths:
        raise ValueError(f"No stage configs were found in {setup.setup_dir}")

    copied_paths: list[Path] = []
    for source_path in source_stage_paths:
        target_path = setup.run_dir / source_path.name
        if start_over or not target_path.exists():
            shutil.copy2(source_path, target_path)
        copied_paths.append(target_path)
    return copied_paths


def load_state(run_dir: Path) -> dict[str, Any]:
    state_path = run_dir / STATE_FILENAME
    if not state_path.exists():
        return {"stages": {}}
    with open(state_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid state payload in {state_path}")
    payload.setdefault("stages", {})
    return payload


def save_state(run_dir: Path, state: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / STATE_FILENAME, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)


def apply_winner_to_next_stage(next_stage_path: Path, winner_resolved_path: Path) -> None:
    winner_payload = _load_yaml(winner_resolved_path)
    next_payload = _load_yaml(next_stage_path)
    next_defaults = next_payload.get("defaults", {})
    if next_defaults and not isinstance(next_defaults, dict):
        raise ValueError(f"'defaults' must be a mapping in {next_stage_path}")
    next_payload["defaults"] = _deep_merge(winner_payload, next_defaults or {})
    _dump_yaml(next_stage_path, next_payload)


def build_stage_result(stage_config_path: Path, runs_root: Path, suite_name: str) -> StageResult:
    run_names = list_experiment_names(stage_config_path)
    definition = StageDefinition(
        suite=suite_name,
        suite_slug=_slugify(suite_name),
        stage=_stage_from_path(stage_config_path),
        config_path=stage_config_path,
        run_names=run_names,
    )
    runs = []
    for run_name in run_names:
        run = load_run_metrics(
            suite_name,
            _slugify(suite_name),
            definition.stage,
            run_name,
            runs_root,
        )
        if run is not None:
            runs.append(run)
    return StageResult(definition=definition, runs=runs)


def run_has_usable_metrics(run_dir: Path) -> bool:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return False
    try:
        with open(metrics_path, "r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                if payload.get("split") == "val" and "loss_total" in payload and "step" in payload:
                    return True
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return False


def stage_is_complete(stage_config_path: Path, runs_root: Path) -> bool:
    return all(run_has_usable_metrics(runs_root / run_name) for run_name in list_experiment_names(stage_config_path))


def query_slurm_states(
    job_ids: list[str],
    *,
    squeue_bin: str = "squeue",
    sacct_bin: str = "sacct",
) -> dict[str, str]:
    if not job_ids:
        return {}

    normalized_ids = [job_id for job_id in job_ids if job_id]
    if not normalized_ids:
        return {}
    joined_ids = ",".join(normalized_ids)
    states: dict[str, str] = {}

    squeue_command = [squeue_bin, "-h", "-o", "%i|%T", "-j", joined_ids]
    try:
        completed = subprocess.run(
            squeue_command,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        completed = None
    if completed and completed.returncode == 0:
        for line in completed.stdout.splitlines():
            if "|" not in line:
                continue
            job_id, state = line.strip().split("|", 1)
            states[job_id.strip()] = state.strip().upper()

    sacct_command = [sacct_bin, "-n", "-P", "-j", joined_ids, "--format=JobIDRaw,State"]
    try:
        completed = subprocess.run(
            sacct_command,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return states
    if completed.returncode != 0:
        return states
    for line in completed.stdout.splitlines():
        if "|" not in line:
            continue
        raw_job_id, state = line.strip().split("|", 1)
        base_job_id = raw_job_id.split(".", 1)[0].strip()
        if base_job_id in normalized_ids and base_job_id not in states:
            states[base_job_id] = state.strip().upper()
    return states


def wait_for_stage_completion(
    stage_config_path: Path,
    runs_root: Path,
    job_ids: list[str],
    *,
    poll_seconds: int,
    squeue_bin: str = "squeue",
    sacct_bin: str = "sacct",
) -> None:
    expected_runs = list_experiment_names(stage_config_path)
    while True:
        states = query_slurm_states(job_ids, squeue_bin=squeue_bin, sacct_bin=sacct_bin)
        failed_jobs = {
            job_id: state
            for job_id, state in states.items()
            if state.split()[0] in FAILURE_SLURM_STATES
        }
        if failed_jobs:
            formatted = ", ".join(f"{job_id}={state}" for job_id, state in sorted(failed_jobs.items()))
            raise RuntimeError(f"Stage {_stage_from_path(stage_config_path).upper()} failed in SLURM: {formatted}")

        all_outputs_ready = all(run_has_usable_metrics(runs_root / run_name) for run_name in expected_runs)
        active_jobs = {
            job_id: state
            for job_id, state in states.items()
            if state.split()[0] in ACTIVE_SLURM_STATES
        }
        if all_outputs_ready and not active_jobs:
            return
        time.sleep(poll_seconds)


def _winner_resolved_path(best_run_name: str, runs_root: Path) -> Path:
    return runs_root / best_run_name / "resolved_experiment.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an entire hyperparameter tuning setup stage by stage."
    )
    parser.add_argument(
        "--setup",
        required=True,
        help="Setup name or relative subfolder under experiments/ (for example: main, lora, frozen wilor).",
    )
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=DEFAULT_EXPERIMENTS_ROOT,
        help="Root directory that contains setup folders and hparam stage YAMLs.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="Override the run output root. Defaults to a setup-specific folder under outputs/wilor_finetune.",
    )
    parser.add_argument(
        "--train-script",
        type=Path,
        default=SCRIPT_DIR / "train.sh",
        help="Path to the training wrapper script.",
    )
    parser.add_argument(
        "--sbatch-bin",
        type=str,
        default="sbatch",
        help="SLURM submission binary.",
    )
    parser.add_argument(
        "--squeue-bin",
        type=str,
        default="squeue",
        help="SLURM queue inspection binary.",
    )
    parser.add_argument(
        "--sacct-bin",
        type=str,
        default="sacct",
        help="SLURM accounting binary.",
    )
    parser.add_argument(
        "--poll-minutes",
        type=int,
        default=10,
        help="How often to poll SLURM and output artifacts.",
    )
    parser.add_argument(
        "--start-over",
        action="store_true",
        help="Recreate the setup-local run workspace and restart the suite from stage 1.",
    )
    parser.add_argument(
        "--continue",
        dest="resume",
        action="store_true",
        help="Resume from an existing setup-local run workspace if present.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Submit nothing; only print the stage submissions that would happen.",
    )
    return parser.parse_args(argv)


def run_suite(args: argparse.Namespace) -> None:
    setup = find_setup(args.setup, args.experiments_root)
    runs_root = args.runs_root.resolve() if args.runs_root else setup.output_root.resolve()
    run_stage_paths = initialize_run_workspace(setup, start_over=args.start_over)
    state = {} if args.start_over else load_state(setup.run_dir)
    state.setdefault("setup", setup.name)
    state.setdefault("runs_root", str(runs_root))
    state.setdefault("stages", {})
    save_state(setup.run_dir, state)

    for index, stage_path in enumerate(run_stage_paths):
        stage_key = _stage_from_path(stage_path)
        stage_state = state["stages"].setdefault(stage_key, {})

        if index > 0 and not stage_state:
            previous_stage_key = _stage_from_path(run_stage_paths[index - 1])
            previous_winner = state["stages"].get(previous_stage_key, {}).get("winner")
            if not previous_winner:
                raise RuntimeError(
                    f"Cannot initialize Stage {stage_key.upper()} without a winner from Stage "
                    f"{previous_stage_key.upper()}."
                )
            apply_winner_to_next_stage(stage_path, _winner_resolved_path(previous_winner, runs_root))

        if stage_state.get("status") == "completed" and stage_is_complete(stage_path, runs_root):
            continue

        if not stage_state.get("job_ids"):
            records = submit_experiments(
                stage_path,
                list_experiment_names(stage_path),
                args.train_script.resolve(),
                sbatch_bin=args.sbatch_bin,
                dry_run=args.dry_run,
                extra_env={"OUTPUT_ROOT": str(runs_root)},
            )
            stage_state["job_ids"] = [record.job_id for record in records if record.job_id]
            stage_state["status"] = "submitted"
            stage_state["experiments"] = [record.experiment_name for record in records]
            save_state(setup.run_dir, state)
            if args.dry_run:
                print(json.dumps([record.to_dict() for record in records], indent=2))
                return

        wait_for_stage_completion(
            stage_path,
            runs_root,
            stage_state.get("job_ids", []),
            poll_seconds=max(1, args.poll_minutes) * 60,
            squeue_bin=args.squeue_bin,
            sacct_bin=args.sacct_bin,
        )
        stage_result = build_stage_result(stage_path, runs_root, setup.name)
        best_run = select_best_run(stage_result)
        if best_run is None:
            raise RuntimeError(f"Stage {stage_key.upper()} completed without usable validation metrics.")
        stage_state["winner"] = best_run.name
        stage_state["status"] = "completed"
        save_state(setup.run_dir, state)

    print(f"Finished hyperparameter sweep for setup '{setup.name}'. Outputs root: {runs_root}")


def main() -> None:
    run_suite(parse_args())


if __name__ == "__main__":
    main()
