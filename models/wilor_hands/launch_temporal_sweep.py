import argparse
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from experiment_config import list_experiment_names, resolve_experiment_config
from finetune_runtime_estimator import estimate_runtime

SCRIPT_DIR = Path(__file__).resolve().parent
LOG_DIR = SCRIPT_DIR / "SLURM_logs"
SBATCH_JOB_ID_RE = re.compile(r"Submitted batch job (\d+)")


@dataclass(frozen=True)
class SubmissionRecord:
    experiment_name: str
    loss_config: str
    train_mode: str
    walltime: str
    walltime_source: str
    command: list[str]
    dry_run: bool
    job_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch WiLoR temporal ablation experiments through train.sh."
    )
    parser.add_argument(
        "--loss-config",
        type=str,
        default="experiments/temporal_ablations.yaml",
        help="Path to the temporal ablation YAML file.",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        dest="experiments",
        default=[],
        help="Select one or more named experiments. Defaults to all experiments in the config.",
    )
    parser.add_argument(
        "--train-script",
        type=str,
        default="train.sh",
        help="Path to the SLURM training wrapper script.",
    )
    parser.add_argument(
        "--sbatch-bin",
        type=str,
        default="sbatch",
        help="SLURM submission binary.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would be submitted instead of calling sbatch.",
    )
    parser.add_argument(
        "--output-format",
        choices=["text", "json"],
        default="text",
        help="How submission results should be printed.",
    )
    return parser


def resolve_launcher_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.resolve()
    cwd_candidate = path.resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (SCRIPT_DIR / path).resolve()


def submit_experiments(
    loss_config: Path,
    experiment_names: list[str],
    train_script: Path,
    *,
    sbatch_bin: str = "sbatch",
    dry_run: bool = False,
    extra_env: dict[str, str] | None = None,
) -> list[SubmissionRecord]:
    records: list[SubmissionRecord] = []
    for experiment_name in experiment_names:
        resolved = resolve_experiment_config(loss_config, experiment_name)
        estimate = estimate_runtime(resolved, LOG_DIR)
        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)
        env["LOSS_CONFIG"] = str(loss_config)
        env["EXPERIMENT_NAME"] = experiment_name
        command = [sbatch_bin, f"--time={estimate.time_str}", str(train_script)]

        job_id: str | None = None
        if not dry_run:
            completed = subprocess.run(
                command,
                check=True,
                env=env,
                capture_output=True,
                text=True,
            )
            stdout = completed.stdout.strip()
            stderr = completed.stderr.strip()
            combined = "\n".join(part for part in (stdout, stderr) if part)
            match = SBATCH_JOB_ID_RE.search(combined)
            if match is not None:
                job_id = match.group(1)

        records.append(
            SubmissionRecord(
                experiment_name=experiment_name,
                loss_config=str(loss_config),
                train_mode=str(resolved["train_mode"]),
                walltime=estimate.time_str,
                walltime_source=estimate.source,
                command=command,
                dry_run=dry_run,
                job_id=job_id,
            )
        )
    return records


def main() -> None:
    args = make_argparser().parse_args()
    loss_config = resolve_launcher_path(args.loss_config)
    train_script = resolve_launcher_path(args.train_script)

    if not loss_config.exists():
        raise FileNotFoundError(f"Loss config not found: {loss_config}")
    if not train_script.exists():
        raise FileNotFoundError(f"Train script not found: {train_script}")

    selected_experiments = args.experiments or list_experiment_names(loss_config)
    if not selected_experiments:
        raise ValueError(f"No experiments found in {loss_config}")

    records = submit_experiments(
        loss_config,
        selected_experiments,
        train_script,
        sbatch_bin=args.sbatch_bin,
        dry_run=args.dry_run,
    )
    if args.output_format == "json":
        print(json.dumps([record.to_dict() for record in records], indent=2))
        return
    for record in records:
        if record.dry_run:
            print(
                f"DRY_RUN: LOSS_CONFIG={record.loss_config} "
                f"EXPERIMENT_NAME={record.experiment_name} "
                f"TRAIN_MODE={record.train_mode} TIME={record.walltime} "
                f"({record.walltime_source}) {' '.join(record.command)}"
            )
            continue
        job_suffix = f" job_id={record.job_id}" if record.job_id else ""
        print(
            f"Submitted experiment '{record.experiment_name}' "
            f"(train_mode={record.train_mode}, time={record.walltime}, "
            f"basis={record.walltime_source}) via {train_script}{job_suffix}"
        )


if __name__ == "__main__":
    main()
