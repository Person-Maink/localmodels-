import argparse
import os
import subprocess
from pathlib import Path

from experiment_config import list_experiment_names, resolve_experiment_config
from finetune_runtime_estimator import estimate_runtime

SCRIPT_DIR = Path(__file__).resolve().parent
LOG_DIR = SCRIPT_DIR / "SLURM_logs"


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
    return parser


def resolve_launcher_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.resolve()
    cwd_candidate = path.resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (SCRIPT_DIR / path).resolve()


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

    for experiment_name in selected_experiments:
        resolved = resolve_experiment_config(loss_config, experiment_name)
        estimate = estimate_runtime(resolved, LOG_DIR)
        env = os.environ.copy()
        env["LOSS_CONFIG"] = str(loss_config)
        env["EXPERIMENT_NAME"] = experiment_name

        command = [args.sbatch_bin, f"--time={estimate.time_str}", str(train_script)]
        if args.dry_run:
            print(
                f"DRY_RUN: LOSS_CONFIG={loss_config} EXPERIMENT_NAME={experiment_name} "
                f"TRAIN_MODE={resolved['train_mode']} TIME={estimate.time_str} "
                f"({estimate.source}) {args.sbatch_bin} --time={estimate.time_str} {train_script}"
            )
            continue

        subprocess.run(command, check=True, env=env)
        print(
            f"Submitted experiment '{experiment_name}' "
            f"(train_mode={resolved['train_mode']}, time={estimate.time_str}, "
            f"basis={estimate.source}) via {train_script}"
        )


if __name__ == "__main__":
    main()
