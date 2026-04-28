# WiLoR Hands

This directory contains the WiLoR inference and finetuning workflow used for the
temporal hand-training experiments in this repo. The most important pieces for
hyperparameter work are:

- [train.sh](/home/mayank/Documents/Uni/TUD/Thesis%20Extra/comparative%20study/models/wilor_hands/train.sh): SLURM-friendly training entrypoint
- [experiment_config.py](/home/mayank/Documents/Uni/TUD/Thesis%20Extra/comparative%20study/models/wilor_hands/experiment_config.py): resolves YAML experiment configs into concrete training settings
- [launch_temporal_sweep.py](/home/mayank/Documents/Uni/TUD/Thesis%20Extra/comparative%20study/models/wilor_hands/launch_temporal_sweep.py): submits all experiments from one stage config
- [run_hparam_suite.py](/home/mayank/Documents/Uni/TUD/Thesis%20Extra/comparative%20study/models/wilor_hands/run_hparam_suite.py): runs an entire multi-stage setup end to end
- [plot_finetune_losses.py](/home/mayank/Documents/Uni/TUD/Thesis%20Extra/comparative%20study/models/wilor_hands/plot_finetune_losses.py): compares validation curves and picks the best run per stage

## Experiment Layout

Hyperparameter sweeps live under [experiments](/home/mayank/Documents/Uni/TUD/Thesis%20Extra/comparative%20study/models/wilor_hands/experiments).

- The root folder is the `main` setup.
- Any subfolder that contains `hparam_stage_*.yaml` is treated as a separate setup.
- Current examples include `lora` and `frozen wilor`.
- New setups can be added just by creating another subfolder with stage YAMLs.

Typical files:

- `hparam_stage_*.yaml`: one tuning stage
- `temporal_ablations.yaml`: a standalone config for manual testing or one-off runs
- `README.md`: setup-specific notes such as stage ordering or scope differences

The setup-specific docs are:

- [experiments/lora/README.md](/home/mayank/Documents/Uni/TUD/Thesis%20Extra/comparative%20study/models/wilor_hands/experiments/lora/README.md)
- [experiments/frozen wilor/README.md](</home/mayank/Documents/Uni/TUD/Thesis Extra/comparative study/models/wilor_hands/experiments/frozen wilor/README.md>)

## YAML Format

Stage and ablation configs use the same basic schema:

- `defaults`: shared settings merged into every experiment
- `experiments`: a list or mapping of named overrides

The full option reference is documented in:

- [experiments/temporal_ablations_options.md](/home/mayank/Documents/Uni/TUD/Thesis%20Extra/comparative%20study/models/wilor_hands/experiments/temporal_ablations_options.md)
- [experiments/lora/temporal_ablations_options.md](/home/mayank/Documents/Uni/TUD/Thesis%20Extra/comparative%20study/models/wilor_hands/experiments/lora/temporal_ablations_options.md)

You can inspect or resolve a config directly:

```bash
python3 wilor_hands/experiment_config.py list \
  --loss-config wilor_hands/experiments/hparam_stage_a_windows.yaml

python3 wilor_hands/experiment_config.py resolve \
  --loss-config wilor_hands/experiments/hparam_stage_a_windows.yaml \
  --experiment-name hp_a_ws3_s2_b8 \
  --format json
```

`train.sh` uses the same resolver internally when `LOSS_CONFIG` and
`EXPERIMENT_NAME` are provided.

## Old Workflow: Run One Stage

Use [launch_temporal_sweep.py](/home/mayank/Documents/Uni/TUD/Thesis%20Extra/comparative%20study/models/wilor_hands/launch_temporal_sweep.py)
when you want to submit every experiment from a single stage YAML.

Examples:

```bash
python3 wilor_hands/launch_temporal_sweep.py \
  --loss-config wilor_hands/experiments/hparam_stage_a_windows.yaml

python3 wilor_hands/launch_temporal_sweep.py \
  --loss-config wilor_hands/experiments/lora/hparam_stage_h_lora_adapters.yaml \
  --experiment lora_hp_h_qkv_last8_r8 \
  --experiment lora_hp_h_qkv_last8_r16
```

Useful flags:

- `--dry-run`: print the `sbatch` commands without submitting
- `--sbatch-bin`: override the submission binary
- `--output-format json`: emit structured submission records, including parsed job IDs when available

Dry-run example:

```bash
python3 wilor_hands/launch_temporal_sweep.py \
  --loss-config wilor_hands/experiments/hparam_stage_b_optimizer.yaml \
  --dry-run
```

## New Workflow: Run A Whole Setup

Use [run_hparam_suite.py](/home/mayank/Documents/Uni/TUD/Thesis%20Extra/comparative%20study/models/wilor_hands/run_hparam_suite.py)
to run an entire setup stage by stage.

What it does:

- discovers setups dynamically from `experiments/`
- copies the selected setup’s `hparam_stage_*.yaml` files into a local `run/` subfolder
- submits every experiment in the first stage
- polls SLURM every 10 minutes by default
- waits for both SLURM completion and usable `metrics.jsonl` output
- selects the best run using lowest best validation `loss_total`
- copies the winner’s resolved config into the next stage’s `defaults`
- repeats until the final stage is complete

Examples:

```bash
python3 wilor_hands/run_hparam_suite.py --setup main --start-over

python3 wilor_hands/run_hparam_suite.py --setup lora --start-over

python3 wilor_hands/run_hparam_suite.py --setup "frozen wilor" --start-over
```

Resume behavior:

- `--start-over` recreates the setup-local `run/` workspace and restarts from Stage A
- rerunning without `--start-over` reuses the existing `run/` copies and saved state
- `--continue` is accepted as an explicit resume-style invocation for readability

Examples:

```bash
python3 wilor_hands/run_hparam_suite.py --setup lora --continue

python3 wilor_hands/run_hparam_suite.py --setup "frozen wilor"
```

Useful flags:

- `--poll-minutes 10`: SLURM polling interval
- `--runs-root <path>`: override the output root for run directories
- `--train-script <path>`: alternate training wrapper
- `--sbatch-bin`, `--squeue-bin`, `--sacct-bin`: alternate SLURM binaries
- `--dry-run`: print the first stage submissions that would happen and exit

## Run Workspace And Outputs

Each setup keeps its editable execution copies inside its own `run/` folder.

Examples:

- `wilor_hands/experiments/run/`
- `wilor_hands/experiments/lora/run/`
- `wilor_hands/experiments/frozen wilor/run/`

These `run/` folders contain:

- copied stage YAMLs that are safe to mutate during orchestration
- `sweep_state.json` with stage status, job IDs, and winners

The original checked-in YAMLs under `experiments/` remain untouched.

Training outputs are written under setup-specific folders in
`outputs/wilor_finetune` so suites with overlapping experiment names do not
collide. For example:

- `outputs/wilor_finetune/main/`
- `outputs/wilor_finetune/lora/`
- `outputs/wilor_finetune/frozen_wilor/`

Each run directory typically contains:

- `metrics.jsonl`
- `resolved_experiment.yaml`
- `best.ckpt`
- `latest.ckpt`
- detection cache files

## How “Best Run” Is Chosen

The stage winner uses the same logic as [plot_finetune_losses.py](/home/mayank/Documents/Uni/TUD/Thesis%20Extra/comparative%20study/models/wilor_hands/plot_finetune_losses.py):

- look only at validation rows in `metrics.jsonl`
- use the minimum validation `loss_total` reached by each run
- break ties by earlier step
- break remaining ties by the stage YAML experiment order

The winning run’s `resolved_experiment.yaml` becomes the base for the next stage
in the setup-local `run/` copy.

## Plotting And Review

Generate validation plots for all discovered stages:

```bash
python3 wilor_hands/plot_finetune_losses.py
```

Filter to one setup or stage:

```bash
python3 wilor_hands/plot_finetune_losses.py --suite lora

python3 wilor_hands/plot_finetune_losses.py --suite "frozen wilor" --stage j
```

By default this reads stage configs from
[experiments](/home/mayank/Documents/Uni/TUD/Thesis%20Extra/comparative%20study/models/wilor_hands/experiments)
and run outputs from `outputs/wilor_finetune`.

## Manual Training

If you want to bypass the sweep helpers and run one experiment directly, export
`LOSS_CONFIG` and `EXPERIMENT_NAME` into [train.sh](/home/mayank/Documents/Uni/TUD/Thesis%20Extra/comparative%20study/models/wilor_hands/train.sh):

```bash
LOSS_CONFIG=wilor_hands/experiments/temporal_ablations.yaml \
EXPERIMENT_NAME=test_temporal_all_losses \
bash wilor_hands/train.sh
```

`train.sh` resolves the config, builds the python command, and writes the run
artifacts to `outputs/wilor_finetune/<run_name>` unless `OUTPUT_ROOT` or
`RUN_OUTPUT_DIR` is overridden.

## Notes

- Full-suite orchestration expects SLURM tools such as `sbatch`, `squeue`, and `sacct`.
- Stage completion requires both finished SLURM jobs and valid validation metrics.
- If a job fails or never produces usable validation metrics, the suite runner stops instead of silently picking from partial results.
