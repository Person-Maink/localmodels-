# Temporal Ablations Config Options

This file documents the experiment-config schema used by
[temporal_ablations.yaml](/home/mayank/Documents/Uni/TUD/Thesis Extra/comparative study/models/wilor_hands/experiments/temporal_ablations.yaml).

The config has two top-level sections:

- `defaults`: shared settings merged into every experiment
- `experiments`: a list of named experiment overrides

## Top-Level Experiment Keys

### `name`
- Type: `string`
- Required in each experiment entry
- Example: `temporal_camera_only`
- Used as the experiment identifier when resolving configs and, by default, as the run name in the training wrapper.
- Give each experiment a unique name so its outputs are easy to trace back to the exact setting.

### `run_name_suffix`
- Type: `string`
- Optional
- Any string is allowed
- A lightweight label you can use to make related experiments easier to group or compare.
- This is mainly for naming and organization; it does not change model behavior by itself.

### `train_mode`
- Type: `string`
- Allowed values:
  - `distill`
  - `test`
- `distill` is the normal training mode for actual finetuning runs, while `test` is a short smoke-test mode with smaller defaults.
- Use `test` when you want to verify that a configuration runs end-to-end before spending time on a full experiment.

### `videos`
- Type: `list[string]` or `string`
- Meaning: explicitly chosen video names
- Example:
```yaml
videos:
  - clip_2
  - clip_4
```
- Use this when you want a controlled subset instead of letting the pipeline discover everything available.
- This is especially useful for debugging, quick sweeps, or reproducing a run on a fixed set of videos.

### `all_videos`
- Type: `bool`
- Allowed values:
  - `true`
  - `false`
- Note: cannot be set together with `videos`
- This tells the pipeline to train on every eligible discovered video rather than a manually chosen subset.
- It is convenient for large training runs, but less ideal for targeted debugging because it makes runs slower and harder to compare.

### `train_scope`
- Type: `string`
- Allowed values:
  - `camera_head`
  - `refine_net`
  - `full`
- This controls which part of the WiLoR model is allowed to update during finetuning.
- Smaller scopes are cheaper and safer for early tuning, while `full` is the most flexible but also the easiest to destabilize.

### `validation_split`
- Type: `float`
- Allowed values: any float in `[0.0, 1.0)`
- Meaning: fraction of unique frames held out for validation
- This is your main mechanism for getting a validation signal without needing a separate dataset split file.
- If it is too high relative to your available frames and temporal window settings, you can end up with no usable validation windows.

### `sample_limit`
- Type: `int`
- Allowed values: any integer
- Common values:
  - `0` means no explicit limit
  - positive integer limits sample count
- This caps how many frame paths are used before detector samples and temporal windows are built.
- It is mainly useful for debugging and fast sweeps; for real finetuning you usually want `0` so the run can use all available frames.

### `detection_conf`
- Type: `float`
- Allowed values: any float
- Typical range: `0.0` to `1.0`
- This is the YOLO confidence threshold used when collecting hand detections from frames.
- Lower values give you more candidate detections but can add noise, while higher values are stricter but may drop valid hands.

### `rescale_factor`
- Type: `float`
- Allowed values: any positive float
- This affects how much context is kept around each detected hand crop before it is passed to WiLoR.
- Larger values include more surrounding image content, which can help context but may also dilute the hand region.

### `batch_size`
- Type: `int`
- Allowed values: any positive integer
- Meaning: temporal windows per optimization step
- This is one of the main memory-pressure knobs because each batch item is itself a temporal window, not just a single frame.
- If you hit GPU OOM, this is usually one of the first settings to reduce.

### `num_workers`
- Type: `int`
- Allowed values: any non-negative integer
- This controls how many worker processes the PyTorch dataloader uses to prepare batches.
- More workers can improve throughput, but on small jobs or constrained nodes they can also add overhead or instability.

### `max_steps`
- Type: `int`
- Allowed values: any positive integer
- Training is step-based in this pipeline, so this directly controls how many optimizer updates the run performs.
- A useful pattern is to keep this small for smoke tests and tuning, then raise it once the configuration looks stable.

### `log_every`
- Type: `int`
- Allowed values: any positive integer
- This controls how often training metrics are printed and appended to `metrics.jsonl`.
- Smaller values make debugging easier, while larger values reduce log volume for long runs.

### `save_every`
- Type: `int`
- Allowed values: any positive integer
- This controls how often `latest.ckpt` is refreshed during training.
- Lower values are safer for debugging and preemptible jobs, but they also create more I/O overhead.

### `seed`
- Type: `int`
- Allowed values: any integer
- This seeds the random components of sampling and splitting so runs are easier to reproduce.
- Keep it fixed during tuning if you want cleaner comparisons across experiments.

## `optimizer`

### `optimizer.lr`
- Type: `float`
- Allowed values: any positive float
- This is usually the most sensitive optimization hyperparameter and often has the largest effect on stability.
- If loss barely moves or explodes, `lr` is one of the first things to revisit.

### `optimizer.weight_decay`
- Type: `float`
- Allowed values: any non-negative float
- This adds regularization to the optimizer and can help prevent overfitting or overly sharp updates.
- It usually matters less than `lr`, but it is still worth tuning once the basic setup is behaving sensibly.

## `temporal`

### `temporal.window_size`
- Type: `int`
- Allowed values: integer `>= 3`
- This sets how many consecutive frames are grouped into each temporal training example.
- Larger windows capture longer motion structure, but they also increase memory use and require more consecutive valid frames.

### `temporal.window_stride`
- Type: `int`
- Allowed values: integer `>= 1`
- This sets how far the sliding window moves between neighboring temporal samples.
- Smaller strides produce more overlapping windows and more training data, while larger strides reduce redundancy and runtime.

### `temporal.max_frame_gap`
- Type: `int`
- Allowed values: integer `>= 1`
- This defines how tolerant a temporal stream is to missing or skipped frame indices when building windows.
- Lower values make the temporal supervision stricter; higher values allow looser continuity but may weaken the motion signal.

### `temporal.reduction`
- Type: `string`
- Allowed values:
  - `l1`
  - `l2`
  - `smooth_l1`
- This chooses how the temporal residual is reduced into a scalar loss.
- `smooth_l1` is often a reasonable default because it is less sensitive to outliers than pure `l2` while still giving smooth gradients.

### `temporal.scorer_hidden_dim`
- Type: `int`
- Allowed values: any positive integer
- Used only when at least one temporal family is `learnable`
- This controls the width of the learned temporal scorer network.
- Larger values increase model capacity, but they also cost more memory and may be unnecessary for small tuning runs.

### `temporal.scorer_layers`
- Type: `int`
- Allowed values: any positive integer
- Used only when at least one temporal family is `learnable`
- This controls the depth of the learned temporal scorer network.
- More layers let the scorer model more complex patterns, but they also make the temporal head heavier and slower.

### `temporal.scorer_dropout`
- Type: `float`
- Allowed values: any float typically in `[0.0, 1.0]`
- Used only when at least one temporal family is `learnable`
- This is regularization for the learned temporal scorer only.
- In small-data settings it can help prevent the scorer from overfitting, but many short test runs work fine with `0.0`.

## `losses`

There are three supported loss families:

- `vipe_camera`
- `temporal_camera`
- `temporal_bbox_projected`

### Common keys for all loss families

#### `enabled`
- Type: `bool`
- Allowed values:
  - `true`
  - `false`
- Use this to cleanly switch a loss family on or off without changing the rest of its config.
- It is helpful for ablations because you can keep all other settings fixed and toggle one component at a time.

#### `weight`
- Type: `float`
- Allowed values: any non-negative float
- Meaning:
  - `vipe_camera.weight` scales direct ViPE camera supervision
  - temporal family `weight` scales the analytical second-difference term
- This determines how much a loss family contributes to the total objective relative to the others.
- If a family is enabled but its base values are numerically tiny, you may need to increase this by a lot before it meaningfully affects training.

#### `scorer_weight`
- Type: `float`
- Allowed values: any non-negative float
- Meaning:
  - used only by temporal families
  - ignored by `vipe_camera`
  - for temporal families, scales the learned scorer term when `formulation: learnable`
- This is separate from the analytical temporal penalty and only matters when the learned scorer is active.
- It is useful when you want the scorer to help shape the loss without letting it dominate the simpler temporal term.

### Additional keys for temporal families only

These apply to:

- `temporal_camera`
- `temporal_bbox_projected`

#### `formulation`
- Type: `string`
- Allowed values:
  - `static`
  - `learnable`

Meaning:
- `static`: only the analytical second-difference loss is used
- `learnable`: analytical second-difference plus the learned scorer module
- `static` is the simpler and cheaper option, so it is often a better starting point for stability checks.
- `learnable` adds extra flexibility, but it also adds parameters and another set of weights to tune.

### Loss-family-specific notes

#### `losses.vipe_camera`
- Supported keys:
  - `enabled`
  - `weight`
  - `scorer_weight`
- Note:
  - `formulation` is not supported here
  - `scorer_weight` is effectively unused
- This is the anchor supervision coming from ViPE camera targets and is currently the strongest training signal in most runs.
- In practice, it often stabilizes training, but if it dominates too much it can drown out the temporal objectives.

#### `losses.temporal_camera`
- Supported keys:
  - `enabled`
  - `formulation`
  - `weight`
  - `scorer_weight`
- This encourages temporal smoothness in the predicted camera translation sequence.
- Use it when you want camera motion itself to be temporally consistent across adjacent frames.

#### `losses.temporal_bbox_projected`
- Supported keys:
  - `enabled`
  - `formulation`
  - `weight`
  - `scorer_weight`
- This applies temporal consistency to bounding boxes derived from projected keypoints.
- It is useful when you care about temporal stability in the model's projected image-space behavior rather than only camera parameters.

## Important Rules

- `videos` and `all_videos` are mutually exclusive.
- `train_mode` must be `distill` or `test`.
- Temporal `formulation` must be `static` or `learnable`.
- The scorer network is only instantiated if at least one enabled temporal family is:
  - `formulation: learnable`
  - and has `scorer_weight > 0.0`

## Latest Synced Baseline

The most recent directly comparable finetune reruns are the Stage A jobs from April 23, 2026:

- [wilor-train_079.out](</home/mayank/Documents/Uni/TUD/Thesis Extra/comparative study/models/wilor_hands/SLURM_logs/wilor-train_079.out>): `hp_a_ws3_s1_b8` reached `best val_loss_total=0.8864`
- [wilor-train_080.out](</home/mayank/Documents/Uni/TUD/Thesis Extra/comparative study/models/wilor_hands/SLURM_logs/wilor-train_080.out>): `hp_a_ws3_s2_b8` reached `best val_loss_total=0.8492`
- [wilor-train_081.out](</home/mayank/Documents/Uni/TUD/Thesis Extra/comparative study/models/wilor_hands/SLURM_logs/wilor-train_081.out>), [wilor-train_082.out](</home/mayank/Documents/Uni/TUD/Thesis Extra/comparative study/models/wilor_hands/SLURM_logs/wilor-train_082.out>), [wilor-train_083.out](</home/mayank/Documents/Uni/TUD/Thesis Extra/comparative study/models/wilor_hands/SLURM_logs/wilor-train_083.out>), and [wilor-train_084.out](</home/mayank/Documents/Uni/TUD/Thesis Extra/comparative study/models/wilor_hands/SLURM_logs/wilor-train_084.out>) used larger windows and ended up with `0` validation windows, so their train-loss-only numbers are not suitable for model selection

Parameters to carry forward from that rerun:

- `train_scope=refine_net`
- `videos=[0KuJ2t4S_TY, 0gzQAgjx39o, 0rq4nkWlO2E, 0ElPjqY4Cq8, SlksdQT5JRE]`
- `validation_split=0.2`, `sample_limit=256`, `detection_conf=0.3`, `rescale_factor=2.0`
- `batch_size=8`, `num_workers=2`, `max_steps=200`, `log_every=10`, `save_every=100`, `seed=42`
- `optimizer.lr=1e-5`, `optimizer.weight_decay=1e-4`
- `temporal.window_size=3`, `temporal.window_stride=2`, `temporal.max_frame_gap=1`, `temporal.reduction=smooth_l1`
- scorer setup: `hidden_dim=64`, `layers=2`, `dropout=0.0`
- `losses.vipe_camera.enabled=true`, `losses.vipe_camera.weight=0.005`
- `losses.temporal_camera.enabled=true`, `losses.temporal_camera.formulation=learnable`, `losses.temporal_camera.weight=0.03`, `losses.temporal_camera.scorer_weight=0.001`
- `losses.temporal_bbox_projected.enabled=true`, `losses.temporal_bbox_projected.formulation=learnable`, `losses.temporal_bbox_projected.weight=0.03`, `losses.temporal_bbox_projected.scorer_weight=0.001`

The original Stage B logs from April 15, 2026 are not a valid learning-rate comparison. [wilor-train_065.out](</home/mayank/Documents/Uni/TUD/Thesis Extra/comparative study/models/wilor_hands/SLURM_logs/wilor-train_065.out>), [wilor-train_066.out](</home/mayank/Documents/Uni/TUD/Thesis Extra/comparative study/models/wilor_hands/SLURM_logs/wilor-train_066.out>), and [wilor-train_067.out](</home/mayank/Documents/Uni/TUD/Thesis Extra/comparative study/models/wilor_hands/SLURM_logs/wilor-train_067.out>) all launched with `--lr 1e-5`, so the earlier `lr=3e-5` winner should be treated as invalid. Stage B should be rerun with the fixed optimizer export path before promoting any learning-rate winner downstream.

## Suggested Staged Sweep

If you are tuning this regime manually, a good order is:

- Stage A: [hparam_stage_a_windows.yaml](/home/mayank/Documents/Uni/TUD/Thesis%20Extra/comparative%20study/models/wilor_hands/experiments/hparam_stage_a_windows.yaml)
- Stage B: [hparam_stage_b_optimizer.yaml](/home/mayank/Documents/Uni/TUD/Thesis%20Extra/comparative%20study/models/wilor_hands/experiments/hparam_stage_b_optimizer.yaml)
- Stage C: [hparam_stage_c_temporal_weights.yaml](/home/mayank/Documents/Uni/TUD/Thesis%20Extra/comparative%20study/models/wilor_hands/experiments/hparam_stage_c_temporal_weights.yaml)
- Stage D: [hparam_stage_d_vipe_camera.yaml](/home/mayank/Documents/Uni/TUD/Thesis%20Extra/comparative%20study/models/wilor_hands/experiments/hparam_stage_d_vipe_camera.yaml)
- Stage E: [hparam_stage_e_tuning.yaml](/home/mayank/Documents/Uni/TUD/Thesis%20Extra/comparative%20study/models/wilor_hands/experiments/hparam_stage_e_tuning.yaml)
- Stage F: [hparam_stage_f_weight_decay.yaml](/home/mayank/Documents/Uni/TUD/Thesis%20Extra/comparative%20study/models/wilor_hands/experiments/hparam_stage_f_weight_decay.yaml)
- Stage G: [hparam_stage_g_train_scope.yaml](/home/mayank/Documents/Uni/TUD/Thesis%20Extra/comparative%20study/models/wilor_hands/experiments/hparam_stage_g_train_scope.yaml)
- Stage H: [hparam_stage_h_temporal_families.yaml](/home/mayank/Documents/Uni/TUD/Thesis%20Extra/comparative%20study/models/wilor_hands/experiments/hparam_stage_h_temporal_families.yaml)
- Stage I: [hparam_stage_i_temporal_dynamics.yaml](/home/mayank/Documents/Uni/TUD/Thesis%20Extra/comparative%20study/models/wilor_hands/experiments/hparam_stage_i_temporal_dynamics.yaml)
- Stage J: [hparam_stage_j_scorer_architecture.yaml](/home/mayank/Documents/Uni/TUD/Thesis%20Extra/comparative%20study/models/wilor_hands/experiments/hparam_stage_j_scorer_architecture.yaml)
- Stage K: [hparam_stage_k_detection_crop.yaml](/home/mayank/Documents/Uni/TUD/Thesis%20Extra/comparative%20study/models/wilor_hands/experiments/hparam_stage_k_detection_crop.yaml)

Keep the ViPE camera weight in its own stage if you want a clean comparison, since it changes the balance of the base supervision rather than the temporal regularizers.

The stage YAMLs now carry the latest synced forward baseline in `defaults`. Stage B is the next required rerun, because its older logs never actually varied `lr`; later stages are still documented so the sweep order stays reproducible once that optimizer pass is refreshed.

| Stage | Tune | Values tried / planned | Historical winner / status | Config |
| --- | --- | --- | --- | --- |
| A | `temporal.window_size`, `temporal.window_stride`, `batch_size` | `ws3_s1_b8`, `ws3_s2_b8`, `ws5_s2_b4`, `ws5_s4_b4`, `ws8_s2_b2`, `ws8_s4_b2` | Apr 23, 2026 rerun winner: `window_size=3`, `window_stride=2`, `batch_size=8` (`hp_a_ws3_s2_b8`, best `val_loss_total=0.8492`) | [hparam_stage_a_windows.yaml](/home/mayank/Documents/Uni/TUD/Thesis%20Extra/comparative%20study/models/wilor_hands/experiments/hparam_stage_a_windows.yaml) |
| B | `optimizer.lr` | `3e-6`, `1e-5`, `3e-5` | Apr 15, 2026 sweep invalid: all three runs launched with `lr=1e-5`; rerun required. Forward baseline stays `lr=1e-5`, `weight_decay=1e-4`. | [hparam_stage_b_optimizer.yaml](/home/mayank/Documents/Uni/TUD/Thesis%20Extra/comparative%20study/models/wilor_hands/experiments/hparam_stage_b_optimizer.yaml) |
| C | shared temporal base weight and shared scorer weight across both temporal families | `(0.01, 0.001)`, `(0.03, 0.001)`, `(0.10, 0.001)`, `(0.03, 0.01)` | temporal weight `0.03`, scorer weight `0.001` | [hparam_stage_c_temporal_weights.yaml](/home/mayank/Documents/Uni/TUD/Thesis%20Extra/comparative%20study/models/wilor_hands/experiments/hparam_stage_c_temporal_weights.yaml) |
| D | `losses.vipe_camera.weight` | `0.005`, `0.01`, `0.02`, `0.05` | `0.005` | [hparam_stage_d_vipe_camera.yaml](/home/mayank/Documents/Uni/TUD/Thesis%20Extra/comparative%20study/models/wilor_hands/experiments/hparam_stage_d_vipe_camera.yaml) |
| E | video coverage | `all_videos=true` vs the 5-video stage subset | best available completed run: 5-video subset; `all_videos` still pending locally | [hparam_stage_e_tuning.yaml](/home/mayank/Documents/Uni/TUD/Thesis%20Extra/comparative%20study/models/wilor_hands/experiments/hparam_stage_e_tuning.yaml) |
| F | `optimizer.weight_decay` | `0`, `1e-5`, `1e-4`, `1e-3` | planned | [hparam_stage_f_weight_decay.yaml](/home/mayank/Documents/Uni/TUD/Thesis%20Extra/comparative%20study/models/wilor_hands/experiments/hparam_stage_f_weight_decay.yaml) |
| G | `train_scope` | `camera_head`, `refine_net`, `full` | planned | [hparam_stage_g_train_scope.yaml](/home/mayank/Documents/Uni/TUD/Thesis%20Extra/comparative%20study/models/wilor_hands/experiments/hparam_stage_g_train_scope.yaml) |
| H | temporal family enable/formulation | all learnable, all static, and every non-empty learnable family subset | planned | [hparam_stage_h_temporal_families.yaml](/home/mayank/Documents/Uni/TUD/Thesis%20Extra/comparative%20study/models/wilor_hands/experiments/hparam_stage_h_temporal_families.yaml) |
| I | `temporal.max_frame_gap`, `temporal.reduction` | `gap={1,2,3}`, `reduction in {smooth_l1,l1,l2}` | planned | [hparam_stage_i_temporal_dynamics.yaml](/home/mayank/Documents/Uni/TUD/Thesis%20Extra/comparative%20study/models/wilor_hands/experiments/hparam_stage_i_temporal_dynamics.yaml) |
| J | scorer architecture | `hidden_dim={32,64,128}`, `layers={1,2,3}`, `dropout={0.0,0.1}` | planned | [hparam_stage_j_scorer_architecture.yaml](/home/mayank/Documents/Uni/TUD/Thesis%20Extra/comparative%20study/models/wilor_hands/experiments/hparam_stage_j_scorer_architecture.yaml) |
| K | `detection_conf`, `rescale_factor` | `detection_conf={0.2,0.3,0.5}`, `rescale_factor={1.5,2.0,2.5}` | planned | [hparam_stage_k_detection_crop.yaml](/home/mayank/Documents/Uni/TUD/Thesis%20Extra/comparative%20study/models/wilor_hands/experiments/hparam_stage_k_detection_crop.yaml) |

## Historical Tuning Results

| Stage | Experiments compared | Best validation result | Winning value(s) | Notes |
| --- | --- | --- | --- | --- |
| A | `hp_a_ws3_s1_b8`, `hp_a_ws3_s2_b8`, `hp_a_ws5_s2_b4`, `hp_a_ws5_s4_b4`, `hp_a_ws8_s2_b2`, `hp_a_ws8_s4_b2` | `hp_a_ws3_s2_b8` with `loss_total=1.5752` at step `200` | `window_size=3`, `window_stride=2`, `batch_size=8` | The `ws5` and `ws8` runs produced no validation windows, so the real comparison was between the two `ws3` runs. |
| B | `hp_b_lr_3e6`, `hp_b_lr_1e5`, `hp_b_lr_3e5` | `hp_b_lr_3e5` with `loss_total=1.4672` at step `190` | `optimizer.lr=3e-5` | `optimizer.weight_decay` stayed fixed at `1e-4`. |
| C | `hp_c_tw_0p01_sw_0p001`, `hp_c_tw_0p03_sw_0p001`, `hp_c_tw_0p10_sw_0p001`, `hp_c_tw_0p03_sw_0p01` | `hp_c_tw_0p03_sw_0p001` with `loss_total=1.4648` at step `200` | temporal weight `0.03`, scorer weight `0.001` | The same winning values apply to `temporal_camera` and `temporal_bbox_projected`. |
| D | `hp_d_vipe_0p005`, `hp_d_vipe_0p01`, `hp_d_vipe_0p02`, `hp_d_vipe_0p05` | `hp_d_vipe_0p005` with `loss_total=0.8515` at step `200` | `losses.vipe_camera.weight=0.005` | This is the strongest improvement across the staged sweeps. |
| E | `tune_all_videos`, `tune_stage_5_videos` | `tune_stage_5_videos` with `loss_total=1.6595` at step `400` | best available completed run: 5-video subset | Only `tune_stage_5_videos` is present in the synced outputs right now, so the Stage E comparison is still incomplete until the `all_videos` run is available locally. |

## Current Best-Known Parameter Set

This is the current best-known baseline from Stages A-D. Stage E keeps these values fixed and only changes video coverage, and the planned F-K sweeps also start from this same baseline unless noted otherwise.

| Parameter | Final value |
| --- | --- |
| `train_mode` | `distill` |
| `train_scope` | `refine_net` |
| `validation_split` | `0.2` |
| `detection_conf` | `0.3` |
| `rescale_factor` | `2.0` |
| `batch_size` | `8` |
| `num_workers` | `2` |
| `optimizer.lr` | `3e-5` |
| `optimizer.weight_decay` | `1e-4` |
| `temporal.window_size` | `3` |
| `temporal.window_stride` | `2` |
| `temporal.max_frame_gap` | `1` |
| `temporal.reduction` | `smooth_l1` |
| `temporal.scorer_hidden_dim` | `64` |
| `temporal.scorer_layers` | `2` |
| `temporal.scorer_dropout` | `0.0` |
| `losses.vipe_camera.enabled` | `true` |
| `losses.vipe_camera.weight` | `0.005` |
| `losses.temporal_camera.enabled` | `true` |
| `losses.temporal_camera.formulation` | `learnable` |
| `losses.temporal_camera.weight` | `0.03` |
| `losses.temporal_camera.scorer_weight` | `0.001` |
| `losses.temporal_bbox_projected.enabled` | `true` |
| `losses.temporal_bbox_projected.formulation` | `learnable` |
| `losses.temporal_bbox_projected.weight` | `0.03` |
| `losses.temporal_bbox_projected.scorer_weight` | `0.001` |

For the Stage E tuning pass in [hparam_stage_e_tuning.yaml](/home/mayank/Documents/Uni/TUD/Thesis%20Extra/comparative%20study/models/wilor_hands/experiments/hparam_stage_e_tuning.yaml), the run-specific settings are:

| Setting | Value |
| --- | --- |
| `sample_limit` | `0` |
| `max_steps` | `500` |
| `log_every` | `25` |
| `save_every` | `100` |
| `video coverage variants` | `all_videos=true` or the 5-video subset |
| `best available completed Stage E run` | `tune_stage_5_videos` |
| `best available completed Stage E val loss` | `1.6595` at step `400` |
| `best available completed Stage E video set` | `0KuJ2t4S_TY`, `0gzQAgjx39o`, `0rq4nkWlO2E`, `0ElPjqY4Cq8`, `SlksdQT5JRE` |

## Planned Additional Sweeps

These files are ready to run next on top of the current best-known baseline:

| Stage | Main question | Config |
| --- | --- | --- |
| F | Is `weight_decay=1e-4` actually best, or should regularization be weaker or stronger? | [hparam_stage_f_weight_decay.yaml](/home/mayank/Documents/Uni/TUD/Thesis%20Extra/comparative%20study/models/wilor_hands/experiments/hparam_stage_f_weight_decay.yaml) |
| G | Is `refine_net` really the best trainable scope, or does `camera_head` / `full` work better? | [hparam_stage_g_train_scope.yaml](/home/mayank/Documents/Uni/TUD/Thesis%20Extra/comparative%20study/models/wilor_hands/experiments/hparam_stage_g_train_scope.yaml) |
| H | Which temporal families are actually helping, and do they need to be `learnable`? | [hparam_stage_h_temporal_families.yaml](/home/mayank/Documents/Uni/TUD/Thesis%20Extra/comparative%20study/models/wilor_hands/experiments/hparam_stage_h_temporal_families.yaml) |
| I | Should temporal continuity be stricter or looser, and is `smooth_l1` still the best reduction? | [hparam_stage_i_temporal_dynamics.yaml](/home/mayank/Documents/Uni/TUD/Thesis%20Extra/comparative%20study/models/wilor_hands/experiments/hparam_stage_i_temporal_dynamics.yaml) |
| J | Is the learnable scorer under- or over-sized? | [hparam_stage_j_scorer_architecture.yaml](/home/mayank/Documents/Uni/TUD/Thesis%20Extra/comparative%20study/models/wilor_hands/experiments/hparam_stage_j_scorer_architecture.yaml) |
| K | Are the hand detection threshold and crop context holding back the downstream training signal? | [hparam_stage_k_detection_crop.yaml](/home/mayank/Documents/Uni/TUD/Thesis%20Extra/comparative%20study/models/wilor_hands/experiments/hparam_stage_k_detection_crop.yaml) |

## Minimal Example

```yaml
defaults:
  train_mode: distill
  train_scope: refine_net
  videos:
    - clip_2
  temporal:
    window_size: 8
    window_stride: 4
    max_frame_gap: 1
    reduction: smooth_l1
  losses:
    vipe_camera:
      enabled: true
      weight: 0.01
    temporal_camera:
      enabled: true
      formulation: static
      weight: 0.01
      scorer_weight: 0.0

experiments:
  - name: camera_static

  - name: camera_learnable
    losses:
      temporal_camera:
        formulation: learnable
        scorer_weight: 0.001
```
