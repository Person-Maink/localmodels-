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

There are four supported loss families:

- `vipe_camera`
- `temporal_camera`
- `temporal_bbox_projected`
- `temporal_bbox_input`

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
- `temporal_bbox_input`

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

#### `losses.temporal_bbox_input`
- Supported keys:
  - `enabled`
  - `formulation`
  - `weight`
  - `scorer_weight`
- This applies temporal consistency to the input bounding-box sequence itself.
- It can act as a weaker or more geometry-agnostic temporal prior compared with projected-output-based constraints.

## Important Rules

- `videos` and `all_videos` are mutually exclusive.
- `train_mode` must be `distill` or `test`.
- Temporal `formulation` must be `static` or `learnable`.
- The scorer network is only instantiated if at least one enabled temporal family is:
  - `formulation: learnable`
  - and has `scorer_weight > 0.0`

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
