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

### `run_name_suffix`
- Type: `string`
- Optional
- Any string is allowed

### `train_mode`
- Type: `string`
- Allowed values:
  - `distill`
  - `test`

### `videos`
- Type: `list[string]` or `string`
- Meaning: explicitly chosen video names
- Example:
```yaml
videos:
  - clip_2
  - clip_4
```

### `all_videos`
- Type: `bool`
- Allowed values:
  - `true`
  - `false`
- Note: cannot be set together with `videos`

### `train_scope`
- Type: `string`
- Allowed values:
  - `camera_head`
  - `refine_net`
  - `full`

### `validation_split`
- Type: `float`
- Allowed values: any float in `[0.0, 1.0)`
- Meaning: fraction of unique frames held out for validation

### `sample_limit`
- Type: `int`
- Allowed values: any integer
- Common values:
  - `0` means no explicit limit
  - positive integer limits sample count

### `detection_conf`
- Type: `float`
- Allowed values: any float
- Typical range: `0.0` to `1.0`

### `rescale_factor`
- Type: `float`
- Allowed values: any positive float

### `batch_size`
- Type: `int`
- Allowed values: any positive integer
- Meaning: temporal windows per optimization step

### `num_workers`
- Type: `int`
- Allowed values: any non-negative integer

### `max_steps`
- Type: `int`
- Allowed values: any positive integer

### `log_every`
- Type: `int`
- Allowed values: any positive integer

### `save_every`
- Type: `int`
- Allowed values: any positive integer

### `seed`
- Type: `int`
- Allowed values: any integer

## `optimizer`

### `optimizer.lr`
- Type: `float`
- Allowed values: any positive float

### `optimizer.weight_decay`
- Type: `float`
- Allowed values: any non-negative float

## `temporal`

### `temporal.window_size`
- Type: `int`
- Allowed values: integer `>= 3`

### `temporal.window_stride`
- Type: `int`
- Allowed values: integer `>= 1`

### `temporal.max_frame_gap`
- Type: `int`
- Allowed values: integer `>= 1`

### `temporal.reduction`
- Type: `string`
- Allowed values:
  - `l1`
  - `l2`
  - `smooth_l1`

### `temporal.scorer_hidden_dim`
- Type: `int`
- Allowed values: any positive integer
- Used only when at least one temporal family is `learnable`

### `temporal.scorer_layers`
- Type: `int`
- Allowed values: any positive integer
- Used only when at least one temporal family is `learnable`

### `temporal.scorer_dropout`
- Type: `float`
- Allowed values: any float typically in `[0.0, 1.0]`
- Used only when at least one temporal family is `learnable`

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

#### `weight`
- Type: `float`
- Allowed values: any non-negative float
- Meaning:
  - `vipe_camera.weight` scales direct ViPE camera supervision
  - temporal family `weight` scales the analytical second-difference term

#### `scorer_weight`
- Type: `float`
- Allowed values: any non-negative float
- Meaning:
  - used only by temporal families
  - ignored by `vipe_camera`
  - for temporal families, scales the learned scorer term when `formulation: learnable`

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

### Loss-family-specific notes

#### `losses.vipe_camera`
- Supported keys:
  - `enabled`
  - `weight`
  - `scorer_weight`
- Note:
  - `formulation` is not supported here
  - `scorer_weight` is effectively unused

#### `losses.temporal_camera`
- Supported keys:
  - `enabled`
  - `formulation`
  - `weight`
  - `scorer_weight`

#### `losses.temporal_bbox_projected`
- Supported keys:
  - `enabled`
  - `formulation`
  - `weight`
  - `scorer_weight`

#### `losses.temporal_bbox_input`
- Supported keys:
  - `enabled`
  - `formulation`
  - `weight`
  - `scorer_weight`

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
