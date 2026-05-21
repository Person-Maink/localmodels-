# LoRA Temporal Ablations Config Options

This file documents the current experiment-config schema used by the LoRA
temporal-ablation stage YAMLs in this directory:

- [finetuning.yaml](./finetuning.yaml)
- [hparam_stage_a_windows.yaml](./hparam_stage_a_windows.yaml)
- [hparam_stage_b_optimizer.yaml](./hparam_stage_b_optimizer.yaml)
- [hparam_stage_c_temporal_weights.yaml](./hparam_stage_c_temporal_weights.yaml)
- [hparam_stage_d_vipe_camera.yaml](./hparam_stage_d_vipe_camera.yaml)
- [hparam_stage_e_tuning.yaml](./hparam_stage_e_tuning.yaml)
- [hparam_stage_f_weight_decay.yaml](./hparam_stage_f_weight_decay.yaml)
- [hparam_stage_g_train_scope.yaml](./hparam_stage_g_train_scope.yaml)
- [hparam_stage_h_lora_adapters.yaml](./hparam_stage_h_lora_adapters.yaml)
- [hparam_stage_i_temporal_families.yaml](./hparam_stage_i_temporal_families.yaml)
- [hparam_stage_j_temporal_dynamics.yaml](./hparam_stage_j_temporal_dynamics.yaml)
- [hparam_stage_k_scorer_architecture.yaml](./hparam_stage_k_scorer_architecture.yaml)
- [hparam_stage_l_detection_crop.yaml](./hparam_stage_l_detection_crop.yaml)

The config format has two top-level sections:

- `defaults`: shared settings merged into every experiment
- `experiments`: a list of named per-run overrides

There is no separate top-level `lora` section. LoRA settings live inside
`defaults.lora` and can be overridden inside an individual experiment entry.

## Top-Level Experiment Keys

### `name`
- Type: `string`
- Required in each experiment entry
- Used as the experiment identifier when resolving configs.

### `run_name_suffix`
- Type: `string`
- Optional
- Used only for naming and grouping runs.

### `train_mode`
- Type: `string`
- Allowed values:
  - `distill`
  - `test`

### `videos`
- Type: `list[string]` or `string`
- Explicit video-name selection.

### `all_videos`
- Type: `bool`
- Cannot be set together with `videos`.

### `train_scope`
- Type: `string`
- Allowed values:
  - `temporal_only`
  - `refine_net`
- `refine_net` trains WiLoR's refine net, plus LoRA adapters when `lora.enabled=true`.
- `temporal_only` freezes the normal WiLoR trainable scope and relies on the
  learnable temporal scorer, plus LoRA adapters when enabled.

### Standard scalar run settings

These keys are supported in `defaults` and may be overridden per experiment:

- `validation_split`
- `sample_limit`
- `detection_conf`
- `rescale_factor`
- `batch_size`
- `num_workers`
- `max_steps`
- `log_every`
- `save_every`
- `seed`

## `optimizer`

### `optimizer.lr`
- Type: `float`
- Any positive float.

### `optimizer.weight_decay`
- Type: `float`
- Any non-negative float.

## `temporal`

### `temporal.window_size`
- Type: `int`
- Must be `>= 3`.

### `temporal.window_stride`
- Type: `int`
- Must be `>= 1`.

### `temporal.max_frame_gap`
- Type: `int`
- Must be `>= 1`.

### `temporal.reduction`
- Type: `string`
- Allowed values:
  - `l1`
  - `l2`
  - `smooth_l1`

### `temporal.scorer_hidden_dim`
- Type: `int`
- Used only when a learnable temporal family is active.

### `temporal.scorer_layers`
- Type: `int`
- Used only when a learnable temporal family is active.

### `temporal.scorer_dropout`
- Type: `float`
- Used only when a learnable temporal family is active.

## `lora`

These keys live under `defaults.lora` and may be overridden per experiment.

### `lora.enabled`
- Type: `bool`
- Enables LoRA adapters on the WiLoR ViT attention blocks.

### `lora.rank`
- Type: `int`

### `lora.alpha`
- Type: `float`

### `lora.dropout`
- Type: `float`

### `lora.block_start`
- Type: `int`
- First ViT block index that receives LoRA adapters.

### `lora.block_end`
- Type: `int`
- Exclusive end index of the adapted ViT block range.

### `lora.target_modules`
- Type: `list[string]` or comma-separated `string`
- Allowed values:
  - `qkv`
  - `proj`

## `losses`

There are currently three supported loss families:

- `vipe_camera`
- `temporal_camera`
- `temporal_bbox_projected`

`temporal_vipe_camera` is no longer supported in this suite.

### Common keys

#### `enabled`
- Type: `bool`

#### `weight`
- Type: `float`
- For `vipe_camera`, this scales direct ViPE camera supervision.
- For temporal families, this scales the analytical second-difference term.

#### `scorer_weight`
- Type: `float`
- Only meaningful for temporal families with `formulation: learnable`.
- Accepted on `vipe_camera` for schema consistency, but unused by training.

### Temporal-family-only keys

These apply to:

- `temporal_camera`
- `temporal_bbox_projected`

#### `formulation`
- Type: `string`
- Allowed values:
  - `static`
  - `learnable`

Meaning:

- `static`: analytical second-difference only
- `learnable`: analytical second-difference plus the learned temporal scorer

### Family-specific notes

#### `losses.vipe_camera`
- Direct supervision from ViPE camera targets.
- This is not part of the temporal scorer flow.

#### `losses.temporal_camera`
- Applies temporal smoothness to the predicted camera translation sequence.

#### `losses.temporal_bbox_projected`
- Applies temporal smoothness to projected image-space bounding-box behavior.

## Important Rules

- `videos` and `all_videos` are mutually exclusive.
- `train_mode` must be `distill` or `test`.
- `train_scope` must be `temporal_only` or `refine_net`.
- Temporal `formulation` must be `static` or `learnable`.
- The temporal scorer is instantiated only if at least one enabled temporal
  family is both:
  - `formulation: learnable`
  - and has `scorer_weight > 0.0`
- In practice, `train_scope=temporal_only` should therefore be paired with at
  least one enabled learnable temporal family with `scorer_weight > 0.0`.

## Current Stage Inventory

These are the checked-in LoRA stage files and what they vary:

| Stage | Purpose | Config |
| --- | --- | --- |
| A | temporal window size / stride / batch size | [hparam_stage_a_windows.yaml](./hparam_stage_a_windows.yaml) |
| B | learning rate | [hparam_stage_b_optimizer.yaml](./hparam_stage_b_optimizer.yaml) |
| C | shared temporal-family base weight and scorer weight | [hparam_stage_c_temporal_weights.yaml](./hparam_stage_c_temporal_weights.yaml) |
| D | `losses.vipe_camera.weight` | [hparam_stage_d_vipe_camera.yaml](./hparam_stage_d_vipe_camera.yaml) |
| E | stage subset vs `all_videos` long-run tuning | [hparam_stage_e_tuning.yaml](./hparam_stage_e_tuning.yaml) |
| F | optimizer weight decay | [hparam_stage_f_weight_decay.yaml](./hparam_stage_f_weight_decay.yaml) |
| G | additive WiLoR trainable scope with LoRA enabled | [hparam_stage_g_train_scope.yaml](./hparam_stage_g_train_scope.yaml) |
| H | LoRA adapter recipe | [hparam_stage_h_lora_adapters.yaml](./hparam_stage_h_lora_adapters.yaml) |
| I | temporal-family ablations across surviving temporal families | [hparam_stage_i_temporal_families.yaml](./hparam_stage_i_temporal_families.yaml) |
| J | temporal continuity assumptions and reduction | [hparam_stage_j_temporal_dynamics.yaml](./hparam_stage_j_temporal_dynamics.yaml) |
| K | scorer architecture | [hparam_stage_k_scorer_architecture.yaml](./hparam_stage_k_scorer_architecture.yaml) |
| L | detection threshold and crop scale | [hparam_stage_l_detection_crop.yaml](./hparam_stage_l_detection_crop.yaml) |

Notes:

- Stage G now compares only `temporal_only` and `refine_net`.
- Stage I now compares only the surviving temporal families:
  - all learnable
  - all static
  - camera only
  - bbox-projected only
  - camera + bbox-projected
- There is no longer a Stage M for temporal ViPE camera supervision.

## Current Checked-In Finetuning Baseline

The winner-locked long-run LoRA baseline is defined in
[finetuning.yaml](./finetuning.yaml), not in the early sweep stages. The
checked-in defaults there are:

| Parameter | Value |
| --- | --- |
| `train_mode` | `distill` |
| `train_scope` | `refine_net` |
| `all_videos` | `true` |
| `validation_split` | `0.15` |
| `sample_limit` | `0` |
| `detection_conf` | `0.3` |
| `rescale_factor` | `2.0` |
| `batch_size` | `8` |
| `num_workers` | `2` |
| `max_steps` | `10000` |
| `log_every` | `25` |
| `save_every` | `250` |
| `seed` | `42` |
| `optimizer.lr` | `3e-5` |
| `optimizer.weight_decay` | `0.0` |
| `temporal.window_size` | `3` |
| `temporal.window_stride` | `2` |
| `temporal.max_frame_gap` | `1` |
| `temporal.reduction` | `smooth_l1` |
| `temporal.scorer_hidden_dim` | `64` |
| `temporal.scorer_layers` | `2` |
| `temporal.scorer_dropout` | `0.0` |
| `lora.enabled` | `true` |
| `lora.rank` | `8` |
| `lora.alpha` | `16.0` |
| `lora.dropout` | `0.0` |
| `lora.block_start` | `24` |
| `lora.block_end` | `32` |
| `lora.target_modules` | `qkv` |
| `losses.vipe_camera.enabled` | `true` |
| `losses.vipe_camera.weight` | `0.005` |
| `losses.temporal_camera.enabled` | `true` |
| `losses.temporal_camera.formulation` | `learnable` |
| `losses.temporal_camera.weight` | `0.1` |
| `losses.temporal_camera.scorer_weight` | `0.001` |
| `losses.temporal_bbox_projected.enabled` | `true` |
| `losses.temporal_bbox_projected.formulation` | `learnable` |
| `losses.temporal_bbox_projected.weight` | `0.1` |
| `losses.temporal_bbox_projected.scorer_weight` | `0.001` |

The stage YAML defaults do not all match this exactly. That is intentional:
early stages freeze the baseline that was current when that sweep was defined,
while `finetuning.yaml` captures the current long-run checked-in baseline.

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
  lora:
    enabled: true
    rank: 8
    alpha: 16.0
    dropout: 0.0
    block_start: 24
    block_end: 32
    target_modules:
      - qkv
  losses:
    vipe_camera:
      enabled: true
      weight: 0.005
    temporal_camera:
      enabled: true
      formulation: static
      weight: 0.1
      scorer_weight: 0.0
    temporal_bbox_projected:
      enabled: true
      formulation: static
      weight: 0.1
      scorer_weight: 0.0

experiments:
  - name: lora_camera_static

  - name: lora_camera_learnable
    losses:
      temporal_camera:
        formulation: learnable
        scorer_weight: 0.001
```
