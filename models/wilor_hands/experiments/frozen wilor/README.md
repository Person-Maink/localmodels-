# Frozen WiLoR Experiment Suite

This folder contains a temporal-head-only copy of the WiLoR temporal tuning suite.

Every config here uses:

- `train_scope: temporal_only`
- a completely frozen WiLoR model
- learnable temporal modules only

## What Changed

Two original stages were intentionally removed because they are not meaningful
when WiLoR never unfreezes:

- Original Stage `D` (`vipe_camera`) was removed because direct `vipe_camera`
  loss does not update temporal-only modules when WiLoR is frozen.
- Original Stage `G` (`train_scope`) was removed because the trainable scope is
  fixed to `temporal_only` for the whole frozen suite.

The remaining stages were renamed to close the gaps:

- Original `E` -> Frozen `D`
- Original `F` -> Frozen `E`
- Original `H` -> Frozen `F`
- Original `I` -> Frozen `G`
- Original `J` -> Frozen `H`
- Original `K` -> Frozen `I`

An extra Frozen `J` stage was added for the new `temporal_vipe_camera` head.

## Objective Notes

To keep this suite focused on frozen-WiLoR temporal-module training:

- `vipe_camera` is disabled in the defaults
- analytical temporal base weights stay at `0.0` in the scorer-only stages
- the new `temporal_vipe_camera` branch trains a dedicated temporal camera head
  from frozen WiLoR outputs plus ViPE camera targets

The fixed teacher/distillation losses still appear in the total loss reported by
the training script, but the trainable part of the frozen-WiLoR setup remains
outside WiLoR itself.
