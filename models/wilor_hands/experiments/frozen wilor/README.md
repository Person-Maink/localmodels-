# Frozen WiLoR Experiment Suite

This folder contains a scorer-only copy of the WiLoR temporal tuning suite.

Every config here uses:

- `train_scope: temporal_only`
- a completely frozen WiLoR model
- learnable temporal scorer training only

## What Changed

Two original stages were intentionally removed because they are not meaningful
when WiLoR never unfreezes:

- Original Stage `D` (`vipe_camera`) was removed because `vipe_camera` loss does
  not update the temporal scorer when WiLoR is frozen.
- Original Stage `G` (`train_scope`) was removed because the trainable scope is
  fixed to `temporal_only` for the whole frozen suite.

The remaining stages were renamed to close the gaps:

- Original `E` -> Frozen `D`
- Original `F` -> Frozen `E`
- Original `H` -> Frozen `F`
- Original `I` -> Frozen `G`
- Original `J` -> Frozen `H`
- Original `K` -> Frozen `I`

## Objective Notes

To keep this suite focused on scorer-only training:

- `vipe_camera` is disabled in the defaults
- analytical temporal base weights are set to `0.0`
- only the temporal scorer terms are tuned through `scorer_weight`

The fixed teacher/distillation losses still appear in the total loss reported by
the training script, but the trainable part of the frozen-WiLoR setup is the
temporal scorer only.
