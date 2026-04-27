# LoRA Experiment Suite

This folder mirrors the main WiLoR finetuning suite, but treats LoRA as an
additive adaptation path rather than a replacement for the normal finetuning
scope.

Every run here updates:

- the usual WiLoR parameters selected by `train_scope`
- the LoRA adapter parameters at the same time

## Baseline LoRA setup

The suite defaults to a conservative adapter recipe on the ViT attention stack:

- `lora.enabled: true`
- `lora.rank: 8`
- `lora.alpha: 16.0`
- `lora.dropout: 0.0`
- `lora.block_start: 24`
- `lora.block_end: 32`
- `lora.target_modules: [qkv]`

## How `train_scope` works here

LoRA is always trained in addition to the selected scope:

- `camera_head`: camera head modules plus LoRA adapters train
- `refine_net`: refine net plus LoRA adapters train
- `full`: the whole WiLoR model plus LoRA adapters train

## Recommended Order

The suite is organized around the following progression:

- Stages `A-F`: retune the ordinary optimization and temporal settings with the
  stable default `refine_net + LoRA` setup
- Stage `G`: compare additive scopes, starting from `refine_net + LoRA`
  before trying `camera_head + LoRA` and only then `full + LoRA`
- Stage `H`: tune the LoRA adapter capacity itself
- Stages `I-L`: continue the temporal-family, dynamics, scorer, and crop sweeps
- Stage `M`: tune the newer `temporal_vipe_camera` loss under the chosen
  LoRA-backed finetuning regime

Experiment names are prefixed with `lora_` so their run directories do not
collide with the non-LoRA suite.
