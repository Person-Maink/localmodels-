# LoRA Experiment Suite

This folder mirrors the main WiLoR finetuning experiment suite, but enables
LoRA adapters on the core ViT attention blocks while keeping the rest of each
stage as close as possible to the original setup.

## Defaults used in this suite

- `lora.enabled: true`
- `lora.rank: 8`
- `lora.alpha: 16.0`
- `lora.dropout: 0.0`
- `lora.block_start: 24`
- `lora.block_end: 32`
- `lora.target_modules: [qkv]`

## How this interacts with `train_scope`

The original `train_scope` behavior is preserved:

- `camera_head`: camera head modules plus LoRA adapters train
- `refine_net`: refine net plus LoRA adapters train
- `full`: the whole model trains, and LoRA adapters are also present

Experiment names are prefixed with `lora_` so their run directories do not
collide with the non-LoRA suite.
