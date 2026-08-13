# STRIDE two-stage hyperparameter sweep

`generate_stage1_assets.py` creates editable split manifests, 48 HMP configs,
and one config submitter per stage-1 YAML. It validates the 120 processed videos
and their cached WiLoR meshes before writing assets.

```bash
python3 generate_stage1_assets.py
STAGE=stage1 DRY_RUN=true bash submit_stage.sh
STAGE=stage1 bash submit_stage.sh
```

Results are isolated at `outputs/stride_search/<stage>/<config-id>/`. To score a
completed config:

```bash
uv run python3 analysis/dataset_tremor_metrics.py \
  --family stride \
  --family-output-root outputs/stride_search/stage1/s01_baseline \
  --output-dir analysis/analysis_images/custom/stride_search/stage1/s01_baseline
```

After ranking stage 1, copy the chosen YAMLs into `configs/stage2/`. Then stage 2
automatically creates `submitters/stage2/submit_<config-id>.sh` wrappers and uses
all 96 development clips. The 24 Niti videos listed in `splits/test.json` are
never used by either submission stage.
