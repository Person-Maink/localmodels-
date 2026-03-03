#!/bin/bash

VIDEO_DIR="../../data/images"
TEMPLATE="template_hamba.sh"
OUT_DIR="generated_jobs"

mkdir -p "$OUT_DIR"

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

for video in "$VIDEO_DIR"/*.mp4; do
    [ -f "$video" ] || continue

    filename=$(basename "$video")
    name="${filename%.*}"

    TIME_FMT=$(python estimate_time.py "$video")

    job_script="$OUT_DIR/demo_${name}.sh"

    sed \
        -e "s/__NAME__/${name}/g" \
        -e "s/__TIME__/${TIME_FMT}/g" \
        "$TEMPLATE" > "$job_script"

    sbatch "$job_script"
done
