#!/bin/bash

VIDEO_DIR="../../data/images"
TEMPLATE="template_hamba.sh"
OUT_DIR="generated_jobs"

mkdir -p "$OUT_DIR"

if [ -x .venv/bin/python3.10 ]; then
    PYTHON_BIN=".venv/bin/python3.10"
elif [ -x .venv/bin/python ]; then
    PYTHON_BIN=".venv/bin/python"
else
    PYTHON_BIN="python3"
fi

for video in "$VIDEO_DIR"/*.mp4; do
    [ -f "$video" ] || continue

    filename=$(basename "$video")
    name="${filename%.*}"

    TIME_FMT=$("$PYTHON_BIN" estimate_time.py "$video")
    if [ $? -ne 0 ] || [ -z "$TIME_FMT" ]; then
        echo "Skipping $video: failed to estimate runtime with $PYTHON_BIN"
        continue
    fi

    job_script="$OUT_DIR/demo_${name}.sh"

    sed \
        -e "s/__NAME__/${name}/g" \
        -e "s/__TIME__/${TIME_FMT}/g" \
        "$TEMPLATE" > "$job_script"

    sbatch "$job_script"
done
