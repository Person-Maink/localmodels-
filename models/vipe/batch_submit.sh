#!/bin/bash

VIDEO_DIR="../../data/images"
TEMPLATE="template_vipe.sh"
OUT_DIR="generated_jobs"

mkdir -p "$OUT_DIR"

# Keep this simple and close to wilor: use local venv if available.
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

shopt -s nullglob nocaseglob
videos=("$VIDEO_DIR"/*.mp4 "$VIDEO_DIR"/*.avi "$VIDEO_DIR"/*.mts)

for video in "${videos[@]}"; do
    [ -f "$video" ] || continue

    filename=$(basename "$video")
    name="${filename%.*}"

    TIME_FMT=$(python estimate_time.py "$video")
    job_script="$OUT_DIR/demo_${name}.sh"

    escaped_name=$(printf '%s\n' "$name" | sed 's/[&|]/\\&/g')
    escaped_file=$(printf '%s\n' "$filename" | sed 's/[&|]/\\&/g')

    sed \
        -e "s|__NAME__|${escaped_name}|g" \
        -e "s|__TIME__|${TIME_FMT}|g" \
        -e "s|__VIDEO_FILE__|${escaped_file}|g" \
        "$TEMPLATE" > "$job_script"

    sbatch "$job_script"
done
