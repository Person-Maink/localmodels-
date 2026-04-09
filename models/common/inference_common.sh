#!/bin/bash

is_truthy() {
    [[ "${1:-}" =~ ^([Tt][Rr][Uu][Ee]|[Yy]([Ee][Ss])?|[Oo][Nn]|1)$ ]]
}

pick_temp_root() {
    if [[ -n "${SLURM_TMPDIR:-}" ]]; then
        printf '%s\n' "${SLURM_TMPDIR}"
    elif [[ -n "${TMPDIR:-}" ]]; then
        printf '%s\n' "${TMPDIR}"
    else
        printf '/tmp\n'
    fi
}

resolve_video_path() {
    local video_dir="$1"
    local video_name="${2:-}"
    local video_file="${3:-}"
    local candidate

    if [[ -n "${video_file}" ]]; then
        if [[ -f "${video_dir}/${video_file}" ]]; then
            printf '%s\n' "${video_dir}/${video_file}"
            return 0
        fi
        shopt -s nullglob nocaseglob
        for candidate in "${video_dir}"/*; do
            [[ -f "${candidate}" ]] || continue
            if [[ "${candidate##*/}" == "${video_file}" ]] || [[ "${candidate##*/,,}" == "${video_file,,}" ]]; then
                printf '%s\n' "${candidate}"
                return 0
            fi
        done
        shopt -u nullglob nocaseglob
        return 1
    fi

    if [[ -z "${video_name}" ]]; then
        return 1
    fi

    shopt -s nullglob nocaseglob
    for candidate in "${video_dir}/${video_name}.mp4" "${video_dir}/${video_name}.avi" "${video_dir}/${video_name}.mts" "${video_dir}/${video_name}.mov"; do
        if [[ -f "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            shopt -u nullglob nocaseglob
            return 0
        fi
    done
    shopt -u nullglob nocaseglob
    return 1
}

completion_marker_path() {
    local output_root="$1"
    local video_name="$2"
    printf '%s\n' "${output_root%/}/_completed/${video_name}.json"
}

write_simple_completion_marker() {
    local marker_path="$1"
    local model_name="$2"
    local video_name="$3"
    local retained_root="$4"
    local timestamp

    timestamp="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    mkdir -p "$(dirname "${marker_path}")"
    cat >"${marker_path}" <<EOF
{
  "model": "${model_name}",
  "video": "${video_name}",
  "completed_at_utc": "${timestamp}",
  "retained_root": "${retained_root}"
}
EOF
}
