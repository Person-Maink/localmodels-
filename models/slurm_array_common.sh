#!/bin/bash

escape_sed_replacement() {
    printf '%s\n' "$1" | sed 's/[\\&|]/\\&/g'
}

safe_job_name() {
    local raw_name="$1"
    local safe_name
    safe_name=$(printf '%s\n' "$raw_name" | tr ' /()' '_' | tr -cd '[:alnum:]_.-')
    if [[ -z "$safe_name" ]]; then
        safe_name="job"
    fi
    printf '%s\n' "$safe_name"
}

hms_to_seconds() {
    local time_string="$1"
    local hours minutes seconds
    IFS=':' read -r hours minutes seconds <<< "$time_string"
    printf '%s\n' $((10#$hours * 3600 + 10#$minutes * 60 + 10#$seconds))
}

seconds_to_hms() {
    local total_seconds="$1"
    local hours minutes seconds

    if (( total_seconds < 0 )); then
        total_seconds=0
    fi

    hours=$((total_seconds / 3600))
    minutes=$(((total_seconds % 3600) / 60))
    seconds=$((total_seconds % 60))
    printf '%02d:%02d:%02d\n' "$hours" "$minutes" "$seconds"
}

round_up_bucket_seconds() {
    local total_seconds="$1"
    local bucket_seconds="${2:-1800}"
    local rounded_seconds

    if (( total_seconds <= 0 )); then
        total_seconds=$bucket_seconds
    fi

    rounded_seconds=$((((total_seconds + bucket_seconds - 1) / bucket_seconds) * bucket_seconds))
    if (( rounded_seconds < bucket_seconds )); then
        rounded_seconds=$bucket_seconds
    fi
    printf '%s\n' "$rounded_seconds"
}

normalize_positive_int() {
    local raw_value="$1"
    local default_value="$2"

    if [[ "$raw_value" =~ ^[0-9]+$ ]] && (( raw_value > 0 )); then
        printf '%s\n' "$raw_value"
    else
        printf '%s\n' "$default_value"
    fi
}

is_truthy() {
    local value="${1:-0}"
    shopt -s nocasematch
    case "$value" in
        1|true|yes|y|on)
            shopt -u nocasematch
            return 0
            ;;
        *)
            shopt -u nocasematch
            return 1
            ;;
    esac
}

submit_batch_script() {
    local job_script="$1"
    local submitter="${SBATCH_BIN:-sbatch}"

    if is_truthy "${DRY_RUN:-0}"; then
        printf 'DRY_RUN:'
        printf ' %q' "$submitter" "$job_script"
        printf '\n'
    else
        "$submitter" "$job_script"
    fi
}
