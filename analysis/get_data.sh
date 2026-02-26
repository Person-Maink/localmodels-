#!/usr/bin/env bash
set -euo pipefail

# --- EDIT THIS ---
REMOTE_HOST="delftblue"
REMOTE_DIR="/scratch/mthakur/manifold/outputs/"
LOCAL_OUTPUT_DIR="/home/mayank/Documents/Uni/TUD/Thesis Extra/comparative study/outputs"
# -----------------

# Optional: set DRY_RUN=1 to preview changes without modifying local files.
DRY_RUN="${DRY_RUN:-0}"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: required command '$1' not found in PATH." >&2
    exit 1
  fi
}

require_cmd ssh
require_cmd rsync

mkdir -p "$LOCAL_OUTPUT_DIR"

REMOTE_SOURCE="${REMOTE_HOST}:${REMOTE_DIR%/}/"
LOCAL_TARGET="${LOCAL_OUTPUT_DIR%/}/"

RSYNC_OPTS=(
  -a
  -z
  --delete
  --info=stats2,progress2
)

if [[ "$DRY_RUN" == "1" ]]; then
  RSYNC_OPTS+=(--dry-run --itemize-changes)
  echo "Running in dry-run mode (DRY_RUN=1). No local files will be modified."
fi

echo "Syncing from ${REMOTE_SOURCE} to ${LOCAL_TARGET}"

if rsync "${RSYNC_OPTS[@]}" "$REMOTE_SOURCE" "$LOCAL_TARGET"; then
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "Dry-run complete."
  else
    echo "Sync complete: ${LOCAL_OUTPUT_DIR}"
  fi
else
  echo "Error: rsync sync failed. Check SSH access to '${REMOTE_HOST}' and path '${REMOTE_DIR}'." >&2
  exit 1
fi
