#!/usr/bin/env bash
set -euo pipefail
SEARCH_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
exec env STAGE=stage1 CONFIG_ID=s14 "${SEARCH_ROOT}/submit_config.sh" "$@"
