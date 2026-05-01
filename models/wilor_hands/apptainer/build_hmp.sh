#!/bin/bash
# Build the HMP-capable WiLoR Apptainer image.

set -euo pipefail

if [[ ! -f "params_hmp.sh" ]]; then
    echo "Error: params_hmp.sh not found in current directory" >&2
    exit 1
fi
source params_hmp.sh

if [[ -z "${NAME:-}" ]]; then
    echo "Error: NAME variable not set in params_hmp.sh" >&2
    exit 1
fi

if [[ ! -f "Apptainer_hmp.def" ]]; then
    echo "Error: Apptainer_hmp.def not found in current directory" >&2
    exit 1
fi

if ! command -v apptainer &> /dev/null; then
    echo "Error: apptainer command not found. Please install Apptainer first." >&2
    exit 1
fi

echo "Gathering version information..."

if git rev-parse --git-dir > /dev/null 2>&1; then
    VERSION=$(git describe --tags --always --dirty 2>/dev/null || echo "unknown")
    VERSION=${VERSION#v}
    GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")

    if [[ -n $(git status --porcelain) ]]; then
        GIT_DIRTY="true"
        GIT_STATUS="dirty (uncommitted changes)"
        if [[ ! "${VERSION}" =~ -dirty$ ]]; then
            VERSION="${VERSION}-dirty"
        fi
    else
        GIT_DIRTY="false"
        GIT_STATUS="clean"
    fi

    GIT_TAG=$(git describe --exact-match --tags HEAD 2>/dev/null || echo "")
else
    VERSION="unknown"
    GIT_COMMIT="not-in-git"
    GIT_BRANCH="unknown"
    GIT_DIRTY="unknown"
    GIT_STATUS="not in git repository"
    GIT_TAG=""
fi

BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
BUILD_HOST=$(hostname)

cat > .build_info <<EOF
VERSION=${VERSION}
GIT_COMMIT=${GIT_COMMIT}
GIT_BRANCH=${GIT_BRANCH}
GIT_DIRTY=${GIT_DIRTY}
GIT_TAG=${GIT_TAG}
BUILD_DATE=${BUILD_DATE}
BUILD_HOST=${BUILD_HOST}
CONTAINER_NAME=${NAME}
EOF

CONTAINER_FILE="${NAME}-${VERSION}.sif"

echo "========================================="
echo "Building HMP WiLoR Apptainer container: ${CONTAINER_FILE}"
echo "========================================="
echo "Version: ${VERSION}"
echo "Git commit: ${GIT_COMMIT}"
if [[ -n "${GIT_TAG}" ]]; then
    echo "Git tag: ${GIT_TAG}"
fi
echo "Git branch: ${GIT_BRANCH}"
echo "Git status: ${GIT_STATUS}"
echo "Build date: ${BUILD_DATE}"
echo "Start time: $(date)"
echo ""

export BUILD_VERSION="${VERSION}"
export BUILD_GIT_COMMIT="${GIT_COMMIT}"
export BUILD_GIT_BRANCH="${GIT_BRANCH}"
export BUILD_GIT_DIRTY="${GIT_DIRTY}"
export BUILD_GIT_TAG="${GIT_TAG}"
export BUILD_DATE="${BUILD_DATE}"

if apptainer build "${CONTAINER_FILE}" Apptainer_hmp.def 2>&1 | tee build-hmp.log; then
    echo ""
    echo "========================================="
    echo "Build completed successfully!"
    echo "Container: ${CONTAINER_FILE}"
    echo "Log file: build-hmp.log"
    echo "End time: $(date)"
    echo "========================================="
    rm -f .build_info
else
    echo ""
    echo "=========================================" >&2
    echo "Build failed! Check build-hmp.log for details." >&2
    echo "End time: $(date)" >&2
    echo "=========================================" >&2
    rm -f .build_info
    exit 1
fi
