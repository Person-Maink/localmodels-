#!/bin/bash
# Test the Apptainer container functionality
# Usage: ./test.sh

# Exit on error, undefined variables, and pipe failures
set -euo pipefail

# Load configuration
if [[ ! -f "params.sh" ]]; then
    echo "Error: params.sh not found in current directory" >&2
    exit 1
fi
source params.sh

# Validate NAME variable
if [[ -z "${NAME:-}" ]]; then
    echo "Error: NAME variable not set in params.sh" >&2
    exit 1
fi

resolve_container_file() {
    local newest_versioned=""

    if [[ $# -gt 0 && -n "$1" ]]; then
        echo "$1"
        return
    fi

    if compgen -G "${NAME}-*.sif" > /dev/null; then
        newest_versioned=$(ls -1t "${NAME}"-*.sif 2>/dev/null | head -n 1)
    fi
    if [[ -n "${newest_versioned}" ]]; then
        echo "${newest_versioned}"
        return
    fi

    echo "${NAME}.sif"
}

CONTAINER_FILE="$(resolve_container_file "${1:-}")"

if [[ ! -f "${CONTAINER_FILE}" ]]; then
    echo "Error: Container file ${CONTAINER_FILE} not found" >&2
    echo "Please run build.sh first to create the container" >&2
    exit 1
fi

# Check if apptainer command is available
if ! command -v apptainer &> /dev/null; then
    echo "Error: apptainer command not found. Please install Apptainer first." >&2
    exit 1
fi

echo "========================================="
echo "Testing container: ${CONTAINER_FILE}"
echo "========================================="
echo ""

# Track test results
TESTS_PASSED=0
TESTS_FAILED=0

# Helper function to run tests
run_test() {
    local test_name=$1
    local test_command=$2

    echo "Running test: ${test_name}"
    echo "Command: ${test_command}"

    if eval "${test_command}"; then
        echo "✓ PASSED: ${test_name}"
        ((TESTS_PASSED++))
    else
        echo "✗ FAILED: ${test_name}" >&2
        ((TESTS_FAILED++))
    fi
    echo ""
}

# Test 1: Check if Python is available
run_test "Python availability" \
    "apptainer exec ${CONTAINER_FILE} which python"

# Test 2: Check that the isolated Dyn-HaMR venv is the runtime interpreter
run_test "Dyn-HaMR venv interpreter" \
    "apptainer exec ${CONTAINER_FILE} python -c \"import sys; print(sys.executable); assert sys.executable.startswith('/opt/dynhamr-venv/')\""

# Test 3: Validate the scientific stack used by Dyn-HaMR
run_test "Dyn-HaMR runtime stack" \
    "apptainer exec ${CONTAINER_FILE} python -c \"import numpy, scipy, trimesh; print('numpy', numpy.__version__); print('scipy', scipy.__version__); print('trimesh', trimesh.__version__); assert numpy.__version__ == '1.22.4'; assert scipy.__version__ == '1.8.1'\""

# Test 4: Check PyTorch availability
run_test "PyTorch availability" \
    "apptainer exec ${CONTAINER_FILE} python -c \"import torch; print(torch.__version__)\""

# Test 5: Smoke-test Dyn-HaMR imports that used to fail
run_test "Dyn-HaMR import smoke test" \
    "apptainer exec ${CONTAINER_FILE} python -c \"import numpy, scipy, trimesh, torch; print('import smoke test ok')\""

# Print summary
echo "========================================="
echo "Test Summary"
echo "========================================="
echo "Tests passed: ${TESTS_PASSED}"
echo "Tests failed: ${TESTS_FAILED}"
echo "========================================="

if [[ ${TESTS_FAILED} -eq 0 ]]; then
    echo "All tests passed successfully!"
    exit 0
else
    echo "Some tests failed. Please review the output above." >&2
    exit 1
fi
