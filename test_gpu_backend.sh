#!/bin/bash
# GPU Backend Testing Script
# Tests both GPU renderer and GPU PDE smoothing paths

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
cd "${BUILD_DIR}"

echo "=========================================="
echo "GPU Backend Testing Script"
echo "=========================================="
echo ""

# Check if binary exists
if [ ! -f "./msbg_demo" ]; then
    echo "ERROR: msbg_demo not found in build directory"
    echo "Please build first: cd build && ../mk"
    exit 1
fi

# Test parameters
TEST_CASE=1
BLOCK_SIZE=16
RESOLUTION=64
LOG_LEVEL=3

echo "Test Configuration:"
echo "  Test case: ${TEST_CASE} (bunny demo)"
echo "  Block size: ${BLOCK_SIZE}"
echo "  Resolution: ${RESOLUTION}"
echo "  Log level: ${LOG_LEVEL}"
echo ""

# Test 1: CPU renderer (baseline)
echo "=========================================="
echo "Test 1: CPU Renderer (Baseline)"
echo "=========================================="
CPU_OUTPUT=$(./msbg_demo -g cpu -c${TEST_CASE} -b${BLOCK_SIZE} -r${RESOLUTION} -v${LOG_LEVEL} 2>&1)
CPU_CHECKSUM=$(echo "${CPU_OUTPUT}" | grep -i "checksum.*cpu" | tail -1 | grep -oE "0x[0-9a-f]{16}" || echo "NOT_FOUND")
echo "CPU Render Checksum: ${CPU_CHECKSUM}"
echo ""

# Test 2: GPU renderer with validation
echo "=========================================="
echo "Test 2: GPU Renderer with Validation (-V)"
echo "=========================================="
if GPU_V_OUTPUT=$(./msbg_demo -g gpu -V -c${TEST_CASE} -b${BLOCK_SIZE} -r${RESOLUTION} -v${LOG_LEVEL} 2>&1); then
    echo "${GPU_V_OUTPUT}" | grep -E "(GPU|CPU|checksum|Metal|renderer)" | head -20
else
    EXIT_CODE=$?
    echo "WARNING: GPU test exited with code ${EXIT_CODE}"
    echo "This may indicate a crash in the GPU path."
    echo "Last 20 lines of output:"
    echo "${GPU_V_OUTPUT}" | tail -20
    GPU_V_OUTPUT=""  # Clear for checksum extraction
fi

GPU_CHECKSUM=$(echo "${GPU_V_OUTPUT}" | grep -i "gpu.*checksum" | tail -1 | grep -oE "0x[0-9a-f]{16}" || echo "NOT_FOUND")
CPU_VAL_CHECKSUM=$(echo "${GPU_V_OUTPUT}" | grep -i "cpu.*checksum" | tail -1 | grep -oE "0x[0-9a-f]{16}" || echo "NOT_FOUND")
MATCH_STATUS=$(echo "${GPU_V_OUTPUT}" | grep -i "MATCH\|MISMATCH" | tail -1 || echo "NOT_FOUND")

echo ""
echo "GPU Render Checksum: ${GPU_CHECKSUM}"
echo "CPU Validation Checksum: ${CPU_VAL_CHECKSUM}"
echo "Match Status: ${MATCH_STATUS}"
echo ""

# Test 3: GPU renderer without validation
echo "=========================================="
echo "Test 3: GPU Renderer (GPU-only)"
echo "=========================================="
if GPU_OUTPUT=$(./msbg_demo -g gpu -c${TEST_CASE} -b${BLOCK_SIZE} -r${RESOLUTION} -v${LOG_LEVEL} 2>&1); then
    echo "${GPU_OUTPUT}" | grep -E "(GPU|Metal|renderer|checksum)" | head -15
else
    EXIT_CODE=$?
    echo "WARNING: GPU-only test exited with code ${EXIT_CODE}"
    echo "Last 20 lines of output:"
    echo "${GPU_OUTPUT}" | tail -20
    # Try to extract what we can
    GPU_OUTPUT="${GPU_OUTPUT}"  # Keep for analysis
fi
GPU_ONLY_CHECKSUM=$(echo "${GPU_OUTPUT}" | grep -i "gpu.*checksum\|Metal.*checksum" | tail -1 | grep -oE "0x[0-9a-f]{16}" || echo "NOT_FOUND")
echo "GPU-only Checksum: ${GPU_ONLY_CHECKSUM}"
echo ""

# Test 4: Check for GPU PDE execution (reuse GPU_OUTPUT if available)
echo "=========================================="
echo "Test 4: GPU PDE Smoothing"
echo "=========================================="
if [ -z "${GPU_OUTPUT}" ]; then
    # Run again if we don't have output
    if PDE_OUTPUT=$(./msbg_demo -g gpu -c${TEST_CASE} -b${BLOCK_SIZE} -r${RESOLUTION} -v${LOG_LEVEL} 2>&1); then
        echo "${PDE_OUTPUT}" | grep -E "(GPU PDE|PDE|smoothing|Metal|packed.*blocks)" | head -20
    else
        echo "WARNING: GPU PDE test failed"
        echo "${PDE_OUTPUT}" | tail -20
        PDE_OUTPUT=""
    fi
else
    PDE_OUTPUT="${GPU_OUTPUT}"
    echo "${PDE_OUTPUT}" | grep -E "(GPU PDE|PDE|smoothing|Metal|packed.*blocks)" | head -20
fi

PDE_CHECKSUM=$(echo "${PDE_OUTPUT}" | grep -i "gpu pde.*checksum" | tail -1 | grep -oE "0x[0-9a-f]{16}" || echo "NOT_FOUND")
PDE_BLOCKS=$(echo "${PDE_OUTPUT}" | grep -i "gpu pde.*packed.*blocks" | tail -1 | grep -oE "[0-9]+ blocks" || echo "NOT_FOUND")
echo ""
echo "PDE Checksum: ${PDE_CHECKSUM}"
echo "PDE Blocks Processed: ${PDE_BLOCKS}"
echo ""

# Summary
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo "CPU Render Checksum:     ${CPU_CHECKSUM}"
echo "GPU Render Checksum:     ${GPU_CHECKSUM}"
echo "GPU-only Checksum:       ${GPU_ONLY_CHECKSUM}"
echo "CPU Validation Checksum: ${CPU_VAL_CHECKSUM}"
echo "Match Status:            ${MATCH_STATUS}"
echo "PDE Checksum:            ${PDE_CHECKSUM}"
echo "PDE Blocks:              ${PDE_BLOCKS}"
echo ""

# Check if Metal is available
METAL_AVAILABLE=$(echo "${GPU_OUTPUT}" | grep -i "Metal.*detected\|Metal.*available" | head -1 || echo "")
if [ -n "${METAL_AVAILABLE}" ]; then
    echo "✓ Metal device detected"
else
    echo "✗ Metal device not detected or unavailable"
fi

# Check if GPU renderer executed
if echo "${GPU_OUTPUT}" | grep -qi "gpu renderer.*checksum\|metal.*checksum"; then
    echo "✓ GPU renderer executed"
else
    echo "✗ GPU renderer may have fallen back to CPU"
fi

# Check if GPU PDE executed
if echo "${PDE_OUTPUT}" | grep -qi "gpu pde.*completed\|gpu pde.*packed"; then
    echo "✓ GPU PDE path executed"
else
    echo "✗ GPU PDE may have fallen back to CPU"
fi

echo ""
echo "=========================================="
echo "Testing Complete"
echo "=========================================="

