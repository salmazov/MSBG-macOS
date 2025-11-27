#!/bin/bash
# Simple test to check Metal availability

cd "$(dirname "$0")/build"

echo "Testing Metal availability..."
echo ""

# Check if binary exists
if [ ! -f "./msbg_demo" ]; then
    echo "ERROR: msbg_demo not found"
    exit 1
fi

# Try to get Metal availability info
echo "Running with minimal parameters to check Metal detection..."
./msbg_demo -g gpu -c1 -b16 -r32 -v3 -h 2>&1 | head -20 || echo "Command failed"

echo ""
echo "Checking for Metal symbols in binary..."
nm msbg_demo 2>/dev/null | grep -i "Metal\|renderSceneMetal" | head -3

echo ""
echo "Checking binary type..."
file msbg_demo

