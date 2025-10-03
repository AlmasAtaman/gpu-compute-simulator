#!/bin/bash
# Build script for GPU Compute Simulator using MinGW on Git Bash

echo "======================================"
echo "  GPU Compute Simulator - Build Script"
echo "======================================"
echo ""

# Navigate to build directory
cd "$(dirname "$0")/build" || exit 1

# Clean build directory
echo "Cleaning build directory..."
rm -rf *

# Configure with CMake
echo "Configuring with CMake..."
cmake .. -G "MinGW Makefiles"

if [ $? -ne 0 ]; then
    echo "ERROR: CMake configuration failed!"
    exit 1
fi

# Build
echo ""
echo "Building gpu_simulator..."
mingw32-make gpu_simulator -j4

if [ $? -ne 0 ]; then
    echo "ERROR: Build failed!"
    exit 1
fi

echo ""
echo "======================================"
echo "  Build Successful!"
echo "======================================"
echo ""
echo "Executable: $(pwd)/gpu_simulator.exe"
echo "Size: $(ls -lh gpu_simulator.exe | awk '{print $5}')"
echo ""
echo "To run the simulator:"
echo "  cd build"
echo "  ./gpu_simulator.exe"
echo ""
