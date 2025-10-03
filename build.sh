#!/bin/bash

# Build script for Linux/macOS

echo "========================================"
echo "Building GPU Compute Simulator"
echo "========================================"

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake ..
if [ $? -ne 0 ]; then
    echo "Error: CMake configuration failed"
    echo "Please ensure CMake and a C++ compiler are installed"
    exit 1
fi

# Build
echo "Building project..."
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
if [ $? -ne 0 ]; then
    echo "Error: Build failed"
    exit 1
fi

echo ""
echo "========================================"
echo "Build completed successfully!"
echo "Executable: build/gpu_simulator"
echo "========================================"
echo ""
echo "To run: cd build && ./gpu_simulator"
