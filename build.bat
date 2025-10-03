@echo off
REM Build script for Windows

echo ========================================
echo Building GPU Compute Simulator
echo ========================================

REM Create build directory
if not exist build mkdir build
cd build

REM Configure with CMake
echo Configuring with CMake...
cmake .. -G "MinGW Makefiles"
if %ERRORLEVEL% NEQ 0 (
    echo Error: CMake configuration failed
    echo Please ensure CMake and a C++ compiler are installed
    pause
    exit /b 1
)

REM Build
echo Building project...
mingw32-make
if %ERRORLEVEL% NEQ 0 (
    echo Error: Build failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo Build completed successfully!
echo Executable: build\gpu_simulator.exe
echo ========================================
echo.
echo To run: cd build ^&^& gpu_simulator.exe
pause
