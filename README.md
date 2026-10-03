# GPU Compute Simulator

A multi-threaded C++17 model of a GPU, built to see how different scheduling policies change utilization and runtime on ML-style workloads.

The default device is shaped like an RTX 3080: 68 compute units, 64 warps per unit, 32 threads per warp and a 10 GB global memory space. Each compute unit runs on its own `std::thread` and executes warps in SIMT fashion against a three-level memory hierarchy.

## What it models

- **Execution:** threads → warps → thread blocks, scheduled onto compute units by a per-unit warp scheduler
- **Memory:** global memory, per-block shared memory (48 KB) and per-thread register files, with a memory controller tracking bandwidth
- **Scheduling:** FIFO, priority, shortest-job-first and round-robin, all behind a common `Scheduler` interface and created through a factory
- **Workloads:** matrix multiply, convolution, vector add, reduction, plus custom kernels
- **Metrics:** per-workload runtime, compute-unit utilization, throughput (instructions/ms) and memory bandwidth utilization, with a side-by-side scheduler comparison

## Layout

```
include/            public headers (device, compute unit, warp, memory, scheduler, metrics)
src/architecture/   GPUDevice, ComputeUnit, Warp, Workload
src/memory/         global/shared memory, register file, memory controller
src/scheduler/      scheduling policies + factory
src/metrics/        PerformanceAnalyzer and SchedulerComparison
src/main.cpp        interactive menu
```

## Build and run

Requires CMake 3.12+ and a C++17 compiler.

```bash
./build.sh            # Linux / macOS
build.bat             # Windows (MSVC)
./build-mingw.sh      # Windows (MinGW)

cd build && ./gpu_simulator
```

The menu lets you run:

1. **Basic simulation:** a few workloads under FIFO
2. **Scheduler comparison:** the same workload mix under all four policies
3. **ML workload:** a ResNet-style sequence of convolution and matrix-multiply layers
4. **Custom benchmark:** mixed workload sizes with a performance breakdown

> The default profile reserves a 10 GB global memory space. On machines with less RAM, lower `global_memory_size` in `GPUConfig` (`include/gpu_device.h`).
