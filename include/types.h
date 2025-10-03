#ifndef TYPES_H
#define TYPES_H

#include <cstdint>
#include <memory>
#include <vector>
#include <string>

namespace GPUSim {

// Basic types
using ThreadID = uint32_t;
using WarpID = uint32_t;
using BlockID = uint32_t;
using CoreID = uint32_t;
using MemoryAddress = uint64_t;
using Timestamp = uint64_t;

// GPU configuration constants
constexpr size_t WARP_SIZE = 32;
constexpr size_t MAX_THREADS_PER_BLOCK = 1024;
constexpr size_t MAX_BLOCKS_PER_GRID = 65535;

// Memory sizes (in bytes)
constexpr size_t GLOBAL_MEMORY_SIZE = 8ULL * 1024 * 1024 * 1024; // 8GB
constexpr size_t SHARED_MEMORY_PER_BLOCK = 48 * 1024; // 48KB
constexpr size_t REGISTERS_PER_THREAD = 255;

// Workload types
enum class WorkloadType {
    MATRIX_MULTIPLY,
    CONVOLUTION,
    VECTOR_ADD,
    REDUCTION,
    CUSTOM
};

// Scheduling algorithms
enum class SchedulingAlgorithm {
    FIFO,
    PRIORITY,
    ROUND_ROBIN,
    SHORTEST_JOB_FIRST
};

// Thread/Warp states
enum class ExecutionState {
    IDLE,
    READY,
    RUNNING,
    MEMORY_STALLED,
    COMPLETED
};

} // namespace GPUSim

#endif // TYPES_H
