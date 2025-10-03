#ifndef WORKLOAD_H
#define WORKLOAD_H

#include "types.h"
#include "warp.h"
#include <string>
#include <functional>
#include <vector>
#include <memory>
#include <chrono>

namespace GPUSim {

// Kernel configuration (similar to CUDA's dim3)
struct KernelConfig {
    size_t grid_dim_x;
    size_t grid_dim_y;
    size_t grid_dim_z;
    size_t block_dim_x;
    size_t block_dim_y;
    size_t block_dim_z;

    KernelConfig(size_t gx = 1, size_t gy = 1, size_t gz = 1,
                 size_t bx = 256, size_t by = 1, size_t bz = 1)
        : grid_dim_x(gx), grid_dim_y(gy), grid_dim_z(gz),
          block_dim_x(bx), block_dim_y(by), block_dim_z(bz) {}

    size_t getTotalBlocks() const {
        return grid_dim_x * grid_dim_y * grid_dim_z;
    }

    size_t getThreadsPerBlock() const {
        return block_dim_x * block_dim_y * block_dim_z;
    }

    size_t getTotalThreads() const {
        return getTotalBlocks() * getThreadsPerBlock();
    }
};

// Workload represents a GPU kernel/task
class Workload {
private:
    std::string name_;
    WorkloadType type_;
    KernelConfig config_;
    int priority_;
    size_t estimated_instructions_;
    size_t estimated_memory_ops_;

    // Execution tracking
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point end_time_;
    bool completed_;

    // Thread blocks for this workload
    std::vector<std::unique_ptr<ThreadBlock>> thread_blocks_;

public:
    Workload(const std::string& name, WorkloadType type, const KernelConfig& config);

    const std::string& getName() const { return name_; }
    WorkloadType getType() const { return type_; }
    const KernelConfig& getConfig() const { return config_; }

    int getPriority() const { return priority_; }
    void setPriority(int priority) { priority_ = priority; }

    size_t getEstimatedInstructions() const { return estimated_instructions_; }
    void setEstimatedInstructions(size_t count) { estimated_instructions_ = count; }

    size_t getEstimatedMemoryOps() const { return estimated_memory_ops_; }
    void setEstimatedMemoryOps(size_t count) { estimated_memory_ops_ = count; }

    // Block management
    void generateThreadBlocks();
    std::unique_ptr<ThreadBlock> getNextBlock();
    bool hasMoreBlocks() const;
    size_t getRemainingBlocks() const { return thread_blocks_.size(); }

    // Execution tracking
    void start();
    void complete();
    bool isCompleted() const { return completed_; }
    double getExecutionTime() const; // in milliseconds

    // Create common workload types
    static std::unique_ptr<Workload> createMatrixMultiply(size_t M, size_t N, size_t K);
    static std::unique_ptr<Workload> createConvolution(size_t batch, size_t channels, size_t height, size_t width);
    static std::unique_ptr<Workload> createVectorAdd(size_t size);
    static std::unique_ptr<Workload> createReduction(size_t size);
};

} // namespace GPUSim

#endif // WORKLOAD_H
