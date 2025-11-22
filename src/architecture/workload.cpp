#include "workload.h"
#include <algorithm>
#include <cmath>

namespace GPUSim {

Workload::Workload(const std::string& name, WorkloadType type, const KernelConfig& config)
    : name_(name),
      type_(type),
      config_(config),
      priority_(0),
      estimated_instructions_(0),
      estimated_memory_ops_(0),
      completed_(false) {
}

void Workload::generateThreadBlocks() {
    thread_blocks_.clear();
    size_t total_blocks = config_.getTotalBlocks();

    for (size_t i = 0; i < total_blocks; ++i) {
        size_t threads_per_block = config_.getThreadsPerBlock();
        auto block = std::make_unique<ThreadBlock>(i, threads_per_block);

        // Calculate 3D grid position
        size_t grid_xy = config_.grid_dim_x * config_.grid_dim_y;
        size_t z = i / grid_xy;
        size_t remaining = i % grid_xy;
        size_t y = remaining / config_.grid_dim_x;
        size_t x = remaining % config_.grid_dim_x;

        block->setGridPosition(x, y, z);
        thread_blocks_.push_back(std::move(block));
    }
}

std::unique_ptr<ThreadBlock> Workload::getNextBlock() {
    if (thread_blocks_.empty()) {
        return nullptr;
    }

    auto block = std::move(thread_blocks_.back());
    thread_blocks_.pop_back();
    return block;
}

bool Workload::hasMoreBlocks() const {
    return !thread_blocks_.empty();
}

void Workload::start() {
    start_time_ = std::chrono::high_resolution_clock::now();
}

void Workload::complete() {
    end_time_ = std::chrono::high_resolution_clock::now();
    completed_ = true;
}

double Workload::getExecutionTime() const {
    if (!completed_) return 0.0;

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time_ - start_time_);
    return duration.count() / 1000.0; // Convert to milliseconds
}

// Factory methods for common workloads
std::unique_ptr<Workload> Workload::createMatrixMultiply(size_t M, size_t N, size_t K) {

    size_t grid_x = (M + 15) / 16;
    size_t grid_y = (N + 15) / 16;

    KernelConfig config(grid_x, grid_y, 1, 16, 16, 1);

    auto workload = std::make_unique<Workload>(
        "MatrixMultiply_" + std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K),
        WorkloadType::MATRIX_MULTIPLY,
        config
    );

    // Estimate: Each thread does K multiply-adds, plus memory ops
    workload->setEstimatedInstructions(M * N * K * 2);
    workload->setEstimatedMemoryOps(M * N * (K + 2));

    return workload;
}

std::unique_ptr<Workload> Workload::createConvolution(size_t batch, size_t channels, size_t height, size_t width) {
    // Simplified: one thread per output pixel
    size_t total_outputs = batch * channels * height * width;
    size_t threads_per_block = 256;
    size_t num_blocks = (total_outputs + threads_per_block - 1) / threads_per_block;

    KernelConfig config(num_blocks, 1, 1, threads_per_block, 1, 1);

    auto workload = std::make_unique<Workload>(
        "Convolution_" + std::to_string(batch) + "x" + std::to_string(channels) +
        "x" + std::to_string(height) + "x" + std::to_string(width),
        WorkloadType::CONVOLUTION,
        config
    );

    // Estimate: 3x3 kernel, 9 multiply-adds per output
    workload->setEstimatedInstructions(total_outputs * 9 * 2);
    workload->setEstimatedMemoryOps(total_outputs * 10);

    return workload;
}

std::unique_ptr<Workload> Workload::createVectorAdd(size_t size) {
    size_t threads_per_block = 256;
    size_t num_blocks = (size + threads_per_block - 1) / threads_per_block;

    KernelConfig config(num_blocks, 1, 1, threads_per_block, 1, 1);

    auto workload = std::make_unique<Workload>(
        "VectorAdd_" + std::to_string(size),
        WorkloadType::VECTOR_ADD,
        config
    );

    workload->setEstimatedInstructions(size * 2); // Load, add, store
    workload->setEstimatedMemoryOps(size * 3); // 2 reads, 1 write

    return workload;
}

std::unique_ptr<Workload> Workload::createReduction(size_t size) {
    size_t threads_per_block = 256;
    size_t num_blocks = (size + threads_per_block - 1) / threads_per_block;

    KernelConfig config(num_blocks, 1, 1, threads_per_block, 1, 1);

    auto workload = std::make_unique<Workload>(
        "Reduction_" + std::to_string(size),
        WorkloadType::REDUCTION,
        config
    );

    // Reduction requires log2(size) steps
    size_t steps = static_cast<size_t>(std::log2(size));
    workload->setEstimatedInstructions(size * steps);
    workload->setEstimatedMemoryOps(size * 2);

    return workload;
}

} // namespace GPUSim
