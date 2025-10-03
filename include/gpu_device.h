#ifndef GPU_DEVICE_H
#define GPU_DEVICE_H

#include "types.h"
#include "compute_unit.h"
#include "memory.h"
#include "scheduler.h"
#include "metrics.h"
#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <condition_variable>

namespace GPUSim {

// GPU Device Configuration
struct GPUConfig {
    size_t num_compute_units;
    size_t warps_per_cu;
    size_t threads_per_warp;
    size_t max_blocks_per_cu;
    size_t global_memory_size;
    size_t shared_memory_per_block;
    std::string device_name;

    // Default: Similar to NVIDIA RTX 3080
    GPUConfig()
        : num_compute_units(68),
          warps_per_cu(64),
          threads_per_warp(32),
          max_blocks_per_cu(16),
          global_memory_size(10ULL * 1024 * 1024 * 1024), // 10GB
          shared_memory_per_block(48 * 1024),
          device_name("GPU Simulator - RTX 3080 Profile") {}
};

// Main GPU Device class
class GPUDevice {
private:
    GPUConfig config_;
    std::vector<std::unique_ptr<ComputeUnit>> compute_units_;
    std::shared_ptr<MemoryController> memory_controller_;
    std::unique_ptr<Scheduler> scheduler_;

    // Execution control
    std::vector<std::thread> cu_threads_;
    std::atomic<bool> running_;
    std::atomic<bool> simulation_active_;
    std::mutex device_mutex_;
    std::condition_variable work_cv_;

    // Performance tracking
    std::unique_ptr<PerformanceAnalyzer> performance_analyzer_;
    std::atomic<uint64_t> global_cycle_count_;

    // Private methods
    void initializeComputeUnits();
    void distributorThread(); // Distributes blocks to CUs
    void cuExecutionThread(ComputeUnit* cu);

public:
    GPUDevice(const GPUConfig& config = GPUConfig());
    ~GPUDevice();

    // Device information
    const GPUConfig& getConfig() const { return config_; }
    size_t getNumComputeUnits() const { return compute_units_.size(); }

    // Scheduler management
    void setScheduler(std::unique_ptr<Scheduler> scheduler);
    Scheduler* getScheduler() { return scheduler_.get(); }

    // Workload management
    void submitWorkload(std::shared_ptr<Workload> workload);
    void executeWorkloads();
    void waitForCompletion();

    // Execution control
    void start();
    void stop();
    bool isRunning() const { return running_.load(); }

    // Performance metrics
    PerformanceAnalyzer* getPerformanceAnalyzer() { return performance_analyzer_.get(); }
    const PerformanceAnalyzer* getPerformanceAnalyzer() const { return performance_analyzer_.get(); }

    // Resource queries
    size_t getTotalActiveBlocks() const;
    size_t getTotalActiveWarps() const;
    double getAverageUtilization() const;

    const std::vector<std::unique_ptr<ComputeUnit>>& getComputeUnits() const {
        return compute_units_;
    }

    MemoryController* getMemoryController() { return memory_controller_.get(); }
    const MemoryController* getMemoryController() const { return memory_controller_.get(); }

    // Utility
    void printDeviceInfo() const;
    void reset();
};

} // namespace GPUSim

#endif // GPU_DEVICE_H
