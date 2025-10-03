#include "gpu_device.h"
#include <iostream>
#include <iomanip>
#include <algorithm>

namespace GPUSim {

GPUDevice::GPUDevice(const GPUConfig& config)
    : config_(config),
      memory_controller_(std::make_shared<MemoryController>()),
      scheduler_(std::make_unique<FIFOScheduler>()),
      running_(false),
      simulation_active_(false),
      performance_analyzer_(std::make_unique<PerformanceAnalyzer>()),
      global_cycle_count_(0) {

    initializeComputeUnits();
}

GPUDevice::~GPUDevice() {
    stop();
}

void GPUDevice::initializeComputeUnits() {
    compute_units_.reserve(config_.num_compute_units);

    for (size_t i = 0; i < config_.num_compute_units; ++i) {
        compute_units_.push_back(std::make_unique<ComputeUnit>(i, memory_controller_));
    }

    std::cout << "Initialized " << config_.num_compute_units << " compute units\n";
}

void GPUDevice::setScheduler(std::unique_ptr<Scheduler> scheduler) {
    std::lock_guard<std::mutex> lock(device_mutex_);
    scheduler_ = std::move(scheduler);
}

void GPUDevice::submitWorkload(std::shared_ptr<Workload> workload) {
    if (!workload) return;

    // Generate thread blocks for the workload
    workload->generateThreadBlocks();

    // Add to scheduler
    scheduler_->addWorkload(workload);

    std::cout << "Submitted workload: " << workload->getName()
              << " (" << workload->getConfig().getTotalBlocks() << " blocks, "
              << workload->getConfig().getTotalThreads() << " threads)\n";
}

void GPUDevice::distributorThread() {
    while (running_.load()) {
        // Check if there are pending workloads
        if (!scheduler_->hasPendingWorkloads()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        // Get next workload from scheduler
        auto workload = scheduler_->getNextWorkload();
        if (!workload) continue;

        std::cout << "Starting workload: " << workload->getName() << "\n";
        workload->start();

        // Distribute blocks to available compute units
        while (workload->hasMoreBlocks()) {
            auto block = workload->getNextBlock();
            if (!block) break;

            // Find an available compute unit
            bool assigned = false;
            while (!assigned && running_.load()) {
                for (auto& cu : compute_units_) {
                    if (cu->canAcceptBlock(block.get())) {
                        cu->assignBlock(std::move(block));
                        assigned = true;
                        break;
                    }
                }

                if (!assigned) {
                    // Wait a bit and clean up completed blocks
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    for (auto& cu : compute_units_) {
                        cu->removeCompletedBlocks();
                    }
                }
            }
        }

        // Wait for all blocks to complete
        bool all_idle = false;
        while (!all_idle && running_.load()) {
            all_idle = true;
            for (auto& cu : compute_units_) {
                cu->removeCompletedBlocks();
                if (!cu->isIdle()) {
                    all_idle = false;
                }
            }

            if (!all_idle) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }

        // Workload completed
        workload->complete();
        scheduler_->markWorkloadCompleted(workload);

        std::cout << "Completed workload: " << workload->getName()
                  << " in " << std::fixed << std::setprecision(2)
                  << workload->getExecutionTime() << " ms\n";

        // Record metrics
        performance_analyzer_->recordWorkloadMetrics(workload.get(), this);
    }
}

void GPUDevice::cuExecutionThread(ComputeUnit* cu) {
    if (!cu) return;
    cu->run();
}

void GPUDevice::executeWorkloads() {
    if (running_.load()) {
        std::cerr << "GPU is already running\n";
        return;
    }

    start();
}

void GPUDevice::waitForCompletion() {
    // Wait until all workloads are completed
    while (scheduler_->hasPendingWorkloads() || scheduler_->getRunningCount() > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    stop();
}

void GPUDevice::start() {
    if (running_.load()) return;

    running_.store(true);
    simulation_active_.store(true);

    performance_analyzer_->startSimulation();

    // Start compute unit threads
    for (auto& cu : compute_units_) {
        cu_threads_.emplace_back(&GPUDevice::cuExecutionThread, this, cu.get());
    }

    // Start distributor thread
    cu_threads_.emplace_back(&GPUDevice::distributorThread, this);

    std::cout << "GPU Device started with " << config_.num_compute_units << " compute units\n";
}

void GPUDevice::stop() {
    if (!running_.load()) return;

    running_.store(false);

    // Stop all compute units FIRST
    for (auto& cu : compute_units_) {
        cu->stop();
    }

    // Then join all threads
    for (auto& thread : cu_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    cu_threads_.clear();

    if (simulation_active_.load()) {
        performance_analyzer_->endSimulation();
        performance_analyzer_->recordGPUMetrics(this);
        simulation_active_.store(false);
    }

    std::cout << "GPU Device stopped\n";
}

size_t GPUDevice::getTotalActiveBlocks() const {
    size_t total = 0;
    for (const auto& cu : compute_units_) {
        total += cu->getActiveBlockCount();
    }
    return total;
}

size_t GPUDevice::getTotalActiveWarps() const {
    size_t total = 0;
    for (const auto& cu : compute_units_) {
        total += cu->getActiveWarpCount();
    }
    return total;
}

double GPUDevice::getAverageUtilization() const {
    if (compute_units_.empty()) return 0.0;

    double total_util = 0.0;
    for (const auto& cu : compute_units_) {
        total_util += cu->getUtilization();
    }

    return total_util / compute_units_.size();
}

void GPUDevice::printDeviceInfo() const {
    std::cout << "\n========================================\n";
    std::cout << "  GPU DEVICE INFORMATION\n";
    std::cout << "========================================\n";
    std::cout << "Device Name: " << config_.device_name << "\n";
    std::cout << "Compute Units: " << config_.num_compute_units << "\n";
    std::cout << "Warps per CU: " << config_.warps_per_cu << "\n";
    std::cout << "Threads per Warp: " << config_.threads_per_warp << "\n";
    std::cout << "Max Blocks per CU: " << config_.max_blocks_per_cu << "\n";
    std::cout << "Global Memory: " << (config_.global_memory_size / (1024*1024*1024)) << " GB\n";
    std::cout << "Shared Memory per Block: " << (config_.shared_memory_per_block / 1024) << " KB\n";
    std::cout << "========================================\n\n";
}

void GPUDevice::reset() {
    stop();

    for (auto& cu : compute_units_) {
        cu->resetMetrics();
    }

    performance_analyzer_->reset();
    global_cycle_count_ = 0;

    std::cout << "GPU Device reset\n";
}

} // namespace GPUSim
