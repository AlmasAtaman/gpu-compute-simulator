#include "compute_unit.h"
#include <algorithm>

namespace GPUSim {

// WarpScheduler implementation
WarpScheduler::WarpScheduler(size_t max_warps)
    : max_warps_(max_warps) {
}

bool WarpScheduler::addWarp(Warp* warp) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (ready_queue_.size() >= max_warps_) {
        return false;
    }
    if (warp && warp->getState() == ExecutionState::READY) {
        ready_queue_.push(warp);
        return true;
    }
    return false;
}

Warp* WarpScheduler::getNextWarp() {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (ready_queue_.empty()) {
        return nullptr;
    }

    Warp* warp = ready_queue_.front();
    ready_queue_.pop();
    return warp;
}

bool WarpScheduler::hasReadyWarps() const {
    return !ready_queue_.empty();
}

size_t WarpScheduler::getQueueSize() const {
    return ready_queue_.size();
}

// ComputeUnit implementation
ComputeUnit::ComputeUnit(CoreID id, std::shared_ptr<MemoryController> mem_ctrl)
    : core_id_(id),
      warp_scheduler_(64),
      max_warps_per_cu_(64),
      max_threads_per_cu_(2048),
      max_blocks_per_cu_(16),
      state_(ExecutionState::IDLE),
      running_(false),
      cycles_executed_(0),
      instructions_executed_(0),
      warps_executed_(0),
      idle_cycles_(0),
      memory_controller_(mem_ctrl) {
}

bool ComputeUnit::canAcceptBlock(const ThreadBlock* block) const {
    if (!block) return false;

    size_t current_blocks = active_blocks_.size();
    if (current_blocks >= max_blocks_per_cu_) {
        return false;
    }

    // Check if we have enough resources
    size_t current_warps = getActiveWarpCount();
    if (current_warps + block->getNumWarps() > max_warps_per_cu_) {
        return false;
    }

    return true;
}

bool ComputeUnit::assignBlock(std::unique_ptr<ThreadBlock> block) {
    std::lock_guard<std::mutex> lock(cu_mutex_);

    if (!canAcceptBlock(block.get())) {
        return false;
    }

    // Add all warps from the block to the scheduler
    for (const auto& warp : block->getWarps()) {
        warp_scheduler_.addWarp(warp.get());
    }

    active_blocks_.push_back(std::move(block));
    state_ = ExecutionState::RUNNING;
    return true;
}

void ComputeUnit::removeCompletedBlocks() {
    std::lock_guard<std::mutex> lock(cu_mutex_);

    active_blocks_.erase(
        std::remove_if(active_blocks_.begin(), active_blocks_.end(),
            [](const std::unique_ptr<ThreadBlock>& block) {
                return block->isCompleted();
            }),
        active_blocks_.end()
    );

    if (active_blocks_.empty()) {
        state_ = ExecutionState::IDLE;
    }
}

void ComputeUnit::executeWarp(Warp* warp, size_t num_instructions) {
    if (!warp) return;

    warp->setState(ExecutionState::RUNNING);

    for (size_t i = 0; i < num_instructions; ++i) {
        warp->recordInstruction();
        warp->incrementPC();
        instructions_executed_++;

        // Simulate occasional memory accesses (20% of instructions)
        if (i % 5 == 0) {
            memory_controller_->recordMemoryOp();

            // Simulate memory stalls (10% chance)
            if (i % 10 == 0) {
                warp->setState(ExecutionState::MEMORY_STALLED);
                warp->recordStall();
                cycles_stalled_++;

                // Simulate stall duration
                for (size_t s = 0; s < memory_controller_->getGlobalMemory()->getLatency() / 10; ++s) {
                    cycles_executed_++;
                }

                warp->setState(ExecutionState::RUNNING);
            }
        }
    }

    warp->setState(ExecutionState::READY);
    warps_executed_++;
}

void ComputeUnit::simulateCycle() {
    cycles_executed_++;

    // Try to fetch and execute a warp
    Warp* warp = warp_scheduler_.getNextWarp();

    if (warp) {
        // Execute one instruction batch (simulate SIMD execution)
        executeWarp(warp, 8); // Execute 8 instructions per cycle

        // Check if warp is completed (simplified: after certain instructions)
        if (warp->getInstructionsExecuted() >= 1000) {
            warp->setState(ExecutionState::COMPLETED);

            // Check if all warps in the block are completed (with mutex protection)
            std::lock_guard<std::mutex> lock(cu_mutex_);
            for (auto& block : active_blocks_) {
                bool all_warps_done = true;
                for (const auto& w : block->getWarps()) {
                    if (w->getState() != ExecutionState::COMPLETED) {
                        all_warps_done = false;
                        break;
                    }
                }
                if (all_warps_done) {
                    block->markCompleted();
                }
            }
        } else {
            // Re-add to scheduler if not completed
            warp_scheduler_.addWarp(warp);
        }
    } else {
        idle_cycles_++;
    }
}

void ComputeUnit::run() {
    running_.store(true);
    while (running_.load()) {
        if (!active_blocks_.empty() && warp_scheduler_.hasReadyWarps()) {
            simulateCycle();
        } else {
            // Sleep briefly if no work
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
}

void ComputeUnit::stop() {
    running_.store(false);
}

size_t ComputeUnit::getActiveWarpCount() const {
    size_t count = 0;
    for (const auto& block : active_blocks_) {
        count += block->getNumWarps();
    }
    return count;
}

size_t ComputeUnit::getActiveThreadCount() const {
    size_t count = 0;
    for (const auto& block : active_blocks_) {
        for (const auto& warp : block->getWarps()) {
            count += warp->getNumThreads();
        }
    }
    return count;
}

double ComputeUnit::getUtilization() const {
    uint64_t total_cycles = cycles_executed_.load();
    if (total_cycles == 0) return 0.0;

    uint64_t active_cycles = total_cycles - idle_cycles_.load();
    return static_cast<double>(active_cycles) / total_cycles * 100.0;
}

void ComputeUnit::resetMetrics() {
    cycles_executed_ = 0;
    instructions_executed_ = 0;
    warps_executed_ = 0;
    idle_cycles_ = 0;
}

} // namespace GPUSim
