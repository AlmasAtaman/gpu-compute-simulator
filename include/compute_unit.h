#ifndef COMPUTE_UNIT_H
#define COMPUTE_UNIT_H

#include "types.h"
#include "warp.h"
#include "memory.h"
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <chrono>

namespace GPUSim {

// Warp Scheduler: Selects which warp to execute on a compute unit
class WarpScheduler {
private:
    std::queue<Warp*> ready_queue_;
    std::mutex queue_mutex_;
    size_t max_warps_;

public:
    WarpScheduler(size_t max_warps = 64);

    bool addWarp(Warp* warp);
    Warp* getNextWarp();
    bool hasReadyWarps() const;
    size_t getQueueSize() const;
};

// Compute Unit (equivalent to NVIDIA's SM - Streaming Multiprocessor)
class ComputeUnit {
private:
    CoreID core_id_;
    std::vector<std::unique_ptr<ThreadBlock>> active_blocks_;
    WarpScheduler warp_scheduler_;

    // Hardware resources
    size_t max_warps_per_cu_;
    size_t max_threads_per_cu_;
    size_t max_blocks_per_cu_;

    // Execution state
    ExecutionState state_;
    std::atomic<bool> running_;
    std::mutex cu_mutex_;

    // Performance metrics
    std::atomic<uint64_t> cycles_executed_;
    std::atomic<uint64_t> instructions_executed_;
    std::atomic<uint64_t> warps_executed_;
    std::atomic<uint64_t> idle_cycles_;
    std::atomic<uint64_t> cycles_stalled_;

    // Memory controller reference
    std::shared_ptr<MemoryController> memory_controller_;

public:
    ComputeUnit(CoreID id, std::shared_ptr<MemoryController> mem_ctrl);

    CoreID getCoreID() const { return core_id_; }
    ExecutionState getState() const { return state_; }

    // Block management
    bool canAcceptBlock(const ThreadBlock* block) const;
    bool assignBlock(std::unique_ptr<ThreadBlock> block);
    void removeCompletedBlocks();

    // Execution
    void executeWarp(Warp* warp, size_t num_instructions);
    void simulateCycle();
    void run();
    void stop();

    // Resource queries
    size_t getActiveBlockCount() const { return active_blocks_.size(); }
    size_t getActiveWarpCount() const;
    size_t getActiveThreadCount() const;

    bool isIdle() const { return active_blocks_.empty() && state_ == ExecutionState::IDLE; }
    bool isRunning() const { return running_.load(); }

    // Metrics
    uint64_t getCyclesExecuted() const { return cycles_executed_.load(); }
    uint64_t getInstructionsExecuted() const { return instructions_executed_.load(); }
    uint64_t getWarpsExecuted() const { return warps_executed_.load(); }
    uint64_t getIdleCycles() const { return idle_cycles_.load(); }
    double getUtilization() const;

    void resetMetrics();
};

} 
#endif // COMPUTE_UNIT_H
