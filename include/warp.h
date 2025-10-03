#ifndef WARP_H
#define WARP_H

#include "types.h"
#include "memory.h"
#include <vector>
#include <functional>
#include <atomic>

namespace GPUSim {

// Represents a single GPU thread
class Thread {
private:
    ThreadID thread_id_;
    WarpID warp_id_;
    BlockID block_id_;
    ExecutionState state_;
    std::unique_ptr<RegisterFile> registers_;

public:
    Thread(ThreadID tid, WarpID wid, BlockID bid);

    ThreadID getThreadID() const { return thread_id_; }
    WarpID getWarpID() const { return warp_id_; }
    BlockID getBlockID() const { return block_id_; }
    ExecutionState getState() const { return state_; }

    void setState(ExecutionState state) { state_ = state; }
    RegisterFile* getRegisters() { return registers_.get(); }
};

// Warp: Group of threads that execute in lockstep (SIMT)
class Warp {
private:
    WarpID warp_id_;
    BlockID block_id_;
    std::vector<std::unique_ptr<Thread>> threads_;
    ExecutionState state_;
    size_t program_counter_;
    size_t active_mask_; // Bitmask for active threads
    std::atomic<uint64_t> instructions_executed_;
    std::atomic<uint64_t> cycles_stalled_;

public:
    Warp(WarpID wid, BlockID bid, size_t num_threads = WARP_SIZE);

    WarpID getWarpID() const { return warp_id_; }
    BlockID getBlockID() const { return block_id_; }
    ExecutionState getState() const { return state_; }
    void setState(ExecutionState state) { state_ = state; }

    size_t getNumThreads() const { return threads_.size(); }
    size_t getActiveMask() const { return active_mask_; }
    void setActiveMask(size_t mask) { active_mask_ = mask; }

    size_t getProgramCounter() const { return program_counter_; }
    void incrementPC() { program_counter_++; }

    void recordInstruction() { instructions_executed_++; }
    void recordStall() { cycles_stalled_++; }

    uint64_t getInstructionsExecuted() const { return instructions_executed_.load(); }
    uint64_t getCyclesStalled() const { return cycles_stalled_.load(); }

    const std::vector<std::unique_ptr<Thread>>& getThreads() const { return threads_; }
};

// Thread Block: Collection of warps
class ThreadBlock {
private:
    BlockID block_id_;
    std::vector<std::unique_ptr<Warp>> warps_;
    std::shared_ptr<SharedMemory> shared_memory_;
    ExecutionState state_;
    size_t grid_x_, grid_y_, grid_z_; // Position in grid
    std::atomic<bool> completed_;

public:
    ThreadBlock(BlockID bid, size_t num_threads);

    BlockID getBlockID() const { return block_id_; }
    ExecutionState getState() const { return state_; }
    void setState(ExecutionState state) { state_ = state; }

    size_t getNumWarps() const { return warps_.size(); }
    const std::vector<std::unique_ptr<Warp>>& getWarps() const { return warps_; }
    Warp* getWarp(size_t index);

    SharedMemory* getSharedMemory() { return shared_memory_.get(); }

    void setGridPosition(size_t x, size_t y, size_t z) {
        grid_x_ = x; grid_y_ = y; grid_z_ = z;
    }

    bool isCompleted() const { return completed_.load(); }
    void markCompleted() { completed_.store(true); }
};

} // namespace GPUSim

#endif // WARP_H
