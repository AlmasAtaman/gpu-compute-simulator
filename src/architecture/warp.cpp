#include "warp.h"
#include <algorithm>

namespace GPUSim {

// Thread implementation
Thread::Thread(ThreadID tid, WarpID wid, BlockID bid)
    : thread_id_(tid),
      warp_id_(wid),
      block_id_(bid),
      state_(ExecutionState::READY),
      registers_(std::make_unique<RegisterFile>()) {
    registers_->setOwner(tid);
}

// Warp implementation
Warp::Warp(WarpID wid, BlockID bid, size_t num_threads)
    : warp_id_(wid),
      block_id_(bid),
      state_(ExecutionState::READY),
      program_counter_(0),
      active_mask_((1ULL << num_threads) - 1), // All threads active initially
      instructions_executed_(0),
      cycles_stalled_(0) {

    threads_.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
        ThreadID tid = bid * MAX_THREADS_PER_BLOCK + wid * WARP_SIZE + i;
        threads_.push_back(std::make_unique<Thread>(tid, wid, bid));
    }
}

// ThreadBlock implementation
ThreadBlock::ThreadBlock(BlockID bid, size_t num_threads)
    : block_id_(bid),
      shared_memory_(std::make_shared<SharedMemory>()),
      state_(ExecutionState::READY),
      grid_x_(0), grid_y_(0), grid_z_(0),
      completed_(false) {

    shared_memory_->setOwner(bid);

    // Calculate number of warps needed
    size_t num_warps = (num_threads + WARP_SIZE - 1) / WARP_SIZE;

    warps_.reserve(num_warps);
    for (size_t i = 0; i < num_warps; ++i) {
        size_t threads_in_warp = std::min(WARP_SIZE, num_threads - i * WARP_SIZE);
        warps_.push_back(std::make_unique<Warp>(i, bid, threads_in_warp));
    }
}

Warp* ThreadBlock::getWarp(size_t index) {
    if (index >= warps_.size()) {
        return nullptr;
    }
    return warps_[index].get();
}

} // namespace GPUSim
