#ifndef MEMORY_H
#define MEMORY_H

#include "types.h"
#include <vector>
#include <mutex>
#include <atomic>

namespace GPUSim {

// Base memory class
class Memory {
protected:
    size_t size_;
    size_t latency_cycles_;
    std::atomic<uint64_t> access_count_;
    mutable std::mutex mutex_;

public:
    Memory(size_t size, size_t latency)
        : size_(size), latency_cycles_(latency), access_count_(0) {}

    virtual ~Memory() = default;

    virtual bool read(MemoryAddress address, size_t bytes) = 0;
    virtual bool write(MemoryAddress address, size_t bytes) = 0;

    size_t getSize() const { return size_; }
    size_t getLatency() const { return latency_cycles_; }
    uint64_t getAccessCount() const { return access_count_.load(); }
};

// Global GPU memory (GDDR/HBM)
class GlobalMemory : public Memory {
private:
    std::vector<uint8_t> data_;
    std::atomic<uint64_t> read_count_;
    std::atomic<uint64_t> write_count_;
    std::atomic<uint64_t> bytes_read_;
    std::atomic<uint64_t> bytes_written_;

public:
    GlobalMemory(size_t size = GLOBAL_MEMORY_SIZE);

    bool read(MemoryAddress address, size_t bytes) override;
    bool write(MemoryAddress address, size_t bytes) override;

    uint64_t getReadCount() const { return read_count_.load(); }
    uint64_t getWriteCount() const { return write_count_.load(); }
    uint64_t getBytesRead() const { return bytes_read_.load(); }
    uint64_t getBytesWritten() const { return bytes_written_.load(); }

    void reset();
};

// Shared memory (per thread block)
class SharedMemory : public Memory {
private:
    std::vector<uint8_t> data_;
    BlockID owner_block_;

public:
    SharedMemory(size_t size = SHARED_MEMORY_PER_BLOCK);

    bool read(MemoryAddress address, size_t bytes) override;
    bool write(MemoryAddress address, size_t bytes) override;

    void setOwner(BlockID block_id) { owner_block_ = block_id; }
    BlockID getOwner() const { return owner_block_; }
    void clear();
};

// Register file (per thread)
class RegisterFile {
private:
    std::vector<uint32_t> registers_;
    ThreadID owner_thread_;
    size_t num_registers_;

public:
    RegisterFile(size_t num_regs = REGISTERS_PER_THREAD);

    bool readRegister(size_t reg_index, uint32_t& value);
    bool writeRegister(size_t reg_index, uint32_t value);

    void setOwner(ThreadID thread_id) { owner_thread_ = thread_id; }
    ThreadID getOwner() const { return owner_thread_; }
    void clear();
};

// Memory controller for managing memory hierarchy
class MemoryController {
private:
    std::shared_ptr<GlobalMemory> global_memory_;
    std::atomic<uint64_t> total_memory_ops_;
    std::atomic<uint64_t> cache_hits_;
    std::atomic<uint64_t> cache_misses_;

public:
    MemoryController();

    std::shared_ptr<GlobalMemory> getGlobalMemory() { return global_memory_; }

    void recordMemoryOp() { total_memory_ops_++; }
    void recordCacheHit() { cache_hits_++; }
    void recordCacheMiss() { cache_misses_++; }

    double getCacheHitRate() const;
    uint64_t getTotalMemoryOps() const { return total_memory_ops_.load(); }
};

} // namespace GPUSim

#endif // MEMORY_H
