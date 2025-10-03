#include "memory.h"
#include <algorithm>
#include <stdexcept>

namespace GPUSim {

// GlobalMemory implementation
GlobalMemory::GlobalMemory(size_t size)
    : Memory(size, 400), // ~400 cycles latency for global memory
      data_(size, 0),
      read_count_(0),
      write_count_(0),
      bytes_read_(0),
      bytes_written_(0) {
}

bool GlobalMemory::read(MemoryAddress address, size_t bytes) {
    if (address + bytes > size_) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    access_count_++;
    read_count_++;
    bytes_read_ += bytes;

    // Simulate memory read (actual data access not needed for simulation)
    return true;
}

bool GlobalMemory::write(MemoryAddress address, size_t bytes) {
    if (address + bytes > size_) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    access_count_++;
    write_count_++;
    bytes_written_ += bytes;

    // Simulate memory write
    return true;
}

void GlobalMemory::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    access_count_ = 0;
    read_count_ = 0;
    write_count_ = 0;
    bytes_read_ = 0;
    bytes_written_ = 0;
    std::fill(data_.begin(), data_.end(), 0);
}

// SharedMemory implementation
SharedMemory::SharedMemory(size_t size)
    : Memory(size, 4), // ~4 cycles latency for shared memory
      data_(size, 0),
      owner_block_(0) {
}

bool SharedMemory::read(MemoryAddress address, size_t bytes) {
    if (address + bytes > size_) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    access_count_++;
    return true;
}

bool SharedMemory::write(MemoryAddress address, size_t bytes) {
    if (address + bytes > size_) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    access_count_++;
    return true;
}

void SharedMemory::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::fill(data_.begin(), data_.end(), 0);
    access_count_ = 0;
}

// RegisterFile implementation
RegisterFile::RegisterFile(size_t num_regs)
    : registers_(num_regs, 0),
      owner_thread_(0),
      num_registers_(num_regs) {
}

bool RegisterFile::readRegister(size_t reg_index, uint32_t& value) {
    if (reg_index >= num_registers_) {
        return false;
    }
    value = registers_[reg_index];
    return true;
}

bool RegisterFile::writeRegister(size_t reg_index, uint32_t value) {
    if (reg_index >= num_registers_) {
        return false;
    }
    registers_[reg_index] = value;
    return true;
}

void RegisterFile::clear() {
    std::fill(registers_.begin(), registers_.end(), 0);
}

// MemoryController implementation
MemoryController::MemoryController()
    : global_memory_(std::make_shared<GlobalMemory>()),
      total_memory_ops_(0),
      cache_hits_(0),
      cache_misses_(0) {
}

double MemoryController::getCacheHitRate() const {
    uint64_t total = cache_hits_.load() + cache_misses_.load();
    if (total == 0) return 0.0;
    return static_cast<double>(cache_hits_.load()) / total;
}

} // namespace GPUSim
