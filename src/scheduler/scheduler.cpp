#include "scheduler.h"
#include <algorithm>

namespace GPUSim {

// Base Scheduler implementation
void Scheduler::addWorkload(std::shared_ptr<Workload> workload) {
    std::lock_guard<std::mutex> lock(scheduler_mutex_);
    pending_workloads_.push_back(workload);
}

bool Scheduler::hasPendingWorkloads() const {
    return !pending_workloads_.empty();
}

void Scheduler::markWorkloadRunning(std::shared_ptr<Workload> workload) {
    std::lock_guard<std::mutex> lock(scheduler_mutex_);

    auto it = std::find(pending_workloads_.begin(), pending_workloads_.end(), workload);
    if (it != pending_workloads_.end()) {
        pending_workloads_.erase(it);
        running_workloads_.push_back(workload);
    }
}

void Scheduler::markWorkloadCompleted(std::shared_ptr<Workload> workload) {
    std::lock_guard<std::mutex> lock(scheduler_mutex_);

    auto it = std::find(running_workloads_.begin(), running_workloads_.end(), workload);
    if (it != running_workloads_.end()) {
        running_workloads_.erase(it);
        completed_workloads_.push_back(workload);
    }
}

// FIFOScheduler implementation
std::shared_ptr<Workload> FIFOScheduler::getNextWorkload() {
    std::lock_guard<std::mutex> lock(scheduler_mutex_);

    if (pending_workloads_.empty()) {
        return nullptr;
    }

    auto workload = pending_workloads_.front();
    pending_workloads_.erase(pending_workloads_.begin());
    running_workloads_.push_back(workload);

    return workload;
}

// PriorityScheduler implementation
std::shared_ptr<Workload> PriorityScheduler::getNextWorkload() {
    std::lock_guard<std::mutex> lock(scheduler_mutex_);

    if (pending_workloads_.empty()) {
        return nullptr;
    }

    // Find workload with highest priority
    auto it = std::max_element(pending_workloads_.begin(), pending_workloads_.end(),
        [](const std::shared_ptr<Workload>& a, const std::shared_ptr<Workload>& b) {
            return a->getPriority() < b->getPriority();
        });

    auto workload = *it;
    pending_workloads_.erase(it);
    running_workloads_.push_back(workload);

    return workload;
}

// RoundRobinScheduler implementation
std::shared_ptr<Workload> RoundRobinScheduler::getNextWorkload() {
    std::lock_guard<std::mutex> lock(scheduler_mutex_);

    if (pending_workloads_.empty()) {
        return nullptr;
    }

    // Use round-robin index
    current_index_ = current_index_ % pending_workloads_.size();
    auto workload = pending_workloads_[current_index_];

    pending_workloads_.erase(pending_workloads_.begin() + current_index_);
    running_workloads_.push_back(workload);

    return workload;
}

// ShortestJobFirstScheduler implementation
std::shared_ptr<Workload> ShortestJobFirstScheduler::getNextWorkload() {
    std::lock_guard<std::mutex> lock(scheduler_mutex_);

    if (pending_workloads_.empty()) {
        return nullptr;
    }

    // Find workload with fewest estimated instructions
    auto it = std::min_element(pending_workloads_.begin(), pending_workloads_.end(),
        [](const std::shared_ptr<Workload>& a, const std::shared_ptr<Workload>& b) {
            return a->getEstimatedInstructions() < b->getEstimatedInstructions();
        });

    auto workload = *it;
    pending_workloads_.erase(it);
    running_workloads_.push_back(workload);

    return workload;
}

// SchedulerFactory implementation
std::unique_ptr<Scheduler> SchedulerFactory::createScheduler(SchedulingAlgorithm algorithm) {
    switch (algorithm) {
        case SchedulingAlgorithm::FIFO:
            return std::make_unique<FIFOScheduler>();
        case SchedulingAlgorithm::PRIORITY:
            return std::make_unique<PriorityScheduler>();
        case SchedulingAlgorithm::ROUND_ROBIN:
            return std::make_unique<RoundRobinScheduler>();
        case SchedulingAlgorithm::SHORTEST_JOB_FIRST:
            return std::make_unique<ShortestJobFirstScheduler>();
        default:
            return std::make_unique<FIFOScheduler>();
    }
}

} // namespace GPUSim
