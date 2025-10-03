#ifndef SCHEDULER_H
#define SCHEDULER_H

#include "types.h"
#include "workload.h"
#include "compute_unit.h"
#include <queue>
#include <vector>
#include <mutex>
#include <memory>
#include <functional>

namespace GPUSim {

// Base scheduler interface
class Scheduler {
protected:
    std::mutex scheduler_mutex_;
    std::vector<std::shared_ptr<Workload>> pending_workloads_;
    std::vector<std::shared_ptr<Workload>> running_workloads_;
    std::vector<std::shared_ptr<Workload>> completed_workloads_;

public:
    virtual ~Scheduler() = default;

    virtual void addWorkload(std::shared_ptr<Workload> workload);
    virtual std::shared_ptr<Workload> getNextWorkload() = 0;
    virtual const char* getName() const = 0;

    bool hasPendingWorkloads() const;
    size_t getPendingCount() const { return pending_workloads_.size(); }
    size_t getRunningCount() const { return running_workloads_.size(); }
    size_t getCompletedCount() const { return completed_workloads_.size(); }

    void markWorkloadRunning(std::shared_ptr<Workload> workload);
    void markWorkloadCompleted(std::shared_ptr<Workload> workload);

    const std::vector<std::shared_ptr<Workload>>& getCompletedWorkloads() const {
        return completed_workloads_;
    }
};

// First-In-First-Out scheduler
class FIFOScheduler : public Scheduler {
public:
    std::shared_ptr<Workload> getNextWorkload() override;
    const char* getName() const override { return "FIFO"; }
};

// Priority-based scheduler (higher priority first)
class PriorityScheduler : public Scheduler {
public:
    std::shared_ptr<Workload> getNextWorkload() override;
    const char* getName() const override { return "Priority"; }
};

// Round-robin scheduler
class RoundRobinScheduler : public Scheduler {
private:
    size_t current_index_;

public:
    RoundRobinScheduler() : current_index_(0) {}

    std::shared_ptr<Workload> getNextWorkload() override;
    const char* getName() const override { return "Round-Robin"; }
};

// Shortest Job First scheduler
class ShortestJobFirstScheduler : public Scheduler {
public:
    std::shared_ptr<Workload> getNextWorkload() override;
    const char* getName() const override { return "Shortest-Job-First"; }
};

// Factory for creating schedulers
class SchedulerFactory {
public:
    static std::unique_ptr<Scheduler> createScheduler(SchedulingAlgorithm algorithm);
};

} // namespace GPUSim

#endif // SCHEDULER_H
