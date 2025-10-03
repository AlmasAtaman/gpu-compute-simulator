#ifndef METRICS_H
#define METRICS_H

#include "types.h"
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <chrono>

namespace GPUSim {

class Workload;
class GPUDevice;

// Performance metrics for a single workload
struct WorkloadMetrics {
    std::string workload_name;
    WorkloadType type;
    double execution_time_ms;
    uint64_t instructions_executed;
    uint64_t memory_operations;
    uint64_t cycles_executed;
    double average_cu_utilization;
    size_t total_threads;
    size_t total_blocks;
    double throughput; // Instructions per millisecond
};

// GPU-wide performance metrics
struct GPUMetrics {
    uint64_t total_cycles;
    uint64_t total_instructions;
    uint64_t total_memory_ops;
    double total_execution_time_ms;
    double average_utilization;
    double memory_bandwidth_utilization;
    size_t total_workloads_executed;
};

// Performance analyzer for collecting and analyzing metrics
class PerformanceAnalyzer {
private:
    std::vector<WorkloadMetrics> workload_metrics_;
    GPUMetrics gpu_metrics_;
    std::chrono::high_resolution_clock::time_point sim_start_time_;
    std::chrono::high_resolution_clock::time_point sim_end_time_;

public:
    PerformanceAnalyzer();

    void recordWorkloadMetrics(const Workload* workload, const GPUDevice* device);
    void recordGPUMetrics(const GPUDevice* device);

    void startSimulation();
    void endSimulation();

    const std::vector<WorkloadMetrics>& getWorkloadMetrics() const {
        return workload_metrics_;
    }

    const GPUMetrics& getGPUMetrics() const {
        return gpu_metrics_;
    }

    // Analysis functions
    double getTotalSimulationTime() const;
    double getAverageThroughput() const;
    double getAverageWorkloadTime() const;
    WorkloadMetrics getFastestWorkload() const;
    WorkloadMetrics getSlowestWorkload() const;

    // Reporting
    void printSummary() const;
    void printDetailedReport() const;
    void exportToCSV(const std::string& filename) const;

    void reset();
};

// Comparison framework for different scheduling strategies
class SchedulerComparison {
private:
    std::map<std::string, std::unique_ptr<PerformanceAnalyzer>> analyzer_map_;

public:
    void addAnalyzer(const std::string& scheduler_name, std::unique_ptr<PerformanceAnalyzer> analyzer);

    void printComparison() const;
    void exportComparisonCSV(const std::string& filename) const;

    std::string getBestScheduler() const; // Based on total execution time
};

} // namespace GPUSim

#endif // METRICS_H
