#include "metrics.h"
#include "workload.h"
#include "gpu_device.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <numeric>

namespace GPUSim {

PerformanceAnalyzer::PerformanceAnalyzer()
    : gpu_metrics_{} {
    gpu_metrics_.total_cycles = 0;
    gpu_metrics_.total_instructions = 0;
    gpu_metrics_.total_memory_ops = 0;
    gpu_metrics_.total_execution_time_ms = 0.0;
    gpu_metrics_.average_utilization = 0.0;
    gpu_metrics_.memory_bandwidth_utilization = 0.0;
    gpu_metrics_.total_workloads_executed = 0;
}

void PerformanceAnalyzer::recordWorkloadMetrics(const Workload* workload, const GPUDevice* device) {
    if (!workload || !device) return;

    WorkloadMetrics metrics;
    metrics.workload_name = workload->getName();
    metrics.type = workload->getType();
    metrics.execution_time_ms = workload->getExecutionTime();
    metrics.total_threads = workload->getConfig().getTotalThreads();
    metrics.total_blocks = workload->getConfig().getTotalBlocks();

    // Aggregate metrics from all compute units
    metrics.instructions_executed = 0;
    metrics.cycles_executed = 0;
    double total_utilization = 0.0;

    for (const auto& cu : device->getComputeUnits()) {
        metrics.instructions_executed += cu->getInstructionsExecuted();
        metrics.cycles_executed += cu->getCyclesExecuted();
        total_utilization += cu->getUtilization();
    }

    metrics.average_cu_utilization = total_utilization / device->getNumComputeUnits();

    // Calculate throughput
    if (metrics.execution_time_ms > 0) {
        metrics.throughput = static_cast<double>(metrics.instructions_executed) / metrics.execution_time_ms;
    } else {
        metrics.throughput = 0.0;
    }

    metrics.memory_operations = device->getMemoryController()->getTotalMemoryOps();

    workload_metrics_.push_back(metrics);
}

void PerformanceAnalyzer::recordGPUMetrics(const GPUDevice* device) {
    if (!device) return;

    gpu_metrics_.total_cycles = 0;
    gpu_metrics_.total_instructions = 0;
    double total_utilization = 0.0;

    for (const auto& cu : device->getComputeUnits()) {
        gpu_metrics_.total_cycles += cu->getCyclesExecuted();
        gpu_metrics_.total_instructions += cu->getInstructionsExecuted();
        total_utilization += cu->getUtilization();
    }

    gpu_metrics_.average_utilization = total_utilization / device->getNumComputeUnits();
    gpu_metrics_.total_memory_ops = device->getMemoryController()->getTotalMemoryOps();
    gpu_metrics_.total_workloads_executed = workload_metrics_.size();
}

void PerformanceAnalyzer::startSimulation() {
    sim_start_time_ = std::chrono::high_resolution_clock::now();
}

void PerformanceAnalyzer::endSimulation() {
    sim_end_time_ = std::chrono::high_resolution_clock::now();
    gpu_metrics_.total_execution_time_ms = getTotalSimulationTime();
}

double PerformanceAnalyzer::getTotalSimulationTime() const {
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        sim_end_time_ - sim_start_time_);
    return duration.count();
}

double PerformanceAnalyzer::getAverageThroughput() const {
    if (workload_metrics_.empty()) return 0.0;

    double total_throughput = 0.0;
    for (const auto& metrics : workload_metrics_) {
        total_throughput += metrics.throughput;
    }

    return total_throughput / workload_metrics_.size();
}

double PerformanceAnalyzer::getAverageWorkloadTime() const {
    if (workload_metrics_.empty()) return 0.0;

    double total_time = 0.0;
    for (const auto& metrics : workload_metrics_) {
        total_time += metrics.execution_time_ms;
    }

    return total_time / workload_metrics_.size();
}

WorkloadMetrics PerformanceAnalyzer::getFastestWorkload() const {
    if (workload_metrics_.empty()) return WorkloadMetrics{};

    return *std::min_element(workload_metrics_.begin(), workload_metrics_.end(),
        [](const WorkloadMetrics& a, const WorkloadMetrics& b) {
            return a.execution_time_ms < b.execution_time_ms;
        });
}

WorkloadMetrics PerformanceAnalyzer::getSlowestWorkload() const {
    if (workload_metrics_.empty()) return WorkloadMetrics{};

    return *std::max_element(workload_metrics_.begin(), workload_metrics_.end(),
        [](const WorkloadMetrics& a, const WorkloadMetrics& b) {
            return a.execution_time_ms < b.execution_time_ms;
        });
}

void PerformanceAnalyzer::printSummary() const {
    std::cout << "\n========================================\n";
    std::cout << "      PERFORMANCE SUMMARY\n";
    std::cout << "========================================\n\n";

    std::cout << "Total Simulation Time: " << std::fixed << std::setprecision(2)
              << gpu_metrics_.total_execution_time_ms << " ms\n";
    std::cout << "Workloads Executed: " << gpu_metrics_.total_workloads_executed << "\n";
    std::cout << "Total Instructions: " << gpu_metrics_.total_instructions << "\n";
    std::cout << "Total Memory Operations: " << gpu_metrics_.total_memory_ops << "\n";
    std::cout << "Average GPU Utilization: " << std::fixed << std::setprecision(2)
              << gpu_metrics_.average_utilization << "%\n";
    std::cout << "Average Throughput: " << std::fixed << std::setprecision(2)
              << getAverageThroughput() << " instr/ms\n";

    std::cout << "\n========================================\n\n";
}

void PerformanceAnalyzer::printDetailedReport() const {
    printSummary();

    std::cout << "WORKLOAD DETAILS:\n";
    std::cout << "----------------------------------------\n";

    for (const auto& metrics : workload_metrics_) {
        std::cout << "\nWorkload: " << metrics.workload_name << "\n";
        std::cout << "  Execution Time: " << std::fixed << std::setprecision(2)
                  << metrics.execution_time_ms << " ms\n";
        std::cout << "  Instructions: " << metrics.instructions_executed << "\n";
        std::cout << "  Memory Ops: " << metrics.memory_operations << "\n";
        std::cout << "  Threads: " << metrics.total_threads << "\n";
        std::cout << "  Blocks: " << metrics.total_blocks << "\n";
        std::cout << "  Avg CU Utilization: " << std::fixed << std::setprecision(2)
                  << metrics.average_cu_utilization << "%\n";
        std::cout << "  Throughput: " << std::fixed << std::setprecision(2)
                  << metrics.throughput << " instr/ms\n";
    }

    std::cout << "\n========================================\n";
}

void PerformanceAnalyzer::exportToCSV(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << "\n";
        return;
    }

    // Header
    file << "Workload,Type,Execution_Time_ms,Instructions,Memory_Ops,Threads,Blocks,Utilization_%,Throughput_instr_ms\n";

    // Data
    for (const auto& metrics : workload_metrics_) {
        file << metrics.workload_name << ","
             << static_cast<int>(metrics.type) << ","
             << metrics.execution_time_ms << ","
             << metrics.instructions_executed << ","
             << metrics.memory_operations << ","
             << metrics.total_threads << ","
             << metrics.total_blocks << ","
             << metrics.average_cu_utilization << ","
             << metrics.throughput << "\n";
    }

    file.close();
    std::cout << "Metrics exported to " << filename << "\n";
}

void PerformanceAnalyzer::reset() {
    workload_metrics_.clear();
    gpu_metrics_ = GPUMetrics{};
}

// SchedulerComparison implementation
void SchedulerComparison::addAnalyzer(const std::string& scheduler_name,
                                      std::unique_ptr<PerformanceAnalyzer> analyzer) {
    analyzer_map_[scheduler_name] = std::move(analyzer);
}

void SchedulerComparison::printComparison() const {
    std::cout << "\n========================================\n";
    std::cout << "   SCHEDULER COMPARISON\n";
    std::cout << "========================================\n\n";

    std::cout << std::left << std::setw(20) << "Scheduler"
              << std::setw(15) << "Total Time(ms)"
              << std::setw(15) << "Avg Util(%)"
              << std::setw(15) << "Throughput"
              << "\n";
    std::cout << "----------------------------------------\n";

    for (const auto& [name, analyzer] : analyzer_map_) {
        const auto& metrics = analyzer->getGPUMetrics();
        std::cout << std::left << std::setw(20) << name
                  << std::setw(15) << std::fixed << std::setprecision(2) << metrics.total_execution_time_ms
                  << std::setw(15) << std::fixed << std::setprecision(2) << metrics.average_utilization
                  << std::setw(15) << std::fixed << std::setprecision(2) << analyzer->getAverageThroughput()
                  << "\n";
    }

    std::cout << "\nBest Scheduler: " << getBestScheduler() << "\n";
    std::cout << "========================================\n\n";
}

void SchedulerComparison::exportComparisonCSV(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << "\n";
        return;
    }

    file << "Scheduler,Total_Time_ms,Avg_Utilization_%,Avg_Throughput,Total_Instructions,Total_Memory_Ops\n";

    for (const auto& [name, analyzer] : analyzer_map_) {
        const auto& metrics = analyzer->getGPUMetrics();
        file << name << ","
             << metrics.total_execution_time_ms << ","
             << metrics.average_utilization << ","
             << analyzer->getAverageThroughput() << ","
             << metrics.total_instructions << ","
             << metrics.total_memory_ops << "\n";
    }

    file.close();
    std::cout << "Comparison exported to " << filename << "\n";
}

std::string SchedulerComparison::getBestScheduler() const {
    if (analyzer_map_.empty()) return "None";

    std::string best_scheduler;
    double best_time = std::numeric_limits<double>::max();

    for (const auto& [name, analyzer] : analyzer_map_) {
        double time = analyzer->getGPUMetrics().total_execution_time_ms;
        if (time < best_time && time > 0) {
            best_time = time;
            best_scheduler = name;
        }
    }

    return best_scheduler;
}

} // namespace GPUSim
