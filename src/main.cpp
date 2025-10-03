#include "gpu_device.h"
#include "workload.h"
#include "scheduler.h"
#include "metrics.h"
#include <iostream>
#include <memory>
#include <vector>

using namespace GPUSim;

void runBasicSimulation() {
    std::cout << "\n==============================================\n";
    std::cout << "  BASIC GPU SIMULATION\n";
    std::cout << "==============================================\n\n";

    // Create GPU device with default configuration (RTX 3080 profile)
    GPUConfig config;
    config.num_compute_units = 16; // Smaller for faster simulation
    GPUDevice gpu(config);

    gpu.printDeviceInfo();

    // Create some workloads
    auto matmul = Workload::createMatrixMultiply(512, 512, 512);
    auto vecadd = Workload::createVectorAdd(1024 * 1024);
    auto reduction = Workload::createReduction(1024 * 1024);

    // Submit workloads
    gpu.submitWorkload(std::move(matmul));
    gpu.submitWorkload(std::move(vecadd));
    gpu.submitWorkload(std::move(reduction));

    // Execute and wait for completion
    gpu.executeWorkloads();
    gpu.waitForCompletion();

    // Print performance results
    gpu.getPerformanceAnalyzer()->printDetailedReport();
    gpu.getPerformanceAnalyzer()->exportToCSV("basic_simulation_results.csv");
}

void runSchedulerComparison() {
    std::cout << "\n==============================================\n";
    std::cout << "  SCHEDULER COMPARISON\n";
    std::cout << "==============================================\n\n";

    // Create comparison framework
    SchedulerComparison comparison;

    // Test different scheduling algorithms
    std::vector<SchedulingAlgorithm> algorithms = {
        SchedulingAlgorithm::FIFO,
        SchedulingAlgorithm::PRIORITY,
        SchedulingAlgorithm::SHORTEST_JOB_FIRST,
        SchedulingAlgorithm::ROUND_ROBIN
    };

    const char* algorithm_names[] = {
        "FIFO",
        "Priority",
        "Shortest-Job-First",
        "Round-Robin"
    };

    for (size_t i = 0; i < algorithms.size(); ++i) {
        std::cout << "\nTesting " << algorithm_names[i] << " scheduler...\n";

        GPUConfig config;
        config.num_compute_units = 16;
        GPUDevice gpu(config);

        // Set scheduler
        auto scheduler = SchedulerFactory::createScheduler(algorithms[i]);
        gpu.setScheduler(std::move(scheduler));

        // Create diverse workloads
        auto small_matmul = Workload::createMatrixMultiply(256, 256, 256);
        small_matmul->setPriority(3);

        auto large_matmul = Workload::createMatrixMultiply(1024, 1024, 1024);
        large_matmul->setPriority(1);

        auto conv = Workload::createConvolution(4, 64, 224, 224);
        conv->setPriority(2);

        auto vecadd = Workload::createVectorAdd(2 * 1024 * 1024);
        vecadd->setPriority(2);

        auto reduction = Workload::createReduction(1024 * 1024);
        reduction->setPriority(3);

        // Submit workloads
        gpu.submitWorkload(std::move(small_matmul));
        gpu.submitWorkload(std::move(large_matmul));
        gpu.submitWorkload(std::move(conv));
        gpu.submitWorkload(std::move(vecadd));
        gpu.submitWorkload(std::move(reduction));

        // Execute
        gpu.executeWorkloads();
        gpu.waitForCompletion();

        // Store results
        auto analyzer = std::make_unique<PerformanceAnalyzer>(
            *gpu.getPerformanceAnalyzer()
        );
        comparison.addAnalyzer(algorithm_names[i], std::move(analyzer));
    }

    // Print comparison
    comparison.printComparison();
    comparison.exportComparisonCSV("scheduler_comparison.csv");
}

void runMLWorkloadSimulation() {
    std::cout << "\n==============================================\n";
    std::cout << "  MACHINE LEARNING WORKLOAD SIMULATION\n";
    std::cout << "==============================================\n\n";

    GPUConfig config;
    config.num_compute_units = 32;
    config.device_name = "GPU Simulator - ML Workload Profile";
    GPUDevice gpu(config);

    gpu.printDeviceInfo();

    // Simulate a small neural network forward pass
    std::cout << "Simulating ResNet-like network inference...\n\n";

    // Layer 1: Input convolution
    auto conv1 = Workload::createConvolution(1, 64, 224, 224);
    conv1->setPriority(5);

    // Layer 2: Residual block convolutions
    auto conv2 = Workload::createConvolution(1, 64, 112, 112);
    auto conv3 = Workload::createConvolution(1, 64, 112, 112);

    // Layer 3: Downsampling
    auto conv4 = Workload::createConvolution(1, 128, 56, 56);

    // Layer 4: More residual blocks
    auto conv5 = Workload::createConvolution(1, 128, 56, 56);
    auto conv6 = Workload::createConvolution(1, 256, 28, 28);

    // Final: Fully connected (matrix multiply)
    auto fc = Workload::createMatrixMultiply(1, 1000, 2048);
    fc->setPriority(10);

    // Submit all layers
    gpu.submitWorkload(std::move(conv1));
    gpu.submitWorkload(std::move(conv2));
    gpu.submitWorkload(std::move(conv3));
    gpu.submitWorkload(std::move(conv4));
    gpu.submitWorkload(std::move(conv5));
    gpu.submitWorkload(std::move(conv6));
    gpu.submitWorkload(std::move(fc));

    // Execute
    gpu.executeWorkloads();
    gpu.waitForCompletion();

    // Results
    gpu.getPerformanceAnalyzer()->printDetailedReport();
    gpu.getPerformanceAnalyzer()->exportToCSV("ml_workload_results.csv");
}

void runCustomWorkloadBenchmark() {
    std::cout << "\n==============================================\n";
    std::cout << "  CUSTOM WORKLOAD BENCHMARK\n";
    std::cout << "==============================================\n\n";

    GPUConfig config;
    config.num_compute_units = 24;
    GPUDevice gpu(config);

    // Create a mix of different workload sizes
    std::vector<std::shared_ptr<Workload>> workloads;

    // Small workloads
    for (int i = 0; i < 3; ++i) {
        workloads.push_back(Workload::createVectorAdd(512 * 1024));
    }

    // Medium workloads
    for (int i = 0; i < 3; ++i) {
        workloads.push_back(Workload::createMatrixMultiply(256, 256, 256));
    }

    // Large workloads
    for (int i = 0; i < 2; ++i) {
        workloads.push_back(Workload::createConvolution(2, 32, 128, 128));
    }

    // Mixed priorities
    for (size_t i = 0; i < workloads.size(); ++i) {
        workloads[i]->setPriority(i % 5);
    }

    // Submit all workloads
    for (auto& workload : workloads) {
        gpu.submitWorkload(workload);
    }

    // Execute
    gpu.executeWorkloads();
    gpu.waitForCompletion();

    // Results
    gpu.getPerformanceAnalyzer()->printSummary();

    std::cout << "\nFastest workload: "
              << gpu.getPerformanceAnalyzer()->getFastestWorkload().workload_name << "\n";
    std::cout << "Slowest workload: "
              << gpu.getPerformanceAnalyzer()->getSlowestWorkload().workload_name << "\n";
}

void printMenu() {
    std::cout << "\n==============================================\n";
    std::cout << "  GPU COMPUTE SIMULATOR v1.0\n";
    std::cout << "==============================================\n\n";
    std::cout << "Select a simulation mode:\n\n";
    std::cout << "  1. Basic Simulation\n";
    std::cout << "     - Run simple workloads with FIFO scheduling\n";
    std::cout << "     - Demonstrates core GPU functionality\n\n";
    std::cout << "  2. Scheduler Comparison\n";
    std::cout << "     - Compare all scheduling algorithms\n";
    std::cout << "     - FIFO, Priority, SJF, Round-Robin\n\n";
    std::cout << "  3. ML Workload Simulation\n";
    std::cout << "     - Simulate neural network inference\n";
    std::cout << "     - ResNet-like architecture\n\n";
    std::cout << "  4. Custom Workload Benchmark\n";
    std::cout << "     - Mix of different workload sizes\n";
    std::cout << "     - Performance analysis\n\n";
    std::cout << "  5. Run All Simulations\n\n";
    std::cout << "  0. Exit\n\n";
    std::cout << "Enter your choice: ";
}

int main() {
    int choice;

    while (true) {
        printMenu();
        std::cin >> choice;

        if (std::cin.fail()) {
            std::cin.clear();
            std::cin.ignore(10000, '\n');
            std::cout << "\nInvalid input. Please enter a number.\n";
            continue;
        }

        switch (choice) {
            case 0:
                std::cout << "\nExiting GPU Compute Simulator. Goodbye!\n";
                return 0;

            case 1:
                runBasicSimulation();
                break;

            case 2:
                runSchedulerComparison();
                break;

            case 3:
                runMLWorkloadSimulation();
                break;

            case 4:
                runCustomWorkloadBenchmark();
                break;

            case 5:
                runBasicSimulation();
                runSchedulerComparison();
                runMLWorkloadSimulation();
                runCustomWorkloadBenchmark();
                break;

            default:
                std::cout << "\nInvalid choice. Please select 0-5.\n";
        }

        std::cout << "\nPress Enter to continue...";
        std::cin.ignore(10000, '\n');
        std::cin.get();
    }

    return 0;
}
