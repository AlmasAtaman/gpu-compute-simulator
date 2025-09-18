#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

class SimpleGPU {
private:
    int num_cores;
    
public:
    SimpleGPU(int cores) : num_cores(cores) {
        std::cout << "GPU Simulator initialized with " << cores << " cores\n";
    }
    
    void run_workload(const std::string& task_name) {
        std::cout << "Starting workload: " << task_name << std::endl;
        
        // Simulate work on each core
        std::vector<std::thread> workers;
        
        for (int i = 0; i < num_cores; i++) {
            workers.emplace_back([i, task_name]() {
                std::cout << "Core " << i << " processing " << task_name << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Fake work
                std::cout << "Core " << i << " finished\n";
            });
        }
        
        // Wait for all cores to finish
        for (auto& worker : workers) {
            worker.join();
        }
        
        std::cout << "Workload " << task_name << " completed!\n\n";
    }
};

int main() {
    std::cout << "=== GPU Compute Simulator v0.1 ===\n\n";
    
    // Create a simple GPU with 4 cores
    SimpleGPU gpu(4);
    
    // Run some fake workloads
    gpu.run_workload("Matrix Multiplication");
    gpu.run_workload("Neural Network Layer");
    
    std::cout << "Simulation complete!\n";
    return 0;
}