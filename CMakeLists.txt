cmake_minimum_required(VERSION 3.12)
project(GPUComputeSimulator)

set(CMAKE_CXX_STANDARD 17)
find_package(Threads REQUIRED)

add_executable(gpu_simulator src/main.cpp)
target_link_libraries(gpu_simulator Threads::Threads)