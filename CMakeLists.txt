cmake_minimum_required(VERSION 3.12)
project(GPUComputeSimulator VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Source files
set(SOURCES
    src/memory/memory.cpp
    src/architecture/warp.cpp
    src/architecture/compute_unit.cpp
    src/architecture/workload.cpp
    src/architecture/gpu_device.cpp
    src/scheduler/scheduler.cpp
    src/metrics/metrics.cpp
)

# Main executable
add_executable(gpu_simulator
    src/main.cpp
    ${SOURCES}
)

# Link threading library
if(MINGW)
    # For MinGW on Windows, explicitly link winpthread
    set(CMAKE_THREAD_LIBS_INIT "-lpthread")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
    target_link_libraries(gpu_simulator -static-libgcc -static-libstdc++ -Wl,-Bstatic -lstdc++ -lpthread -Wl,-Bdynamic)
elseif(WIN32)
    # For MSVC and other Windows compilers
    find_package(Threads REQUIRED)
    target_link_libraries(gpu_simulator Threads::Threads)
else()
    # For Unix-like platforms
    find_package(Threads REQUIRED)
    target_link_libraries(gpu_simulator Threads::Threads)
endif()

# Optional: Enable warnings
if(MSVC)
    target_compile_options(gpu_simulator PRIVATE /W4)
else()
    target_compile_options(gpu_simulator PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Optional: Build test executable
add_executable(test_simulator
    test.cpp
)

# Link threading for test executable too
if(MINGW)
    target_link_libraries(test_simulator -static-libgcc -static-libstdc++ -Wl,-Bstatic -lstdc++ -lpthread -Wl,-Bdynamic)
elseif(WIN32)
    find_package(Threads REQUIRED)
    target_link_libraries(test_simulator Threads::Threads)
else()
    find_package(Threads REQUIRED)
    target_link_libraries(test_simulator Threads::Threads)
endif()

message(STATUS "GPU Compute Simulator configured successfully")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
