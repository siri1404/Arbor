cmake_minimum_required(VERSION 3.15)
project(arbor_quant_engine VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Aggressive optimization flags for maximum performance
if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Oi /Ot /GL /arch:AVX2")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -march=native -mtune=native")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -funroll-loops -finline-functions")
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Core library
add_library(arbor_core STATIC
    src/orderbook.cpp
    src/options_pricing.cpp
    src/monte_carlo.cpp
    src/technical_indicators.cpp
    src/market_data_parser.cpp
)

# Platform-specific network libraries
if(WIN32)
    target_link_libraries(arbor_core ws2_32)
endif()

# Lock-free benchmark
add_executable(lockfree_bench benchmarks/lockfree_benchmark.cpp)
target_link_libraries(lockfree_bench arbor_core)

# Node.js Native Addon
find_package(Node QUIET)
if(Node_FOUND)
    include_directories(${Node_INCLUDE_DIRS})
    add_library(arbor_addon SHARED
        src/node_bindings.cpp
    )
    target_link_libraries(arbor_addon arbor_core ${Node_LIBRARIES})
    set_target_properties(arbor_addon PROPERTIES PREFIX "" SUFFIX ".node")
endif()

# Standalone benchmarking executables
add_executable(orderbook_bench benchmarks/orderbook_benchmark.cpp)
target_link_libraries(orderbook_bench arbor_core)

add_executable(options_bench benchmarks/options_benchmark.cpp)
target_link_libraries(options_bench arbor_core)

add_executable(montecarlo_bench benchmarks/montecarlo_benchmark.cpp)
target_link_libraries(montecarlo_bench arbor_core)

# Enable testing
enable_testing()
add_subdirectory(tests)
