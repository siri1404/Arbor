cmake_minimum_required(VERSION 3.16)
project(arbor_quant_engine VERSION 2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================================
# Build Type Configuration
# ============================================================================
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

# ============================================================================
# Platform Detection and SIMD Support
# ============================================================================
include(CheckCXXCompilerFlag)
include(CheckIncludeFileCXX)

# Detect AVX2 support
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)
check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512)

# Check for immintrin.h (SIMD intrinsics header)
check_include_file_cxx("immintrin.h" HAVE_IMMINTRIN_H)

# ============================================================================
# Compiler Flags
# ============================================================================
if(MSVC)
    # MSVC flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /permissive-")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Od /Zi /RTC1")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Oi /Ot /GL /DNDEBUG")
    
    # AVX2 support for MSVC
    if(HAVE_IMMINTRIN_H)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /arch:AVX2")
        add_compile_definitions(ARBOR_AVX2_ENABLED)
    endif()
    
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG")
    set(CMAKE_STATIC_LINKER_FLAGS_RELEASE "${CMAKE_STATIC_LINKER_FLAGS_RELEASE} /LTCG")
else()
    # GCC/Clang flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wconversion -Wshadow")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter -Wno-missing-field-initializers")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g3 -fno-omit-frame-pointer")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address,undefined")
    
    # Release optimizations
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -mtune=native")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffast-math -funroll-loops")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -finline-functions -ftree-vectorize")
    # LTO disabled due to Google Test compatibility issues on MinGW
    # set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -flto=auto -fuse-linker-plugin")
    
    # SIMD flags
    if(COMPILER_SUPPORTS_AVX2 AND HAVE_IMMINTRIN_H)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mavx2")
        add_compile_definitions(ARBOR_AVX2_ENABLED)
        message(STATUS "AVX2 support: ENABLED")
    else()
        message(STATUS "AVX2 support: DISABLED")
    endif()
    
    if(COMPILER_SUPPORTS_FMA)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mfma")
        add_compile_definitions(ARBOR_FMA_ENABLED)
        message(STATUS "FMA support: ENABLED")
    endif()
    
    if(COMPILER_SUPPORTS_AVX512)
        option(ENABLE_AVX512 "Enable AVX-512 instructions" OFF)
        if(ENABLE_AVX512)
            set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mavx512f -mavx512dq")
            add_compile_definitions(ARBOR_AVX512_ENABLED)
            message(STATUS "AVX-512 support: ENABLED")
        endif()
    endif()
    
    # Link-time optimization disabled due to Google Test compatibility issues on MinGW
    # set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -flto=auto")
endif()

# ============================================================================
# Include Directories
# ============================================================================
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ============================================================================
# Core Library
# ============================================================================
add_library(arbor_core STATIC
    src/orderbook.cpp
    src/options_pricing.cpp
    src/monte_carlo.cpp
    src/technical_indicators.cpp
    src/market_data_parser.cpp
)

target_include_directories(arbor_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Threading support
find_package(Threads REQUIRED)
target_link_libraries(arbor_core PUBLIC Threads::Threads)

# Math library (required for lgamma, erf, etc.)
if(UNIX AND NOT APPLE)
    target_link_libraries(arbor_core PUBLIC m)
endif()

# Platform-specific network libraries
if(WIN32)
    target_link_libraries(arbor_core PUBLIC ws2_32)
endif()

# ============================================================================
# Benchmarks
# ============================================================================
add_executable(orderbook_bench benchmarks/orderbook_benchmark.cpp)
target_link_libraries(orderbook_bench PRIVATE arbor_core)

add_executable(options_bench benchmarks/options_benchmark.cpp)
target_link_libraries(options_bench PRIVATE arbor_core)

add_executable(montecarlo_bench benchmarks/montecarlo_benchmark.cpp)
target_link_libraries(montecarlo_bench PRIVATE arbor_core)

add_executable(lockfree_bench benchmarks/lockfree_benchmark.cpp)
target_link_libraries(lockfree_bench PRIVATE arbor_core)

# ============================================================================
# Node.js Native Addon (Optional)
# ============================================================================
option(BUILD_NODE_ADDON "Build Node.js native addon" OFF)
if(BUILD_NODE_ADDON)
    find_package(Node QUIET)
    if(Node_FOUND)
        include_directories(${Node_INCLUDE_DIRS})
        add_library(arbor_addon SHARED src/node_bindings.cpp)
        target_link_libraries(arbor_addon PRIVATE arbor_core ${Node_LIBRARIES})
        set_target_properties(arbor_addon PROPERTIES PREFIX "" SUFFIX ".node")
        message(STATUS "Node.js addon: ENABLED")
    else()
        message(STATUS "Node.js addon: DISABLED (Node.js not found)")
    endif()
endif()

# ============================================================================
# Testing
# ============================================================================
option(BUILD_TESTS "Build unit tests" ON)
if(BUILD_TESTS)
    enable_testing()
    
    # Try to find Google Test
    find_package(GTest QUIET)
    
    if(GTest_FOUND)
        message(STATUS "Google Test: FOUND")
    else()
        # Download and build Google Test
        include(FetchContent)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.14.0
        )
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        # Disable LTO for GTest to avoid linker issues
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
        FetchContent_MakeAvailable(googletest)
        # Re-enable for rest of project if needed
        if(CMAKE_BUILD_TYPE STREQUAL "Release")
            set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
        endif()
        message(STATUS "Google Test: FETCHED")
    endif()
    
    add_subdirectory(tests)
endif()

# ============================================================================
# Installation
# ============================================================================
install(TARGETS arbor_core
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include/arbor)

# ============================================================================
# Summary
# ============================================================================
message(STATUS "")
message(STATUS "=== Arbor Quant Engine Configuration ===")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Tests: ${BUILD_TESTS}")
message(STATUS "Node.js addon: ${BUILD_NODE_ADDON}")
message(STATUS "=========================================")
message(STATUS "")
