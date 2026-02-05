cmake_minimum_required(VERSION 3.16)

# Note: Google Test is fetched by parent CMakeLists.txt if not found

enable_testing()

# Disable LTO for tests to avoid linker issues with GTest
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

# Test executables
add_executable(orderbook_test orderbook_test.cpp)
target_link_libraries(orderbook_test PRIVATE arbor_core gtest gtest_main)
set_target_properties(orderbook_test PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)

add_executable(options_test options_test.cpp)
target_link_libraries(options_test PRIVATE arbor_core gtest gtest_main)
set_target_properties(options_test PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)

add_executable(montecarlo_test montecarlo_test.cpp)
target_link_libraries(montecarlo_test PRIVATE arbor_core gtest gtest_main)
set_target_properties(montecarlo_test PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)

# Lock-free data structures test (standalone, no gtest dependency)
add_executable(lockfree_test lockfree_test.cpp)
target_include_directories(lockfree_test PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_compile_options(lockfree_test PRIVATE -pthread)
target_link_libraries(lockfree_test PRIVATE pthread)

# Math library for tests
if(UNIX AND NOT APPLE)
    target_link_libraries(options_test PRIVATE m)
    target_link_libraries(montecarlo_test PRIVATE m)
endif()

# Register tests with CTest
include(GoogleTest)
gtest_discover_tests(orderbook_test)
gtest_discover_tests(options_test)
gtest_discover_tests(montecarlo_test)

# Add lockfree test (standalone)
add_test(NAME lockfree_test COMMAND lockfree_test)
