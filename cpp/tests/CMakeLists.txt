cmake_minimum_required(VERSION 3.15)
project(arbor_tests)

# Google Test
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

# Test executables
add_executable(orderbook_test orderbook_test.cpp)
target_link_libraries(orderbook_test arbor_core gtest_main)

add_executable(options_test options_test.cpp)
target_link_libraries(options_test arbor_core gtest_main)

add_executable(montecarlo_test montecarlo_test.cpp)
target_link_libraries(montecarlo_test arbor_core gtest_main)

# Register tests
include(GoogleTest)
gtest_discover_tests(orderbook_test)
gtest_discover_tests(options_test)
gtest_discover_tests(montecarlo_test)

# Register tests
include(GoogleTest)
gtest_discover_tests(orderbook_test)
gtest_discover_tests(options_test)
gtest_discover_tests(montecarlo_test)
