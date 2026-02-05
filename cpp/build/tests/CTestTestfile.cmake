# CMake generated Testfile for 
# Source directory: C:/Users/pooja/Arbor/cpp/tests
# Build directory: C:/Users/pooja/Arbor/cpp/build/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
include("C:/Users/pooja/Arbor/cpp/build/tests/orderbook_test[1]_include.cmake")
include("C:/Users/pooja/Arbor/cpp/build/tests/options_test[1]_include.cmake")
include("C:/Users/pooja/Arbor/cpp/build/tests/montecarlo_test[1]_include.cmake")
add_test(lockfree_test "C:/Users/pooja/Arbor/cpp/build/tests/lockfree_test.exe")
set_tests_properties(lockfree_test PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/pooja/Arbor/cpp/tests/CMakeLists.txt;42;add_test;C:/Users/pooja/Arbor/cpp/tests/CMakeLists.txt;0;")
