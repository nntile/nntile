# CMake generated Testfile for 
# Source directory: /workspace/tests
# Build directory: /workspace/build-test/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[tests_constants]=] "/workspace/build-test/tests/test_constants")
set_tests_properties([=[tests_constants]=] PROPERTIES  _BACKTRACE_TRIPLES "/workspace/tests/CMakeLists.txt;102;add_test;/workspace/tests/CMakeLists.txt;141;add_test_set;/workspace/tests/CMakeLists.txt;0;")
subdirs("kernel")
subdirs("starpu")
subdirs("tile")
subdirs("tensor")
