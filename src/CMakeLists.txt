cmake_minimum_required (VERSION 3.0)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

if (Test)
    find_package(GTest REQUIRED)
    add_subdirectory (unit_test)
endif (Test)