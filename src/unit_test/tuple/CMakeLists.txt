cmake_minimum_required (VERSION 3.0)

add_executable (TestRecord TestRecord.cpp ${Sources})
target_link_libraries (TestRecord LINK_PUBLIC "gtest;pthread")
