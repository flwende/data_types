cmake_minimum_required (VERSION 3.0)

add_executable (TestField TestField.cpp ${Sources})
target_link_libraries (TestField LINK_PUBLIC "gtest;pthread")
