cmake_minimum_required (VERSION 3.0)

add_executable (TestRecord TestRecord.cpp ${Sources})
target_link_libraries (TestRecord LINK_PUBLIC "gtest;pthread")

add_executable (TestTuple TestTuple.cpp ${Sources})
target_link_libraries (TestTuple LINK_PUBLIC "gtest;pthread")

add_executable (TestTupleProxy TestTupleProxy.cpp ${Sources})
target_link_libraries (TestTupleProxy LINK_PUBLIC "gtest;pthread")