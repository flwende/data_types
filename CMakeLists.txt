cmake_minimum_required (VERSION 3.0)
project (data_types)
 
set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -ftree-vectorize -ffast-math -Wall -Wextra -pedantic")

include_directories (include/data_types)

set (Test on)

add_subdirectory (src)
