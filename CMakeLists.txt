cmake_minimum_required (VERSION 3.0)
project (data_types)
 
set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -ftree-vectorize -ffast-math -Wall -Wextra -pedantic")

include_directories (include/data_types)

#set (CUDA on)
if (CUDA)
  find_package (CUDA REQUIRED)
  set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3 -std=c++14 -arch=sm_61 --resource-usage --expt-relaxed-constexpr --expt-extended-lambda -ccbin /usr/bin/g++-8 -Xcompiler="-std=c++14")
endif()

set (Test on)

add_subdirectory (src)
