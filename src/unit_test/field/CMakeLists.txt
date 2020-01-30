cmake_minimum_required (VERSION 3.0)

add_executable (TestField TestField.cpp ${Sources})
target_link_libraries (TestField LINK_PUBLIC "gtest;pthread")

find_package (CUDA REQUIRED)
set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3 -std=c++14 -arch=sm_61 --resource-usage --expt-relaxed-constexpr --expt-extended-lambda -ccbin /usr/bin/g++-8 -Xcompiler="-std=c++14")

cuda_add_executable (TestFieldCUDA TestFieldCUDA.cu ${Sources})
target_link_libraries (TestFieldCUDA "gtest;pthread")
