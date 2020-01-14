// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(PLATFORM_TARGET_HPP)
#define PLATFORM_TARGET_HPP

#if defined(__CUDACC__)
#include <cuda_runtime.h>

#define HOST_VERSION __host__
#define CUDA_DEVICE_VERSION __device__
#define CUDA_KERNEL __global__
#else
#define HOST_VERSION 
#define CUDA_DEVICE_VERSION 
#define CUDA_KERNEL 
#endif

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

namespace XXX_NAMESPACE
{
    namespace platform
    {
#if defined(__CUDACC__)
        enum class Identifier {Host, GPU_CUDA};
#else
        enum class Identifier {Host};
#endif
    }
}

#endif