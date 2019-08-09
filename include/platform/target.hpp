// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(PLATFORM_TARGET_HPP)
#define PLATFORM_TARGET_HPP

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

namespace XXX_NAMESPACE
{
    enum class target {Host = 1, GPU_CUDA = 2};
}

#endif