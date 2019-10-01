// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(COMMON_DATA_LAYOUT_HPP)
#define COMMON_DATA_LAYOUT_HPP

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

namespace XXX_NAMESPACE
{
    namespace memory
    {
        enum class DataLayout
        {
            AoS,
            SoA,
            SoAi
        };
    } // namespace memory
} // namespace XXX_NAMESPACE

#endif