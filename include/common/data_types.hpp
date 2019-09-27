// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(COMMON_DATA_TYPES_HPP)
#define COMMON_DATA_TYPES_HPP

#include <cstdint>
#include <cstdlib>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

namespace XXX_NAMESPACE
{
    using size_type = std::size_t;
    //using size_type = std::uint32_t;
    using real_type = float;
}

#endif