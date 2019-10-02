// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(COMMON_DATA_TYPES_HPP)
#define COMMON_DATA_TYPES_HPP

#include <cstdint>
//#include <cstdlib>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

namespace XXX_NAMESPACE
{
    using SizeType = std::size_t;
    using RealType = float;
}

#include <data_types/array/Array.hpp>

namespace XXX_NAMESPACE
{
    template <SizeType C_N>
    using SizeArray = fw::sarray<SizeType, C_N>;
}


#endif