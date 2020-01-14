// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(DATA_TYPES_VEC_VEC_HPP)
#define DATA_TYPES_VEC_VEC_HPP

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#include <tuple/Tuple.hpp>
#include <DataTypes.hpp>

namespace XXX_NAMESPACE
{
    namespace dataTypes
    {
        template <typename ValueT, SizeT N>
        using Vec = Builder<Tuple, ValueT, N>;
    }
}

#endif
