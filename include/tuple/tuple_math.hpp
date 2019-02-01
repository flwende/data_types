// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(TUPLE_TUPLE_MATH_HPP)
#define TUPLE_TUPLE_MATH_HPP

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#if !defined(TUPLE_NAMESPACE)
#define TUPLE_NAMESPACE XXX_NAMESPACE
#endif

#include "tuple.hpp"

namespace TUPLE_NAMESPACE
{
#define MACRO_UNQUALIFIED(OP, IN_T)                                                                                                         \
    template <typename T_1, typename T_2, typename T_3>                                                                                     \
    inline tuple<T_1, T_2, T_3> OP (IN_T<T_1, T_2, T_3>& t)                                                                                 \
    {                                                                                                                                       \
        return tuple<T_1, T_2, T_3>(AUXILIARY_NAMESPACE::math<T_1>:: OP (t.x), AUXILIARY_NAMESPACE::math<T_2>:: OP (t.y), AUXILIARY_NAMESPACE::math<T_3>:: OP (t.z));   \
    }                                                                                                                                       \

#define MACRO_QUALIFIED(OP, IN_T)                                                                                                           \
    MACRO_UNQUALIFIED(OP, IN_T)                                                                                                             \
    MACRO_UNQUALIFIED(OP, const IN_T)                                                                                                       \

#define MACRO(OP, IN_T)                                                                                                                     \
    MACRO_QUALIFIED(OP, IN_T)                                                                                                               \

    MACRO(sqrt, tuple)
    MACRO(log, tuple)
    MACRO(exp, tuple)

    MACRO(sqrt, internal::tuple_proxy)
    MACRO(log, internal::tuple_proxy)
    MACRO(exp, internal::tuple_proxy)

#undef MACRO
#undef MACRO_QUALIFIED
#undef MACRO_UNQUALIFIED
}

#endif