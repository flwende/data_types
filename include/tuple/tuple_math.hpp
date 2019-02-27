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

#if !defined(MATH_NAMESPACE)
#define MATH_NAMESPACE XXX_NAMESPACE
#endif

#include "../auxiliary/math.hpp"

namespace MATH_NAMESPACE
{
#define MACRO_UNQUALIFIED(OP, IN_T_1, IN_T_2)                                                                                               \
    template <typename T_1, typename T_2, typename T_3, typename T_4, typename T_5, typename T_6,                                           \
              typename X_1 = typename XXX_NAMESPACE::internal::compare<T_1, T_4>::stronger_type_unqualified,                                \
              typename X_2 = typename XXX_NAMESPACE::internal::compare<T_2, T_5>::stronger_type_unqualified,                                \
              typename X_3 = typename XXX_NAMESPACE::internal::compare<T_3, T_6>::stronger_type_unqualified>                                \
    inline tuple<X_1, X_2, X_3> operator OP (IN_T_1<T_1, T_2, T_3>& x_1, IN_T_2<T_4, T_5, T_6>& x_2)                                        \
    {                                                                                                                                       \
        tuple<X_1, X_2, X_3> y(x_1);                                                                                                        \
        y OP ## = x_2;                                                                                                                      \
        return y;                                                                                                                           \
    }                                                                                                                                       \

#define MACRO_QUALIFIED(OP, IN_T_1, IN_T_2)                                                                                                 \
    MACRO_UNQUALIFIED(OP, IN_T_1, IN_T_2)                                                                                                   \
    MACRO_UNQUALIFIED(OP, const IN_T_1, IN_T_2)                                                                                             \
    MACRO_UNQUALIFIED(OP, IN_T_1, const IN_T_2)                                                                                             \
    MACRO_UNQUALIFIED(OP, const IN_T_1, const IN_T_2)                                                                                       \

#define MACRO(OP, IN_T_1, IN_T_2)                                                                                                           \
    MACRO_QUALIFIED(OP, IN_T_1, IN_T_1)                                                                                                     \
    MACRO_QUALIFIED(OP, IN_T_2, IN_T_2)                                                                                                     \
    MACRO_QUALIFIED(OP, IN_T_1, IN_T_2)                                                                                                     \
    MACRO_QUALIFIED(OP, IN_T_2, IN_T_1)                                                                                                     \

    MACRO(+, TUPLE_NAMESPACE::tuple, TUPLE_NAMESPACE::internal::tuple_proxy)
    MACRO(-, TUPLE_NAMESPACE::tuple, TUPLE_NAMESPACE::internal::tuple_proxy)
    MACRO(*, TUPLE_NAMESPACE::tuple, TUPLE_NAMESPACE::internal::tuple_proxy)
    MACRO(/, TUPLE_NAMESPACE::tuple, TUPLE_NAMESPACE::internal::tuple_proxy)

#undef MACRO
#undef MACRO_QUALIFIED
#undef MACRO_UNQUALIFIED

#define MACRO_UNQUALIFIED(OP, IN_T)                                                                                                         \
    template <typename T_1, typename T_2, typename T_3, typename T_4,                                                                       \
              typename X_1 = typename XXX_NAMESPACE::internal::compare<T_1, T_4>::stronger_type_unqualified,                                \
              typename X_2 = typename XXX_NAMESPACE::internal::compare<T_2, T_4>::stronger_type_unqualified,                                \
              typename X_3 = typename XXX_NAMESPACE::internal::compare<T_3, T_4>::stronger_type_unqualified>                                \
    inline tuple<X_1, X_2, X_3> operator OP (IN_T<T_1, T_2, T_3>& x_1, const T_4 x_2)                                                       \
    {                                                                                                                                       \
        tuple<X_1, X_2, X_3> y(x_1);                                                                                                        \
        y OP ## = x_2;                                                                                                                      \
        return y;                                                                                                                           \
    }                                                                                                                                       \
                                                                                                                                            \
    template <typename T_1, typename T_2, typename T_3, typename T_4,                                                                       \
              typename X_1 = typename XXX_NAMESPACE::internal::compare<T_1, T_4>::stronger_type_unqualified,                                \
              typename X_2 = typename XXX_NAMESPACE::internal::compare<T_2, T_4>::stronger_type_unqualified,                                \
              typename X_3 = typename XXX_NAMESPACE::internal::compare<T_3, T_4>::stronger_type_unqualified>                                \
    inline tuple<X_1, X_2, X_3> operator OP (const T_1 x_1, IN_T<T_2, T_3, T_4>& x_2)                                                       \
    {                                                                                                                                       \
        tuple<X_1, X_2, X_3> y(x_1);                                                                                                        \
        y OP ## = x_2;                                                                                                                      \
        return y;                                                                                                                           \
    }                                                                                                                                       \

#define MACRO_QUALIFIED(OP, IN_T)                                                                                                           \
    MACRO_UNQUALIFIED(OP, IN_T)                                                                                                             \
    MACRO_UNQUALIFIED(OP, const IN_T)                                                                                                       \

#define MACRO(OP, IN_T)                                                                                                                     \
    MACRO_QUALIFIED(OP, IN_T)                                                                                                               \

    MACRO(+, TUPLE_NAMESPACE::tuple)
    MACRO(-, TUPLE_NAMESPACE::tuple)
    MACRO(*, TUPLE_NAMESPACE::tuple)
    MACRO(/, TUPLE_NAMESPACE::tuple)

    MACRO(+, TUPLE_NAMESPACE::internal::tuple_proxy)
    MACRO(-, TUPLE_NAMESPACE::internal::tuple_proxy)
    MACRO(*, TUPLE_NAMESPACE::internal::tuple_proxy)
    MACRO(/, TUPLE_NAMESPACE::internal::tuple_proxy)

#undef MACRO
#undef MACRO_QUALIFIED
#undef MACRO_UNQUALIFIED

#define MACRO_UNQUALIFIED(OP, IN_T)                                                                                                         \
    template <typename T_1, typename T_2, typename T_3,                                                                                     \
              typename X_1 = typename std::remove_cv<T_1>::type,                                                                            \
              typename X_2 = typename std::remove_cv<T_2>::type,                                                                            \
              typename X_3 = typename std::remove_cv<T_3>::type>                                                                            \
    inline tuple<X_1, X_2, X_3> OP (IN_T<T_1, T_2, T_3>& t)                                                                                 \
    {                                                                                                                                       \
        return tuple<X_1, X_2, X_3>(MATH_NAMESPACE::math<T_1>:: OP (t.x),                                                                   \
                                    MATH_NAMESPACE::math<T_2>:: OP (t.y),                                                                   \
                                    MATH_NAMESPACE::math<T_3>:: OP (t.z));                                                                  \
    }                                                                                                                                       \

#define MACRO_QUALIFIED(OP, IN_T)                                                                                                           \
    MACRO_UNQUALIFIED(OP, IN_T)                                                                                                             \
    MACRO_UNQUALIFIED(OP, const IN_T)                                                                                                       \

#define MACRO(OP, IN_T)                                                                                                                     \
    MACRO_QUALIFIED(OP, IN_T)                                                                                                               \

    MACRO(sqrt, TUPLE_NAMESPACE::tuple)
    MACRO(log, TUPLE_NAMESPACE::tuple)
    MACRO(exp, TUPLE_NAMESPACE::tuple)

    MACRO(sqrt, TUPLE_NAMESPACE::internal::tuple_proxy)
    MACRO(log, TUPLE_NAMESPACE::internal::tuple_proxy)
    MACRO(exp, TUPLE_NAMESPACE::internal::tuple_proxy)

#undef MACRO
#undef MACRO_QUALIFIED
#undef MACRO_UNQUALIFIED

#define MACRO_UNQUALIFIED(IN_T_1, IN_T_2)                                                                                                   \
    template <typename T_1, typename T_2, typename T_3, typename T_4, typename T_5, typename T_6,                                           \
              typename X_1 = typename XXX_NAMESPACE::internal::compare<T_1, T_4>::stronger_type_unqualified,                                \
              typename X_2 = typename XXX_NAMESPACE::internal::compare<T_2, T_5>::stronger_type_unqualified,                                \
              typename X_3 = typename XXX_NAMESPACE::internal::compare<T_3, T_6>::stronger_type_unqualified>                                \
    inline tuple<X_1, X_2, X_3> cross_product(IN_T_1<T_1, T_2, T_3>& x_1, IN_T_2<T_4, T_5, T_6>& x_2)                                       \
    {                                                                                                                                       \
        return tuple<X_1, X_2, X_3>(x_1.y * x_2.z - x_1.z * x_2.y, x_1.z * x_2.x - x_1.x * x_2.z, x_1.x * x_2.y - x_1.y * x_2.x);           \
    }                                                                                                                                       \
                                                                                                                                            \
    template <typename T_1, typename T_2, typename T_3, typename T_4, typename T_5, typename T_6,                                           \
              typename X_1 = typename XXX_NAMESPACE::internal::compare<T_1, T_4>::stronger_type_unqualified,                                \
              typename X_2 = typename XXX_NAMESPACE::internal::compare<T_2, T_5>::stronger_type_unqualified,                                \
              typename X_3 = typename XXX_NAMESPACE::internal::compare<T_3, T_6>::stronger_type_unqualified>                                \
    inline tuple<X_1, X_2, X_3> cross(IN_T_1<T_1, T_2, T_3>& x_1, IN_T_2<T_4, T_5, T_6>& x_2)                                               \
    {                                                                                                                                       \
        return cross_product(x_1, x_2);                                                                                                     \
    }                                                                                                                                       \

#define MACRO_QUALIFIED(IN_T_1, IN_T_2)                                                                                                     \
    MACRO_UNQUALIFIED(IN_T_1, IN_T_2)                                                                                                       \
    MACRO_UNQUALIFIED(const IN_T_1, IN_T_2)                                                                                                 \
    MACRO_UNQUALIFIED(IN_T_1, const IN_T_2)                                                                                                 \
    MACRO_UNQUALIFIED(const IN_T_1, const IN_T_2)                                                                                           \

#define MACRO(IN_T_1, IN_T_2)                                                                                                               \
    MACRO_QUALIFIED(IN_T_1, IN_T_1)                                                                                                         \
    MACRO_QUALIFIED(IN_T_2, IN_T_2)                                                                                                         \
    MACRO_QUALIFIED(IN_T_1, IN_T_2)                                                                                                         \
    MACRO_QUALIFIED(IN_T_2, IN_T_1)                                                                                                         \

    MACRO(TUPLE_NAMESPACE::tuple, TUPLE_NAMESPACE::internal::tuple_proxy)

#undef MACRO
#undef MACRO_QUALIFIED
#undef MACRO_UNQUALIFIED

#define MACRO_UNQUALIFIED(IN_T)                                                                                                             \
    template <typename T_1, typename T_2, typename T_3, typename T_4,                                                                       \
              typename X_1 = typename XXX_NAMESPACE::internal::compare<T_1, T_4>::stronger_type_unqualified,                                \
              typename X_2 = typename XXX_NAMESPACE::internal::compare<T_2, T_4>::stronger_type_unqualified,                                \
              typename X_3 = typename XXX_NAMESPACE::internal::compare<T_3, T_4>::stronger_type_unqualified>                                \
    inline tuple<X_1, X_2, X_3> cross_product(IN_T<T_1, T_2, T_3>& x_1, const T_4 x_2)                                                      \
    {                                                                                                                                       \
        return tuple<X_1, X_2, X_3>(x_1.y - x_1.z, x_1.z - x_1.x, x_1.x - x_1.y) * x_2;                                                     \
    }                                                                                                                                       \
                                                                                                                                            \
    template <typename T_1, typename T_2, typename T_3, typename T_4,                                                                       \
              typename X_1 = typename XXX_NAMESPACE::internal::compare<T_1, T_4>::stronger_type_unqualified,                                \
              typename X_2 = typename XXX_NAMESPACE::internal::compare<T_2, T_4>::stronger_type_unqualified,                                \
              typename X_3 = typename XXX_NAMESPACE::internal::compare<T_3, T_4>::stronger_type_unqualified>                                \
    inline tuple<X_1, X_2, X_3> cross_product(const T_1 x_1, IN_T<T_2, T_2, T_3>& x_2)                                                      \
    {                                                                                                                                       \
        return x_1 * tuple<X_1, X_2, X_3>(x_2.z - x_2.y, x_2.x - x_2.z, x_2.y - x_2.x);                                                     \
    }                                                                                                                                       \
                                                                                                                                            \
    template <typename T_1, typename T_2, typename T_3, typename T_4,                                                                       \
              typename X_1 = typename XXX_NAMESPACE::internal::compare<T_1, T_4>::stronger_type_unqualified,                                \
              typename X_2 = typename XXX_NAMESPACE::internal::compare<T_2, T_4>::stronger_type_unqualified,                                \
              typename X_3 = typename XXX_NAMESPACE::internal::compare<T_3, T_4>::stronger_type_unqualified>                                \
    inline tuple<X_1, X_2, X_3> cross(IN_T<T_1, T_2, T_3>& x_1, const T_4 x_2)                                                              \
    {                                                                                                                                       \
        return cross_product(x_1, x_2);                                                                                                     \
    }                                                                                                                                       \
                                                                                                                                            \
    template <typename T_1, typename T_2, typename T_3, typename T_4,                                                                       \
              typename X_1 = typename XXX_NAMESPACE::internal::compare<T_1, T_4>::stronger_type_unqualified,                                \
              typename X_2 = typename XXX_NAMESPACE::internal::compare<T_2, T_4>::stronger_type_unqualified,                                \
              typename X_3 = typename XXX_NAMESPACE::internal::compare<T_3, T_4>::stronger_type_unqualified>                                \
    inline tuple<X_1, X_2, X_3> cross(const T_1 x_1, IN_T<T_2, T_2, T_3>& x_2)                                                              \
    {                                                                                                                                       \
        return cross_product(x_1, x_2);                                                                                                     \
    }                                                                                                                                       \

#define MACRO_QUALIFIED(IN_T)                                                                                                               \
    MACRO_UNQUALIFIED(IN_T)                                                                                                                 \
    MACRO_UNQUALIFIED(const IN_T)                                                                                                           \
    
#define MACRO(IN_T)                                                                                                                         \
    MACRO_QUALIFIED(IN_T)                                                                                                                   \
    
    MACRO(TUPLE_NAMESPACE::tuple)
    MACRO(TUPLE_NAMESPACE::internal::tuple_proxy)

#undef MACRO
#undef MACRO_QUALIFIED
#undef MACRO_UNQUALIFIED
}

#endif