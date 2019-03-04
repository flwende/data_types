// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(VEC_VEC_MATH_HPP)
#define VEC_VEC_MATH_HPP

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#if !defined(VEC_NAMESPACE)
#define VEC_NAMESPACE XXX_NAMESPACE
#endif

#if !defined(MATH_NAMESPACE)
#define MATH_NAMESPACE XXX_NAMESPACE
#endif

#include "../auxiliary/math.hpp"

namespace MATH_NAMESPACE
{
#define MACRO_UNQUALIFIED(OP, IN_T_1, IN_T_2)                                                                                               \
    template <typename T_1, typename T_2, std::size_t D,                                                                                    \
              typename X = typename XXX_NAMESPACE::internal::compare<T_1, T_2>::stronger_type_unqualified>                                  \
    inline vec<X, D> operator OP (IN_T_1<T_1, D>& x_1, IN_T_2<T_2, D>& x_2)                                                                 \
    {                                                                                                                                       \
        vec<X, D> y(x_1);                                                                                                                   \
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

    MACRO(+, VEC_NAMESPACE::vec, VEC_NAMESPACE::internal::vec_proxy)
    MACRO(-, VEC_NAMESPACE::vec, VEC_NAMESPACE::internal::vec_proxy)
    MACRO(*, VEC_NAMESPACE::vec, VEC_NAMESPACE::internal::vec_proxy)
    MACRO(/, VEC_NAMESPACE::vec, VEC_NAMESPACE::internal::vec_proxy)

#undef MACRO
#undef MACRO_QUALIFIED
#undef MACRO_UNQUALIFIED

#define MACRO_UNQUALIFIED(OP, IN_T)                                                                                                         \
    template <typename T_1, typename T_2, std::size_t D,                                                                                    \
              typename X = typename XXX_NAMESPACE::internal::compare<T_1, T_2>::stronger_type_unqualified>                                  \
    inline vec<X, D> operator OP (IN_T<T_1, D>& x_1, const T_2 x_2)                                                                         \
    {                                                                                                                                       \
        vec<X, D> y(x_1);                                                                                                                   \
        y OP ## = x_2;                                                                                                                      \
        return y;                                                                                                                           \
    }                                                                                                                                       \
                                                                                                                                            \
    template <typename T_1, typename T_2, std::size_t D,                                                                                    \
              typename X = typename XXX_NAMESPACE::internal::compare<T_1, T_2>::stronger_type_unqualified>                                  \
    inline vec<X, D> operator OP (const T_1 x_1, IN_T<T_2, D>& x_2)                                                                         \
    {                                                                                                                                       \
        vec<X, D> y(x_1);                                                                                                                   \
        y OP ## = x_2;                                                                                                                      \
        return y;                                                                                                                           \
    }                                                                                                                                       \

#define MACRO_QUALIFIED(OP, IN_T)                                                                                                           \
    MACRO_UNQUALIFIED(OP, IN_T)                                                                                                             \
    MACRO_UNQUALIFIED(OP, const IN_T)                                                                                                       \

#define MACRO(OP, IN_T)                                                                                                                     \
    MACRO_QUALIFIED(OP, IN_T)                                                                                                               \

    MACRO(+, VEC_NAMESPACE::vec)
    MACRO(-, VEC_NAMESPACE::vec)
    MACRO(*, VEC_NAMESPACE::vec)
    MACRO(/, VEC_NAMESPACE::vec)

    MACRO(+, VEC_NAMESPACE::internal::vec_proxy)
    MACRO(-, VEC_NAMESPACE::internal::vec_proxy)
    MACRO(*, VEC_NAMESPACE::internal::vec_proxy)
    MACRO(/, VEC_NAMESPACE::internal::vec_proxy)

#undef MACRO
#undef MACRO_QUALIFIED
#undef MACRO_UNQUALIFIED

#define MACRO_UNQUALIFIED(OP, IN_T)                                                                                                         \
    template <typename T, typename X = typename std::remove_cv<T>::type>                                                                    \
    inline vec<T, 1> OP (IN_T<T, 1>& v)                                                                                                     \
    {                                                                                                                                       \
        return vec<T, 1>(MATH_NAMESPACE::math<T>:: OP (v.x));                                                                               \
    }                                                                                                                                       \
                                                                                                                                            \
    template <typename T, typename X = typename std::remove_cv<T>::type>                                                                    \
    inline vec<T, 2> OP (IN_T<T, 2>& v)                                                                                                     \
    {                                                                                                                                       \
        return vec<T, 2>(MATH_NAMESPACE::math<T>:: OP (v.x), MATH_NAMESPACE::math<T>:: OP (v.y));                                           \
    }                                                                                                                                       \
                                                                                                                                            \
    template <typename T, typename X = typename std::remove_cv<T>::type>                                                                    \
    inline vec<X, 3> OP (IN_T<T, 3>& v)                                                                                                     \
    {                                                                                                                                       \
        return vec<X, 3>(MATH_NAMESPACE::math<X>:: OP (v.x),                                                                                \
                         MATH_NAMESPACE::math<X>:: OP (v.y),                                                                                \
                         MATH_NAMESPACE::math<X>:: OP (v.z));                                                                               \
    }                                                                                                                                       \

#define MACRO_QUALIFIED(OP, IN_T)                                                                                                           \
    MACRO_UNQUALIFIED(OP, IN_T)                                                                                                             \
    MACRO_UNQUALIFIED(OP, const IN_T)                                                                                                       \

#define MACRO(OP, IN_T)                                                                                                                     \
    MACRO_QUALIFIED(OP, IN_T)                                                                                                               \

    MACRO(sqrt, VEC_NAMESPACE::vec)
    MACRO(log, VEC_NAMESPACE::vec)
    MACRO(exp, VEC_NAMESPACE::vec)

    MACRO(sqrt, VEC_NAMESPACE::internal::vec_proxy)
    MACRO(log, VEC_NAMESPACE::internal::vec_proxy)
    MACRO(exp, VEC_NAMESPACE::internal::vec_proxy)

#undef MACRO
#undef MACRO_QUALIFIED
#undef MACRO_UNQUALIFIED

#define MACRO_UNQUALIFIED(IN_T_1, IN_T_2)                                                                                                   \
    template <typename T_1, typename T_2,                                                                                                   \
              typename X = typename XXX_NAMESPACE::internal::compare<T_1, T_2>::stronger_type_unqualified>                                  \
    inline X dot_product(IN_T_1<T_1, 1>& x_1, IN_T_2<T_2, 1>& x_2)                                                                          \
    {                                                                                                                                       \
        return (x_1.x * x_2.x);                                                                                                             \
    }                                                                                                                                       \
                                                                                                                                            \
    template <typename T_1, typename T_2,                                                                                                   \
              typename X = typename XXX_NAMESPACE::internal::compare<T_1, T_2>::stronger_type_unqualified>                                  \
    inline X dot_product(IN_T_1<T_1, 2>& x_1, IN_T_2<T_2, 2>& x_2)                                                                          \
    {                                                                                                                                       \
        return (x_1.x * x_2.x + x_1.y * x_2.y);                                                                                             \
    }                                                                                                                                       \
                                                                                                                                            \
    template <typename T_1, typename T_2,                                                                                                   \
              typename X = typename XXX_NAMESPACE::internal::compare<T_1, T_2>::stronger_type_unqualified>                                  \
    inline X dot_product(IN_T_1<T_1, 3>& x_1, IN_T_2<T_2, 3>& x_2)                                                                          \
    {                                                                                                                                       \
        return (x_1.x * x_2.x + x_1.y * x_2.y + x_1.z * x_2.z);                                                                             \
    }                                                                                                                                       \
                                                                                                                                            \
    template <typename T_1, typename T_2, std::size_t D,                                                                                    \
              typename X = typename XXX_NAMESPACE::internal::compare<T_1, T_2>::stronger_type_unqualified>                                  \
    inline X dot(IN_T_1<T_1, D>& x_1, IN_T_2<T_2, D>& x_2)                                                                                  \
    {                                                                                                                                       \
        return dot_product(x_1, x_2);                                                                                                       \
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

    MACRO(VEC_NAMESPACE::vec, VEC_NAMESPACE::internal::vec_proxy)

#undef MACRO
#undef MACRO_QUALIFIED
#undef MACRO_UNQUALIFIED

#define MACRO_UNQUALIFIED(IN_T)                                                                                                             \
    template <typename T_1, typename T_2,                                                                                                   \
              typename X = typename XXX_NAMESPACE::internal::compare<T_1, T_2>::stronger_type_unqualified>                                  \
    inline X dot_product(IN_T<T_1, 1>& x_1, const T_2 x_2)                                                                                  \
    {                                                                                                                                       \
        return (x_1.x * x_2);                                                                                                               \
    }                                                                                                                                       \
                                                                                                                                            \
    template <typename T_1, typename T_2,                                                                                                   \
              typename X = typename XXX_NAMESPACE::internal::compare<T_1, T_2>::stronger_type_unqualified>                                  \
    inline X dot_product(IN_T<T_1, 2>& x_1, const T_2 x_2)                                                                                  \
    {                                                                                                                                       \
        return (x_1.x + x_1.y) * x_2;                                                                                                       \
    }                                                                                                                                       \
                                                                                                                                            \
    template <typename T_1, typename T_2,                                                                                                   \
              typename X = typename XXX_NAMESPACE::internal::compare<T_1, T_2>::stronger_type_unqualified>                                  \
    inline X dot_product(IN_T<T_1, 3>& x_1, const T_2 x_2)                                                                                  \
    {                                                                                                                                       \
        return (x_1.x + x_1.y + x_1.z) * x_2;                                                                                               \
    }                                                                                                                                       \
                                                                                                                                            \
    template <typename T_1, typename T_2,                                                                                                   \
              typename X = typename XXX_NAMESPACE::internal::compare<T_1, T_2>::stronger_type_unqualified>                                  \
    inline X dot_product(const T_1 x_1, IN_T<T_2, 1>& x_2)                                                                                  \
    {                                                                                                                                       \
        return (x_1 * x_2.x);                                                                                                               \
    }                                                                                                                                       \
                                                                                                                                            \
    template <typename T_1, typename T_2,                                                                                                   \
              typename X = typename XXX_NAMESPACE::internal::compare<T_1, T_2>::stronger_type_unqualified>                                  \
    inline X dot_product(const T_1 x_1, IN_T<T_2, 2>& x_2)                                                                                  \
    {                                                                                                                                       \
        return x_1 * (x_2.x + x_2.y);                                                                                                       \
    }                                                                                                                                       \
                                                                                                                                            \
    template <typename T_1, typename T_2,                                                                                                   \
              typename X = typename XXX_NAMESPACE::internal::compare<T_1, T_2>::stronger_type_unqualified>                                  \
    inline X dot_product(const T_1 x_1, IN_T<T_2, 3>& x_2)                                                                                  \
    {                                                                                                                                       \
        return x_1 * (x_2.x + x_2.y + x_2.z);                                                                                               \
    }                                                                                                                                       \
                                                                                                                                            \
    template <typename T_1, typename T_2, std::size_t D,                                                                                    \
              typename X = typename XXX_NAMESPACE::internal::compare<T_1, T_2>::stronger_type_unqualified>                                  \
    inline X dot(IN_T<T_1, D>& x_1, const T_2 x_2)                                                                                          \
    {                                                                                                                                       \
        return dot_product(x_1, x_2);                                                                                                       \
    }                                                                                                                                       \
                                                                                                                                            \
    template <typename T_1, typename T_2, std::size_t D,                                                                                    \
              typename X = typename XXX_NAMESPACE::internal::compare<T_1, T_2>::stronger_type_unqualified>                                  \
    inline X dot(const T_1 x_1, IN_T<T_2, D>& x_2)                                                                                          \
    {                                                                                                                                       \
        return dot_product(x_1, x_2);                                                                                                       \
    }                                                                                                                                       \

#define MACRO_QUALIFIED(IN_T)                                                                                                               \
    MACRO_UNQUALIFIED(IN_T)                                                                                                                 \
    MACRO_UNQUALIFIED(const IN_T)                                                                                                           \
    
#define MACRO(IN_T)                                                                                                                         \
    MACRO_QUALIFIED(IN_T)                                                                                                                   \
    
    MACRO(VEC_NAMESPACE::vec)
    MACRO(VEC_NAMESPACE::internal::vec_proxy)

#undef MACRO
#undef MACRO_QUALIFIED
#undef MACRO_UNQUALIFIED

#define MACRO_UNQUALIFIED(IN_T_1, IN_T_2)                                                                                                   \
    template <typename T_1, typename T_2,                                                                                                   \
              typename X = typename XXX_NAMESPACE::internal::compare<T_1, T_2>::stronger_type_unqualified>                                  \
    inline vec<X, 3> cross_product(IN_T_1<T_1, 3>& x_1, IN_T_2<T_2, 3>& x_2)                                                                \
    {                                                                                                                                       \
        return vec<X, 3>(x_1.y * x_2.z - x_1.z * x_2.y, x_1.z * x_2.x - x_1.x * x_2.z, x_1.x * x_2.y - x_1.y * x_2.x);                      \
    }                                                                                                                                       \
                                                                                                                                            \
    template <typename T_1, typename T_2,                                                                                                   \
              typename X = typename XXX_NAMESPACE::internal::compare<T_1, T_2>::stronger_type_unqualified>                                  \
    inline vec<X, 3> cross(IN_T_1<T_1, 3>& x_1, IN_T_2<T_2, 3>& x_2)                                                                        \
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

    MACRO(VEC_NAMESPACE::vec, VEC_NAMESPACE::internal::vec_proxy)

#undef MACRO
#undef MACRO_QUALIFIED
#undef MACRO_UNQUALIFIED

#define MACRO_UNQUALIFIED(IN_T)                                                                                                             \
    template <typename T_1, typename T_2,                                                                                                   \
              typename X = typename XXX_NAMESPACE::internal::compare<T_1, T_2>::stronger_type_unqualified>                                  \
    inline vec<X, 3> cross_product(IN_T<T_1, 3>& x_1, const T_2 x_2)                                                                        \
    {                                                                                                                                       \
        return vec<X, 3>(x_1.y - x_1.z, x_1.z - x_1.x, x_1.x - x_1.y) * x_2;                                                                \
    }                                                                                                                                       \
                                                                                                                                            \
    template <typename T_1, typename T_2,                                                                                                   \
              typename X = typename XXX_NAMESPACE::internal::compare<T_1, T_2>::stronger_type_unqualified>                                  \
    inline vec<X, 3> cross_product(const T_1 x_1, IN_T<T_2, 3>& x_2)                                                                        \
    {                                                                                                                                       \
        return x_1 * vec<X, 3>(x_2.z - x_2.y, x_2.x - x_2.z, x_2.y - x_2.x);                                                                \
    }                                                                                                                                       \
                                                                                                                                            \
    template <typename T_1, typename T_2,                                                                                                   \
              typename X = typename XXX_NAMESPACE::internal::compare<T_1, T_2>::stronger_type_unqualified>                                  \
    inline vec<X, 3> cross(IN_T<T_1, 3>& x_1, const T_2 x_2)                                                                                \
    {                                                                                                                                       \
        return cross_product(x_1, x_2);                                                                                                     \
    }                                                                                                                                       \
                                                                                                                                            \
    template <typename T_1, typename T_2,                                                                                                   \
              typename X = typename XXX_NAMESPACE::internal::compare<T_1, T_2>::stronger_type_unqualified>                                  \
    inline vec<X, 3> cross(const T_1 x_1, IN_T<T_2, 3>& x_2)                                                                                \
    {                                                                                                                                       \
        return cross_product(x_1, x_2);                                                                                                     \
    }                                                                                                                                       \

#define MACRO_QUALIFIED(IN_T)                                                                                                               \
    MACRO_UNQUALIFIED(IN_T)                                                                                                                 \
    MACRO_UNQUALIFIED(const IN_T)                                                                                                           \
    
#define MACRO(IN_T)                                                                                                                         \
    MACRO_QUALIFIED(IN_T)                                                                                                                   \
    
    MACRO(VEC_NAMESPACE::vec)
    MACRO(VEC_NAMESPACE::internal::vec_proxy)

#undef MACRO
#undef MACRO_QUALIFIED
#undef MACRO_UNQUALIFIED
}

#endif