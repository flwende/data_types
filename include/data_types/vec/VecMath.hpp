// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(DATA_TYPES_VEC_VEC_MATH_HPP)
#define DATA_TYPES_VEC_VEC_MATH_HPP

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#include <common/Math.hpp>
#include <data_types/DataTypes.hpp>

namespace XXX_NAMESPACE
{
    namespace math
    {
    #define MACRO_UNQUALIFIED(OP, IN_T_1, IN_T_2)                                                                                               \
        template <typename T_1, typename T_2, SizeT D,                                                                                    \
                typename X = typename XXX_NAMESPACE::dataTypes::Compare<T_1, T_2>::UnqualifiedStrongerT>                                  \
        inline ::XXX_NAMESPACE::dataTypes::Vec<X, D> operator OP (IN_T_1<T_1, D>& x_1, IN_T_2<T_2, D>& x_2)                                                                 \
        {                                                                                                                                       \
            ::XXX_NAMESPACE::dataTypes::Vec<X, D> y(x_1);                                                                                                                   \
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

        MACRO(+, ::XXX_NAMESPACE::dataTypes::Vec, ::XXX_NAMESPACE::dataTypes::internal::VecProxy)
        MACRO(-, ::XXX_NAMESPACE::dataTypes::Vec, ::XXX_NAMESPACE::dataTypes::internal::VecProxy)
        MACRO(*, ::XXX_NAMESPACE::dataTypes::Vec, ::XXX_NAMESPACE::dataTypes::internal::VecProxy)
        MACRO(/, ::XXX_NAMESPACE::dataTypes::Vec, ::XXX_NAMESPACE::dataTypes::internal::VecProxy)

    #undef MACRO
    #undef MACRO_QUALIFIED
    #undef MACRO_UNQUALIFIED

    #define MACRO_UNQUALIFIED(OP, IN_T)                                                                                                         \
        template <typename T_1, typename T_2, SizeT D,                                                                                    \
                typename X = typename XXX_NAMESPACE::dataTypes::Compare<T_1, T_2>::UnqualifiedStrongerT>                                  \
        inline ::XXX_NAMESPACE::dataTypes::Vec<X, D> operator OP (IN_T<T_1, D>& x_1, const T_2 x_2)                                                                         \
        {                                                                                                                                       \
            ::XXX_NAMESPACE::dataTypes::Vec<X, D> y(x_1);                                                                                                                   \
            y OP ## = x_2;                                                                                                                      \
            return y;                                                                                                                           \
        }                                                                                                                                       \
                                                                                                                                                \
        template <typename T_1, typename T_2, SizeT D,                                                                                    \
                typename X = typename XXX_NAMESPACE::dataTypes::Compare<T_1, T_2>::UnqualifiedStrongerT>                                  \
        inline ::XXX_NAMESPACE::dataTypes::Vec<X, D> operator OP (const T_1 x_1, IN_T<T_2, D>& x_2)                                                                         \
        {                                                                                                                                       \
            ::XXX_NAMESPACE::dataTypes::Vec<X, D> y(x_1);                                                                                                                   \
            y OP ## = x_2;                                                                                                                      \
            return y;                                                                                                                           \
        }                                                                                                                                       \

    #define MACRO_QUALIFIED(OP, IN_T)                                                                                                           \
        MACRO_UNQUALIFIED(OP, IN_T)                                                                                                             \
        MACRO_UNQUALIFIED(OP, const IN_T)                                                                                                       \

    #define MACRO(OP, IN_T)                                                                                                                     \
        MACRO_QUALIFIED(OP, IN_T)                                                                                                               \

        MACRO(+, ::XXX_NAMESPACE::dataTypes::Vec)
        MACRO(-, ::XXX_NAMESPACE::dataTypes::Vec)
        MACRO(*, ::XXX_NAMESPACE::dataTypes::Vec)
        MACRO(/, ::XXX_NAMESPACE::dataTypes::Vec)

        MACRO(+, ::XXX_NAMESPACE::dataTypes::internal::VecProxy)
        MACRO(-, ::XXX_NAMESPACE::dataTypes::internal::VecProxy)
        MACRO(*, ::XXX_NAMESPACE::dataTypes::internal::VecProxy)
        MACRO(/, ::XXX_NAMESPACE::dataTypes::internal::VecProxy)

    #undef MACRO
    #undef MACRO_QUALIFIED
    #undef MACRO_UNQUALIFIED

    #define MACRO_UNQUALIFIED(OP, IN_T)                                                                                                         \
        template <typename T, typename X = typename std::remove_cv<T>::type>                                                                    \
        inline ::XXX_NAMESPACE::dataTypes::Vec<T, 1> OP (IN_T<T, 1>& v)                                                                                                     \
        {                                                                                                                                       \
            return ::XXX_NAMESPACE::dataTypes::Vec<T, 1>(::XXX_NAMESPACE::math::internal::Func<T>:: OP (v.x));                                                                               \
        }                                                                                                                                       \
                                                                                                                                                \
        template <typename T, typename X = typename std::remove_cv<T>::type>                                                                    \
        inline ::XXX_NAMESPACE::dataTypes::Vec<T, 2> OP (IN_T<T, 2>& v)                                                                                                     \
        {                                                                                                                                       \
            return ::XXX_NAMESPACE::dataTypes::Vec<T, 2>(::XXX_NAMESPACE::math::internal::Func<T>:: OP (v.x), ::XXX_NAMESPACE::math::internal::Func<T>:: OP (v.y));                                           \
        }                                                                                                                                       \
                                                                                                                                                \
        template <typename T, typename X = typename std::remove_cv<T>::type>                                                                    \
        inline ::XXX_NAMESPACE::dataTypes::Vec<X, 3> OP (IN_T<T, 3>& v)                                                                                                     \
        {                                                                                                                                       \
            return ::XXX_NAMESPACE::dataTypes::Vec<X, 3>(::XXX_NAMESPACE::math::internal::Func<X>:: OP (v.x),                                                                                \
                            ::XXX_NAMESPACE::math::internal::Func<X>:: OP (v.y),                                                                                \
                            ::XXX_NAMESPACE::math::internal::Func<X>:: OP (v.z));                                                                               \
        }                                                                                                                                       \

    #define MACRO_QUALIFIED(OP, IN_T)                                                                                                           \
        MACRO_UNQUALIFIED(OP, IN_T)                                                                                                             \
        MACRO_UNQUALIFIED(OP, const IN_T)                                                                                                       \

    #define MACRO(OP, IN_T)                                                                                                                     \
        MACRO_QUALIFIED(OP, IN_T)                                                                                                               \

        MACRO(sqrt, ::XXX_NAMESPACE::dataTypes::Vec)
        MACRO(log, ::XXX_NAMESPACE::dataTypes::Vec)
        MACRO(exp, ::XXX_NAMESPACE::dataTypes::Vec)

        MACRO(sqrt, ::XXX_NAMESPACE::dataTypes::internal::VecProxy)
        MACRO(log, ::XXX_NAMESPACE::dataTypes::internal::VecProxy)
        MACRO(exp, ::XXX_NAMESPACE::dataTypes::internal::VecProxy)

    #undef MACRO
    #undef MACRO_QUALIFIED
    #undef MACRO_UNQUALIFIED

    #define MACRO_UNQUALIFIED(IN_T_1, IN_T_2)                                                                                                   \
        template <typename T_1, typename T_2,                                                                                                   \
                typename X = typename XXX_NAMESPACE::dataTypes::Compare<T_1, T_2>::UnqualifiedStrongerT>                                  \
        inline X dot_product(IN_T_1<T_1, 1>& x_1, IN_T_2<T_2, 1>& x_2)                                                                          \
        {                                                                                                                                       \
            return (x_1.x * x_2.x);                                                                                                             \
        }                                                                                                                                       \
                                                                                                                                                \
        template <typename T_1, typename T_2,                                                                                                   \
                typename X = typename XXX_NAMESPACE::dataTypes::Compare<T_1, T_2>::UnqualifiedStrongerT>                                  \
        inline X dot_product(IN_T_1<T_1, 2>& x_1, IN_T_2<T_2, 2>& x_2)                                                                          \
        {                                                                                                                                       \
            return (x_1.x * x_2.x + x_1.y * x_2.y);                                                                                             \
        }                                                                                                                                       \
                                                                                                                                                \
        template <typename T_1, typename T_2,                                                                                                   \
                typename X = typename XXX_NAMESPACE::dataTypes::Compare<T_1, T_2>::UnqualifiedStrongerT>                                  \
        inline X dot_product(IN_T_1<T_1, 3>& x_1, IN_T_2<T_2, 3>& x_2)                                                                          \
        {                                                                                                                                       \
            return (x_1.x * x_2.x + x_1.y * x_2.y + x_1.z * x_2.z);                                                                             \
        }                                                                                                                                       \
                                                                                                                                                \
        template <typename T_1, typename T_2, SizeT D,                                                                                    \
                typename X = typename XXX_NAMESPACE::dataTypes::Compare<T_1, T_2>::UnqualifiedStrongerT>                                  \
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

        MACRO(::XXX_NAMESPACE::dataTypes::Vec, ::XXX_NAMESPACE::dataTypes::internal::VecProxy)

    #undef MACRO
    #undef MACRO_QUALIFIED
    #undef MACRO_UNQUALIFIED

    #define MACRO_UNQUALIFIED(IN_T)                                                                                                             \
        template <typename T_1, typename T_2,                                                                                                   \
                typename X = typename XXX_NAMESPACE::dataTypes::Compare<T_1, T_2>::UnqualifiedStrongerT>                                  \
        inline X dot_product(IN_T<T_1, 1>& x_1, const T_2 x_2)                                                                                  \
        {                                                                                                                                       \
            return (x_1.x * x_2);                                                                                                               \
        }                                                                                                                                       \
                                                                                                                                                \
        template <typename T_1, typename T_2,                                                                                                   \
                typename X = typename XXX_NAMESPACE::dataTypes::Compare<T_1, T_2>::UnqualifiedStrongerT>                                  \
        inline X dot_product(IN_T<T_1, 2>& x_1, const T_2 x_2)                                                                                  \
        {                                                                                                                                       \
            return (x_1.x + x_1.y) * x_2;                                                                                                       \
        }                                                                                                                                       \
                                                                                                                                                \
        template <typename T_1, typename T_2,                                                                                                   \
                typename X = typename XXX_NAMESPACE::dataTypes::Compare<T_1, T_2>::UnqualifiedStrongerT>                                  \
        inline X dot_product(IN_T<T_1, 3>& x_1, const T_2 x_2)                                                                                  \
        {                                                                                                                                       \
            return (x_1.x + x_1.y + x_1.z) * x_2;                                                                                               \
        }                                                                                                                                       \
                                                                                                                                                \
        template <typename T_1, typename T_2,                                                                                                   \
                typename X = typename XXX_NAMESPACE::dataTypes::Compare<T_1, T_2>::UnqualifiedStrongerT>                                  \
        inline X dot_product(const T_1 x_1, IN_T<T_2, 1>& x_2)                                                                                  \
        {                                                                                                                                       \
            return (x_1 * x_2.x);                                                                                                               \
        }                                                                                                                                       \
                                                                                                                                                \
        template <typename T_1, typename T_2,                                                                                                   \
                typename X = typename XXX_NAMESPACE::dataTypes::Compare<T_1, T_2>::UnqualifiedStrongerT>                                  \
        inline X dot_product(const T_1 x_1, IN_T<T_2, 2>& x_2)                                                                                  \
        {                                                                                                                                       \
            return x_1 * (x_2.x + x_2.y);                                                                                                       \
        }                                                                                                                                       \
                                                                                                                                                \
        template <typename T_1, typename T_2,                                                                                                   \
                typename X = typename XXX_NAMESPACE::dataTypes::Compare<T_1, T_2>::UnqualifiedStrongerT>                                  \
        inline X dot_product(const T_1 x_1, IN_T<T_2, 3>& x_2)                                                                                  \
        {                                                                                                                                       \
            return x_1 * (x_2.x + x_2.y + x_2.z);                                                                                               \
        }                                                                                                                                       \
                                                                                                                                                \
        template <typename T_1, typename T_2, SizeT D,                                                                                    \
                typename X = typename XXX_NAMESPACE::dataTypes::Compare<T_1, T_2>::UnqualifiedStrongerT>                                  \
        inline X dot(IN_T<T_1, D>& x_1, const T_2 x_2)                                                                                          \
        {                                                                                                                                       \
            return dot_product(x_1, x_2);                                                                                                       \
        }                                                                                                                                       \
                                                                                                                                                \
        template <typename T_1, typename T_2, SizeT D,                                                                                    \
                typename X = typename XXX_NAMESPACE::dataTypes::Compare<T_1, T_2>::UnqualifiedStrongerT>                                  \
        inline X dot(const T_1 x_1, IN_T<T_2, D>& x_2)                                                                                          \
        {                                                                                                                                       \
            return dot_product(x_1, x_2);                                                                                                       \
        }                                                                                                                                       \

    #define MACRO_QUALIFIED(IN_T)                                                                                                               \
        MACRO_UNQUALIFIED(IN_T)                                                                                                                 \
        MACRO_UNQUALIFIED(const IN_T)                                                                                                           \
        
    #define MACRO(IN_T)                                                                                                                         \
        MACRO_QUALIFIED(IN_T)                                                                                                                   \
        
        MACRO(::XXX_NAMESPACE::dataTypes::Vec)
        MACRO(::XXX_NAMESPACE::dataTypes::internal::VecProxy)

    #undef MACRO
    #undef MACRO_QUALIFIED
    #undef MACRO_UNQUALIFIED

    #define MACRO_UNQUALIFIED(IN_T_1, IN_T_2)                                                                                                   \
        template <typename T_1, typename T_2,                                                                                                   \
                typename X = typename XXX_NAMESPACE::dataTypes::Compare<T_1, T_2>::UnqualifiedStrongerT>                                  \
        inline ::XXX_NAMESPACE::dataTypes::Vec<X, 3> cross_product(IN_T_1<T_1, 3>& x_1, IN_T_2<T_2, 3>& x_2)                                                                \
        {                                                                                                                                       \
            return ::XXX_NAMESPACE::dataTypes::Vec<X, 3>(x_1.y * x_2.z - x_1.z * x_2.y, x_1.z * x_2.x - x_1.x * x_2.z, x_1.x * x_2.y - x_1.y * x_2.x);                      \
        }                                                                                                                                       \
                                                                                                                                                \
        template <typename T_1, typename T_2,                                                                                                   \
                typename X = typename XXX_NAMESPACE::dataTypes::Compare<T_1, T_2>::UnqualifiedStrongerT>                                  \
        inline ::XXX_NAMESPACE::dataTypes::Vec<X, 3> cross(IN_T_1<T_1, 3>& x_1, IN_T_2<T_2, 3>& x_2)                                                                        \
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

        MACRO(::XXX_NAMESPACE::dataTypes::Vec, ::XXX_NAMESPACE::dataTypes::internal::VecProxy)

    #undef MACRO
    #undef MACRO_QUALIFIED
    #undef MACRO_UNQUALIFIED

    #define MACRO_UNQUALIFIED(IN_T)                                                                                                             \
        template <typename T_1, typename T_2,                                                                                                   \
                typename X = typename XXX_NAMESPACE::dataTypes::Compare<T_1, T_2>::UnqualifiedStrongerT>                                  \
        inline ::XXX_NAMESPACE::dataTypes::Vec<X, 3> cross_product(IN_T<T_1, 3>& x_1, const T_2 x_2)                                                                        \
        {                                                                                                                                       \
            return ::XXX_NAMESPACE::dataTypes::Vec<X, 3>(x_1.y - x_1.z, x_1.z - x_1.x, x_1.x - x_1.y) * x_2;                                                                \
        }                                                                                                                                       \
                                                                                                                                                \
        template <typename T_1, typename T_2,                                                                                                   \
                typename X = typename XXX_NAMESPACE::dataTypes::Compare<T_1, T_2>::UnqualifiedStrongerT>                                  \
        inline ::XXX_NAMESPACE::dataTypes::Vec<X, 3> cross_product(const T_1 x_1, IN_T<T_2, 3>& x_2)                                                                        \
        {                                                                                                                                       \
            return x_1 * ::XXX_NAMESPACE::dataTypes::Vec<X, 3>(x_2.z - x_2.y, x_2.x - x_2.z, x_2.y - x_2.x);                                                                \
        }                                                                                                                                       \
                                                                                                                                                \
        template <typename T_1, typename T_2,                                                                                                   \
                typename X = typename XXX_NAMESPACE::dataTypes::Compare<T_1, T_2>::UnqualifiedStrongerT>                                  \
        inline ::XXX_NAMESPACE::dataTypes::Vec<X, 3> cross(IN_T<T_1, 3>& x_1, const T_2 x_2)                                                                                \
        {                                                                                                                                       \
            return cross_product(x_1, x_2);                                                                                                     \
        }                                                                                                                                       \
                                                                                                                                                \
        template <typename T_1, typename T_2,                                                                                                   \
                typename X = typename XXX_NAMESPACE::dataTypes::Compare<T_1, T_2>::UnqualifiedStrongerT>                                  \
        inline ::XXX_NAMESPACE::dataTypes::Vec<X, 3> cross(const T_1 x_1, IN_T<T_2, 3>& x_2)                                                                                \
        {                                                                                                                                       \
            return cross_product(x_1, x_2);                                                                                                     \
        }                                                                                                                                       \

    #define MACRO_QUALIFIED(IN_T)                                                                                                               \
        MACRO_UNQUALIFIED(IN_T)                                                                                                                 \
        MACRO_UNQUALIFIED(const IN_T)                                                                                                           \
        
    #define MACRO(IN_T)                                                                                                                         \
        MACRO_QUALIFIED(IN_T)                                                                                                                   \
        
        MACRO(::XXX_NAMESPACE::dataTypes::Vec)
        MACRO(::XXX_NAMESPACE::dataTypes::internal::VecProxy)

    #undef MACRO
    #undef MACRO_QUALIFIED
    #undef MACRO_UNQUALIFIED

        namespace internal
        {
            template <typename T, SizeT D>
            struct Func<::XXX_NAMESPACE::dataTypes::Vec<T, D>>
            {
                using type = ::XXX_NAMESPACE::dataTypes::Vec<T, D>;
                using value_type = typename std::remove_cv<typename type::Type>::type;

                static constexpr value_type One = Func<value_type>::One;
                static constexpr value_type MinusOne = Func<value_type>::MinusOne;

                template <typename X>
                static type sqrt(X x)
                {
                    return ::XXX_NAMESPACE::math::sqrt(x);
                }

                template <typename X>
                static type log(X x)
                {
                    return ::XXX_NAMESPACE::math::log(x);
                }

                template <typename X>
                static type exp(X x)
                {
                    return ::XXX_NAMESPACE::math::exp(x);
                }
            };
        }
    }
}

#endif