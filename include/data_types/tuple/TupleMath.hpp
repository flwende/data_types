// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(DATA_TYPES_TUPLE_TUPLE_MATH_HPP)
#define DATA_TYPES_TUPLE_TUPLE_MATH_HPP

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
        template <typename T_1, typename T_2, typename T_3, typename T_4, typename T_5, typename T_6,                                           \
                typename X_1 = typename ::XXX_NAMESPACE::dataTypes::Compare<T_1, T_4>::UnqualifiedStrongerT,                                \
                typename X_2 = typename ::XXX_NAMESPACE::dataTypes::Compare<T_2, T_5>::UnqualifiedStrongerT,                                \
                typename X_3 = typename ::XXX_NAMESPACE::dataTypes::Compare<T_3, T_6>::UnqualifiedStrongerT>                                \
        inline ::XXX_NAMESPACE::dataTypes::Tuple<X_1, X_2, X_3> operator OP (IN_T_1<T_1, T_2, T_3>& x_1, IN_T_2<T_4, T_5, T_6>& x_2)                                        \
        {                                                                                                                                       \
            ::XXX_NAMESPACE::dataTypes::Tuple<X_1, X_2, X_3> y(x_1);                                                                                                        \
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

        MACRO(+, ::XXX_NAMESPACE::dataTypes::Tuple, ::XXX_NAMESPACE::dataTypes::internal::TupleProxy)
        MACRO(-, ::XXX_NAMESPACE::dataTypes::Tuple, ::XXX_NAMESPACE::dataTypes::internal::TupleProxy)
        MACRO(*, ::XXX_NAMESPACE::dataTypes::Tuple, ::XXX_NAMESPACE::dataTypes::internal::TupleProxy)
        MACRO(/, ::XXX_NAMESPACE::dataTypes::Tuple, ::XXX_NAMESPACE::dataTypes::internal::TupleProxy)

    #undef MACRO
    #undef MACRO_QUALIFIED
    #undef MACRO_UNQUALIFIED

    #define MACRO_UNQUALIFIED(OP, IN_T)                                                                                                         \
        template <typename T_1, typename T_2, typename T_3, typename T_4,                                                                       \
                typename X_1 = typename ::XXX_NAMESPACE::dataTypes::Compare<T_1, T_4>::UnqualifiedStrongerT,                                \
                typename X_2 = typename ::XXX_NAMESPACE::dataTypes::Compare<T_2, T_4>::UnqualifiedStrongerT,                                \
                typename X_3 = typename ::XXX_NAMESPACE::dataTypes::Compare<T_3, T_4>::UnqualifiedStrongerT>                                \
        inline ::XXX_NAMESPACE::dataTypes::Tuple<X_1, X_2, X_3> operator OP (IN_T<T_1, T_2, T_3>& x_1, const T_4 x_2)                                                       \
        {                                                                                                                                       \
            ::XXX_NAMESPACE::dataTypes::Tuple<X_1, X_2, X_3> y(x_1);                                                                                                        \
            y OP ## = x_2;                                                                                                                      \
            return y;                                                                                                                           \
        }                                                                                                                                       \
                                                                                                                                                \
        template <typename T_1, typename T_2, typename T_3, typename T_4,                                                                       \
                typename X_1 = typename ::XXX_NAMESPACE::dataTypes::Compare<T_1, T_2>::UnqualifiedStrongerT,                                \
                typename X_2 = typename ::XXX_NAMESPACE::dataTypes::Compare<T_1, T_3>::UnqualifiedStrongerT,                                \
                typename X_3 = typename ::XXX_NAMESPACE::dataTypes::Compare<T_1, T_4>::UnqualifiedStrongerT>                                \
        inline ::XXX_NAMESPACE::dataTypes::Tuple<X_1, X_2, X_3> operator OP (const T_1 x_1, IN_T<T_2, T_3, T_4>& x_2)                                                       \
        {                                                                                                                                       \
            ::XXX_NAMESPACE::dataTypes::Tuple<X_1, X_2, X_3> y(x_1);                                                                                                        \
            y OP ## = x_2;                                                                                                                      \
            return y;                                                                                                                           \
        }                                                                                                                                       \

    #define MACRO_QUALIFIED(OP, IN_T)                                                                                                           \
        MACRO_UNQUALIFIED(OP, IN_T)                                                                                                             \
        MACRO_UNQUALIFIED(OP, const IN_T)                                                                                                       \

    #define MACRO(OP, IN_T)                                                                                                                     \
        MACRO_QUALIFIED(OP, IN_T)                                                                                                               \

        MACRO(+, ::XXX_NAMESPACE::dataTypes::Tuple)
        MACRO(-, ::XXX_NAMESPACE::dataTypes::Tuple)
        MACRO(*, ::XXX_NAMESPACE::dataTypes::Tuple)
        MACRO(/, ::XXX_NAMESPACE::dataTypes::Tuple)

        MACRO(+, ::XXX_NAMESPACE::dataTypes::internal::TupleProxy)
        MACRO(-, ::XXX_NAMESPACE::dataTypes::internal::TupleProxy)
        MACRO(*, ::XXX_NAMESPACE::dataTypes::internal::TupleProxy)
        MACRO(/, ::XXX_NAMESPACE::dataTypes::internal::TupleProxy)

    #undef MACRO
    #undef MACRO_QUALIFIED
    #undef MACRO_UNQUALIFIED

    #define MACRO_UNQUALIFIED(OP, IN_T)                                                                                                         \
        template <typename T_1, typename T_2, typename T_3,                                                                                     \
                typename X_1 = typename std::remove_cv<T_1>::type,                                                                            \
                typename X_2 = typename std::remove_cv<T_2>::type,                                                                            \
                typename X_3 = typename std::remove_cv<T_3>::type>                                                                            \
        inline ::XXX_NAMESPACE::dataTypes::Tuple<X_1, X_2, X_3> OP (IN_T<T_1, T_2, T_3>& t)                                                                                 \
        {                                                                                                                                       \
            return ::XXX_NAMESPACE::dataTypes::Tuple<X_1, X_2, X_3>(::XXX_NAMESPACE::math::Func<T_1>:: OP (t.x),                                                                   \
                                        ::XXX_NAMESPACE::math::Func<T_2>:: OP (t.y),                                                                   \
                                        ::XXX_NAMESPACE::math::Func<T_3>:: OP (t.z));                                                                  \
        }                                                                                                                                       \

    #define MACRO_QUALIFIED(OP, IN_T)                                                                                                           \
        MACRO_UNQUALIFIED(OP, IN_T)                                                                                                             \
        MACRO_UNQUALIFIED(OP, const IN_T)                                                                                                       \

    #define MACRO(OP, IN_T)                                                                                                                     \
        MACRO_QUALIFIED(OP, IN_T)                                                                                                               \

        MACRO(sqrt, ::XXX_NAMESPACE::dataTypes::Tuple)
        MACRO(log, ::XXX_NAMESPACE::dataTypes::Tuple)
        MACRO(exp, ::XXX_NAMESPACE::dataTypes::Tuple)

        MACRO(sqrt, ::XXX_NAMESPACE::dataTypes::internal::TupleProxy)
        MACRO(log, ::XXX_NAMESPACE::dataTypes::internal::TupleProxy)
        MACRO(exp, ::XXX_NAMESPACE::dataTypes::internal::TupleProxy)

    #undef MACRO
    #undef MACRO_QUALIFIED
    #undef MACRO_UNQUALIFIED

    #define MACRO_UNQUALIFIED(IN_T_1, IN_T_2)                                                                                                   \
        template <typename T_1, typename T_2, typename T_3, typename T_4, typename T_5, typename T_6,                                           \
                typename X_1 = typename ::XXX_NAMESPACE::dataTypes::Compare<T_1, T_4>::UnqualifiedStrongerT,                                \
                typename X_2 = typename ::XXX_NAMESPACE::dataTypes::Compare<T_2, T_5>::UnqualifiedStrongerT,                                \
                typename X_3 = typename ::XXX_NAMESPACE::dataTypes::Compare<T_3, T_6>::UnqualifiedStrongerT>                                \
        inline ::XXX_NAMESPACE::dataTypes::Tuple<X_1, X_2, X_3> cross_product(IN_T_1<T_1, T_2, T_3>& x_1, IN_T_2<T_4, T_5, T_6>& x_2)                                       \
        {                                                                                                                                       \
            return ::XXX_NAMESPACE::dataTypes::Tuple<X_1, X_2, X_3>(x_1.y * x_2.z - x_1.z * x_2.y, x_1.z * x_2.x - x_1.x * x_2.z, x_1.x * x_2.y - x_1.y * x_2.x);           \
        }                                                                                                                                       \
                                                                                                                                                \
        template <typename T_1, typename T_2, typename T_3, typename T_4, typename T_5, typename T_6,                                           \
                typename X_1 = typename ::XXX_NAMESPACE::dataTypes::Compare<T_1, T_4>::UnqualifiedStrongerT,                                \
                typename X_2 = typename ::XXX_NAMESPACE::dataTypes::Compare<T_2, T_5>::UnqualifiedStrongerT,                                \
                typename X_3 = typename ::XXX_NAMESPACE::dataTypes::Compare<T_3, T_6>::UnqualifiedStrongerT>                                \
        inline ::XXX_NAMESPACE::dataTypes::Tuple<X_1, X_2, X_3> cross(IN_T_1<T_1, T_2, T_3>& x_1, IN_T_2<T_4, T_5, T_6>& x_2)                                               \
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

        MACRO(::XXX_NAMESPACE::dataTypes::Tuple, ::XXX_NAMESPACE::dataTypes::internal::TupleProxy)

    #undef MACRO
    #undef MACRO_QUALIFIED
    #undef MACRO_UNQUALIFIED

    #define MACRO_UNQUALIFIED(IN_T)                                                                                                             \
        template <typename T_1, typename T_2, typename T_3, typename T_4,                                                                       \
                typename X_1 = typename ::XXX_NAMESPACE::dataTypes::Compare<T_1, T_4>::UnqualifiedStrongerT,                                \
                typename X_2 = typename ::XXX_NAMESPACE::dataTypes::Compare<T_2, T_4>::UnqualifiedStrongerT,                                \
                typename X_3 = typename ::XXX_NAMESPACE::dataTypes::Compare<T_3, T_4>::UnqualifiedStrongerT>                                \
        inline ::XXX_NAMESPACE::dataTypes::Tuple<X_1, X_2, X_3> cross_product(IN_T<T_1, T_2, T_3>& x_1, const T_4 x_2)                                                      \
        {                                                                                                                                       \
            return ::XXX_NAMESPACE::dataTypes::Tuple<X_1, X_2, X_3>(x_1.y - x_1.z, x_1.z - x_1.x, x_1.x - x_1.y) * x_2;                                                     \
        }                                                                                                                                       \
                                                                                                                                                \
        template <typename T_1, typename T_2, typename T_3, typename T_4,                                                                       \
                typename X_1 = typename ::XXX_NAMESPACE::dataTypes::Compare<T_1, T_4>::UnqualifiedStrongerT,                                \
                typename X_2 = typename ::XXX_NAMESPACE::dataTypes::Compare<T_2, T_4>::UnqualifiedStrongerT,                                \
                typename X_3 = typename ::XXX_NAMESPACE::dataTypes::Compare<T_3, T_4>::UnqualifiedStrongerT>                                \
        inline ::XXX_NAMESPACE::dataTypes::Tuple<X_1, X_2, X_3> cross_product(const T_1 x_1, IN_T<T_2, T_2, T_3>& x_2)                                                      \
        {                                                                                                                                       \
            return x_1 * ::XXX_NAMESPACE::dataTypes::Tuple<X_1, X_2, X_3>(x_2.z - x_2.y, x_2.x - x_2.z, x_2.y - x_2.x);                                                     \
        }                                                                                                                                       \
                                                                                                                                                \
        template <typename T_1, typename T_2, typename T_3, typename T_4,                                                                       \
                typename X_1 = typename ::XXX_NAMESPACE::dataTypes::Compare<T_1, T_4>::UnqualifiedStrongerT,                                \
                typename X_2 = typename ::XXX_NAMESPACE::dataTypes::Compare<T_2, T_4>::UnqualifiedStrongerT,                                \
                typename X_3 = typename ::XXX_NAMESPACE::dataTypes::Compare<T_3, T_4>::UnqualifiedStrongerT>                                \
        inline ::XXX_NAMESPACE::dataTypes::Tuple<X_1, X_2, X_3> cross(IN_T<T_1, T_2, T_3>& x_1, const T_4 x_2)                                                              \
        {                                                                                                                                       \
            return cross_product(x_1, x_2);                                                                                                     \
        }                                                                                                                                       \
                                                                                                                                                \
        template <typename T_1, typename T_2, typename T_3, typename T_4,                                                                       \
                typename X_1 = typename ::XXX_NAMESPACE::dataTypes::Compare<T_1, T_4>::UnqualifiedStrongerT,                                \
                typename X_2 = typename ::XXX_NAMESPACE::dataTypes::Compare<T_2, T_4>::UnqualifiedStrongerT,                                \
                typename X_3 = typename ::XXX_NAMESPACE::dataTypes::Compare<T_3, T_4>::UnqualifiedStrongerT>                                \
        inline ::XXX_NAMESPACE::dataTypes::Tuple<X_1, X_2, X_3> cross(const T_1 x_1, IN_T<T_2, T_2, T_3>& x_2)                                                              \
        {                                                                                                                                       \
            return cross_product(x_1, x_2);                                                                                                     \
        }                                                                                                                                       \

    #define MACRO_QUALIFIED(IN_T)                                                                                                               \
        MACRO_UNQUALIFIED(IN_T)                                                                                                                 \
        MACRO_UNQUALIFIED(const IN_T)                                                                                                           \
        
    #define MACRO(IN_T)                                                                                                                         \
        MACRO_QUALIFIED(IN_T)                                                                                                                   \
        
        MACRO(::XXX_NAMESPACE::dataTypes::Tuple)
        MACRO(::XXX_NAMESPACE::dataTypes::internal::TupleProxy)

    #undef MACRO
    #undef MACRO_QUALIFIED
    #undef MACRO_UNQUALIFIED

        template <typename TupleT>
        struct Func;

        template <>
        struct Func<::XXX_NAMESPACE::dataTypes::Tuple<>> {};

        template <typename ...ValueT>
        struct Func<::XXX_NAMESPACE::dataTypes::Tuple<ValueT...>>
        {
            using Tuple = ::XXX_NAMESPACE::dataTypes::Tuple<ValueT...>;
            
            template <typename X>
            static auto sqrt(X x) -> Tuple
            {
                return ::XXX_NAMESPACE::math::sqrt(x);
            }

            template <typename X>
            static auto log(X x) -> Tuple
            {
                return ::XXX_NAMESPACE::math::log(x);
            }

            template <typename X>
            static auto exp(X x) -> Tuple
            {
                return ::XXX_NAMESPACE::math::exp(x);
            }
        };
    }
}

#endif