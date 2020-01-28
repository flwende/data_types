// Copyright (c) 2020 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(DATA_TYPES_TUPLE_TUPLE_MATH_HPP)
#define DATA_TYPES_TUPLE_TUPLE_MATH_HPP

#include <type_traits>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#include <common/Math.hpp>
#include <tuple/Tuple.hpp>
#include <DataTypes.hpp>

namespace XXX_NAMESPACE
{
    namespace math
    {
        using ::XXX_NAMESPACE::compileTime::Loop;
        using ::XXX_NAMESPACE::dataTypes::Compare;
        using ::XXX_NAMESPACE::dataTypes::SizeT;
        using ::XXX_NAMESPACE::dataTypes::Tuple;
        using ::XXX_NAMESPACE::dataTypes::Get;

        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        //
        // Arithmetic on `Tuple` types.
        //
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
#define MACRO(OP)                                                                                                                                                                                                          \
    template <typename... T_1, typename... T_2>                                                                                                                                                                            \
HOST_VERSION CUDA_DEVICE_VERSION inline constexpr auto operator OP(const Tuple<T_1...>& x_1, const Tuple<T_2...>& x_2)                                                                                                     \
    {                                                                                                                                                                                                                      \
        constexpr SizeT N_1 = sizeof...(T_1);                                                                                                                                                                              \
        constexpr SizeT N_2 = sizeof...(T_2);                                                                                                                                                                              \
                                                                                                                                                                                                                           \
        static_assert(N_1 == N_2, "error: parameter lists have different size.");                                                                                                                                          \
                                                                                                                                                                                                                           \
        Tuple<typename Compare<std::decay_t<T_1>, std::decay_t<T_2>>::UnqualifiedStrongerT...> y{x_1};                                                                                                                     \
                                                                                                                                                                                                                           \
        y OP## = x_2;                                                                                                                                                                                                      \
                                                                                                                                                                                                                           \
        return y;                                                                                                                                                                                                          \
    }

        MACRO(+)
        MACRO(-)
        MACRO(*)
        MACRO(/)

#undef MACRO

        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        //
        // Arithmetic on `Tuple` types and scalars.
        //
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
#define MACRO(OP)                                                                                                                                                                                                          \
    template <typename T_2, typename... T_1, typename EnableType = std::enable_if_t<std::is_fundamental<T_2>::value>>                                                                                                      \
    HOST_VERSION CUDA_DEVICE_VERSION inline constexpr auto operator OP(const Tuple<T_1...>& x_1, const T_2 x_2)                                                                                                            \
    {                                                                                                                                                                                                                      \
        Tuple<typename Compare<std::decay_t<T_1>, std::decay_t<T_2>>::UnqualifiedStrongerT...> y{x_1};                                                                                                                     \
                                                                                                                                                                                                                           \
        y OP## = x_2;                                                                                                                                                                                                      \
                                                                                                                                                                                                                           \
        return y;                                                                                                                                                                                                          \
    }                                                                                                                                                                                                                      \
                                                                                                                                                                                                                           \
    template <typename T_1, typename... T_2, typename EnableType = std::enable_if_t<std::is_fundamental<T_1>::value>>                                                                                                      \
    HOST_VERSION CUDA_DEVICE_VERSION inline constexpr auto operator OP(const T_1 x_1, const Tuple<T_2...>& x_2)                                                                                                            \
    {                                                                                                                                                                                                                      \
        Tuple<typename Compare<std::decay_t<T_2>, std::decay_t<T_1>>::UnqualifiedStrongerT...> y(x_1);                                                                                                                     \
                                                                                                                                                                                                                           \
        y OP## = x_2;                                                                                                                                                                                                      \
                                                                                                                                                                                                                           \
        return y;                                                                                                                                                                                                          \
    }                                                                                                                                                                                                                      \

        MACRO(+)
        MACRO(-)
        MACRO(*)
        MACRO(/)

#undef MACRO

        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        //
        // Cross product on `Tuple` types.
        //
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        template <typename... T_1, typename... T_2>
        HOST_VERSION CUDA_DEVICE_VERSION inline constexpr auto CrossProduct(const Tuple<T_1...>& x_1, const Tuple<T_2...>& x_2)
        {
            constexpr SizeT N_1 = sizeof...(T_1);
            constexpr SizeT N_2 = sizeof...(T_2);

            static_assert(N_1 == 3, "error: implementation for 3 components only.");
            static_assert(N_1 == N_2, "error: parameter lists have different size.");
                                                                                                                                                                                                                            
            Tuple<typename Compare<T_1, T_2>::UnqualifiedStrongerT...> y;

            Get<0>(y) = Get<1>(x_1) * Get<2>(x_2) - Get<2>(x_1) * Get<1>(x_2);
            Get<1>(y) = Get<2>(x_1) * Get<0>(x_2) - Get<0>(x_1) * Get<2>(x_2);
            Get<2>(y) = Get<0>(x_1) * Get<1>(x_2) - Get<1>(x_1) * Get<0>(x_2);

            return y;
        }

        template <typename TupleT_1, typename TupleT_2>
        HOST_VERSION CUDA_DEVICE_VERSION inline constexpr auto Cross(const TupleT_1& x_1, const TupleT_2& x_2)
        {
            return CrossProduct(x_1, x_2);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        //
        // Dot produc on `Tuple` types.
        //
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        template <typename... T_1, typename... T_2>
        HOST_VERSION CUDA_DEVICE_VERSION inline constexpr auto DotProduct(const Tuple<T_1...>& x_1, const Tuple<T_2...>& x_2)
        {
            constexpr SizeT N_1 = sizeof...(T_1);
            constexpr SizeT N_2 = sizeof...(T_2);

            static_assert(N_1 == N_2, "error: parameter lists have different size.");
                                                                                                                                                                                                                            
            typename Compare<T_1..., T_2...>::UnqualifiedStrongerT y{};

            Loop<N_1>::Execute([&y, &x_1, &x_2] (const auto I) {
                y += Get<I>(x_1) * Get<I>(x_2);
            });
            
            return y;
        }

        template <typename TupleT_1, typename TupleT_2>
        HOST_VERSION CUDA_DEVICE_VERSION inline constexpr auto Dot(const TupleT_1& x_1, const TupleT_2& x_2)
        {
            return DotProduct(x_1, x_2);
        }

        namespace internal
        {
            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            //
            // Sqrt, Log, Exp, Abs functions for `Tuple` types.
            //
            ////////////////////////////////////////////////////////////////////////////////////////////////////////
#define MACRO(OP)                                                                                                                                                                                                          \
    template <typename... T>                                                                                                                                                                                               \
    HOST_VERSION CUDA_DEVICE_VERSION inline constexpr auto OP(const Tuple<T...>& x)                                                                                                                                        \
    {                                                                                                                                                                                                                      \
        constexpr SizeT N = sizeof...(T);                                                                                                                                                                                  \
                                                                                                                                                                                                                           \
        Tuple<std::decay_t<T>...> y;                                                                                                                                                                                       \
                                                                                                                                                                                                                           \
        Loop<N>::Execute([&y, &x](const auto I) {                                                                                                                                                                          \
            using ElementT = std::decay_t<decltype(Get<I>(y))>;                                                                                                                                                            \
                                                                                                                                                                                                                           \
            Get<I>(y) = ::XXX_NAMESPACE::math::internal::Func<ElementT>::OP(Get<I>(x));                                                                                                                                    \
        });                                                                                                                                                                                                                \
                                                                                                                                                                                                                           \
        return y;                                                                                                                                                                                                          \
    }

            MACRO(Sqrt)
            MACRO(Log)
            MACRO(Exp)
            MACRO(Abs)

#undef MACRO

            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            //
            // Max, Min functions for `Tuple` types.
            //
            ////////////////////////////////////////////////////////////////////////////////////////////////////////
#define MACRO(OP)                                                                                                                                                                                                          \
    template <typename... T>                                                                                                                                                                                               \
    HOST_VERSION CUDA_DEVICE_VERSION inline constexpr auto OP(const Tuple<T...>& x_1, const Tuple<T...>& x_2)                                                                                                              \
    {                                                                                                                                                                                                                      \
        constexpr SizeT N = sizeof...(T);                                                                                                                                                                                  \
                                                                                                                                                                                                                           \
        Tuple<std::decay_t<T>...> y;                                                                                                                                                                                       \
                                                                                                                                                                                                                           \
        Loop<N>::Execute([&y, &x_1, &x_2](const auto I) {                                                                                                                                                                  \
            using ElementT = std::decay_t<decltype(Get<I>(y))>;                                                                                                                                                            \
                                                                                                                                                                                                                           \
            Get<I>(y) = ::XXX_NAMESPACE::math::internal::Func<ElementT>::OP(Get<I>(x_1), Get<I>(x_2));                                                                                                                     \
        });                                                                                                                                                                                                                \
                                                                                                                                                                                                                           \
        return y;                                                                                                                                                                                                          \
    }

            MACRO(Max)
            MACRO(Min)

#undef MACRO

            //!
            //! \brief Specialization of the `Func` data structure for the `Tuple` type.
            //!
            template <typename... ValueT>
            struct Func<Tuple<ValueT...>>
            {
                using ElementT = typename ::XXX_NAMESPACE::dataTypes::Compare<ValueT...>::UnqualifiedStrongerT;

                static constexpr ElementT One = static_cast<ElementT>(1);
                static constexpr ElementT MinusOne = static_cast<ElementT>(-1);

                HOST_VERSION
                CUDA_DEVICE_VERSION
                static inline auto Sqrt(const Tuple<ValueT...>& x_1) { return ::XXX_NAMESPACE::math::internal::Sqrt(x_1); }

                HOST_VERSION
                CUDA_DEVICE_VERSION
                static inline auto Log(const Tuple<ValueT...>& x_1) { return ::XXX_NAMESPACE::math::internal::Log(x_1); }

                HOST_VERSION
                CUDA_DEVICE_VERSION
                static inline auto Exp(const Tuple<ValueT...>& x_1) { return ::XXX_NAMESPACE::math::internal::Exp(x_1); }

                HOST_VERSION
                CUDA_DEVICE_VERSION
                static inline auto Max(const Tuple<ValueT...>& x_1, const Tuple<ValueT...>& x_2) { return ::XXX_NAMESPACE::math::internal::Max(x_1, x_2); }

                HOST_VERSION
                CUDA_DEVICE_VERSION
                static inline auto Min(const Tuple<ValueT...>& x_1, const Tuple<ValueT...>& x_2) { return ::XXX_NAMESPACE::math::internal::Min(x_1, x_2); }

                HOST_VERSION
                CUDA_DEVICE_VERSION
                static inline auto Abs(const Tuple<ValueT...>& x_1) { return ::XXX_NAMESPACE::math::internal::Abs(x_1); }

                template <typename T, typename EnableType = std::enable_if_t<std::is_fundamental<T>::value>>
                HOST_VERSION CUDA_DEVICE_VERSION static inline auto Sqrt(T x_1) -> Tuple<ValueT...>
                {
                    return {::XXX_NAMESPACE::math::internal::Func<ElementT>::Sqrt(x_1)};
                }

                template <typename T, typename EnableType = std::enable_if_t<std::is_fundamental<T>::value>>
                HOST_VERSION CUDA_DEVICE_VERSION static inline auto Log(T x_1) -> Tuple<ValueT...>
                {
                    return {::XXX_NAMESPACE::math::internal::Func<ElementT>::Log(x_1)};
                }

                template <typename T, typename EnableType = std::enable_if_t<std::is_fundamental<T>::value>>
                HOST_VERSION CUDA_DEVICE_VERSION static inline auto Exp(T x_1) -> Tuple<ValueT...>
                {
                    return {::XXX_NAMESPACE::math::internal::Func<ElementT>::Exp(x_1)};
                }

                template <typename T, typename EnableType = std::enable_if_t<std::is_fundamental<T>::value>>
                HOST_VERSION CUDA_DEVICE_VERSION static inline auto Max(T x_1, T x_2) -> Tuple<ValueT...>
                {
                    return {::XXX_NAMESPACE::math::internal::Func<ElementT>::Max(x_1, x_2)};
                }

                template <typename T, typename EnableType = std::enable_if_t<std::is_fundamental<T>::value>>
                HOST_VERSION CUDA_DEVICE_VERSION static inline auto Min(T x_1, T x_2) -> Tuple<ValueT...>
                {
                    return {::XXX_NAMESPACE::math::internal::Func<ElementT>::Min(x_1, x_2)};
                }

                template <typename T, typename EnableType = std::enable_if_t<std::is_fundamental<T>::value>>
                HOST_VERSION CUDA_DEVICE_VERSION static inline auto Abs(T x_1) -> Tuple<ValueT...>
                {
                    return {::XXX_NAMESPACE::math::internal::Func<ElementT>::Abs(x_1)};
                }
            };

            template <>
            struct Func<Tuple<>>
            {
            };
        } // namespace internal
    } // namespace math
} // namespace XXX_NAMESPACE

#endif