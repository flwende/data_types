// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
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
#include <data_types/DataTypes.hpp>
#include <data_types/tuple/Tuple.hpp>

namespace XXX_NAMESPACE
{
    namespace math
    {
        using namespace ::XXX_NAMESPACE::dataTypes;

        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        //
        // Arithmetic on `Tuple` types.
        //
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
#define MACRO(OP)                                                                                                                                                                                                          \
    template <typename... T_1, typename... T_2>                                                                                                                                                                            \
    HOST_VERSION CUDA_DEVICE_VERSION inline auto operator OP(const ::XXX_NAMESPACE::dataTypes::Tuple<T_1...>& x_1, const ::XXX_NAMESPACE::dataTypes::Tuple<T_2...>& x_2)                                                   \
    {                                                                                                                                                                                                                      \
        using namespace ::XXX_NAMESPACE::dataTypes;                                                                                                                                                                        \
                                                                                                                                                                                                                           \
        constexpr SizeT N_1 = sizeof...(T_1);                                                                                                                                                                              \
        constexpr SizeT N_2 = sizeof...(T_2);                                                                                                                                                                              \
                                                                                                                                                                                                                           \
        static_assert(N_1 == N_2, "error: parameter lists have different size.");                                                                                                                                          \
                                                                                                                                                                                                                           \
        Tuple<typename Compare<T_1, T_2>::UnqualifiedStrongerT...> y(x_1);                                                                                                                                                 \
                                                                                                                                                                                                                           \
        ::XXX_NAMESPACE::compileTime::Loop<N_1>::Execute([&y, &x_2](const auto I) { Get<I>(y) OP## = Get<I>(x_2); });                                                                                                      \
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
    HOST_VERSION CUDA_DEVICE_VERSION inline auto operator OP(const ::XXX_NAMESPACE::dataTypes::Tuple<T_1...>& x_1, const T_2 x_2)                                                                                          \
    {                                                                                                                                                                                                                      \
        using namespace ::XXX_NAMESPACE::dataTypes;                                                                                                                                                                        \
                                                                                                                                                                                                                           \
        constexpr SizeT N = sizeof...(T_1);                                                                                                                                                                                \
                                                                                                                                                                                                                           \
        Tuple<typename Compare<T_1, T_2>::UnqualifiedStrongerT...> y(x_1);                                                                                                                                                 \
                                                                                                                                                                                                                           \
        ::XXX_NAMESPACE::compileTime::Loop<N>::Execute([&y, x_2](const auto I) { Get<I>(y) OP## = x_2; });                                                                                                                 \
                                                                                                                                                                                                                           \
        return y;                                                                                                                                                                                                          \
    }                                                                                                                                                                                                                      \
                                                                                                                                                                                                                           \
    template <typename T_1, typename... T_2, typename EnableType = std::enable_if_t<std::is_fundamental<T_1>::value>>                                                                                                      \
    HOST_VERSION CUDA_DEVICE_VERSION inline auto operator OP(const T_1 x_1, const ::XXX_NAMESPACE::dataTypes::Tuple<T_2...>& x_2)                                                                                          \
    {                                                                                                                                                                                                                      \
        using namespace ::XXX_NAMESPACE::dataTypes;                                                                                                                                                                        \
                                                                                                                                                                                                                           \
        constexpr SizeT N = sizeof...(T_2);                                                                                                                                                                                \
                                                                                                                                                                                                                           \
        Tuple<typename Compare<T_2, T_1>::UnqualifiedStrongerT...> y(x_1);                                                                                                                                                 \
                                                                                                                                                                                                                           \
        ::XXX_NAMESPACE::compileTime::Loop<N>::Execute([&y, &x_2](const auto I) { Get<I>(y) OP## = Get<I>(x_2); });                                                                                                        \
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
        // Cross product on `Tuple` types.
        //
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        template <typename... T_1, typename... T_2>
        HOST_VERSION CUDA_DEVICE_VERSION inline auto CrossProduct(const ::XXX_NAMESPACE::dataTypes::Tuple<T_1...>& x_1, const ::XXX_NAMESPACE::dataTypes::Tuple<T_2...>& x_2)
        {
            using namespace ::XXX_NAMESPACE::dataTypes;

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
        HOST_VERSION CUDA_DEVICE_VERSION inline auto Cross(const TupleT_1& x_1, const TupleT_2& x_2)
        {
            return CrossProduct(x_1, x_2);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        //
        // Dot produc on `Tuple` types.
        //
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        template <typename... T_1, typename... T_2>
        HOST_VERSION CUDA_DEVICE_VERSION inline auto DotProduct(const ::XXX_NAMESPACE::dataTypes::Tuple<T_1...>& x_1, const ::XXX_NAMESPACE::dataTypes::Tuple<T_2...>& x_2)
        {
            using namespace ::XXX_NAMESPACE::dataTypes;

            constexpr SizeT N_1 = sizeof...(T_1);
            constexpr SizeT N_2 = sizeof...(T_2);

            static_assert(N_1 == N_2, "error: parameter lists have different size.");
                                                                                                                                                                                                                            
            typename Compare<T_1..., T_2...>::UnqualifiedStrongerT y{};

            ::XXX_NAMESPACE::compileTime::Loop<N_1>::Execute([&y, &x_1, &x_2] (const auto I) { y += Get<I>(x_1) * Get<I>(x_2); });
            
            return y;
        }

        template <typename TupleT_1, typename TupleT_2>
        HOST_VERSION CUDA_DEVICE_VERSION inline auto Dot(const TupleT_1& x_1, const TupleT_2& x_2)
        {
            return DotProduct(x_1, x_2);
        }

        namespace internal
        {
            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            //
            // Sqrt, Log, Exp functions `Tuple` types.
            //
            ////////////////////////////////////////////////////////////////////////////////////////////////////////
#define MACRO(OP)                                                                                                                                                                                                          \
    template <typename... T>                                                                                                                                                                                               \
    HOST_VERSION CUDA_DEVICE_VERSION inline auto OP(const ::XXX_NAMESPACE::dataTypes::Tuple<T...>& x)                                                                                                                      \
    {                                                                                                                                                                                                                      \
        using namespace ::XXX_NAMESPACE::dataTypes;                                                                                                                                                                        \
                                                                                                                                                                                                                           \
        constexpr SizeT N = sizeof...(T);                                                                                                                                                                                  \
                                                                                                                                                                                                                           \
        Tuple<std::decay_t<T>...> y;                                                                                                                                                                                       \
                                                                                                                                                                                                                           \
        ::XXX_NAMESPACE::compileTime::Loop<N>::Execute([&y, &x](const auto I) {                                                                                                                                            \
            using ElementT = std::decay_t<decltype(Get<I>(y))>;                                                                                                                                                            \
                                                                                                                                                                                                                           \
            Get<I>(y) = ::XXX_NAMESPACE::math::internal::Func<ElementT>::OP(Get<I>(x));                                                                                                                                    \
        });                                                                                                                                                                                                                \
                                                                                                                                                                                                                           \
        return y;                                                                                                                                                                                                          \
    }

            MACRO(sqrt)
            MACRO(log)
            MACRO(exp)

#undef MACRO

            //!
            //! \brief Specialization of the `Func` data structure for the `Tuple` type.
            //!
            template <typename... ValueT>
            struct Func<::XXX_NAMESPACE::dataTypes::Tuple<ValueT...>>
            {
                using Tuple = ::XXX_NAMESPACE::dataTypes::Tuple<ValueT...>;
                using ElementT = typename ::XXX_NAMESPACE::dataTypes::Compare<ValueT...>::UnqualifiedStrongerT;

                static constexpr ElementT One = static_cast<ElementT>(1);
                static constexpr ElementT MinusOne = static_cast<ElementT>(-1);

                HOST_VERSION
                CUDA_DEVICE_VERSION
                static inline auto sqrt(const Tuple& tuple) { return ::XXX_NAMESPACE::math::internal::sqrt(tuple); }

                HOST_VERSION
                CUDA_DEVICE_VERSION
                static inline auto log(const Tuple& tuple) { return ::XXX_NAMESPACE::math::internal::log(tuple); }

                HOST_VERSION
                CUDA_DEVICE_VERSION
                static inline auto exp(const Tuple& tuple) { return ::XXX_NAMESPACE::math::internal::exp(tuple); }

                template <typename T, typename EnableType = std::enable_if_t<std::is_fundamental<T>::value>>
                HOST_VERSION CUDA_DEVICE_VERSION static inline auto sqrt(T value) -> Tuple
                {
                    return {::XXX_NAMESPACE::math::internal::Func<ElementT>::sqrt(value)};
                }

                template <typename T, typename EnableType = std::enable_if_t<std::is_fundamental<T>::value>>
                HOST_VERSION CUDA_DEVICE_VERSION static inline auto log(T value) -> Tuple
                {
                    return {::XXX_NAMESPACE::math::internal::Func<ElementT>::log(value)};
                }

                template <typename T, typename EnableType = std::enable_if_t<std::is_fundamental<T>::value>>
                HOST_VERSION CUDA_DEVICE_VERSION static inline auto exp(T value) -> Tuple
                {
                    return {::XXX_NAMESPACE::math::internal::Func<ElementT>::exp(value)};
                }
            };

            template <>
            struct Func<::XXX_NAMESPACE::dataTypes::Tuple<>>
            {
            };
        } // namespace internal
    } // namespace math
} // namespace XXX_NAMESPACE

#endif