// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(COMMON_MATH_HPP)
#define COMMON_MATH_HPP

#include <cassert>
#include <cmath>
#include <type_traits>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#include <auxiliary/CPPStandard.hpp>
#include <data_types/DataTypes.hpp>
#include <platform/Target.hpp>

namespace XXX_NAMESPACE
{
    namespace math
    {
        using SizeT = ::XXX_NAMESPACE::dataTypes::SizeT;

        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        //!
        //! \brief Greatest common divisor (gcd).
        //!
        //! \tparam T the argument type (must be an unsigned integer)
        //! \param x_1 argument
        //! \param x_2 argument
        //! \return gcd(x_1, x_2)
        //!
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        template <typename T>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        constexpr inline auto GreatestCommonDivisor(T x_1, T x_2)
        {
            static_assert(std::is_integral<T>::value && std::is_unsigned<T>::value, "error: only unsigned integers allowed");

            while (true)
            {
                if (x_1 == 0)
                    return x_2;

                x_2 %= x_1;

                if (x_2 == 0)
                    return x_1;

                x_1 %= x_2;
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        //!
        //! \brief Least common multiple (lcm).
        //!
        //! It holds 'gcd(x_1, x_2) * lcm(x_1, x_2) = |x_1 * x_2|'
        //!
        //! \tparam T the argument type (must be an unsigned integer)
        //! \param x_1 argument
        //! \param x_2 argument
        //! \return lcm(x_1, x_2)
        //!
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        template <typename T>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        constexpr inline auto LeastCommonMultiple(T x_1, T x_2)
        {
            static_assert(std::is_integral<T>::value && std::is_unsigned<T>::value, "error: only unsigned integers allowed");

            const auto gcd = GreatestCommonDivisor(x_1, x_2);

            assert(gcd > static_cast<T>(0));

            return (x_1 * x_2) / gcd;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        //!
        //! \brief Power of N test.
        //!
        //! \tparam N the base
        //! \tparam T the argument type (must be an unsigned integer)
        //! \param x argument
        //! \return `true` if `x` is power of `N`, otherwise `false`
        //!
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        template <SizeT N, typename T>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        constexpr inline auto IsPowerOf(T x)
        {
            static_assert(std::is_integral<T>::value && std::is_unsigned<T>::value, "error: only unsigned integers allowed");
            static_assert(N > 0, "error: N must be at least 1");

            if (x == 1 || x == N)
                return true;

            if (x <= 0 || x < N || N == 1)
                return false;

            while (true)
            {
                if (x == 0 || x % N)
                    return false;

                x /= N;

                if (x == 1)
                    return true;
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        //!
        //! \brief Prefix sum calculation.
        //!
        //! \tparam N the array extent
        //! \param x argument
        //! \return an array holding the prefix sums
        //!
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        template <SizeT N>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        constexpr inline auto PrefixSum(const ::XXX_NAMESPACE::dataTypes::SizeArray<N>& x)
        {
            ::XXX_NAMESPACE::dataTypes::SizeArray<N> y{0};

            for (SizeT i = 1; i < N; ++i)
            {
                y[i] = y[i - 1] + x[i - 1];
            }

            return y;
        }

        namespace internal
        {
            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            //!
            //! \brief Definition of some math functions and constants for different FP types
            //!
            //! \tparam T a floating point type
            //!
            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            template <typename T>
            struct Func
            {
                static constexpr T One = static_cast<T>(1.0);
                static constexpr T MinusOne = static_cast<T>(-1.0);

                //!
                //! \brief Calculate the square root.
                //!
                //! \param x argument
                //! \return the square root of `x`
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                static inline auto sqrt(const T x)
                { 
                    assert(x >= static_cast<T>(0));

                    return std::sqrt(x);
                }

                //!
                //! \brief Calculate the logarithm.
                //!
                //! \param x argument
                //! \return the logarithm of `x`
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                static inline auto log(const T x)
                { 
                    assert(x > static_cast<T>(0));

                    return std::log(x);
                }

                //!
                //! \brief Calculate the exponential.
                //!
                //! \param x argument
                //! \return the exponential of `x`
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                static inline auto exp(const T x) { return std::exp(x); }
            };

            //!
            //! \brief Specialization for `T=float`.
            //!
            template <>
            struct Func<float>
            {
                static constexpr float One = 1.0F;
                static constexpr float MinusOne = -1.0F;

                HOST_VERSION
                CUDA_DEVICE_VERSION
                static inline auto sqrt(const float x)
                { 
                    assert(x >= 0.0F);

                    return sqrtf(x);
                }

                HOST_VERSION
                CUDA_DEVICE_VERSION
                static inline auto log(const float x)
                {
                    assert(x > 0.0F);

                    return logf(x);
                }

                HOST_VERSION
                CUDA_DEVICE_VERSION
                static inline auto exp(const float x) { return expf(x); }
            };
        }

        template <typename T>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        static inline auto Sqrt(const T& value)
        {
            return internal::Func<T>::sqrt(value);
        }

        template <typename T>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        static inline auto Log(const T& value)
        {
            return internal::Func<T>::log(value);
        }

        template <typename T>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        static inline auto Exp(const T& value)
        {
            return internal::Func<T>::exp(value);
        }
    } // namespace math
} // namespace XXX_NAMESPACE

#endif