// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(COMMON_MATH_HPP)
#define COMMON_MATH_HPP

#include <cmath>
#include <type_traits>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#include <data_types/DataTypes.hpp>
#include <platform/Target.hpp>

namespace XXX_NAMESPACE
{
    namespace math
    {
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
        constexpr auto GreatestCommonDivisor(T x_1, T x_2)
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
        constexpr auto LeastCommonMultiple(T x_1, T x_2)
        {
            static_assert(std::is_integral<T>::value && std::is_unsigned<T>::value, "error: only unsigned integers allowed");

            return (x_1 * x_2) / GreatestCommonDivisor(x_1, x_2);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        //!
        //! \brief Power of C_N test.
        //!
        //! \tparam C_N the base
        //! \tparam T the argument type (must be an unsigned integer)
        //! \param x argument
        //! \return `true` if `x` is power of `C_N`, otherwise `false`
        //!
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        template <SizeType C_N, typename T>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        constexpr auto IsPowerOf(T x)
        {
            static_assert(std::is_integral<T>::value && std::is_unsigned<T>::value, "error: only unsigned integers allowed");
            static_assert(C_N > 0, "error: C_N must be at least 1");

            if (x == 1 || x == C_N)
                return true;

            if (x <= 0 || x < C_N || C_N == 1)
                return false;

            while (true)
            {
                if (x == 0 || x % C_N)
                    return false;

                x /= C_N;

                if (x == 1)
                    return true;
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        //!
        //! \brief Prefix sum calculation.
        //!
        //! \tparam C_N the array extent
        //! \param x argument
        //! \return an array holding the prefix sums
        //!
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        template <SizeType C_N>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        constexpr auto PrefixSum(const ::XXX_NAMESPACE::dataTypes::SizeArray<C_N>& x)
        {
            ::XXX_NAMESPACE::dataTypes::SizeArray<C_N> y{0};

            for (SizeType i = 1; i < C_N; ++i)
            {
                y[i] = y[i - 1] + x[i - 1];
            }

            return y;
        }

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
            static auto sqrt(const T x) { return std::sqrt(x); }

            //!
            //! \brief Calculate the logarithm.
            //!
            //! \param x argument
            //! \return the logarithm of `x`
            //!
            HOST_VERSION
            CUDA_DEVICE_VERSION
            static auto log(const T x) { return std::log(x); }

            //!
            //! \brief Calculate the exponential.
            //!
            //! \param x argument
            //! \return the exponential of `x`
            //!
            HOST_VERSION
            CUDA_DEVICE_VERSION
            static auto exp(const T x) { return std::exp(x); }
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
            static auto sqrt(const float x) { return sqrtf(x); }

            HOST_VERSION
            CUDA_DEVICE_VERSION
            static auto log(const float x) { return logf(x); }

            HOST_VERSION
            CUDA_DEVICE_VERSION
            static auto exp(const float x) { return expf(x); }
        };
    } // namespace math
} // namespace XXX_NAMESPACE

#endif