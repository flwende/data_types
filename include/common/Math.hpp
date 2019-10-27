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
        using ::XXX_NAMESPACE::dataTypes::SizeArray;

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
        //! \param x_1 argument
        //! \return `true` if `x_1` is power of `N`, otherwise `false`
        //!
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        template <SizeT N, typename T>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        constexpr inline auto IsPowerOf(T x_1)
        {
            static_assert(std::is_integral<T>::value && std::is_unsigned<T>::value, "error: only unsigned integers allowed");
            static_assert(N > 0, "error: N must be at least 1");

            if (x_1 == 1 || x_1 == N)
                return true;

            if (x_1 <= 0 || x_1 < N || N == 1)
                return false;

            while (true)
            {
                if (x_1 == 0 || x_1 % N)
                    return false;

                x_1 /= N;

                if (x_1 == 1)
                    return true;
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        //!
        //! \brief Prefix sum calculation.
        //!
        //! \tparam N the array extent
        //! \param x_1 argument
        //! \return an array holding the prefix sums
        //!
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        template <SizeT N>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        constexpr inline auto PrefixSum(const SizeArray<N>& x_1)
        {
            SizeArray<N> y{0};

            for (SizeT i = 1; i < N; ++i)
            {
                y[i] = y[i - 1] + x_1[i - 1];
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
                //! \param x_1 argument
                //! \return the square root of `x_1`
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                static inline auto Sqrt(const T x_1)
                { 
                    assert(x_1 >= static_cast<T>(0));

                    return std::sqrt(x_1);
                }

                //!
                //! \brief Calculate the logarithm.
                //!
                //! \param x_1 argument
                //! \return the logarithm of `x_1`
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                static inline auto Log(const T x_1)
                { 
                    assert(x_1 > static_cast<T>(0));

                    return std::log(x_1);
                }

                //!
                //! \brief Calculate the exponential.
                //!
                //! \param x_1 argument
                //! \return the exponential of `x_1`
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                static inline auto Exp(const T x_1) { return std::exp(x_1); }

                //!
                //! \brief Calculate the maximum.
                //!
                //! \param x_1 argument
                //! \param x_2 argument
                //! \return the maximum of `x_1` and `x_2`
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                static inline auto Max(const T x_1, const T x_2) { return std::max(x_1, x_2); }

                //!
                //! \brief Calculate the minimum.
                //!
                //! \param x_1 argument
                //! \param x_2 argument
                //! \return the minimum of `x_1` and `x_2`
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                static inline auto Min(const T x_1, const T x_2) { return std::min(x_1, x_2); }

                //!
                //! \brief Calculate the absolute value.
                //!
                //! \param x_1 argument
                //! \return the absolute value of `x_1`
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                static inline auto Abs(const T x_1) { return std::abs(x_1); }
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
                static inline auto Sqrt(const float x_1)
                { 
                    assert(x_1 >= 0.0F);

                    return sqrtf(x_1);
                }

                HOST_VERSION
                CUDA_DEVICE_VERSION
                static inline auto Log(const float x_1)
                {
                    assert(x_1 > 0.0F);

                    return logf(x_1);
                }

                HOST_VERSION
                CUDA_DEVICE_VERSION
                static inline auto Exp(const float x_1) { return expf(x_1); }

                HOST_VERSION
                CUDA_DEVICE_VERSION
                static inline auto Max(const float x_1, const float x_2) { return fmax(x_1, x_2); }

                HOST_VERSION
                CUDA_DEVICE_VERSION
                static inline auto Min(const float x_1, const float x_2) { return fmin(x_1, x_2); }

                HOST_VERSION
                CUDA_DEVICE_VERSION
                static inline auto Abs(const float x_1) { return fabs(x_1); }
            };
        }

        template <typename T>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        static inline auto Sqrt(const T& x_1)
        {
            return internal::Func<T>::Sqrt(x_1);
        }

        template <typename T>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        static inline auto Log(const T& x_1)
        {
            return internal::Func<T>::Log(x_1);
        }

        template <typename T>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        static inline auto Exp(const T& x_1)
        {
            return internal::Func<T>::Exp(x_1);
        }

        template <typename T>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        static inline auto Max(const T& x_1, const T& x_2)
        {
            return internal::Func<T>::Max(x_1, x_2);
        }

        template <typename T>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        static inline auto Min(const T& x_1, const T& x_2)
        {
            return internal::Func<T>::Min(x_1, x_2);
        }

        template <typename T>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        static inline auto Abs(const T& x_1)
        {
            return internal::Func<T>::Abs(x_1);
        }
    } // namespace math
} // namespace XXX_NAMESPACE

#endif