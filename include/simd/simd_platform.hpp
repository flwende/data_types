// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(SIMD_PLATFORM_HPP)
#define SIMD_PLATFORM_HPP

#if defined(__AVX512F__)
    #define SIMD_WIDTH_NATIVE_64BIT 8
    #define SIMD_WIDTH_NATIVE_32BIT 16
    #define SIMD_ALIGNMENT 64
#elif defined(__AVX__)
    #define SIMD_WIDTH_NATIVE_64BIT 4
    #define SIMD_WIDTH_NATIVE_32BIT 8
    #define SIMD_ALIGNMENT 32
#elif defined(__SSE__)
    #define SIMD_WIDTH_NATIVE_64BIT 2
    #define SIMD_WIDTH_NATIVE_32BIT 4
    #define SIMD_ALIGNMENT 16
#else
    #define SIMD_WIDTH_NATIVE_64BIT 1
    #define SIMD_WIDTH_NATIVE_32BIT 1
    #define SIMD_ALIGNMENT 8
#endif

#include <cstdint>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

namespace XXX_NAMESPACE
{
    namespace simd
    {
        //! Memory alignment
        static constexpr size_t alignment = SIMD_ALIGNMENT;

        //! \brief Test for which fundamental data types there is SIMD equivalents
        //!
        //! \tparam T (fundamental) data type
        template <typename T>
        struct implementation
        {
            static constexpr bool available = false;
        };

        #define MACRO(T)                                            \
        template <>                                                 \
        struct implementation<T>                                    \
        {                                                           \
            static constexpr bool available = true;                 \
        };

        MACRO(double)
        MACRO(float)
        MACRO(std::uint64_t)
        MACRO(std::int64_t)
        MACRO(std::uint32_t)
        MACRO(std::int32_t)

        #undef MACRO

        //! \brief Get information for fundamental data types when used in the SIMD context
        //!
        //! \tparam T (fundamental) data type
        template <typename T>
        struct type
        {
            static_assert(implementation<T>::available, "error: there is no implementation for data type T");
        };

        #define MACRO(T, SW, SWE)                                   \
        template <>                                                 \
        struct type<T>                                              \
        {                                                           \
            static constexpr std::size_t width = SW;                \
            static constexpr std::size_t width_elements = SWE;      \
        };

        MACRO(double, SIMD_WIDTH_NATIVE_64BIT, 8)
        MACRO(float, SIMD_WIDTH_NATIVE_32BIT, 16)
        MACRO(std::uint64_t, SIMD_WIDTH_NATIVE_64BIT, 8)
        MACRO(std::int64_t, SIMD_WIDTH_NATIVE_64BIT, 8)
        MACRO(std::uint32_t, SIMD_WIDTH_NATIVE_32BIT, 16)
        MACRO(std::int32_t, SIMD_WIDTH_NATIVE_32BIT, 16)

        #undef MACRO
    }
}

#undef SIMD_WIDTH_NATIVE_64BIT
#undef SIMD_WIDTH_NATIVE_32BIT
#undef SIMD_ALIGNMENT
#undef XXX_NAMESPACE

#endif
