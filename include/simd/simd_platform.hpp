// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(SIMD_PLATFORM_HPP)
#define SIMD_PLATFORM_HPP

#include <cstdint>

#if defined(__AVX512F__)
	#define SIMD_WIDTH_NATIVE_64BIT 8
	#define SIMD_WIDTH_NATIVE_32BIT 16
	#define SIMD_WIDTH_NATIVE_16BIT 32
	#define SIMD_WIDTH_NATIVE_8BIT 64
	#define SIMD_ALIGNMENT 64
#elif defined(__AVX__)
	#define SIMD_WIDTH_NATIVE_64BIT 4
	#define SIMD_WIDTH_NATIVE_32BIT 8
	#define SIMD_WIDTH_NATIVE_16BIT 16
	#define SIMD_WIDTH_NATIVE_8BIT 32
	#define SIMD_ALIGNMENT 32
#elif defined(__SSE__)
	#define SIMD_WIDTH_NATIVE_64BIT 2
	#define SIMD_WIDTH_NATIVE_32BIT 4
	#define SIMD_WIDTH_NATIVE_16BIT 8
	#define SIMD_WIDTH_NATIVE_8BIT 16
	#define SIMD_ALIGNMENT 16
#else
	#define SIMD_WIDTH_NATIVE_64BIT 1
	#define SIMD_WIDTH_NATIVE_32BIT 1
	#define SIMD_WIDTH_NATIVE_16BIT 1
	#define SIMD_WIDTH_NATIVE_8BIT 1
	#define SIMD_ALIGNMENT 8
#endif

#if !defined(SIMD_NAMESPACE)
#if !defined(XXX_NAMESPACE)
#define SIMD_NAMESPACE fw
#else
#define SIMD_NAMESPACE XXX_NAMESPACE
#endif
#endif

namespace SIMD_NAMESPACE
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

		#define MACRO(T)					\
		template <>						\
		struct implementation<T>				\
		{							\
			static constexpr bool available = true;		\
		};							\

		#define MACRO_C(T)					\
		MACRO(T)						\
		MACRO(const T)						\

		MACRO_C(double)
		MACRO_C(float)
		MACRO_C(std::uint64_t)
		MACRO_C(std::int64_t)
		MACRO_C(std::uint32_t)
		MACRO_C(std::int32_t)
		MACRO_C(std::uint16_t)
		MACRO_C(std::int16_t)
		MACRO_C(std::uint8_t)
		MACRO_C(std::int8_t)

		#undef MACRO_C
		#undef MACRO

		//! \brief Get information for fundamental data types when used in the SIMD context
		//!
		//! \tparam T (fundamental) data type
		template <typename T>
		struct type
		{
			static_assert(implementation<T>::available, "error: there is no implementation for data type T");
		};

		#define MACRO(T, SW)					\
		template <>						\
		struct type<T>						\
		{							\
			static constexpr std::size_t width = SW;	\
		};							\

		#define MACRO_C(T, SW)					\
		MACRO(T, SW)						\
		MACRO(const T, SW)					\

		MACRO_C(double, SIMD_WIDTH_NATIVE_64BIT)
		MACRO_C(float, SIMD_WIDTH_NATIVE_32BIT)
		MACRO_C(std::uint64_t, SIMD_WIDTH_NATIVE_64BIT)
		MACRO_C(std::int64_t, SIMD_WIDTH_NATIVE_64BIT)
		MACRO_C(std::uint32_t, SIMD_WIDTH_NATIVE_32BIT)
		MACRO_C(std::int32_t, SIMD_WIDTH_NATIVE_32BIT)
		MACRO_C(std::uint16_t, SIMD_WIDTH_NATIVE_16BIT)
		MACRO_C(std::int16_t, SIMD_WIDTH_NATIVE_16BIT)
		MACRO_C(std::uint8_t, SIMD_WIDTH_NATIVE_8BIT)
		MACRO_C(std::int8_t, SIMD_WIDTH_NATIVE_8BIT)

		#undef MACRO_C
		#undef MACRO
	}
}

#undef SIMD_WIDTH_NATIVE_64BIT
#undef SIMD_WIDTH_NATIVE_32BIT
#undef SIMD_WIDTH_NATIVE_16BIT
#undef SIMD_WIDTH_NATIVE_8BIT
#undef SIMD_ALIGNMENT

#endif
