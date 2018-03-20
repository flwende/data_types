// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(KERNEL_HPP)
#define KERNEL_HPP

#include <buffer/buffer.hpp>

// data types
using real_t = float;

#if defined(__INTEL_SDLT)
	#include <sdlt/sdlt.h>
	typedef struct
	{
		real_t x;
		real_t y;
		real_t z;
	} sdlt_real3_t;

	SDLT_PRIMITIVE(sdlt_real3_t, x, y, z)

	inline sdlt_real3_t exp(const sdlt_real3_t& x)
	{
		sdlt_real3_t y;
		y.x = fw::math<real_t>::exp(x.x);
		y.y = fw::math<real_t>::exp(x.y);
		y.z = fw::math<real_t>::exp(x.z);
		return y;
	}

	inline sdlt_real3_t log(const sdlt_real3_t& x)
	{
		sdlt_real3_t y;
		y.x = fw::math<real_t>::log(x.x);
		y.y = fw::math<real_t>::log(x.y);
		y.z = fw::math<real_t>::log(x.z);
		return y;
	}

	using real3_t = sdlt_real3_t;

	template <typename T, std::size_t D>
	using buffer_type = sdlt::soa1d_container<T>;

#else
	using real3_t = fw::vec<real_t, 3>;

	template <typename T, std::size_t D>
	using buffer_type = fw::buffer<T, D, fw::target::host, fw::data_layout::SoA>;
#endif

// data layout
constexpr fw::data_layout layout = fw::data_layout::SoA;
//constexpr fw::data_layout layout = fw::data_layout::AoS;

// prototypes
template <typename T>
struct kernel
{
	template <std::size_t D>
	static double exp(buffer_type<T, D>& x);

	template <std::size_t D>
	static double log(buffer_type<T, D>& x);
};

#endif