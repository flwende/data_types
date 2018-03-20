// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(KERNEL_HPP)
#define KERNEL_HPP

#include <buffer/buffer.hpp>

// data types
using real_t = float;
using real3_t = fw::vec<real_t, 3>;

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
	y.x = exp(x.x);
	y.y = exp(x.y);
	y.z = exp(x.z);
	return y;
}

inline sdlt_real3_t log(const sdlt_real3_t& x)
{
	sdlt_real3_t y;
	y.x = log(x.x);
	y.y = log(x.y);
	y.z = log(x.z);
	return y;
}
#endif

// data layout
constexpr fw::data_layout layout = fw::data_layout::SoA;
//constexpr fw::data_layout layout = fw::data_layout::AoS;

// prototypes
template <typename T>
struct kernel
{
	#if defined(__INTEL_SDLT)
	template <std::size_t D>
	static double exp(sdlt::soa1d_container<T>& x);
	template <std::size_t D>
	static double log(sdlt::soa1d_container<T>& x);
	#else
	template <std::size_t D>
	static double exp(fw::buffer<T, D, fw::target::host, layout>& x);
	template <std::size_t D>
	static double log(fw::buffer<T, D, fw::target::host, layout>& x);
	#endif
};

#endif