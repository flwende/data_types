// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(KERNEL_HPP)
#define KERNEL_HPP

#include <buffer/buffer.hpp>

// data types and layout
using real_t = float;
constexpr fw::buffer_type Buffer_type = fw::buffer_type::host; 
constexpr fw::data_layout Data_layout = fw::data_layout::SoA;
//constexpr fw::data_layout Data_layout = fw::data_layout::AoS;

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
	using buffer_type = fw::buffer<T, D, Buffer_type, Data_layout>;
#endif

// prototypes
template <typename T>
struct kernel
{
	template <std::size_t D>
	static double exp(fw::buffer<T, D, fw::buffer_type::host, fw::data_layout::AoS>& x);

	template <std::size_t D>
	static double exp(fw::buffer<T, D, fw::buffer_type::host, fw::data_layout::SoA>& x);

	template <std::size_t D>
	static double log(fw::buffer<T, D, fw::buffer_type::host, fw::data_layout::AoS>& x);

	template <std::size_t D>
	static double log(fw::buffer<T, D, fw::buffer_type::host, fw::data_layout::SoA>& x);

	#if defined(HAVE_SYCL)
	template <std::size_t D>
	static double exp(fw::buffer<T, D, fw::buffer_type::host_device, fw::data_layout::AoS>& x);

	template <std::size_t D>
	static double exp(fw::buffer<T, D, fw::buffer_type::host_device, fw::data_layout::SoA>& x);
	#endif
};

#endif