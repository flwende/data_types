// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(KERNEL_HPP)
#define KERNEL_HPP

#include <buffer/buffer.hpp>
#include <vec/vec.hpp>

// data types and layout
//using type = float;
using type = double;
using element_type = fw::vec<type, 3>;

#if defined(AOS_LAYOUT)
constexpr fw::data_layout layout = fw::data_layout::AoS;
#elif defined(SOA_LAYOUT)
constexpr fw::data_layout layout = fw::data_layout::SoA;
#endif

template <typename T, std::size_t D>
using buffer_type = fw::buffer<T, D, layout>;

// prototypes
template <typename T>
struct kernel
{
#if defined(AOS_LAYOUT)
	template <std::size_t D>
	static double exp(fw::buffer<T, D, fw::data_layout::AoS>& x);

    template <std::size_t D>
	static double log(fw::buffer<T, D, fw::data_layout::AoS>& x);
#elif defined(SOA_LAYOUT)
	template <std::size_t D>
	static double exp(fw::buffer<T, D, fw::data_layout::SoA>& x);

	template <std::size_t D>
	static double log(fw::buffer<T, D, fw::data_layout::SoA>& x);
#endif
};

#endif
