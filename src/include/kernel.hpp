// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(KERNEL_HPP)
#define KERNEL_HPP

#include <buffer/buffer.hpp>
#include <vec/vec.hpp>
#include <tuple/tuple.hpp>

// data types and layout
using type = double;
/*
using type_x = type;
using type_y = type;
using type_z = type;
using element_type = fw::vec<type, 3>;
*/
/*
using type_x = std::uint32_t;
using type_y = std::uint32_t;
using type_z = std::uint32_t;
using element_type = fw::vec<std::uint32_t, 3>;
*/
/*
using type_x = std::uint32_t;
using type_y = std::uint32_t;
using type_z = std::uint32_t;
using element_type = fw::tuple<type_x, type_y, type_z>;
*/

using type_x = std::uint16_t;
using type_y = double;
using type_z = std::uint32_t;
using element_type = fw::tuple<type_x, type_y, type_z>;

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
    #if defined(VECTOR_PRODUCT)
        template <std::size_t D>
        static double cross(const fw::buffer<T, D, fw::data_layout::AoS>& x_1, const fw::buffer<T, D, fw::data_layout::AoS>& x_2, fw::buffer<T, D, fw::data_layout::AoS>& y);
    #else
        template <std::size_t D>
        static double exp(fw::buffer<T, D, fw::data_layout::AoS>& x);

        template <std::size_t D>
        static double log(fw::buffer<T, D, fw::data_layout::AoS>& x);

        template <std::size_t D>
        static double exp(const fw::buffer<T, D, fw::data_layout::AoS>& x, fw::buffer<T, D, fw::data_layout::AoS>& y);

        template <std::size_t D>
        static double log(const fw::buffer<T, D, fw::data_layout::AoS>& x, fw::buffer<T, D, fw::data_layout::AoS>& y);
    #endif
#elif defined(SOA_LAYOUT)
    #if defined(VECTOR_PRODUCT)
        template <std::size_t D>
        static double cross(const fw::buffer<T, D, fw::data_layout::SoA>& x_1, const fw::buffer<T, D, fw::data_layout::SoA>& x_2, fw::buffer<T, D, fw::data_layout::SoA>& y);
    #else
        template <std::size_t D>
        static double exp(fw::buffer<T, D, fw::data_layout::SoA>& x);

        template <std::size_t D>
        static double log(fw::buffer<T, D, fw::data_layout::SoA>& x);

        template <std::size_t D>
        static double exp(const fw::buffer<T, D, fw::data_layout::SoA>& x, fw::buffer<T, D, fw::data_layout::SoA>& y);

        template <std::size_t D>
        static double log(const fw::buffer<T, D, fw::data_layout::SoA>& x, fw::buffer<T, D, fw::data_layout::SoA>& y);
    #endif
#endif
};

#endif