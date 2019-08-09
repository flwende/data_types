// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(KERNEL_HPP)
#define KERNEL_HPP

#include <cstdint>
#include <field/field.hpp>
#include <vec/vec.hpp>
#include <tuple/tuple.hpp>

// data types and layout
using type = double;

using type_x = type;
using type_y = type;
using type_z = type;
using element_type = fw::vec<type, 3>;

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
/*
using type_x = std::uint16_t;
using type_y = double;
using type_z = std::uint32_t;
using element_type = fw::tuple<type_x, type_y, type_z>;
*/
using const_element_type = const element_type;

#if defined(AOS_LAYOUT)
constexpr fw::data_layout layout = fw::data_layout::AoS;
#elif defined(SOA_LAYOUT)
constexpr fw::data_layout layout = fw::data_layout::SoA;
#elif defined(SOAI_LAYOUT)
constexpr fw::data_layout layout = fw::data_layout::SoAi;
#endif

template <typename T, std::size_t D>
using buffer_type = fw::field<T, D, layout>;

template <std::size_t D>
using array_type = fw::sarray<std::size_t, D>;

// prototypes
template <typename T>
struct kernel
{
#if defined(AOS_LAYOUT)
    #if defined(VECTOR_PRODUCT)
        template <std::size_t D, std::size_t DD = D>
        static double cross(const fw::field<T, DD, fw::data_layout::AoS>& x_1, const fw::field<T, DD, fw::data_layout::AoS>& x_2, fw::field<T, DD, fw::data_layout::AoS>& y, const array_type<D>& n);
    #else
        template <std::size_t D, std::size_t DD = D>
        static double exp(fw::field<T, DD, fw::data_layout::AoS>& x, const array_type<D>& n);

        template <std::size_t D, std::size_t DD = D>
        static double log(fw::field<T, DD, fw::data_layout::AoS>& x, const array_type<D>& n);

        template <std::size_t D, std::size_t DD = D>
        static double exp(const fw::field<T, DD, fw::data_layout::AoS>& x, fw::field<T, DD, fw::data_layout::AoS>& y, const array_type<D>& n);

        template <std::size_t D, std::size_t DD = D>
        static double log(const fw::field<T, DD, fw::data_layout::AoS>& x, fw::field<T, DD, fw::data_layout::AoS>& y, const array_type<D>& n);
    #endif
#elif defined(SOA_LAYOUT)
    #if defined(VECTOR_PRODUCT)
        template <std::size_t D, std::size_t DD = D>
        static double cross(const fw::field<T, DD, fw::data_layout::SoA>& x_1, const fw::field<T, DD, fw::data_layout::SoA>& x_2, fw::field<T, DD, fw::data_layout::SoA>& y, const array_type<D>& n);
    #else
        template <std::size_t D, std::size_t DD = D>
        static double exp(fw::field<T, DD, fw::data_layout::SoA>& x, const array_type<D>& n);

        template <std::size_t D, std::size_t DD = D>
        static double log(fw::field<T, DD, fw::data_layout::SoA>& x, const array_type<D>& n);

        template <std::size_t D, std::size_t DD = D>
        static double exp(const fw::field<T, DD, fw::data_layout::SoA>& x, fw::field<T, DD, fw::data_layout::SoA>& y, const array_type<D>& n);

        template <std::size_t D, std::size_t DD = D>
        static double log(const fw::field<T, DD, fw::data_layout::SoA>& x, fw::field<T, DD, fw::data_layout::SoA>& y, const array_type<D>& n);
    #endif
#elif defined(SOAI_LAYOUT)
    #if defined(VECTOR_PRODUCT)
        template <std::size_t D, std::size_t DD = D>
        static double cross(const fw::field<T, DD, fw::data_layout::SoAi>& x_1, const fw::field<T, DD, fw::data_layout::SoAi>& x_2, fw::field<T, DD, fw::data_layout::SoAi>& y, const array_type<D>& n);
    #else
        template <std::size_t D, std::size_t DD = D>
        static double exp(fw::field<T, DD, fw::data_layout::SoAi>& x, const array_type<D>& n);

        template <std::size_t D, std::size_t DD = D>
        static double log(fw::field<T, DD, fw::data_layout::SoAi>& x, const array_type<D>& n);

        template <std::size_t D, std::size_t DD = D>
        static double exp(const fw::field<T, DD, fw::data_layout::SoAi>& x, fw::field<T, DD, fw::data_layout::SoAi>& y, const array_type<D>& n);

        template <std::size_t D, std::size_t DD = D>
        static double log(const fw::field<T, DD, fw::data_layout::SoAi>& x, fw::field<T, DD, fw::data_layout::SoAi>& y, const array_type<D>& n);
    #endif    
#endif
};

#endif
