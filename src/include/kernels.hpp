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
#include <common/data_types.hpp>

// Data types and layouts.
using size_type = fw::size_type;
using real_type = fw::real_type;

using type_x = real_type;
using type_y = real_type;
using type_z = real_type;
using element_type = fw::vec<real_type, 3>;

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
/*
using type_x = std::uint16_t;
using type_y = std::int8_t;
using type_z = std::uint32_t;
using element_type = fw::tuple<type_x, type_y, type_z>;
*/

using const_element_type = const element_type;

#if defined(AOS_LAYOUT)
constexpr fw::data_layout DataLayout = fw::data_layout::AoS;
#elif defined(SOA_LAYOUT)
constexpr fw::data_layout DataLayout = fw::data_layout::SoA;
#elif defined(SOAI_LAYOUT)
constexpr fw::data_layout DataLayout = fw::data_layout::SoAi;
#endif

template <typename T, size_type C_Dimension>
using field_type = fw::field<T, C_Dimension, DataLayout>;

template <size_type C_Dimension>
using array_type = fw::sarray<size_type, C_Dimension>;
/*
// prototypes
template <typename T>
struct kernel
{
#if defined(AOS_LAYOUT)
    #if defined(VECTOR_PRODUCT)
        template <size_type D, size_type DD = D>
        static double cross(const fw::field<T, DD, fw::data_layout::AoS>& x_1, const fw::field<T, DD, fw::data_layout::AoS>& x_2, fw::field<T, DD, fw::data_layout::AoS>& y, const array_type<D>& n);
    #else
        template <size_type D, size_type DD = D>
        static double exp(fw::field<T, DD, fw::data_layout::AoS>& x, const array_type<D>& n);

        template <size_type D, size_type DD = D>
        static double log(fw::field<T, DD, fw::data_layout::AoS>& x, const array_type<D>& n);

        template <size_type D, size_type DD = D>
        static double exp(const fw::field<T, DD, fw::data_layout::AoS>& x, fw::field<T, DD, fw::data_layout::AoS>& y, const array_type<D>& n);

        template <size_type D, size_type DD = D>
        static double log(const fw::field<T, DD, fw::data_layout::AoS>& x, fw::field<T, DD, fw::data_layout::AoS>& y, const array_type<D>& n);
    #endif
#elif defined(SOA_LAYOUT)
    #if defined(VECTOR_PRODUCT)
        template <size_type D, size_type DD = D>
        static double cross(const fw::field<T, DD, fw::data_layout::SoA>& x_1, const fw::field<T, DD, fw::data_layout::SoA>& x_2, fw::field<T, DD, fw::data_layout::SoA>& y, const array_type<D>& n);
    #else
        template <size_type D, size_type DD = D>
        static double exp(fw::field<T, DD, fw::data_layout::SoA>& x, const array_type<D>& n);

        template <size_type D, size_type DD = D>
        static double log(fw::field<T, DD, fw::data_layout::SoA>& x, const array_type<D>& n);

        template <size_type D, size_type DD = D>
        static double exp(const fw::field<T, DD, fw::data_layout::SoA>& x, fw::field<T, DD, fw::data_layout::SoA>& y, const array_type<D>& n);

        template <size_type D, size_type DD = D>
        static double log(const fw::field<T, DD, fw::data_layout::SoA>& x, fw::field<T, DD, fw::data_layout::SoA>& y, const array_type<D>& n);
    #endif
#elif defined(SOAI_LAYOUT)
    #if defined(VECTOR_PRODUCT)
        template <size_type D, size_type DD = D>
        static double cross(const fw::field<T, DD, fw::data_layout::SoAi>& x_1, const fw::field<T, DD, fw::data_layout::SoAi>& x_2, fw::field<T, DD, fw::data_layout::SoAi>& y, const array_type<D>& n);
    #else
        template <size_type D, size_type DD = D>
        static double exp(fw::field<T, DD, fw::data_layout::SoAi>& x, const array_type<D>& n);

        template <size_type D, size_type DD = D>
        static double log(fw::field<T, DD, fw::data_layout::SoAi>& x, const array_type<D>& n);

        template <size_type D, size_type DD = D>
        static double exp(const fw::field<T, DD, fw::data_layout::SoAi>& x, fw::field<T, DD, fw::data_layout::SoAi>& y, const array_type<D>& n);

        template <size_type D, size_type DD = D>
        static double log(const fw::field<T, DD, fw::data_layout::SoAi>& x, fw::field<T, DD, fw::data_layout::SoAi>& y, const array_type<D>& n);
    #endif    
#endif
};
*/

#endif
