// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(KERNEL_HPP)
#define KERNEL_HPP

#include <cstdint>
#include <data_types/field/Field.hpp>
#include <data_types/vec/Vec.hpp>
#include <data_types/tuple/Tuple.hpp>
#include <data_types/DataTypes.hpp>

// Data types and layout
using SizeType = fw::SizeType;
//using type = double;
using type = float;

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
/*
using type_x = std::uint16_t;
using type_y = std::int8_t;
using type_z = std::uint32_t;
using element_type = fw::tuple<type_x, type_y, type_z>;
*/
using const_element_type = const element_type;

#if defined(AOS_LAYOUT)
constexpr ::fw::memory::DataLayout layout = ::fw::memory::DataLayout::AoS;
#elif defined(SOA_LAYOUT)
constexpr ::fw::memory::DataLayout layout = ::fw::memory::DataLayout::SoA;
#elif defined(SOAI_LAYOUT)
constexpr ::fw::memory::DataLayout layout = ::fw::memory::DataLayout::SoAi;
#endif

template <typename T, SizeType D>
using buffer_type = fw::Field<T, D, layout>;

template <SizeType D>
using array_type = fw::sarray<SizeType, D>;

// prototypes
template <typename T>
struct kernel
{
#if defined(AOS_LAYOUT)
    #if defined(VECTOR_PRODUCT)
        template <SizeType D, SizeType DD = D>
        static double cross(const fw::Field<T, DD, ::fw::memory::DataLayout::AoS>& x_1, const fw::Field<T, DD, ::fw::memory::DataLayout::AoS>& x_2, fw::Field<T, DD, ::fw::memory::DataLayout::AoS>& y, const array_type<D>& n);
    #else
        template <SizeType D, SizeType DD = D>
        static double exp(fw::Field<T, DD, ::fw::memory::DataLayout::AoS>& x, const array_type<D>& n);

        template <SizeType D, SizeType DD = D>
        static double log(fw::Field<T, DD, ::fw::memory::DataLayout::AoS>& x, const array_type<D>& n);

        template <SizeType D, SizeType DD = D>
        static double exp(const fw::Field<T, DD, ::fw::memory::DataLayout::AoS>& x, fw::Field<T, DD, ::fw::memory::DataLayout::AoS>& y, const array_type<D>& n);

        template <SizeType D, SizeType DD = D>
        static double log(const fw::Field<T, DD, ::fw::memory::DataLayout::AoS>& x, fw::Field<T, DD, ::fw::memory::DataLayout::AoS>& y, const array_type<D>& n);
    #endif
#elif defined(SOA_LAYOUT)
    #if defined(VECTOR_PRODUCT)
        template <SizeType D, SizeType DD = D>
        static double cross(const fw::Field<T, DD, ::fw::memory::DataLayout::SoA>& x_1, const fw::Field<T, DD, ::fw::memory::DataLayout::SoA>& x_2, fw::Field<T, DD, ::fw::memory::DataLayout::SoA>& y, const array_type<D>& n);
    #else
        template <SizeType D, SizeType DD = D>
        static double exp(fw::Field<T, DD, ::fw::memory::DataLayout::SoA>& x, const array_type<D>& n);

        template <SizeType D, SizeType DD = D>
        static double log(fw::Field<T, DD, ::fw::memory::DataLayout::SoA>& x, const array_type<D>& n);

        template <SizeType D, SizeType DD = D>
        static double exp(const fw::Field<T, DD, ::fw::memory::DataLayout::SoA>& x, fw::Field<T, DD, ::fw::memory::DataLayout::SoA>& y, const array_type<D>& n);

        template <SizeType D, SizeType DD = D>
        static double log(const fw::Field<T, DD, ::fw::memory::DataLayout::SoA>& x, fw::Field<T, DD, ::fw::memory::DataLayout::SoA>& y, const array_type<D>& n);
    #endif
#elif defined(SOAI_LAYOUT)
    #if defined(VECTOR_PRODUCT)
        template <SizeType D, SizeType DD = D>
        static double cross(const fw::Field<T, DD, ::fw::memory::DataLayout::SoAi>& x_1, const fw::Field<T, DD, ::fw::memory::DataLayout::SoAi>& x_2, fw::Field<T, DD, ::fw::memory::DataLayout::SoAi>& y, const array_type<D>& n);
    #else
        template <SizeType D, SizeType DD = D>
        static double exp(fw::Field<T, DD, ::fw::memory::DataLayout::SoAi>& x, const array_type<D>& n);

        template <SizeType D, SizeType DD = D>
        static double log(fw::Field<T, DD, ::fw::memory::DataLayout::SoAi>& x, const array_type<D>& n);

        template <SizeType D, SizeType DD = D>
        static double exp(const fw::Field<T, DD, ::fw::memory::DataLayout::SoAi>& x, fw::Field<T, DD, ::fw::memory::DataLayout::SoAi>& y, const array_type<D>& n);

        template <SizeType D, SizeType DD = D>
        static double log(const fw::Field<T, DD, ::fw::memory::DataLayout::SoAi>& x, fw::Field<T, DD, ::fw::memory::DataLayout::SoAi>& y, const array_type<D>& n);
    #endif    
#endif
};

#endif
