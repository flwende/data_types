// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(KERNEL_HPP)
#define KERNEL_HPP

#include <cstdint>

#include <data_types/DataTypes.hpp>
#include <data_types/field/Field.hpp>
#include <data_types/vec/Vec.hpp>
#include <data_types/tuple/Tuple.hpp>
#include <platform/Target.hpp>

// Data types and layouts.
using SizeT = ::fw::dataTypes::SizeT;
using RealT = ::fw::dataTypes::RealT;

using TypeX = RealT;
using TypeY = RealT;
using TypeZ = RealT;
using ElementT = ::fw::dataTypes::Vec<RealT, 3>;

/*
using TypeX = std::uint32_t;
using TypeY = std::uint32_t;
using TypeZ = std::uint32_t;
using ElementT = ::fw::dataTypes::Vec<std::uint32_t, 3>;
*/
/*
using TypeX = std::uint32_t;
using TypeY = std::uint32_t;
using TypeZ = std::uint32_t;
using ElementT = ::fw::dataTypes::Tuple<TypeX, TypeY, TypeZ>;
*/
/*
using TypeX = std::uint16_t;
using TypeY = std::uint8_t;
using TypeZ = std::uint32_t;
using ElementT = ::fw::dataTypes::Tuple<TypeX, TypeY, TypeZ>;
*/
/*
using TypeX = std::uint16_t;
using TypeY = double;
using TypeZ = std::uint32_t;
using ElementT = ::fw::dataTypes::Tuple<TypeX, TypeY, TypeZ>;
*/
using ConstElementT = const ElementT;

#if defined(AOS_LAYOUT)
constexpr ::fw::memory::DataLayout Layout = ::fw::memory::DataLayout::AoS;
#elif defined(SOA_LAYOUT)
constexpr ::fw::memory::DataLayout Layout = ::fw::memory::DataLayout::SoA;
#elif defined(SOAI_LAYOUT)
constexpr ::fw::memory::DataLayout Layout = ::fw::memory::DataLayout::SoAi;
#endif

template <typename ValueT, SizeT Dimension>
using Field = ::fw::dataTypes::Field<ValueT, Dimension, Layout>;

template <SizeT N>
using SizeArray = ::fw::dataTypes::SizeArray<N>;

#if defined(__CUDACC__)
template <typename ValueT, SizeT Dimension>
using DeviceField = ::fw::dataTypes::internal::Container<ValueT, Dimension, Layout, ::fw::platform::Identifier::GPU_CUDA>;

template <SizeT Dimension>
auto GetGridSize(const SizeArray<Dimension>& size, const dim3& block);

template <>
auto GetGridSize<1>(const SizeArray<1>& size, const dim3& block)
{
    return dim3((size[0] + block.x - 1) / block.x, 1, 1);
}

template <>
auto GetGridSize<2>(const SizeArray<2>& size, const dim3& block)
{
    return dim3((size[0] + block.x - 1) / block.x, (size[1] + block.y - 1) / block.y, 1);
}

template <>
auto GetGridSize<3>(const SizeArray<3>& size, const dim3& block)
{
    return dim3((size[0] + block.x - 1) / block.x, (size[1] + block.y - 1) / block.y, (size[2] + block.z - 1) / block.z);
}
#endif
/*
// prototypes
template <typename T>
struct kernel
{
#if defined(AOS_LAYOUT)
    #if defined(VECTOR_PRODUCT)
        template <SizeT D, SizeT DD = D>
        static double cross(const ::fw::dataTypes::Field<T, DD, ::fw::memory::DataLayout::AoS>& x_1, const ::fw::dataTypes::Field<T, DD, ::fw::memory::DataLayout::AoS>& x_2, ::fw::dataTypes::Field<T, DD, ::fw::memory::DataLayout::AoS>& y, const SizeArray<D>& n);
    #else
        template <SizeT D, SizeT DD = D>
        static double exp(::fw::dataTypes::Field<T, DD, ::fw::memory::DataLayout::AoS>& x, const SizeArray<D>& n);

        template <SizeT D, SizeT DD = D>
        static double log(::fw::dataTypes::Field<T, DD, ::fw::memory::DataLayout::AoS>& x, const SizeArray<D>& n);

        template <SizeT D, SizeT DD = D>
        static double exp(const ::fw::dataTypes::Field<T, DD, ::fw::memory::DataLayout::AoS>& x, ::fw::dataTypes::Field<T, DD, ::fw::memory::DataLayout::AoS>& y, const SizeArray<D>& n);

        template <SizeT D, SizeT DD = D>
        static double log(const ::fw::dataTypes::Field<T, DD, ::fw::memory::DataLayout::AoS>& x, ::fw::dataTypes::Field<T, DD, ::fw::memory::DataLayout::AoS>& y, const SizeArray<D>& n);
    #endif
#elif defined(SOA_LAYOUT)
    #if defined(VECTOR_PRODUCT)
        template <SizeT D, SizeT DD = D>
        static double cross(const ::fw::dataTypes::Field<T, DD, ::fw::memory::DataLayout::SoA>& x_1, const ::fw::dataTypes::Field<T, DD, ::fw::memory::DataLayout::SoA>& x_2, ::fw::dataTypes::Field<T, DD, ::fw::memory::DataLayout::SoA>& y, const SizeArray<D>& n);
    #else
        template <SizeT D, SizeT DD = D>
        static double exp(::fw::dataTypes::Field<T, DD, ::fw::memory::DataLayout::SoA>& x, const SizeArray<D>& n);

        template <SizeT D, SizeT DD = D>
        static double log(::fw::dataTypes::Field<T, DD, ::fw::memory::DataLayout::SoA>& x, const SizeArray<D>& n);

        template <SizeT D, SizeT DD = D>
        static double exp(const ::fw::dataTypes::Field<T, DD, ::fw::memory::DataLayout::SoA>& x, ::fw::dataTypes::Field<T, DD, ::fw::memory::DataLayout::SoA>& y, const SizeArray<D>& n);

        template <SizeT D, SizeT DD = D>
        static double log(const ::fw::dataTypes::Field<T, DD, ::fw::memory::DataLayout::SoA>& x, ::fw::dataTypes::Field<T, DD, ::fw::memory::DataLayout::SoA>& y, const SizeArray<D>& n);
    #endif
#elif defined(SOAI_LAYOUT)
    #if defined(VECTOR_PRODUCT)
        template <SizeT D, SizeT DD = D>
        static double cross(const ::fw::dataTypes::Field<T, DD, ::fw::memory::DataLayout::SoAi>& x_1, const ::fw::dataTypes::Field<T, DD, ::fw::memory::DataLayout::SoAi>& x_2, ::fw::dataTypes::Field<T, DD, ::fw::memory::DataLayout::SoAi>& y, const SizeArray<D>& n);
    #else
        template <SizeT D, SizeT DD = D>
        static double exp(::fw::dataTypes::Field<T, DD, ::fw::memory::DataLayout::SoAi>& x, const SizeArray<D>& n);

        template <SizeT D, SizeT DD = D>
        static double log(::fw::dataTypes::Field<T, DD, ::fw::memory::DataLayout::SoAi>& x, const SizeArray<D>& n);

        template <SizeT D, SizeT DD = D>
        static double exp(const ::fw::dataTypes::Field<T, DD, ::fw::memory::DataLayout::SoAi>& x, ::fw::dataTypes::Field<T, DD, ::fw::memory::DataLayout::SoAi>& y, const SizeArray<D>& n);

        template <SizeT D, SizeT DD = D>
        static double log(const ::fw::dataTypes::Field<T, DD, ::fw::memory::DataLayout::SoAi>& x, ::fw::dataTypes::Field<T, DD, ::fw::memory::DataLayout::SoAi>& y, const SizeArray<D>& n);
    #endif    
#endif
};
*/

#endif
