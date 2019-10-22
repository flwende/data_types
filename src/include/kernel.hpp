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
using TypeZ = float;
using ElementT = ::fw::dataTypes::Tuple<TypeX, TypeY, TypeZ>;
*/
/*
using TypeX = std::uint16_t;
using TypeY = std::uint16_t;
using TypeZ = std::uint16_t;
using ElementT = ::fw::dataTypes::Tuple<TypeX, TypeY, TypeZ>;
*/
/*
using TypeX = std::uint16_t;
using TypeY = double;
using TypeZ = std::uint32_t;
using ElementT = ::fw::dataTypes::Tuple<TypeX, TypeY, TypeZ>;
*/
/*
using TypeX = float;
using TypeY = std::int16_t;
using TypeZ = std::int32_t;
using ElementT = ::fw::dataTypes::Tuple<TypeX, TypeY, TypeZ>;
*/
using ConstElementT = const ElementT;

#if defined(AOS_LAYOUT)
constexpr ::fw::memory::DataLayout Layout = ::fw::memory::DataLayout::AoS;
#elif defined(SOA_LAYOUT)
constexpr ::fw::memory::DataLayout Layout = ::fw::memory::DataLayout::SoA;
#elif defined(SOAI_LAYOUT)
constexpr ::fw::memory::DataLayout Layout = ::fw::memory::DataLayout::SoAi;
#elif defined(AOSOA_LAYOUT)
constexpr ::fw::memory::DataLayout Layout = ::fw::memory::DataLayout::AoSoA;
#endif

template <typename ValueT, SizeT Dimension>
using Field = ::fw::dataTypes::Field<ValueT, Dimension, Layout>;

template <SizeT N>
using SizeArray = ::fw::dataTypes::SizeArray<N>;

#if defined(__CUDACC__)
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

#endif
