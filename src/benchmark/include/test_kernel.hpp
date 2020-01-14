// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(TEST_KERNEL_HPP)
#define TEST_KERNEL_HPP

#include <cstdint>
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
//using element_type = type;

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

using argument_type = typename fw::internal::traits<element_type, layout>::proxy_type;

struct loop
{
    static constexpr auto foreach = [] (auto& field, auto f)
    { 
        const std::size_t nx = field.n[0];

        for (auto it = field.at(0).begin(); it != field.at(2).end(); ++it)
        {
            auto x = (*it).begin();

            #pragma omp simd
            for (std::size_t i = 0; i < nx; ++i)
            {
                f(*x++);
            }
        }
    };
};

#endif