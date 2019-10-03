// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(COMMON_DATA_TYPES_HPP)
#define COMMON_DATA_TYPES_HPP

#include <cstdint>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

namespace XXX_NAMESPACE
{
    using SizeT = std::size_t;
    using RealT = float;

    namespace dataTypes
    {
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //!
        //! \brief Get the data type that is the stronger one (larger in size or 'double' over 'float')
        //!
        //! Note: the weaker type is the other one :)
        //!
        //! \tparam T_1 some type
        //! \tparam T_2 some other type
        //!
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        template <typename T_1, typename T_2, typename Enabled = void>
        struct Compare
        {
            using StrongerT = std::conditional_t<(sizeof(T_1) > sizeof(T_2)), T_1, T_2>;
            using UnqualifiedStrongerT = std::decay_t<StrongerT>;
            using WeakerT = std::conditional_t<(sizeof(T_1) > sizeof(T_2)), T_2, T_1>;
            using UnqualifiedWeakerT = std::decay_t<WeakerT>;
        };

        template <typename T_1, typename T_2>
        struct Compare<T_1, T_2, std::enable_if_t<std::is_floating_point<T_1>::value && !std::is_floating_point<T_2>::value>>
        {
            using StrongerT = T_1;
            using UnqualifiedStrongerT = std::decay_t<StrongerT>;
            using WeakerT = T_2;
            using UnqualifiedWeakerT = std::decay_t<WeakerT>;
        };

        template <typename T_1, typename T_2>
        struct Compare<T_1, T_2, std::enable_if_t<!std::is_floating_point<T_1>::value && std::is_floating_point<T_2>::value>>
        {
            using StrongerT = T_2;
            using UnqualifiedStrongerT = std::decay_t<StrongerT>;
            using WeakerT = T_1;
            using UnqualifiedWeakerT = std::decay_t<WeakerT>;
        };
    }
}

#include <data_types/array/Array.hpp>

#endif