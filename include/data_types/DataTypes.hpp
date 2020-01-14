// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(COMMON_DATA_TYPES_HPP)
#define COMMON_DATA_TYPES_HPP

#include <cstdint>
#include <tuple>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

namespace XXX_NAMESPACE
{
    namespace dataTypes
    {
        using SizeT = std::size_t;
        using RealT = float;

        namespace
        {
            template <typename T_1, typename T_2, typename Enabled = void>
            struct CompareImplementation
            {
                using StrongerT = std::conditional_t<(sizeof(T_1) > sizeof(T_2)), T_1, T_2>;
                using UnqualifiedStrongerT = std::decay_t<StrongerT>;
                using WeakerT = std::conditional_t<(sizeof(T_1) > sizeof(T_2)), T_2, T_1>;
                using UnqualifiedWeakerT = std::decay_t<WeakerT>;
            };

            template <typename T_1, typename T_2>
            struct CompareImplementation<T_1, T_2, std::enable_if_t<std::is_floating_point<T_1>::value && !std::is_floating_point<T_2>::value>>
            {
                using StrongerT = T_1;
                using UnqualifiedStrongerT = std::decay_t<StrongerT>;
                using WeakerT = T_2;
                using UnqualifiedWeakerT = std::decay_t<WeakerT>;
            };

            template <typename T_1, typename T_2>
            struct CompareImplementation<T_1, T_2, std::enable_if_t<!std::is_floating_point<T_1>::value && std::is_floating_point<T_2>::value>>
            {
                using StrongerT = T_2;
                using UnqualifiedStrongerT = std::decay_t<StrongerT>;
                using WeakerT = T_1;
                using UnqualifiedWeakerT = std::decay_t<WeakerT>;
            };
        }

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
        template <typename ...T>
        struct Compare;
        
        template <>
        struct Compare<>
        {
            using StrongerT = void;
            using UnqualifiedStrongerT = void;
            using WeakerT = void;
            using UnqualifiedWeakerT = void;
        };

        template <typename Head, typename ...Tail>
        struct Compare<Head, Tail...>
        {
            using StrongerT = typename Compare<Head, typename Compare<Tail...>::StrongerT>::StrongerT;
            using UnqualifiedStrongerT = std::decay_t<StrongerT>;
            using WeakerT = typename Compare<Head, typename Compare<Tail...>::WeakerT>::WeakerT;
            using UnqualifiedWeakerT = std::decay_t<WeakerT>;
        };

        template <typename T_1, typename T_2>
        struct Compare<T_1, T_2>
        {
            using StrongerT = typename CompareImplementation<T_1, T_2>::StrongerT;
            using UnqualifiedStrongerT = std::decay_t<StrongerT>;
            using WeakerT = typename CompareImplementation<T_1, T_2>::WeakerT;
            using UnqualifiedWeakerT = std::decay_t<WeakerT>;
        };

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //!
        //! \brief Build type with homogeneous template parameter list.
        //!
        //! \tparam T the actual templated type 
        //! \tparam ParamT the type of the parameters
        //! \tparam N the number of parameters 
        //! \tparam List the current parameter list
        //!
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        namespace
        {
            template <template <typename...> class T, typename ParamT, SizeT N, typename ...List>
            struct BuilderImplementation
            {
                using Type = typename BuilderImplementation<T, ParamT, N - 1, ParamT, List...>::Type;
            };

            template <template <typename...> class T, typename ParamT, typename ...List>
            struct BuilderImplementation<T, ParamT, 0, List...>
            {
                using Type = T<List...>;
            };
        }

        template <template <typename...> class T, typename ParamT, SizeT N>
        using Builder = typename BuilderImplementation<T, ParamT, N>::Type;
    }
}

#include <array/Array.hpp>

#endif