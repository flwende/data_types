// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(TRAITS_TRAITS_HPP)
#define TRAITS_TRAITS_HPP

#include <type_traits>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#include "../misc/misc_memory.hpp"

namespace XXX_NAMESPACE
{
    namespace internal
    {
        template <typename T>
        struct provides_proxy_type
        {
            static constexpr bool value = false;
        };

        template <typename T, data_layout L, typename Enabled = void>
        struct traits
        {
            using type = T;
            using const_type = const T;
            using proxy_type = T;

            using memory = typename std::conditional<std::is_const<T>::value, const typename MISC_NAMESPACE::memory<T>, typename MISC_NAMESPACE::memory<T>>::type;
        };

        template <typename T>
        struct traits<T, data_layout::SoA, typename std::enable_if<provides_proxy_type<T>::value>::type>
        {
            using type = T;
            using const_type = const T;
            using proxy_type = typename std::conditional<std::is_const<T>::value, typename T::proxy_type::const_type, typename T::proxy_type>::type;

            using memory = typename std::conditional<std::is_const<T>::value, const typename proxy_type::memory, typename proxy_type::memory>::type;
        };

        template <typename T_1, typename T_2, typename Enabled = void>
        struct compare
        {
            using stronger_type = typename std::conditional<(sizeof(T_1) > sizeof(T_2)), T_1, T_2>::type;
        };

        template <typename T_1, typename T_2>
        struct compare<T_1, T_2, typename std::enable_if<std::is_floating_point<T_1>::value && !std::is_floating_point<T_2>::value>::type>
        {
            using stronger_type = T_1;
        };

        template <typename T_1, typename T_2>
        struct compare<T_1, T_2, typename std::enable_if<!std::is_floating_point<T_1>::value && std::is_floating_point<T_2>::value>::type>
        {
            using stronger_type = T_2;
        };
    }
}

#endif