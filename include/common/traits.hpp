// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(COMMON_TRAITS_HPP)
#define COMMON_TRAITS_HPP

#include <type_traits>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#include "data_layout.hpp"
#include "memory.hpp"

namespace XXX_NAMESPACE
{
    namespace internal
    {
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //! \brief Test if there exist a proxy type for type T
        //!
        //! Note: specialize this struct if your data type provides a proxy type
        //! 
        //! \tparam T data type
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        template <typename T>
        struct provides_proxy_type
        {
            static constexpr bool value = false;
        };

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //! \brief Query information for data type T (particularly relevant for data layout stuff)
        //!
        //! \tparam T data type
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        template <typename T, data_layout L, typename Enabled = void>
        struct traits
        {
            using type = T;
            using const_type = const T;
            using proxy_type = T;
            using base_pointer = typename std::conditional<std::is_const<T>::value, const typename XXX_NAMESPACE::pointer<T>, typename XXX_NAMESPACE::pointer<T>>::type;
        };

        template <typename T>
        struct traits<T, data_layout::SoA, typename std::enable_if<provides_proxy_type<T>::value>::type>
        {
            using type = T;
            using const_type = const T;
            using proxy_type = typename std::conditional<std::is_const<T>::value, typename T::proxy_type::const_type, typename T::proxy_type>::type;
            using base_pointer = typename std::conditional<std::is_const<T>::value, const typename proxy_type::base_pointer, typename proxy_type::base_pointer>::type;
        };

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //! \brief Get the data type that is the stronger one (larger in size or 'double' over 'float')
        //!
        //! Note: the weaker type is the other one :)
        //!
        //! \tparam T_1 data type
        //! \tparam T_2 data type
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        template <typename T_1, typename T_2, typename Enabled = void>
        struct compare
        {
            using stronger_type = typename std::conditional<(sizeof(T_1) > sizeof(T_2)), T_1, T_2>::type;
            using stronger_type_unqualified = typename std::remove_cv<stronger_type>::type;
            using weaker_type = typename std::conditional<(sizeof(T_1) > sizeof(T_2)), T_2, T_1>::type;
            using weaker_type_unqualified = typename std::remove_cv<weaker_type>::type;
        };

        template <typename T_1, typename T_2>
        struct compare<T_1, T_2, typename std::enable_if<std::is_floating_point<T_1>::value && !std::is_floating_point<T_2>::value>::type>
        {
            using stronger_type = T_1;
            using stronger_type_unqualified = typename std::remove_cv<stronger_type>::type;
            using weaker_type = T_2;
            using weaker_type_unqualified = typename std::remove_cv<weaker_type>::type;
        };

        template <typename T_1, typename T_2>
        struct compare<T_1, T_2, typename std::enable_if<!std::is_floating_point<T_1>::value && std::is_floating_point<T_2>::value>::type>
        {
            using stronger_type = T_2;
            using stronger_type_unqualified = typename std::remove_cv<stronger_type>::type;
            using weaker_type = T_1;
            using weaker_type_unqualified = typename std::remove_cv<weaker_type>::type;
        };
    }
}

#endif