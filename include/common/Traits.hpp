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

#include <common/DataLayout.hpp>

namespace XXX_NAMESPACE
{
    // forward declaration: definition in common/memory.hpp
    template <typename ...T>
    class pointer;

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
        template <typename T, ::XXX_NAMESPACE::memory::DataLayout L, typename Enabled = void>
        struct traits
        {
            using Type = T;
            using ConstType = const T;
            using ProxyType = T;
            using BasePointerType = typename XXX_NAMESPACE::pointer<T>;
        };

        template <typename T, ::XXX_NAMESPACE::memory::DataLayout L>
        struct traits<T, L, std::enable_if_t<(L != ::XXX_NAMESPACE::memory::DataLayout::AoS && provides_proxy_type<T>::value)>>
        {
            using Type = T;
            using ConstType = const T;
            using ProxyType = std::conditional_t<std::is_const<T>::value, typename T::ProxyType::ConstType, typename T::ProxyType>;
            using BasePointerType = typename ProxyType::BasePointerType;
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
            using stronger_type = std::conditional_t<(sizeof(T_1) > sizeof(T_2)), T_1, T_2>;
            using stronger_type_unqualified = std::decay_t<stronger_type>;
            using weaker_type = std::conditional_t<(sizeof(T_1) > sizeof(T_2)), T_2, T_1>;
            using weaker_type_unqualified = std::decay_t<weaker_type>;
        };

        template <typename T_1, typename T_2>
        struct compare<T_1, T_2, std::enable_if_t<std::is_floating_point<T_1>::value && !std::is_floating_point<T_2>::value>>
        {
            using stronger_type = T_1;
            using stronger_type_unqualified = std::decay_t<stronger_type>;
            using weaker_type = T_2;
            using weaker_type_unqualified = std::decay_t<weaker_type>;
        };

        template <typename T_1, typename T_2>
        struct compare<T_1, T_2, std::enable_if_t<!std::is_floating_point<T_1>::value && std::is_floating_point<T_2>::value>>
        {
            using stronger_type = T_2;
            using stronger_type_unqualified = std::decay_t<stronger_type>;
            using weaker_type = T_1;
            using weaker_type_unqualified = std::decay_t<weaker_type>;
        };
    }
}

#endif