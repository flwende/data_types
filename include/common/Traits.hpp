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
#include <common/Memory.hpp>

namespace XXX_NAMESPACE
{
    namespace internal
    {
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //!
        //! \brief Test if there exist a proxy type for type T.
        //!
        //! You can specialize this struct if your data type provides a proxy type.
        //!
        //! \tparam T some type
        //!
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        template <typename T>
        struct ProvidesProxyType
        {
            static constexpr bool value = false;
        };

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //!
        //! \brief Query information for data type T (particularly relevant for data layout stuff).
        //!
        //! \tparam T some type
        //! \tparam C_Layout the data layout
        //! \tparam Enable template parameter used for multi-versioning
        //!
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        template <typename T, ::XXX_NAMESPACE::memory::DataLayout C_Layout, typename Enabled = void>
        struct Traits
        {
            using Type = T;
            using ConstType = const T;
            using ProxyType = T;
            using BasePointerType = typename ::XXX_NAMESPACE::memory::Pointer<T>;
        };

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //!
        //! \brief Query information for data type T (particularly relevant for data layout stuff).
        //!
        //! Specialization for non-AoS data layouts.
        //!
        //! \tparam T some type
        //! \tparam C_Layout the data layout
        //!
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        template <typename T, ::XXX_NAMESPACE::memory::DataLayout C_Layout>
        struct Traits<T, C_Layout, std::enable_if_t<(C_Layout != ::XXX_NAMESPACE::memory::DataLayout::AoS && ProvidesProxyType<T>::value)>>
        {
            using Type = T;
            using ConstType = const T;
            using ProxyType = std::conditional_t<std::is_const<T>::value, typename T::ProxyType::ConstType, typename T::ProxyType>;
            using BasePointerType = typename ProxyType::BasePointerType;
        };
    } // namespace internal
} // namespace XXX_NAMESPACE

#endif