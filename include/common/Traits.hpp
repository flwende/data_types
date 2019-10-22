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

#include <auxiliary/CPPStandard.hpp>
#include <common/DataLayout.hpp>
#include <common/Memory.hpp>

namespace XXX_NAMESPACE
{
    namespace dataTypes
    {
        template <typename ...T>
        class Tuple;

        namespace internal
        {
            template <typename ...T>
            class TupleProxy;
        }
    }

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
        struct ProvidesProxy
        {
            static constexpr bool value = false;
        };

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //!
        //! \brief Query information for data type T (particularly relevant for data layout stuff).
        //!
        //! \tparam T some type
        //! \tparam Layout the data layout
        //! \tparam Enable template parameter used for multi-versioning
        //!
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        template <typename T, ::XXX_NAMESPACE::memory::DataLayout Layout, typename Enabled = void>
        struct Traits
        {
            using ConstT = const T;
            using Proxy = T;
            using BasePointer = typename ::XXX_NAMESPACE::memory::Pointer<0, T>;
        };

        namespace
        {
            template <typename Proxy, ::XXX_NAMESPACE::memory::DataLayout Layout>
            struct GetBasePointer
            {
                using Type = typename Proxy::BasePointer;
            };

            template <::XXX_NAMESPACE::memory::DataLayout Layout, typename ...T>
            struct GetBasePointer<::XXX_NAMESPACE::dataTypes::internal::TupleProxy<T...>, Layout>
            {
                static constexpr bool IsHomogeneous = ::XXX_NAMESPACE::variadic::Pack<T...>::SameSize();

                using Type = std::conditional_t<IsHomogeneous, 
                    std::conditional_t<Layout == ::XXX_NAMESPACE::memory::DataLayout::AoSoA, ::XXX_NAMESPACE::memory::Pointer<32, T...>, ::XXX_NAMESPACE::memory::Pointer<1, T...>>, 
                    ::XXX_NAMESPACE::memory::MultiPointer<T...>>;
            };
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //!
        //! \brief Query information for data type T (particularly relevant for data layout stuff).
        //!
        //! Specialization for non-AoS data layouts.
        //!
        //! \tparam T some type
        //! \tparam Layout the data layout
        //!
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        template <typename T, ::XXX_NAMESPACE::memory::DataLayout Layout>
        struct Traits<T, Layout, std::enable_if_t<(Layout != ::XXX_NAMESPACE::memory::DataLayout::AoS && ProvidesProxy<T>::value)>>
        {
            using ConstT = const T;
            using Proxy = std::conditional_t<std::is_const<T>::value, typename T::Proxy::ConstT, typename T::Proxy>;
            using BasePointer = typename GetBasePointer<std::decay_t<Proxy>, Layout>::Type;
        };
    } // namespace internal
} // namespace XXX_NAMESPACE

#endif