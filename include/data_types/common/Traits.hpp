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
    namespace internal
    {
        using ::XXX_NAMESPACE::memory::DataLayout;
        using ::XXX_NAMESPACE::memory::Pointer;

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

        template <typename T>
        struct IsProxy
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
        template <typename T, DataLayout Layout, typename Enabled = void>
        struct Traits
        {
            using ConstT = const T;
            using Proxy = T;
            using BasePointer = Pointer<1, T>;
        };

        // Helper struct.
        namespace
        {
            using ::XXX_NAMESPACE::memory::MultiPointer;
            using ::XXX_NAMESPACE::variadic::Pack;

            template <typename Proxy, DataLayout Layout>
            struct GetBasePointer
            {
                using Type = typename Proxy::BasePointer;
            };

            template <DataLayout Layout, template <typename...> typename Proxy, typename ...T>
            struct GetBasePointer<Proxy<T...>, Layout>
            {
                static constexpr bool IsHomogeneous = Pack<T...>::SameSize();

                using Type = std::conditional_t<IsHomogeneous, 
                    std::conditional_t<Layout == DataLayout::AoSoA, Pointer<32, T...>, Pointer<1, T...>>, 
                    std::conditional_t<Layout == DataLayout::AoSoA, MultiPointer<32, T...>, MultiPointer<1, T...>>>;
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
        template <typename T, DataLayout Layout>
        struct Traits<T, Layout, std::enable_if_t<(Layout != DataLayout::AoS && ProvidesProxy<T>::value)>>
        {
            using ConstT = const T;
            using Proxy = std::conditional_t<std::is_const<T>::value, typename T::Proxy::ConstT, typename T::Proxy>;
            using BasePointer = typename GetBasePointer<std::decay_t<Proxy>, Layout>::Type;
        };
    } // namespace internal
} // namespace XXX_NAMESPACE

#endif