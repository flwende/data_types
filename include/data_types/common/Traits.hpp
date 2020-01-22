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
#include <auxiliary/Pack.hpp>
#include <common/DataLayout.hpp>
#include <common/Memory.hpp>
#include <DataTypes.hpp>

namespace XXX_NAMESPACE
{
    namespace internal
    {
        using ::XXX_NAMESPACE::dataTypes::SizeT;
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
            using BasePointerValueT = typename BasePointer::ValueT;
            
            static constexpr SizeT BlockingFactor = 1;
            static constexpr SizeT PaddingFactor = 1;
        };

        // Helper struct.
        namespace
        {
            using ::XXX_NAMESPACE::memory::MultiPointer;
            using ::XXX_NAMESPACE::variadic::Pack;

            template <DataLayout Layout>
            struct GetBlockingFactor
            {
                static constexpr SizeT value = 1;
            };

            template <>
            struct GetBlockingFactor<DataLayout::AoSoA>
            {
                static constexpr SizeT value = 32;
            };

            template <typename Proxy, DataLayout Layout>
            struct GetBasePointer
            {
                using Type = typename Proxy::BasePointer;
            };

            template <DataLayout Layout, template <typename...> typename Proxy, typename ...T>
            struct GetBasePointer<Proxy<T...>, Layout>
            {
                static constexpr bool IsHomogeneous = Pack<T...>::SameSize();
                static constexpr SizeT BlockingFactor = GetBlockingFactor<Layout>::value;

                using Type = std::conditional_t<IsHomogeneous, Pointer<BlockingFactor, T...>,  MultiPointer<BlockingFactor, T...>>;
            };

            struct Dummy
            {
                static constexpr SizeT RecordPaddingFactor = 1;
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
            using BasePointerValueT = typename BasePointer::ValueT;

            static constexpr SizeT BlockingFactor = GetBlockingFactor<Layout>::value;
            static constexpr SizeT PaddingFactor = std::conditional_t<BasePointer::IsHomogeneous, Dummy, BasePointer>::RecordPaddingFactor;
        };
    } // namespace internal
} // namespace XXX_NAMESPACE

#endif