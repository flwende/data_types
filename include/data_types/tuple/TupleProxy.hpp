// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(DATA_TYPES_TUPLE_TUPLE_PROXY_HPP)
#define DATA_TYPES_TUPLE_TUPLE_PROXY_HPP

#include <tuple>
#include <type_traits>
#include <utility>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#include <auxiliary/Loop.hpp>
#include <auxiliary/Pack.hpp>
#include <common/DataLayout.hpp>
#include <common/Memory.hpp>
#include <tuple/Tuple.hpp>
#include <DataTypes.hpp>

namespace XXX_NAMESPACE
{
    namespace dataTypes
    {
        namespace internal
        {
            using ::XXX_NAMESPACE::compileTime::Loop;
            using ::XXX_NAMESPACE::dataTypes::Tuple;
            using ::XXX_NAMESPACE::memory::DataLayout;
            using ::XXX_NAMESPACE::variadic::Pack;

            //!
            //! \brief A proxy type for `Tuple`s.
            //!
            //! Unlike the `Tuple` type, the proxy has reference-valued members so that non-contiguous
            //! memory can be accessed while provoding the same functionality as the actual `Tuple` type.
            //! This can be used for the implementation of non-AoS data layouts.
            //!
            //! Note: all functionality of the `Tuple` type is inherited and the proxy type behaves like a `Tuple`.
            //!
            template <typename... ValueT>
            class TupleProxy : public Tuple<ValueT&...>
            {
                using Base = Tuple<ValueT&...>;
                using RecordT = Record<std::decay_t<ValueT>&...>;
                using RecordConstT = Record<const std::decay_t<ValueT>&...>;
                using ConstT = const TupleProxy<const ValueT...>;

                // Member types must be non-void.
                static_assert(!Pack<ValueT...>::IsVoid(), "error: non-void parameter types assumed.");

                //! @{
                //! Friend declarations.
                //!
                template <typename, SizeT, SizeT, DataLayout>
                friend class Accessor;
                template <typename, DataLayout, typename>
                friend struct ::XXX_NAMESPACE::internal::Traits;
                //! @}

              protected:
                //!
                //! This constructor can take `Records` that hold references to temporaries.
                //! It should be used only internally or by friends.
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleProxy(RecordConstT&& record) : Base(record)
                {
                    static_assert(Pack<ValueT...>::IsFundamental(), "error: fundamental parameter types assumed.");
                }

              public:
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleProxy(RecordT&& record) : Base(record)
                {
                    static_assert(Pack<ValueT...>::IsFundamental(), "error: fundamental parameter types assumed.");
                }

                //! @{
                //! \brief Assignment operator.
                //!
                //! The assignment operators defined in the base class (if there are any) are non-virtual.
                //! We thus need to add a version that returns a reference to this `TupleProxy` type.
                //!
                //! Note: the base class assignment operators can handle `TupleProxy` types as arguments due to inheritence.
                //! Note: we use the base class type as argument type as it covers both the `Tuple` and the `TupleProxy` case.
                //!
                template <typename... T>
                HOST_VERSION 
                CUDA_DEVICE_VERSION 
                inline constexpr auto operator=(Tuple<T...>&& tuple) -> TupleProxy&
                {
                    static_assert(sizeof...(T) == sizeof...(ValueT), "error: parameter lists have different size.");
                    static_assert(Pack<T...>::template IsConvertibleTo<ValueT...>(), "error: types are not convertible.");

                    Loop<sizeof...(ValueT)>::Execute([&tuple, this](const auto I) { Get<I>(*this) = Get<I>(tuple); });

                    return *this;
                }

                template <typename T, typename std::enable_if_t<!Base::template IsRecordOrTupleOrProxy<T>, int> = 0>
                HOST_VERSION 
                CUDA_DEVICE_VERSION 
                inline constexpr auto operator=(const T value) -> TupleProxy&
                {
                    static_assert(Pack<ValueT...>::template IsConvertibleFrom<T>(), "error: types are not convertible.");

                    Loop<sizeof...(ValueT)>::Execute([value, this](const auto I) { Get<I>(*this) = value; });

                    return *this;
                }
                //! @}
            };
        } // namespace internal
    }  // namespace dataTypes

    namespace internal
    {
        template <typename... T>
        struct IsProxy<::XXX_NAMESPACE::dataTypes::internal::TupleProxy<T...>>
        {
            static constexpr bool value = true;
        };
    }

    namespace math
    {
        namespace internal
        {
            using ::XXX_NAMESPACE::dataTypes::internal::TupleProxy;

            //!
            //! Specialization of the `Func` data structure for the `TupleProxy` type.
            //!
            template <typename... ValueT>
            struct Func<TupleProxy<ValueT...>> : public Func<Tuple<ValueT...>> {};
        }
    }
} // namespace XXX_NAMESPACE

#endif