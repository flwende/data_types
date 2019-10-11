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

#include <auxiliary/Template.hpp>
#include <common/DataLayout.hpp>
#include <common/Memory.hpp>
#include <data_types/DataTypes.hpp>
#include <data_types/integer_sequence/IntegerSequence.hpp>
#include <data_types/tuple/Tuple.hpp>

namespace XXX_NAMESPACE
{
    namespace dataTypes
    {
        namespace internal
        {
            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            //
            // Forward declarations.
            //
            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            template <typename, SizeT, SizeT, ::XXX_NAMESPACE::memory::DataLayout>
            class Accessor;

            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            //!
            //! \brief A proxy type for `Tuple`s.
            //!
            //! Unlike the `Tuple` type, the proxy has reference-valued members so that non-contiguous
            //! memory can be accessed while provoding the same functionality as the actual `Tuple` type.
            //! This can be used for the implementation of non-AoS data layouts.
            //!
            //! Note: all functionality of the `Tuple` type is inherited and the proxy type behaves like a `Tuple`.
            //!
            //! \tparam ValueT a type parameter list defining the member types
            //!
            template <typename... ValueT>
            class TupleProxy : public ::XXX_NAMESPACE::dataTypes::Tuple<ValueT&...>
            {
                using Base = ::XXX_NAMESPACE::dataTypes::Tuple<ValueT&...>;

                // Member types must be fundamental and non-void.
                static_assert(::XXX_NAMESPACE::variadic::Pack<ValueT...>::IsFundamental(), "error: fundamental parameter types assumed.");
                static_assert(!::XXX_NAMESPACE::variadic::Pack<ValueT...>::IsVoid(), "error: non-void parameter types assumed.");

                // Friend declarations.
                template <typename, SizeT, SizeT, ::XXX_NAMESPACE::memory::DataLayout>
                friend class ::XXX_NAMESPACE::dataTypes::internal::Accessor;

              //private:
              public:
                //!
                //! \brief Constructor.
                //!
                //! This constructor unpacks the references in the tuple argument and forwards them to the base class constructor.
                //!
                //! \tparam I an `IndexSequence` used for unpacking the tuple argument
                //! \param tuple a tuple holding references to memory that is associated with the members of the base class
                //! \param unnamed used for template parameter deduction
                //!
                template <SizeT... I>
                HOST_VERSION CUDA_DEVICE_VERSION TupleProxy(std::tuple<ValueT&...>&& tuple, ::XXX_NAMESPACE::dataTypes::IndexSequence<I...>) : Base(std::get<I>(tuple)...)
                {
                }

                //!
                //! \brief Constructor.
                //!
                //! Create a `TupleProxy` from a tuple of reference to memory.
                //!
                //! \param tuple a tuple holding references to memory that is associated with the members of the base class
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                TupleProxy(std::tuple<ValueT&...> tuple) : TupleProxy(std::move(tuple), ::XXX_NAMESPACE::dataTypes::MakeIndexSequence<sizeof...(ValueT)>()) {}

              public:
                using ConstT = const TupleProxy<const ValueT...>;
                using BasePointer = std::conditional_t<::XXX_NAMESPACE::variadic::Pack<ValueT...>::SameSize(), ::XXX_NAMESPACE::memory::Pointer<ValueT...>, ::XXX_NAMESPACE::memory::MultiPointer<ValueT...>>;

                //!
                //! \brief Assignment operator.
                //!
                //! The assignment operators defined in the base class (if there are any) are non-virtual.
                //! We thus need to add a version that returns a reference to this `TupleProxy` type.
                //!
                //! Note: the base class assignment operators can handle `TupleProxy` types as arguments due to inheritence.
                //! Note: we use the base class type as argument type as it covers both the `Tuple` and the `TupleProxy` case.
                //!
                //! \tparam T a variadic list of type parameters
                //! \param tuple a `Tuple` (or `TupleProxy`) instance
                //! \return a reference to this `TupleProxy` instance
                //!
                template <typename... T>
                HOST_VERSION CUDA_DEVICE_VERSION inline auto operator=(const ::XXX_NAMESPACE::dataTypes::Tuple<T...>& tuple) -> TupleProxy&
                {
                    static_assert(sizeof...(T) == sizeof...(ValueT), "error: parameter lists have different size.");
                    static_assert(::XXX_NAMESPACE::variadic::Pack<T...>::template IsConvertibleTo<ValueT...>(), "error: types are not convertible.");

                    ::XXX_NAMESPACE::compileTime::Loop<sizeof...(ValueT)>::Execute([&tuple, this](const auto I) { Get<I>(*this) = Get<I>(tuple); });

                    return *this;
                }

                //!
                //! \brief Assignment operator.
                //!
                //! The assignment operators defined in the base class (if there are any) are non-virtual.
                //! We thus need to add a version that returns a reference to this `TupleProxy` type.
                //!
                //! Note: the base class assignment operators can handle `TupleProxy` types as arguments due to inheritence.
                //!
                //! \tparam T the type of the value to be assigned
                //! \param value the value to be assigned
                //! \return a reference to this `TupleProxy` instance
                //!
                template <typename T, typename EnableType = std::enable_if_t<std::is_fundamental<T>::value>>
                HOST_VERSION CUDA_DEVICE_VERSION inline auto operator=(const T value) -> TupleProxy&
                {
                    static_assert(::XXX_NAMESPACE::variadic::Pack<ValueT...>::template IsConvertibleFrom<T>(), "error: types are not convertible.");

                    ::XXX_NAMESPACE::compileTime::Loop<sizeof...(ValueT)>::Execute([value, this](const auto I) { Get<I>(*this) = value; });

                    return *this;
                }
            };
        } // namespace internal
    }  // namespace dataTypes
} // namespace XXX_NAMESPACE

#endif