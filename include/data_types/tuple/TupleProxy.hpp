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
#include <integer_sequence/IntegerSequence.hpp>
#include <tuple/Tuple.hpp>
#include <DataTypes.hpp>

namespace XXX_NAMESPACE
{
    namespace dataTypes
    {
        namespace internal
        {
            using ::XXX_NAMESPACE::compileTime::Loop;
            using ::XXX_NAMESPACE::dataTypes::IndexSequence;
            using ::XXX_NAMESPACE::dataTypes::MakeIndexSequence;
            using ::XXX_NAMESPACE::dataTypes::Tuple;
            using ::XXX_NAMESPACE::memory::DataLayout;
            using ::XXX_NAMESPACE::variadic::Pack;

            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            //
            // Forward declarations.
            //
            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            template <typename, SizeT, SizeT, DataLayout>
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
            class TupleProxy : public Tuple<ValueT&...>
            {
                using Base = Tuple<ValueT&...>;
                using TupleT = Tuple<std::decay_t<ValueT>...>;
                using ConstTupleT = const Tuple<std::decay_t<ValueT>...>;
                using RecordT = Record<std::decay_t<ValueT>&...>;
                using RecordConstT = Record<const std::decay_t<ValueT>&...>;

                // Member types must be fundamental and non-void.
                static_assert(!Pack<ValueT...>::IsVoid(), "error: non-void parameter types assumed.");

                // Friend declarations.
                template <typename, SizeT, SizeT, DataLayout>
                friend class Accessor;

              private:
              /*
                //!
                //! \brief Constructor.
                //!
                //! This constructor unpacks the `Record`'s elements and forwards them to the base class constructor.
                //!
                //! \tparam I an `IndexSequence` used for unpacking the tuple argument
                //! \param record a `Record` instance holding references to memory that is associated with the members of the base class
                //! \param unnamed used for template parameter deduction
                //!
                template <SizeT... I>
                HOST_VERSION CUDA_DEVICE_VERSION TupleProxy(RecordT&& record, IndexSequence<I...>) : Base(Get<I>(record)...)
                {
                }

                template <SizeT... I>
                HOST_VERSION CUDA_DEVICE_VERSION TupleProxy(RecordConstT&& record, IndexSequence<I...>) : Base(Get<I>(record)...)
                {
                }
                */
/*
                //!
                //! \brief Constructor.
                //!
                //! Construct a `TupleProxy` from a const `Tuple` instance.
                //! Note: this constructor is not public to avoid incorrect use, e.g. binding to temporaries.
                //!
                //! \param tuple a `Tuple` instance whose data members become the pointees for this `TupleProxy`
                //! \param unnamed used for template parameter deduction
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                TupleProxy(ConstTupleT& tuple) : Base(RecordConstT{tuple.data}) 
                {
                    static_assert(Pack<ValueT...>::IsFundamental(), "error: fundamental parameter types assumed.");
                }
*/
/*
                //!
                //! \brief Constructor.
                //!
                //! Construct a `TupleProxy` from a `Record` instance with const references.
                //! Note: this constructor is not public to avoid incorrect use, e.g. binding to temporaries.
                //!
                //! \tparam I an `IndexSequence` used for unpacking the tuple argument
                //! \param record a `Record` instance holding references to memory that is associated with the members of the base class
                //! \param unnamed used for template parameter deduction
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                TupleProxy(RecordConstT record) : TupleProxy(std::move(record), MakeIndexSequence<sizeof...(ValueT)>())
                {
                    static_assert(Pack<ValueT...>::IsFundamental(), "error: fundamental parameter types assumed.");
                }
*/
                HOST_VERSION
                CUDA_DEVICE_VERSION
                TupleProxy(RecordConstT record) : Base(record)
                {
                    static_assert(Pack<ValueT...>::IsFundamental(), "error: fundamental parameter types assumed.");
                }

                template <typename T>
                static constexpr bool IsRecordOrTupleOrProxy = 
                    ::XXX_NAMESPACE::internal::IsRecord<std::decay_t<T>>::value ||
                    ::XXX_NAMESPACE::internal::IsTuple<std::decay_t<T>>::value ||
                    ::XXX_NAMESPACE::internal::IsProxy<std::decay_t<T>>::value;

              public:
                using ConstT = const TupleProxy<const ValueT...>;
                /*
                //!
                //! \brief Constructor.
                //!
                //! Construct a `TupleProxy` from a `Tuple` instance.
                //!
                //! \param tuple a `Tuple` instance whose data members become the pointees for this `TupleProxy`
                //! \param unnamed used for template parameter deduction
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                TupleProxy(TupleT& tuple) : Base(RecordT{tuple.data}) 
                {
                    static_assert(Pack<ValueT...>::IsFundamental(), "error: fundamental parameter types assumed.");
                }
*/
/*
                //!
                //! \brief Constructor.
                //!
                //! Create a `TupleProxy` from a tuple of reference to memory.
                //!
                //! \param record a `Record` instance holding references to memory that is associated with the members of the base class
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                TupleProxy(RecordT record) : TupleProxy(std::move(record), MakeIndexSequence<sizeof...(ValueT)>())
                {
                    static_assert(Pack<ValueT...>::IsFundamental(), "error: fundamental parameter types assumed.");
                }
*/
                HOST_VERSION
                CUDA_DEVICE_VERSION
                TupleProxy(RecordT record) : Base(record)
                {
                    static_assert(Pack<ValueT...>::IsFundamental(), "error: fundamental parameter types assumed.");
                }

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
                //HOST_VERSION CUDA_DEVICE_VERSION inline auto operator=(const Tuple<T...>& tuple) -> TupleProxy&
                HOST_VERSION CUDA_DEVICE_VERSION inline auto operator=(Tuple<T...>&& tuple) -> TupleProxy&
                {
                    static_assert(sizeof...(T) == sizeof...(ValueT), "error: parameter lists have different size.");
                    static_assert(Pack<T...>::template IsConvertibleTo<ValueT...>(), "error: types are not convertible.");

                    Loop<sizeof...(ValueT)>::Execute([&tuple, this](const auto I) { Get<I>(*this) = Get<I>(tuple); });

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
                //template <typename T, typename EnableType = std::enable_if_t<std::is_fundamental<T>::value>>
                template <typename T, typename std::enable_if_t<!IsRecordOrTupleOrProxy<T>, int> = 0>
                HOST_VERSION CUDA_DEVICE_VERSION inline auto operator=(const T value) -> TupleProxy&
                {
                    static_assert(Pack<ValueT...>::template IsConvertibleFrom<T>(), "error: types are not convertible.");

                    Loop<sizeof...(ValueT)>::Execute([value, this](const auto I) { Get<I>(*this) = value; });

                    return *this;
                }
            };

            /*
            template <typename... ValueT>
            class TupleProxy : public Tuple<ValueT&...>
            {
                using Base = Tuple<ValueT&...>;
                using TupleT = Tuple<std::decay_t<ValueT>...>;
                using ConstTupleT = const Tuple<std::decay_t<ValueT>...>;
                using RecordT = Record<std::decay_t<ValueT>&...>;
                using RecordConstT = Record<const std::decay_t<ValueT>&...>;

                // Member types must be fundamental and non-void.
                static_assert(!Pack<ValueT...>::IsVoid(), "error: non-void parameter types assumed.");

                // Friend declarations.
                template <typename, SizeT, SizeT, DataLayout>
                friend class Accessor;

                template <typename T>
                static constexpr bool IsRecordOrTupleOrProxy = 
                    ::XXX_NAMESPACE::internal::IsRecord<std::decay_t<T>>::value ||
                    ::XXX_NAMESPACE::internal::IsTuple<std::decay_t<T>>::value ||
                    ::XXX_NAMESPACE::internal::IsProxy<std::decay_t<T>>::value;

            public:
                HOST_VERSION
                CUDA_DEVICE_VERSION 
                constexpr TupleProxy(::XXX_NAMESPACE::dataTypes::Tuple<ValueT...>& tuple) : Base(tuple.data) {}

                template <typename T, typename std::enable_if_t<!IsRecordOrTupleOrProxy<T>, int> = 0>
                HOST_VERSION
                CUDA_DEVICE_VERSION 
                inline constexpr auto operator=(T&& value) -> TupleProxy&
                {
                    static_assert(Pack<ValueT...>::template IsConvertibleFrom<T>(), "error: types are not convertible.");

                    Loop<sizeof...(ValueT)>::Execute([value, this](const auto I) { Get<I>(*this) = value; });

                    return *this;
                }
                

                using ConstT = const TupleProxy<const ValueT...>;
            };
            */
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
            //! \brief Specialization of the `Func` data structure for the `TupleProxy` type.
            //!
            template <typename... ValueT>
            struct Func<TupleProxy<ValueT...>> : public Func<Tuple<ValueT...>>
            {
            };
        }
    }
} // namespace XXX_NAMESPACE

#endif
