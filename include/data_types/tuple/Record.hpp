// Copyright (c) 2020 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(DATA_TYPES_TUPLE_RECORD_HPP)
#define DATA_TYPES_TUPLE_RECORD_HPP

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#include <auxiliary/Template.hpp>
#include <integer_sequence/IntegerSequence.hpp>
#include <platform/Target.hpp>
#include <tuple/Get.hpp>

namespace XXX_NAMESPACE
{
    namespace dataTypes::internal
    {
        using ::XXX_NAMESPACE::dataTypes::IndexSequence;
        using ::XXX_NAMESPACE::dataTypes::IndexSequenceT;
        using ::XXX_NAMESPACE::dataTypes::MakeIndexSequence;
        using ::XXX_NAMESPACE::variadic::Pack;

        // Forward declarations.
        template <typename... ValueT>
        class ReverseRecord;

        template <typename... ValueT>
        class Record;

        //!
        //! \brief A Helper type for the construction of a record type with non-reverse order of the members.
        //!
        //! The definition uses an index sequence for the parameter and argument inversion.
        //!
        //! \tparam IndexSequenceT an sequence of indices from 0 to the number of template parameters in the variadic list
        //! \tparam ValueT a variadic list of type parameters
        //!
        template <typename IndexSequenceT, typename... ValueT>
        struct RecordHelperImplementation;

        template <typename... ValueT, SizeT... I>
        struct RecordHelperImplementation<IndexSequence<I...>, ValueT...>
        {
            static constexpr SizeT N = sizeof...(ValueT);

            static_assert(N == sizeof...(I), "error: type parameter count does not match non-type parameter count.");

            using Type = ReverseRecord<typename Pack<ValueT...>::template Type<(N - 1) - I>...>;

            //!@{
            //!
            //! This function inverts the order of the arguments and forwards them to a `ReverseRecord` type with
            //! inverted template parameter list.
            //!
            template <typename... T>
            HOST_VERSION 
            CUDA_DEVICE_VERSION 
            static inline constexpr auto CreateReverseRecord(T&&... values) -> Type
            {
                return {Pack<decltype(values)...>::template Value<(N - 1) - I>(values...)...};
            }

            template <typename... T>
            HOST_VERSION 
            CUDA_DEVICE_VERSION 
            static inline constexpr auto CreateReverseRecord(Record<T...>& record) -> Type
            {
                return {Pack<T...>::template Value<(N - 1) - I>(Get<I>(record)...)...};
            }

            template <typename... T>
            HOST_VERSION 
            CUDA_DEVICE_VERSION 
            static inline constexpr auto CreateReverseRecord(const Record<T...>& record) -> Type
            {
                return {Pack<T...>::template Value<(N - 1) - I>(Get<I>(record)...)...};
            }
            //!@}
        };

        template <typename... ValueT>
        using RecordHelper = RecordHelperImplementation<IndexSequenceT<sizeof...(ValueT)>, ValueT...>;

        //!
        //! \brief A record type with reverse order of the members (recursion).
        //!
        //! The order of the members in memory is reversed:
        //! `ReverseRecord<int, float> {..}` is equivalent to `class {float member_1; int member_2;..}`.
        //!
        //! The recursion level follows from the the number of parameters removed from the original parameter list.
        //! Each level of recursion takes the front most (current) parameter from the list and inherits from
        //! a `ReverseRecord` data using the remaining parameter list.
        //! Accessing the different members of the record type can happen through (implicit)
        //! type casting to the base class. recursively.
        //!
        //! Note: This definition has at least two parameters in the list.
        //!
        //! \tparam ValueT the type of the current member
        //! \tparam Tail the remaining parameter list
        //!
        template <typename ValueT, typename... Tail>
        class ReverseRecord<ValueT, Tail...> : public ReverseRecord<Tail...>
        {
            using Base = ReverseRecord<Tail...>;

            //! @{
            //!
            //! Friend declarations.
            //!
            template <typename, typename...>
            friend struct RecordHelperImplementation;
            template <typename...>
            friend class ReverseRecord;
            template <typename...>
            friend class Record;
            //! @}

            HOST_VERSION
            CUDA_DEVICE_VERSION
            constexpr ReverseRecord() : value{} {};

            template <typename T>
            HOST_VERSION 
            CUDA_DEVICE_VERSION 
            constexpr ReverseRecord(T&& value) : Base(value), value(value)
            {
                static_assert(std::is_convertible<T, ValueT>::value, "error: types are not convertible.");
            }

            template <typename T>
            HOST_VERSION 
            CUDA_DEVICE_VERSION 
            constexpr ReverseRecord(const T& value) : Base(value), value(value)
            {
                static_assert(std::is_convertible<T, ValueT>::value, "error: types are not convertible.");
            }

            HOST_VERSION
            CUDA_DEVICE_VERSION
            constexpr ReverseRecord(const ValueT& value, const Tail&... tail) : Base(tail...), value(value) {}

          public:
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline constexpr auto& Get() { return value; }

            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline constexpr const auto& Get() const { return value; }

          protected:
            ValueT value;
        };

        //!
        //! \brief Definition of a record type with no members.
        //!
        template <>
        class ReverseRecord<>
        {
            //! @{
            //!
            //! Friend declaration:
            //! RecordInverter instantiates this class.
            //! ReverseRecord needs access for the recursion.
            //!
            template <typename, typename...>
            friend struct RecordHelperImplementation;
            template <typename...>
            friend class ReverseRecord;
            //! @}

            HOST_VERSION
            CUDA_DEVICE_VERSION
            constexpr ReverseRecord() = default;

            //! @{
            //!
            //! Just needed for the recursion.
            //! These constructors can only be called by friends.
            //!
            template <typename T>
            HOST_VERSION 
            CUDA_DEVICE_VERSION 
            constexpr ReverseRecord(T&&)
            {
            }

            template <typename T>
            HOST_VERSION 
            CUDA_DEVICE_VERSION 
            constexpr ReverseRecord(const T&)
            {
            }
            //! @}
        };

        //!
        //! \brief A record type with non-reverse order of the members.
        //!
        //! The order of the members in memory is NOT reversed:
        //! `Record<int, float> {..}` is equivalent to `class {int member_1; float member_2;..}`.
        //! This record type inherits from a `ReverseRecord` type with inverted template parameter list.
        //!
        //! Note: This definition has at least one parameter in the list.
        //!
        //! \tparam ValueT a parameter list defining the member types
        //!
        template <typename... ValueT>
        class Record : public RecordHelper<ValueT...>::Type
        {
            using Base = typename RecordHelper<ValueT...>::Type;

            //! @{
            //!
            //! Friend declarations: needed for access to `Base` type.
            //!
            template <SizeT Index, typename... T>
            HOST_VERSION CUDA_DEVICE_VERSION friend constexpr auto& Get(Record<T...>&);
            template <SizeT Index, typename... T>
            HOST_VERSION CUDA_DEVICE_VERSION friend constexpr auto& Get(Record<T...>&&);
            template <SizeT Index, typename... T>
            HOST_VERSION CUDA_DEVICE_VERSION friend constexpr const auto& Get(const Record<T...>&);
            template <SizeT Index, typename... T>
            HOST_VERSION CUDA_DEVICE_VERSION friend constexpr const auto& Get(const Record<T...>&&);
            //! @}

            static constexpr SizeT N = sizeof...(ValueT);

          public:
            HOST_VERSION
            CUDA_DEVICE_VERSION
            constexpr Record() = default;

            //! @{
            //!
            //! Generate a `Record` from a single or from multiple values,
            //! or from other `Record`s with the same or different types.
            //!
            template <typename T>
            HOST_VERSION 
            CUDA_DEVICE_VERSION 
            constexpr Record(T&& value) : Base(value) {}

            template <typename... T>
            HOST_VERSION 
            CUDA_DEVICE_VERSION 
            constexpr Record(T&&... values) : Base(RecordHelper<ValueT...>::CreateReverseRecord(values...)) {}

            template <typename... T>
            HOST_VERSION 
            CUDA_DEVICE_VERSION 
            constexpr Record(Record<T...>& other) : Base(RecordHelper<ValueT...>::CreateReverseRecord(other)) {}

            template <typename... T>
            HOST_VERSION 
            CUDA_DEVICE_VERSION 
            constexpr Record(const Record<T...>& other) : Base(RecordHelper<ValueT...>::CreateReverseRecord(other)) {}

            template <typename... T>
            HOST_VERSION 
            CUDA_DEVICE_VERSION 
            constexpr Record(Record<T...>&& other) : Record(std::move(other), MakeIndexSequence<sizeof...(T)>()) {}

          protected:
            template <typename... T, SizeT... I>
            HOST_VERSION 
            CUDA_DEVICE_VERSION 
            constexpr Record(Record<T...>&& other, IndexSequence<I...>) : Record(std::move(Get<I>(other))...) {}
            //! @}
        };

        //!
        //! \brief Definition of a record type with no members.
        //!
        template <>
        class Record<>
        {
          public:
            HOST_VERSION
            CUDA_DEVICE_VERSION
            constexpr Record() = default;
        };
    } // namespace dataTypes::internal
} // namespace XXX_NAMESPACE

#endif