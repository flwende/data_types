// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(DATA_TYPES_TUPLE_RECORD_HPP)
#define DATA_TYPES_TUPLE_RECORD_HPP

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#include <auxiliary/Template.hpp>
#include <data_types/integer_sequence/IntegerSequence.hpp>
#include <data_types/tuple/Get.hpp>
#include <platform/Target.hpp>

namespace XXX_NAMESPACE
{
    namespace dataTypes
    {
        namespace internal
        {
            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            //!
            //! \brief A record type with reverse order of the members.
            //!
            //! The order of the members in memory is reversed:
            //! `ReverseRecord<int, float> {..}` is equivalent to `class {float member_1; int member_2;..}`.
            //!
            //! \tparam ValueT a parameter list defining the member types
            //!
            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            template <typename... ValueT>
            class ReverseRecord;

            //!
            //! \brief Definition of a record type with reverse order of the members (recursive).
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

              public:
                //!
                //! \brief Standard constructor.
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr ReverseRecord() : value{} {};

                //!
                //! \brief Constructor.
                //!
                //! Assign some `value` to the current member.
                //! All members are assigned the same value.
                //!
                //! \tparam T the type of the value to be assigned (can be different from the member type, but must be convertible)
                //! \param value the value to be assigned to all members
                //!
                template <typename T>
                HOST_VERSION CUDA_DEVICE_VERSION constexpr ReverseRecord(T value) : Base(value), value(value)
                {
                    static_assert(::XXX_NAMESPACE::variadic::Pack<T, ValueT>::IsConvertible(), "error: types are not convertible.");
                }

                //!
                //! \brief Constructor.
                //!
                //! Assign some `value` to the current member.
                //!
                //! \param value the value to be assigned to the current member
                //! \param tail these values are forwarded to the base class
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr ReverseRecord(ValueT value, Tail... tail) : Base(tail...), value(value) {}

                //!
                //! \brief Get the value of the current member.
                //!
                //! \return a reference to the current member
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline constexpr auto& Get() { return value; }

                //!
                //! \brief Get the value of the current member.
                //!
                //! \return a const reference to the current member
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline constexpr const auto& Get() const { return value; }

              protected:
                ValueT value;
            };

            //!
            //! \brief Definition of a record type with reverse order of the members (recursion anchor).
            //!
            //! Note: This definition has a single parameter in the list.
            //!
            //! \tparam ValueT the type of the current member
            //!
            template <typename ValueT>
            class ReverseRecord<ValueT>
            {
              public:
                //!
                //! \brief Standard constructor.
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr ReverseRecord() : value{} {};

                //!
                //! \brief Constructor.
                //!
                //! Assign some `value` to the current member.
                //!
                //! \param value the value to be assigned to the current member
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr ReverseRecord(ValueT value) : value(value) {}

                //!
                //! \brief Get the value of the current member.
                //!
                //! \return a reference to the current member
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline constexpr auto Get() -> ValueT& { return value; }

                //!
                //! \brief Get the value of the current member.
                //!
                //! \return a const reference to the current member
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline constexpr auto Get() const -> const ValueT& { return value; }

              protected:
                ValueT value;
            };

            //!
            //! \brief Definition of a record type with no members.
            //!
            template <>
            class ReverseRecord<>
            {
              public:
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr ReverseRecord() {}
            };

            namespace
            {
                //!
                //! \brief A Helper type for the construction of a record type with non-reverse order of the members.
                //!
                //! The definition uses an index sequence for the parameter and argument inversion.
                //!
                //! \tparam IndexSequenceT an sequence of indices from 0 to the number of template parameters in the variadic list
                //! \tparam ValueT a variadic list of type parameters
                //!
                template <typename IndexSequenceT, typename... ValueT>
                struct RecordHelper;

                //!
                //! \brief Definition of the helper type for the construction of a record type with non-reverse order of the members.
                //!
                //! \tparam ValueT a variadic list of type parameters
                //! \tparam I a list of indices used for the parameter and argument inversion
                //!
                template <typename... ValueT, SizeT... I>
                struct RecordHelper<::XXX_NAMESPACE::dataTypes::IndexSequence<I...>, ValueT...>
                {
                    static constexpr SizeT N = sizeof...(ValueT);

                    static_assert(N == sizeof...(I), "error: type parameter count does not match non-type parameter count.");

                    using Type = ReverseRecord<typename ::XXX_NAMESPACE::variadic::Pack<ValueT...>::template Type<(N - 1) - I>...>;

                    //!
                    //! \brief Argument inversion.
                    //!
                    //! This function inverts the order of the arguments and forwards them to a `ReverseRecord` type with
                    //! inverted template parameter list.
                    //!
                    //! \param values arguments to be forwarded to a `ReverseRecord` type with inverted template parameter list
                    //! \return a `ReverseRecord` type with inverted template parameter list
                    //!
                    HOST_VERSION
                    CUDA_DEVICE_VERSION
                    static inline constexpr auto Make(ValueT... values) -> Type { return {::XXX_NAMESPACE::variadic::Pack<ValueT...>::template Value<(N - 1) - I>(values...)...}; }
                };
            } // namespace

            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            //!
            //! \brief A record type with non-reverse order of the members.
            //!
            //! The order of the members in memory is NOT reversed:
            //! `Record<int, float> {..}` is equivalent to `class {int member_1; float member_2;..}`.
            //! This record type inherits from a `ReverseRecord` type with inverted template parameter list.
            //!
            //! Note: This definition has at least two parameters in the list.
            //!
            //! \tparam ValueT a parameter list defining the member types
            //!
            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            template <typename... ValueT>
            class Record : public RecordHelper<::XXX_NAMESPACE::dataTypes::IndexSequenceT<sizeof...(ValueT)>, ValueT...>::Type
            {
                using Base = typename RecordHelper<::XXX_NAMESPACE::dataTypes::IndexSequenceT<sizeof...(ValueT)>, ValueT...>::Type;

                // Needed for access to `Base` type.
                template <SizeT Index, typename... T>
                HOST_VERSION CUDA_DEVICE_VERSION friend constexpr auto& Get(Record<T...>&);

                template <SizeT Index, typename... T>
                HOST_VERSION CUDA_DEVICE_VERSION friend constexpr const auto& Get(const Record<T...>&);

              public:
                //!
                //! \brief Standard constructor.
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr Record() = default;

                //!
                //! \brief Constructor.
                //!
                //! Assign the same `value` to all members.
                //!
                //! \tparam T the type of the value to be assigned (can be different from the member type, but must be convertible)
                //! \param value the value to be assigned to all members
                //!
                template <typename T, typename EnableType = std::enable_if_t<std::is_fundamental<T>::value>>
                HOST_VERSION CUDA_DEVICE_VERSION constexpr Record(T value) : Base(value)
                {
                    static_assert(::XXX_NAMESPACE::variadic::Pack<T, ValueT...>::IsConvertible(), "error: types are not convertible.");
                }

                //!
                //! \brief Constructor.
                //!
                //! Assign some `values` to the members.
                //!
                //! \param values the values to be assigned to the members
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr Record(ValueT... values) : Base(RecordHelper<::XXX_NAMESPACE::dataTypes::IndexSequenceT<sizeof...(ValueT)>, ValueT...>::Make(values...)) {}
            };

            //!
            //! \brief Definition of a record type with non-reverse order of the members.
            //!
            //! Note: This definition has a single parameter in the list.
            //!
            //! \tparam ValueT the type of the member
            //!
            template <typename ValueT>
            class Record<ValueT> : public ReverseRecord<ValueT>
            {
                using Base = ReverseRecord<ValueT>;

                // Needed for access to `Base` type.
                template <SizeT Index, typename... T>
                HOST_VERSION CUDA_DEVICE_VERSION friend constexpr auto& Get(Record<T...>&);

                template <SizeT Index, typename... T>
                HOST_VERSION CUDA_DEVICE_VERSION friend constexpr const auto& Get(const Record<T...>&);

              public:
                //!
                //! \brief Standard constructor.
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr Record() = default;

                //!
                //! \brief Constructor.
                //!
                //! Assign some `value` to the member.
                //!
                //! \param value the value to be assigned to the member
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr Record(ValueT value) : Base(value) {}
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
                constexpr Record() {}
            };
        } // namespace internal
    } // namespace dataTypes
} // namespace XXX_NAMESPACE

#endif