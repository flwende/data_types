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
    namespace dataTypes
    {
        namespace internal
        {
            using ::XXX_NAMESPACE::dataTypes::IndexSequence;
            using ::XXX_NAMESPACE::dataTypes::IndexSequenceT;
            using ::XXX_NAMESPACE::dataTypes::MakeIndexSequence;
            using ::XXX_NAMESPACE::variadic::Pack;

            //! @{
            //! \brief Generate type `T<..>` with reversed template parameter list.
            //!
            template <typename T, typename... ParameterList>
            struct Reverse;

            template <template <typename...> typename T, typename... InverseParameterList, typename Head, typename... ParameterList>
            struct Reverse<T<InverseParameterList...>, Head, ParameterList...>
            {
                using Type = typename Reverse<T<Head, InverseParameterList...>, ParameterList...>::Type;
            };

            template <template <typename...> typename T, typename... InverseParameterList>
            struct Reverse<T<InverseParameterList...>>
            {
                using Type = T<InverseParameterList...>;
            };
            //! @}

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
            template <typename ValueT, typename... Tail>
            class ReverseRecord<ValueT, Tail...> : public ReverseRecord<Tail...>
            {
                using Base = ReverseRecord<Tail...>;

            protected:
                //! @{
                //! Create a `ReversedRecord` from this instance.
                //! The count and type of the members may be different: if it is larger, fill in zeros.
                //!
                template <SizeT I, typename std::enable_if_t<(I <= sizeof...(Tail)), int> = 0>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline constexpr auto GetMemberValueOrZero() const { return internal::Get<I>(*this); }

                template <SizeT I, typename std::enable_if_t<(I > sizeof...(Tail)), int> = 0>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline constexpr auto GetMemberValueOrZero() const { return 0; }
                

                template <typename... T, SizeT... I>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline constexpr auto Convert(IndexSequence<I...>&&) const -> ReverseRecord<T...>
                {
                    return {static_cast<T>(GetMemberValueOrZero<I>())...};
                }
                //! @}

            public:
                //! @{
                //! Create a `ReverseRecord` from a single or from multiple values.
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr ReverseRecord() : Base(), value{} {}

                template <typename T>
                HOST_VERSION 
                CUDA_DEVICE_VERSION 
                constexpr ReverseRecord(T value) : Base(value), value(value) {}

                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr ReverseRecord(ValueT value, Tail... tail) : Base(tail...), value(value) {}
                //! @}

                //!
                //! Convert this instance into another `ReverseRecord` that may differ in member count and type.
                //!
                template <typename... T>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr operator ReverseRecord<T...>() const { return Convert<T...>(MakeIndexSequence<sizeof...(T)>()); }

                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline constexpr auto& Get() { return value; }

                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline constexpr const auto& Get() const { return value; }

            protected:
                ValueT value;
            };

            template <typename ValueT>
            class ReverseRecord<ValueT>
            {
            protected:
                template <SizeT I, typename std::enable_if_t<(I == 0), int> = 0>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline constexpr auto GetMemberValueOrZero() const { return value; }

                template <SizeT I, typename std::enable_if_t<(I > 0), int> = 0>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline constexpr auto GetMemberValueOrZero() const { return 0; }

                template <typename... T, SizeT... I>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline constexpr auto Convert(IndexSequence<I...>&&) const -> ReverseRecord<T...>
                {
                    return {static_cast<T>(GetMemberValueOrZero<I>())...};
                }

            public:
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr ReverseRecord() : value{} {}

                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr ReverseRecord(ValueT value) : value(value) {}

                template <typename... T>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr operator ReverseRecord<T...>() const { return Convert<T...>(MakeIndexSequence<sizeof...(T)>()); }

                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline constexpr auto& Get() { return value; }

                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline constexpr const auto& Get() const { return value; }

            protected:
                ValueT value;
            };

            template <>
            class ReverseRecord<> {};

            //!
            //! \brief A record type with non-reverse order of the members.
            //!
            //! The order of the members in memory is NOT reversed:
            //! `Record<int, float> {..}` is equivalent to `class {int member_1; float member_2;..}`.
            //! This record type inherits from a `ReverseRecord` type with inverted template parameter list.
            //!
            //! Note: This definition has at least one parameter in the list.
            //!
            template <typename ValueT, typename... Tail>
            class Record<ValueT, Tail...> : public Reverse<ReverseRecord<>, ValueT, Tail...>::Type
            {
                using Base = typename Reverse<ReverseRecord<>, ValueT, Tail...>::Type;

                //! @{
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

            protected:
                template <typename... T, SizeT... I>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline constexpr auto Convert(IndexSequence<I...>&&) const -> Record<T...>
                {
                    // The data members are stored in reverse order: I -> sizeof...(Tail) - I.
                    return {static_cast<T>(Base::template GetMemberValueOrZero<sizeof...(Tail) - I>())...};
                }

                //!
                //! This constructor unpacks the values of the provided `ReverseRecord` in reverse order
                //! and moves them to its base class constructor.
                //! 
                template <SizeT... I>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr Record(ReverseRecord<ValueT, Tail...>&& record, IndexSequence<I...>&&) 
                    : Base(internal::Get<sizeof...(Tail) - I>(record)...) {}

            public:
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr Record() : Base() {}

                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr Record(ValueT value) : Base(value) {}

                //!
                //! Create a `ReverseRecord` from the arguments and move it to this class' constructor
                //! that implements the reverse unpacking of the `ReverseRecord`.
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr Record(ValueT value, Tail... more_values) 
                    : Record(ReverseRecord<ValueT, Tail...>{value, more_values...}, MakeIndexSequence<1 + sizeof...(Tail)>()) {}

                template <typename... T>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline constexpr operator Record<T...>() const { return Convert<T...>(MakeIndexSequence<sizeof...(T)>()); }
            };

            template <typename ValueT>
            class Record<ValueT> : public ReverseRecord<ValueT>
            {
                using Base = ReverseRecord<ValueT>;

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

            protected:
                template <typename... T, SizeT... I>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline constexpr auto Convert(IndexSequence<I...>&&) const -> Record<T...>
                {
                        return {static_cast<T>(Base::template GetMemberValueOrZero<I>())...};
                }

            public:
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr Record() : Base() {}

                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr Record(ValueT value) : Base(value) {}

                template <typename... T>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline constexpr operator Record<T...>() const { return Convert<T...>(MakeIndexSequence<sizeof...(T)>()); }
            };

            template <>
            class Record<> {};

        } // internal
    } // namespace dataTypes
} // namespace XXX_NAMESPACE

#endif