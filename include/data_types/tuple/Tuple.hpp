// Copyright (c) 2020 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(DATA_TYPES_TUPLE_TUPLE_HPP)
#define DATA_TYPES_TUPLE_TUPLE_HPP

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#include <auxiliary/Loop.hpp>
#include <auxiliary/Pack.hpp>
#include <common/Math.hpp>
#include <common/Traits.hpp>
#include <integer_sequence/IntegerSequence.hpp>
#include <platform/Target.hpp>
#include <tuple/Get.hpp>
#include <tuple/Record.hpp>

namespace XXX_NAMESPACE
{
    namespace dataTypes
    {
        template <typename... T>
        class Tuple;
    }

    namespace internal
    {
        //! @{
        //! Type traits stuff.
        //!
        template <typename T>
        struct IsTuple
        {
            static constexpr bool value = false;
        };

        template <typename... T>
        struct IsTuple<::XXX_NAMESPACE::dataTypes::Tuple<T...>>
        {
            static constexpr bool value = true;
        };
        //! @}
    } // internal

    namespace dataTypes
    {
        using ::XXX_NAMESPACE::compileTime::Loop;
        using ::XXX_NAMESPACE::variadic::Pack;

        namespace internal
        {
            //!
            //! \brief A tuple type with non-reverse order of the members (base class).
            //!
            //! The order of the members in memory is NOT reversed:
            //! `TupleBase<int, float> {..}` is equivalent to `class {int member_1; float member_2;..}`.
            //!
            //! Note: This definition has at least five parameters in the list: see specializations below!
            //!
            template <typename... ValueT>
            class TupleBase
            {
              protected:
                HOST_VERSION
                CUDA_DEVICE_VERSION 
                constexpr TupleBase() : data{} {}

                template <typename T>
                HOST_VERSION
                CUDA_DEVICE_VERSION 
                constexpr TupleBase(T value) : data(value) {}

                HOST_VERSION
                CUDA_DEVICE_VERSION 
                constexpr TupleBase(ValueT... values) : data(values...) {}

                template <typename... T>
                HOST_VERSION
                CUDA_DEVICE_VERSION 
                constexpr TupleBase(const Record<T...>& data) : data(data) {}

              public:
                Record<ValueT...> data;
            };

            //!
            //! Note: This definition has four parameters in the list.
            //! The `Record` member overlaps with members `x,y,z,w`.
            //!
            template <typename ValueT_1, typename ValueT_2, typename ValueT_3, typename ValueT_4>
            class TupleBase<ValueT_1, ValueT_2, ValueT_3, ValueT_4>
            {
              protected:
                HOST_VERSION
                CUDA_DEVICE_VERSION 
                constexpr TupleBase() : data{} {}

                template <typename T>
                HOST_VERSION
                CUDA_DEVICE_VERSION 
                constexpr TupleBase(T value) : data(value) {}

                HOST_VERSION
                CUDA_DEVICE_VERSION 
                constexpr TupleBase(ValueT_1 x, ValueT_2 y, ValueT_3 z, ValueT_4 w) : data(x, y, z, w) {}

                template <typename... T>
                HOST_VERSION
                CUDA_DEVICE_VERSION 
                constexpr TupleBase(const Record<T...>& data) : data(data) {}

              public:
                union {
                    struct
                    {
                        ValueT_1 x;
                        ValueT_2 y;
                        ValueT_3 z;
                        ValueT_4 w;
                    };
                    Record<ValueT_1, ValueT_2, ValueT_3, ValueT_4> data;
                };

                using DummyT = struct {ValueT_1 x; ValueT_2 y; ValueT_3 z; ValueT_4 w;};
                static_assert(sizeof(DummyT) == sizeof(Record<ValueT_1, ValueT_2, ValueT_3, ValueT_4>), "error: FATAL union members have different size.");

#define MACRO(FUNC, MEMBER)                                                                                                                                                                                                \
    HOST_VERSION                                                                                                                                                                                                           \
    CUDA_DEVICE_VERSION                                                                                                                                                                                                    \
    constexpr inline auto FUNC() -> std::decay_t<decltype(MEMBER)>& { return MEMBER; }                                                                                                                                     \
                                                                                                                                                                                                                           \
    HOST_VERSION                                                                                                                                                                                                           \
    CUDA_DEVICE_VERSION                                                                                                                                                                                                    \
    constexpr inline auto FUNC() const -> const std::decay_t<decltype(MEMBER)>& { return MEMBER; }

                MACRO(GetX, x)
                MACRO(GetY, y)
                MACRO(GetZ, z)
                MACRO(GetW, w)
#undef MACRO
            };

            //!
            //! Note: This definition has three parameters in the list.
            //! The `Record` member overlaps with members `x,y,z`.
            //!
            template <typename ValueT_1, typename ValueT_2, typename ValueT_3>
            class TupleBase<ValueT_1, ValueT_2, ValueT_3>
            {
              protected:
                HOST_VERSION
                CUDA_DEVICE_VERSION 
                constexpr TupleBase() : data{} {}

                template <typename T>
                HOST_VERSION
                CUDA_DEVICE_VERSION 
                constexpr TupleBase(T value) : data(value) {}

                HOST_VERSION
                CUDA_DEVICE_VERSION 
                constexpr TupleBase(ValueT_1 x, ValueT_2 y, ValueT_3 z) : data(x, y, z) {}

                template <typename... T>
                HOST_VERSION
                CUDA_DEVICE_VERSION 
                constexpr TupleBase(const Record<T...>& data) : data(data) {}
              
              public:
                union {
                    struct
                    {
                        ValueT_1 x;
                        ValueT_2 y;
                        ValueT_3 z;
                    };
                    Record<ValueT_1, ValueT_2, ValueT_3> data;
                };

                using DummyT = struct {ValueT_1 x; ValueT_2 y; ValueT_3 z;};
                static_assert(sizeof(DummyT) == sizeof(Record<ValueT_1, ValueT_2, ValueT_3>), "error: FATAL union members have different size.");

#define MACRO(FUNC, MEMBER)                                                                                                                                                                                                \
    HOST_VERSION                                                                                                                                                                                                           \
    CUDA_DEVICE_VERSION                                                                                                                                                                                                    \
    constexpr inline auto FUNC() -> std::decay_t<decltype(MEMBER)>& { return MEMBER; }                                                                                                                                     \
                                                                                                                                                                                                                           \
    HOST_VERSION                                                                                                                                                                                                           \
    CUDA_DEVICE_VERSION                                                                                                                                                                                                    \
    constexpr inline auto FUNC() const -> const std::decay_t<decltype(MEMBER)>& { return MEMBER; }

                MACRO(GetX, x)
                MACRO(GetY, y)
                MACRO(GetZ, z)
#undef MACRO
            };

            //!
            //! Note: This definition has two parameters in the list.
            //! The `Record` member overlaps with members `x,y`.
            //!
            template <typename ValueT_1, typename ValueT_2>
            class TupleBase<ValueT_1, ValueT_2>
            {
              protected:
                HOST_VERSION
                CUDA_DEVICE_VERSION 
                constexpr TupleBase() : data{} {}

                template <typename T>
                HOST_VERSION
                CUDA_DEVICE_VERSION 
                constexpr TupleBase(T value) : data(value) {}

                HOST_VERSION
                CUDA_DEVICE_VERSION 
                constexpr TupleBase(ValueT_1 x, ValueT_2 y) : data(x, y) {}

                template <typename... T>
                HOST_VERSION
                CUDA_DEVICE_VERSION 
                constexpr TupleBase(const Record<T...>& data) : data(data) {}

              public:
                union {
                    struct
                    {
                        ValueT_1 x;
                        ValueT_2 y;
                    };
                    Record<ValueT_1, ValueT_2> data;
                };

                using DummyT = struct {ValueT_1 x; ValueT_2 y;};
                static_assert(sizeof(DummyT) == sizeof(Record<ValueT_1, ValueT_2>), "error: FATAL union members have different size.");

#define MACRO(FUNC, MEMBER)                                                                                                                                                                                                \
    HOST_VERSION                                                                                                                                                                                                           \
    CUDA_DEVICE_VERSION                                                                                                                                                                                                    \
    constexpr inline auto FUNC() -> std::decay_t<decltype(MEMBER)>& { return MEMBER; }                                                                                                                                     \
                                                                                                                                                                                                                           \
    HOST_VERSION                                                                                                                                                                                                           \
    CUDA_DEVICE_VERSION                                                                                                                                                                                                    \
    constexpr inline auto FUNC() const -> const std::decay_t<decltype(MEMBER)>& { return MEMBER; }

                MACRO(GetX, x)
                MACRO(GetY, y)
#undef MACRO
            };

            //!
            //! Note: This definition has a single parameter in the list.
            //! The `Record` member overlaps with members `x`.
            //!
            template <typename ValueT>
            class TupleBase<ValueT>
            {
              protected:
                HOST_VERSION
                CUDA_DEVICE_VERSION 
                constexpr TupleBase() : data{} {}

                template <typename T>
                HOST_VERSION
                CUDA_DEVICE_VERSION 
                constexpr TupleBase(T x) : data(x) {}

                template <typename... T>
                HOST_VERSION
                CUDA_DEVICE_VERSION 
                constexpr TupleBase(const Record<T...>& data) : data(data) {}

              public:
                union {
                    struct
                    {
                        ValueT x;
                    };
                    Record<ValueT> data;
                };
                
                using DummyT = struct {ValueT x;};
                static_assert(sizeof(DummyT) == sizeof(Record<ValueT>), "error: FATAL union members have different size.");

#define MACRO(FUNC, MEMBER)                                                                                                                                                                                                \
    HOST_VERSION                                                                                                                                                                                                           \
    CUDA_DEVICE_VERSION                                                                                                                                                                                                    \
    constexpr inline auto FUNC() -> std::decay_t<decltype(MEMBER)>& { return MEMBER; }                                                                                                                                     \
                                                                                                                                                                                                                           \
    HOST_VERSION                                                                                                                                                                                                           \
    CUDA_DEVICE_VERSION                                                                                                                                                                                                    \
    constexpr inline auto FUNC() const -> const std::decay_t<decltype(MEMBER)>& { return MEMBER; }

                MACRO(GetX, x)
#undef MACRO                
            };
            
            template <>
            class TupleBase<> {};
        } // namespace internal

        // Forward declaration.
        namespace internal
        {
            template <typename...>
            class TupleProxy;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //!
        //! \brief A tuple type with non-reverse order of the members (implementation).
        //!
        //! The implementation comprises simple arithmetic functions.
        //!
        //! \tparam ValueT a type parameter list defining the member types
        //!
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        template <typename... ValueT>
        class Tuple : public internal::TupleBase<ValueT...>
        {
            using Base = internal::TupleBase<ValueT...>;

            friend class internal::TupleProxy<std::decay_t<ValueT>...>;
            friend class internal::TupleProxy<const std::decay_t<ValueT>...>;

            template <typename T>
            static inline constexpr auto IsRecordOrTupleOrProxy()
            {
                return ::XXX_NAMESPACE::internal::IsRecord<std::decay_t<T>>::value ||
                ::XXX_NAMESPACE::internal::IsTuple<std::decay_t<T>>::value ||
                ::XXX_NAMESPACE::internal::IsProxy<std::decay_t<T>>::value;
            }

          public:
            using Type = Tuple<ValueT...>;
            using Proxy = internal::TupleProxy<ValueT...>;

            HOST_VERSION
            CUDA_DEVICE_VERSION 
            constexpr Tuple() : Base() {}

            template <typename T, typename std::enable_if_t<!IsRecordOrTupleOrProxy<T>(), int> = 0>
            HOST_VERSION
            CUDA_DEVICE_VERSION 
            constexpr Tuple(T value) : Base(value) {}

            HOST_VERSION
            CUDA_DEVICE_VERSION 
            constexpr Tuple(ValueT... values) : Base(values...) {}

            template <typename... T>
            HOST_VERSION
            CUDA_DEVICE_VERSION 
            constexpr Tuple(const internal::Record<T...>& data) : Base(data) {}

            template <typename... T>
            HOST_VERSION
            CUDA_DEVICE_VERSION 
            inline constexpr operator Tuple<T...>() { return Tuple<T...>{Base::data}; }

            template <typename... T>
            HOST_VERSION
            CUDA_DEVICE_VERSION 
            inline constexpr operator Tuple<T...>() const { return Tuple<T...>{Base::data}; }

            //!
            //! \brief Sign operator.
            //!
            //! Change the sign of all members.
            //!
            //! \return a new `Tuple` with sign-changed members
            //!
            HOST_VERSION
            CUDA_DEVICE_VERSION
            constexpr inline auto operator-() const
            {
                Tuple tuple;

                Loop<sizeof...(ValueT)>::Execute([&tuple, this](const auto I) { Get<I>(tuple) = -Get<I>(*this); });

                return tuple;
            }

            //!
            //! \brief Some operator definitions: assignment and arithmetic.
            //!
            //! \tparam T a variadic list of type parameters that are convertibe to this type's type parameters
            //! \param tuple a `Tuple` or `TupleProxy` instance with the same number of members
            //! \return a reference to this `Tuple` after having applied the operation
            //!
#define MACRO(OP, IN_T)                                                                                                                                                                                                    \
    template <typename... T>                                                                                                                                                                                               \
    HOST_VERSION CUDA_DEVICE_VERSION inline constexpr auto operator OP(const IN_T<T...>& tuple)->Tuple&                                                                                                                    \
    {                                                                                                                                                                                                                      \
        static_assert(Pack<T...>::template IsConvertibleTo<ValueT...>(), "error: types are not convertible.");                                                                                                             \
                                                                                                                                                                                                                           \
        Loop<sizeof...(ValueT)>::Execute([&tuple, this](const auto I) { Get<I>(*this) OP Get<I>(tuple); });                                                                                                                \
                                                                                                                                                                                                                           \
        return *this;                                                                                                                                                                                                      \
    }

            MACRO(=, Tuple)
            MACRO(+=, Tuple)
            MACRO(-=, Tuple)
            MACRO(*=, Tuple)
            MACRO(/=, Tuple)
#undef MACRO

            //!
            //! \brief Some operator definitions: assignment and arithmetic.
            //!
            //! \tparam T the type of the value
            //! \param value the value used for the operation
            //! \return a reference to this `Tuple` after having applied the operation
            //!
#define MACRO(OP)                                                                                                                                                                                                          \
    template <typename T, typename EnableType = std::enable_if_t<std::is_fundamental<T>::value>>                                                                                                                           \
    HOST_VERSION CUDA_DEVICE_VERSION inline constexpr auto operator OP(T value)->Tuple&                                                                                                                                    \
    {                                                                                                                                                                                                                      \
        Loop<sizeof...(ValueT)>::Execute([&value, this](const auto I) { Get<I>(*this) OP value; });                                                                                                                        \
                                                                                                                                                                                                                           \
        return *this;                                                                                                                                                                                                      \
    }

            MACRO(=)
            MACRO(+=)
            MACRO(-=)
            MACRO(*=)
            MACRO(/=)
#undef MACRO

            template <typename... T>
            HOST_VERSION CUDA_DEVICE_VERSION inline constexpr auto operator==(const Tuple<T...>& tuple) const
            {
                static_assert(sizeof...(T) == sizeof...(ValueT), "error: types have different size.");

                bool is_equal = true;

                Loop<sizeof...(ValueT)>::Execute([&tuple, &is_equal, this](const auto I) { is_equal &= (Get<I>(*this) == Get<I>(tuple)); });

                return is_equal;
            }

            template <typename T, typename EnableType = std::enable_if_t<std::is_fundamental<T>::value>>
            HOST_VERSION CUDA_DEVICE_VERSION inline constexpr auto operator==(const T& value) const
            {
                static_assert(Pack<T>::template IsConvertibleTo<ValueT...>(), "error: types are not convertible.");

                return (*this) == Tuple<ValueT...>(value);
            }

            template <typename T>
            HOST_VERSION CUDA_DEVICE_VERSION inline constexpr auto operator!=(const T& value) const
            {
                return !((*this) == value);
            }
        };

        //!
        //! \brief Definition of a `Tuple` type with no members (implementation).
        //!
        template <>
        class Tuple<> {};

        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        //
        // Write a `Tuple` to an output stream.
        // We do not use the compile-time loop here as it causes warnings regarding __host__ call from __device__ function.
        //
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        template <typename ValueT>
        auto TupleToStream(std::ostream& os, const ValueT& value) -> int
        {
            using Type = std::conditional_t<sizeof(ValueT) == 1, std::int32_t, ValueT>;

            os << static_cast<Type>(value) << " ";

            return 0;
        }

        template <typename TupleT, SizeT... I>
        std::ostream& TupleToStream(std::ostream& os, const TupleT& tuple, IndexSequence<I...>)
        {
            using dummy = int[];

            (void)dummy{TupleToStream(os, Get<I>(tuple))...};

            return os;
        }

        template <typename... ValueT>
        std::ostream& operator<<(std::ostream& os, const Tuple<ValueT...>& tuple)
        {
            os << "( ";

            TupleToStream(os, tuple, MakeIndexSequence<sizeof...(ValueT)>());

            os << ")";

            return os;
        }
    } // namespace dataTypes
} // namespace XXX_NAMESPACE

#include <common/Traits.hpp>
#include <tuple/TupleMath.hpp>
#include <tuple/TupleProxy.hpp>

namespace XXX_NAMESPACE
{
    namespace internal
    {
        using ::XXX_NAMESPACE::dataTypes::Tuple;

        //!
        //! \brief Specialization of the `ProvidesProxy` type for `Tuple`s.
        //!
        template <typename... ValueT>
        struct ProvidesProxy<Tuple<ValueT...>>
        {
            static constexpr bool value = true;
        };

        template <typename... ValueT>
        struct ProvidesProxy<const Tuple<ValueT...>>
        {
            static constexpr bool value = true;
        };
    } // namespace internal
} // namespace XXX_NAMESPACE

#endif