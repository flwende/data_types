// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(DATA_TYPES_TUPLE_TUPLE_HPP)
#define DATA_TYPES_TUPLE_TUPLE_HPP

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#include <auxiliary/Template.hpp>
#include <common/Math.hpp>
#include <integer_sequence/IntegerSequence.hpp>
#include <platform/Target.hpp>
#include <tuple/Get.hpp>
#include <tuple/Record.hpp>

namespace XXX_NAMESPACE
{
    namespace dataTypes
    {
        using ::XXX_NAMESPACE::compileTime::Loop;
        using ::XXX_NAMESPACE::variadic::Pack;

        namespace internal
        {
            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            //!
            //! \brief A tuple type with non-reverse order of the members (base class).
            //!
            //! The order of the members in memory is NOT reversed:
            //! `TupleBase<int, float> {..}` is equivalent to `class {int member_1; float member_2;..}`.
            //!
            //! Note: This definition has at least five parameters in the list.
            //!
            //! \tparam ValueT a type parameter list defining the member types
            //!
            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            template <typename... ValueT>
            class TupleBase
            {
              protected:
                //!
                //! \brief Standard constructor.
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleBase() : data{} {};

                //!
                //! \brief Constructor.
                //!
                //! Assign the same `value` to all members.
                //!
                //! \tparam T the type of the value to be assigned (can be different from the member type, but must be convertible)
                //! \param value the value to be assigned to all members
                //!
                template <typename T, typename EnableType = std::enable_if_t<std::is_fundamental<T>::value>>
                HOST_VERSION CUDA_DEVICE_VERSION constexpr TupleBase(T value) : data(value)
                {
                    static_assert(Pack<ValueT...>::template IsConvertibleFrom<T>(), "error: types are not convertible.");
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
                constexpr TupleBase(ValueT... values) : data(values...) {}

                //!
                //! \brief Constructor.
                //!
                //! Create this object from a `Record`
                //!
                //! \param tuple another `TupleBase` instance
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleBase(const Record<ValueT...>& data) : data(data) {}

                //!
                //! \brief Copy constructor.
                //!
                //! \tparam a variadic list of type parameters
                //! \param tuple another `TupleBase` instance
                //!
                template <typename... T>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleBase(const TupleBase<T...>& tuple) : data(tuple.data) {}

              public:
                Record<ValueT...> data;
            };

            //!
            //! \brief A tuple type with non-reverse order of the members (base class).
            //!
            //! The order of the members in memory is NOT reversed:
            //! `TupleBase<int, float> {..}` is equivalent to `class {int member_1; float member_2;..}`.
            //!
            //! Note: This definition has four parameters in the list.
            //! The `Record` member overlaps with members `x,y,z,w`.
            //!
            //! \tparam ValueT_1 type of the 1st member
            //! \tparam ValueT_2 type of the 2nd member
            //! \tparam ValueT_3 type of the 3rd member
            //! \tparam ValueT_4 type of the 4th member
            //!
            template <typename ValueT_1, typename ValueT_2, typename ValueT_3, typename ValueT_4>
            class TupleBase<ValueT_1, ValueT_2, ValueT_3, ValueT_4>
            {
              protected:
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleBase() : data{} {}

                template <typename T, typename EnableType = std::enable_if_t<std::is_fundamental<T>::value>>
                HOST_VERSION CUDA_DEVICE_VERSION constexpr TupleBase(T value) : data(value)
                {
                }

                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleBase(ValueT_1 x, ValueT_2 y, ValueT_3 z, ValueT_4 w) : data(x, y, z, w) {}

                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleBase(const Record<ValueT_1, ValueT_2, ValueT_3, ValueT_4>& data) : data(data) {}

                template <typename T_1, typename T_2, typename T_3, typename T_4>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleBase(const TupleBase<T_1, T_2, T_3, T_4>& tuple) : data(tuple.data) {}

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
            //! \brief A tuple type with non-reverse order of the members (base class).
            //!
            //! The order of the members in memory is NOT reversed:
            //! `TupleBase<int, float> {..}` is equivalent to `class {int member_1; float member_2;..}`.
            //!
            //! Note: This definition has three parameters in the list.
            //! The `Record` member overlaps with members `x,y,z`.
            //!
            //! \tparam ValueT_1 type of the 1st member
            //! \tparam ValueT_2 type of the 2nd member
            //! \tparam ValueT_3 type of the 3rd member
            //!
            template <typename ValueT_1, typename ValueT_2, typename ValueT_3>
            class TupleBase<ValueT_1, ValueT_2, ValueT_3>
            {
              protected:
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleBase() : data{} {}

                template <typename T, typename EnableType = std::enable_if_t<std::is_fundamental<T>::value>>
                HOST_VERSION CUDA_DEVICE_VERSION constexpr TupleBase(T value) : data(value)
                {
                }

                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleBase(ValueT_1 x, ValueT_2 y, ValueT_3 z) : data(x, y, z) {}

                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleBase(const Record<ValueT_1, ValueT_2, ValueT_3>& data) : data(data) {}

                template <typename T_1, typename T_2, typename T_3>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleBase(const TupleBase<T_1, T_2, T_3>& tuple) : data(tuple.data) {}

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
            //! \brief A tuple type with non-reverse order of the members (base class).
            //!
            //! The order of the members in memory is NOT reversed:
            //! `TupleBase<int, float> {..}` is equivalent to `class {int member_1; float member_2;..}`.
            //!
            //! Note: This definition has two parameters in the list.
            //! The `Record` member overlaps with members `x,y`.
            //!
            //! \tparam ValueT_1 type of the 1st member
            //! \tparam ValueT_2 type of the 2nd member
            //!
            template <typename ValueT_1, typename ValueT_2>
            class TupleBase<ValueT_1, ValueT_2>
            {
              protected:
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleBase() : data{} {}

                template <typename T, typename EnableType = std::enable_if_t<std::is_fundamental<T>::value>>
                HOST_VERSION CUDA_DEVICE_VERSION constexpr TupleBase(T value) : data(value)
                {
                }

                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleBase(ValueT_1 x, ValueT_2 y) : data(x, y) {}

                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleBase(const Record<ValueT_1, ValueT_2>& data) : data(data) {}

                template <typename T_1, typename T_2>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleBase(const TupleBase<T_1, T_2>& tuple) : data(tuple.data) {}

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
            //! \brief A tuple type with non-reverse order of the members (base class).
            //!
            //! The order of the members in memory is NOT reversed:
            //! `TupleBase<int, float> {..}` is equivalent to `class {int member_1; float member_2;..}`.
            //!
            //! Note: This definition has a single parameter in the list.
            //! The `Record` member overlaps with members `x`.
            //!
            //! \tparam ValueT_1 type of the 1st member
            //!
            template <typename ValueT>
            class TupleBase<ValueT>
            {
              protected:
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleBase() : data{} {}

                template <typename T, typename EnableType = std::enable_if_t<std::is_fundamental<T>::value>>
                HOST_VERSION CUDA_DEVICE_VERSION constexpr TupleBase(T value) : data(value)
                {
                }

                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleBase(const Record<ValueT>& data) : data(data) {}

                template <typename T>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleBase(const TupleBase<T>& tuple) : data(tuple.data) {}

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

            //!
            //! \brief Definition of a tuple type with no members (base class).
            //!
            template <>
            class TupleBase<>
            {
              protected:
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleBase() {}
            };
        } // namespace internal

        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        //
        // Forward declarations.
        //
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
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

            //!
            //! \brief Constructor.
            //!
            //! Create this object from a `Record`.
            //!
            //! \tparam T a variadic list of type parameters
            //! \param record a `Record` instance holding the data
            //!
            template <typename ...T>
            HOST_VERSION
            CUDA_DEVICE_VERSION
            constexpr Tuple(const internal::Record<T...>& record) : Base(record) {}
            
          public:
            using Type = Tuple<ValueT...>;
            using Proxy = internal::TupleProxy<ValueT...>;

            //!
            //! \brief Standard constructor.
            //!
            HOST_VERSION
            CUDA_DEVICE_VERSION
            constexpr Tuple() = default;

            //!
            //! \brief Constructor.
            //!
            //! Assign the same `value` to all members.
            //!
            //! \tparam T the type of the value to be assigned (can be different from the member type, but must be convertible)
            //! \param value the value to be assigned to all members
            //!
            template <typename T, typename EnableType = std::enable_if_t<std::is_fundamental<T>::value>>
            HOST_VERSION CUDA_DEVICE_VERSION explicit constexpr Tuple(T value) : Base(value)
            {
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
            explicit constexpr Tuple(ValueT... values) : Base(values...) {}

            //!
            //! \brief Copy / conversion constructor.
            //!
            //! We do not allow the case where this type has reference template type-parameters!
            //! The only case where this can happen is the base class conversion of a `TupleProxy`. 
            //! If this is the case, the programmer must have requested a hard copy of this type (e.g. a function call with value parameters).
            //! Do not build the program then!
            //!
            //! \tparam T a variadic list of type parameters
            //! \param tuple another `Tuple` instance
            //!
            template <typename... T>
            HOST_VERSION CUDA_DEVICE_VERSION constexpr Tuple(Tuple<T...>& tuple) : Base(tuple)
            {
                static_assert(!(Pack<ValueT...>::IsReference() && Pack<T...>::IsReference()), "error: this type has reference template type-parameters, which is not allowed here!");
            }

            template <typename... T>
            HOST_VERSION CUDA_DEVICE_VERSION constexpr Tuple(const Tuple<T...>& tuple) : Base(tuple)
            {
                static_assert(!(Pack<ValueT...>::IsReference() && Pack<T...>::IsReference()), "error: this type has reference template type-parameters, which is not allowed here!");
            }

            //!
            //! \brief Constructor.
            //!
            //! Create a `Tuple` from a `TupleProxy`.
            //! This constructor enables class template argument deduction with C++17 and later.
            //! 
            //! \param tuple a `ProxyTuple` instance
            //!
            HOST_VERSION CUDA_DEVICE_VERSION constexpr Tuple(internal::TupleProxy<ValueT...>& proxy) : Base(proxy.data)
            {
            }

            HOST_VERSION CUDA_DEVICE_VERSION constexpr Tuple(const internal::TupleProxy<ValueT...>& proxy) : Base(proxy.data)
            {
            }

            //!
            //! \brief Sign operator.
            //!
            //! Change the sign of all members.
            //!
            //! \return a new `Tuple` with sign-changed members
            //!
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline auto operator-() const
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
    HOST_VERSION CUDA_DEVICE_VERSION inline auto operator OP(const IN_T<T...>& tuple)->Tuple&                                                                                                                              \
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
    HOST_VERSION CUDA_DEVICE_VERSION inline auto operator OP(T value)->Tuple&                                                                                                                                              \
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
        };

        //!
        //! \brief Definition of a `Tuple` type with no members (implementation).
        //!
        template <>
        class Tuple<>
        {
          public:
            HOST_VERSION
            CUDA_DEVICE_VERSION
            constexpr Tuple(){};
        };

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