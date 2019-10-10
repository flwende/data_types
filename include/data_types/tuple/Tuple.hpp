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
#include <data_types/integer_sequence/IntegerSequence.hpp>
#include <data_types/tuple/Get.hpp>
#include <data_types/tuple/Record.hpp>
#include <platform/Target.hpp>

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
            template <typename...>
            class TupleProxy;

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
                    static_assert(::XXX_NAMESPACE::variadic::Pack<ValueT...>::template IsConvertibleFrom<T>(), "error: types are not convertible.");
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
                //! \brief Copy constructor.
                //!
                //! \param tuple another `TupleBase` instance
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleBase(const TupleBase& tuple) : data(tuple.data) {}

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
                constexpr TupleBase(const TupleBase& tuple) : data(tuple.data) {}

                template <typename T_1, typename T_2, typename T_3, typename T_4>
                HOST_VERSION CUDA_DEVICE_VERSION constexpr TupleBase(const TupleProxy<T_1, T_2, T_3, T_4>& proxy) : data(proxy.x, proxy.y, proxy.z, proxy.w)
                {
                }

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
                constexpr TupleBase(const TupleBase& tuple) : data(tuple.data) {}

                template <typename T_1, typename T_2, typename T_3>
                HOST_VERSION CUDA_DEVICE_VERSION constexpr TupleBase(const TupleProxy<T_1, T_2, T_3>& proxy) : data(proxy.x, proxy.y, proxy.z)
                {
                }

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
                constexpr TupleBase(const TupleBase& tuple) : data(tuple.data) {}

                template <typename T_1, typename T_2>
                HOST_VERSION CUDA_DEVICE_VERSION constexpr TupleBase(const TupleProxy<T_1, T_2>& proxy) : data(proxy.x, proxy.y)
                {
                }

              public:
                union {
                    struct
                    {
                        ValueT_1 x;
                        ValueT_2 y;
                    };
                    Record<ValueT_1, ValueT_2> data;
                };
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
                constexpr TupleBase(const TupleBase& tuple) : data(tuple.data) {}

                template <typename T>
                HOST_VERSION CUDA_DEVICE_VERSION constexpr TupleBase(const TupleProxy<T>& proxy) : data(proxy.x)
                {
                }

              public:
                union {
                    struct
                    {
                        ValueT x;
                    };
                    Record<ValueT> data;
                };
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
            HOST_VERSION CUDA_DEVICE_VERSION constexpr Tuple(T value) : Base(value)
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
            constexpr Tuple(ValueT... values) : Base(values...) {}

            //!
            //! \brief Copy constructor.
            //!
            //! \param tuple another `Tuple` instance
            //!
            HOST_VERSION
            CUDA_DEVICE_VERSION
            constexpr Tuple(const Tuple& tuple) : Base(tuple) {}

            //!
            //! \brief Constructor.
            //!
            //! Construct a `TupleBase` instance from a `TupleProxy` with a different parameter list.
            //! The number of template type parameters must be convertibe to this type's type parameters.
            //!
            //! \tparam T a variadic list of type parameters that are convertibe to this type's type parameters
            //! \param proxy a `TupleProxy` instance
            //!
            template <typename... T>
            HOST_VERSION CUDA_DEVICE_VERSION constexpr Tuple(const internal::TupleProxy<T...>& proxy) : Base(proxy)
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

                ::XXX_NAMESPACE::compileTime::Loop<sizeof...(ValueT)>::Execute([&tuple, this](const auto I) { internal::Get<I>(tuple.data) = -internal::Get<I>(Base::data); });

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
        static_assert(sizeof...(T) == sizeof...(ValueT), "error: parameter lists have different size.");                                                                                                                   \
        static_assert(::XXX_NAMESPACE::variadic::Pack<ValueT...>::template IsConvertibleFrom<T...>(), "error: types are not convertible.");                                                                                \
                                                                                                                                                                                                                           \
        ::XXX_NAMESPACE::compileTime::Loop<sizeof...(ValueT)>::Execute([&tuple, this](const auto I) { internal::Get<I>(Base::data) OP internal::Get<I>(tuple.data); });                                                    \
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
        ::XXX_NAMESPACE::compileTime::Loop<sizeof...(ValueT)>::Execute([&value, this](const auto I) { internal::Get<I>(Base::data) OP value; });                                                                           \
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
        //! \brief Definition of a tuple type with no members (implementation).
        //!
        template <>
        class Tuple<>
        {
          public:
            HOST_VERSION
            CUDA_DEVICE_VERSION
            constexpr Tuple(){};
        };

        template <typename... ValueT>
        std::ostream& operator<<(std::ostream& os, const Tuple<ValueT...>& tuple)
        {
            os << "( ";

            ::XXX_NAMESPACE::compileTime::Loop<sizeof...(ValueT)>::Execute([&tuple, &os](const auto I) { os << Get<I>(tuple) << " "; });

            os << ")";

            return os;
        }
    } // namespace dataTypes
} // namespace XXX_NAMESPACE

#include <common/Traits.hpp>
#include <data_types/tuple/TupleMath.hpp>
#include <data_types/tuple/TupleProxy.hpp>

namespace XXX_NAMESPACE
{
    namespace internal
    {
        //!
        //! \brief Specialization of the `ProvidesProxy` type for `Tuple`s.
        //!
        template <typename... ValueT>
        struct ProvidesProxy<::XXX_NAMESPACE::dataTypes::Tuple<ValueT...>>
        {
            static constexpr bool value = true;
        };

        template <typename... ValueT>
        struct ProvidesProxy<const ::XXX_NAMESPACE::dataTypes::Tuple<ValueT...>>
        {
            static constexpr bool value = true;
        };
    } // namespace internal
} // namespace XXX_NAMESPACE

#endif