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

            template <typename ...T>
            using Pack = ::XXX_NAMESPACE::variadic::Pack<T...>;
            template <SizeT N>
            using CompileTimeLoop = ::XXX_NAMESPACE::compileTime::Loop<N>;

            //!
            //! \brief Constructor.
            //!
            //! \tparam T the decay types of the tuple members (must be convertibe to this type's type parameters)
            //! \tparam I a list of indices used for the parameter access
            //! \param proxy a `Tuple` instance
            //! \param unnamed used for template parameter deduction
            //!      
            template <typename... T, SizeT... I>
            HOST_VERSION CUDA_DEVICE_VERSION Tuple(const Tuple<T...>& tuple, ::XXX_NAMESPACE::dataTypes::IndexSequence<I...>) : Tuple(Get<I>(tuple)...)
            {
                //static_assert(Pack<ValueT...>::IsReference(), "error: you're trying to copy a proxy where an explicit copy to the original type is needed -> make an explicit copy or use a static cast!");
            }

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
            //! \brief Copy / conversion constructor.
            //!
            //! \tparam T a variadic list of type parameters
            //! \param tuple another `Tuple` instance
            //!
            template <typename ...T>
            HOST_VERSION
            CUDA_DEVICE_VERSION
            constexpr Tuple(const Tuple<T...>& tuple) : Tuple(tuple, ::XXX_NAMESPACE::dataTypes::MakeIndexSequence<sizeof...(T)>()) {}
        
            //!
            //! \brief Constructor (Invalid object construction).
            //!
            //! This constructor handles the case of a base class conversion of a `TupleProxy` where in the calling context a non-reference
            //! instance of the base class is needed, e.g.
            //! ```
            //! template <typename ...T>
            //! void foo(Tuple<T...> tuple) {..}
            //! ..
            //! TupleProxy<int, float> proxy = ..;
            //! foo(proxy); // ERROR
            //! ```
            //! In this case the `TupleProxy` is converted to its base class `Tuple<int&, float&>`, and references to the members are given to the `foo`.
            //! This is errorneous behavior. 
            //! Instead the programmer must make an explicit copy of the `TupleProxy` type to `Tuple`, which would happen anyway when calling `foo`,
            //! or use a static cast to `Tuple<int, float>`
            //!
            //! \tparam T the decay types of the proxy members (must be convertibe to this type's type parameters)
            //! \param proxy a `TupleProxy` instance
            //!
            template <typename... T>
            HOST_VERSION CUDA_DEVICE_VERSION Tuple(internal::TupleProxy<T...> proxy) : Tuple(proxy, ::XXX_NAMESPACE::dataTypes::MakeIndexSequence<sizeof...(T)>())
            {
                static_assert(!Pack<ValueT...>::IsReference(), "error: you're trying to copy a proxy where an explicit copy to the original type is needed -> make an explicit copy or use a static cast!");
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

                CompileTimeLoop<sizeof...(ValueT)>::Execute([&tuple, this](const auto I) { Get<I>(tuple) = -Get<I>(*this); });

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
        static_assert(Pack<T...>::template IsConvertibleTo<ValueT...>(), "error: types are not convertible.");                                                                                                             \
                                                                                                                                                                                                                           \
        CompileTimeLoop<sizeof...(ValueT)>::Execute([&tuple, this](const auto I) { Get<I>(*this) OP Get<I>(tuple); });                                                                                                     \
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
        CompileTimeLoop<sizeof...(ValueT)>::Execute([&value, this](const auto I) { Get<I>(*this) OP value; });                                                                                                             \
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
            os << value << " ";

            return 0;
        }

        template <typename TupleT, SizeT ...I>
        std::ostream& TupleToStream(std::ostream& os, const TupleT& tuple, ::XXX_NAMESPACE::dataTypes::IndexSequence<I...>)
        {
            int dummy[] = {TupleToStream(os, Get<I>(tuple))...};

            return os;
        }

        template <typename... ValueT> 
        std::ostream& operator<<(std::ostream& os, const Tuple<ValueT...>& tuple)
        {
            os << "( ";

            TupleToStream(os, tuple, ::XXX_NAMESPACE::dataTypes::MakeIndexSequence<sizeof...(ValueT)>());

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
