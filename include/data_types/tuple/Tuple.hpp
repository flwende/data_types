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
#include <platform/Target.hpp>

namespace XXX_NAMESPACE
{
    namespace dataTypes
    {
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

        namespace internal
        {
            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            //!
            //! \brief A tuple type with reverse order of the members.
            //!
            //! The order of the members in memory is reversed: 
            //! `TupleReverseData<int, float> {..}` is equivalent to `class {float member_1; int member_2;..}`.
            //!
            //! \tparam ValueT a parameter list defining the member types
            //!
            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            template <typename ...ValueT>
            class TupleReverseData;

            //!
            //! \brief Definition of a tuple type with reverse order of the members (recursive).
            //!
            //! The recursion level follows from the the number of parameters removed from the original parameter list.
            //! Each level of recursion takes the front most (current) parameter from the list and inherits from
            //! a `TupleReverseData` data using the remaining parameter list.
            //! Accessing the different members of the tuple type can happen through (implicit)
            //! type casting to the base class. recursively.
            //!
            //! Note: This definition has at least two parameters in the list.
            //!
            //! \tparam ValueT the type of the current member
            //! \tparam Tail the remaining parameter list
            //!
            template <typename ValueT, typename ...Tail>
            class TupleReverseData<ValueT, Tail...> : public TupleReverseData<Tail...>
            {
                using Base = TupleReverseData<Tail...>;

            public:
                //!
                //! \brief Standard constructor.
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleReverseData() : value{} {};

                //!
                //! \brief Constructor.
                //!
                //! Assign some `value` to the current member.
                //! All members are assigned the same value.
                //!
                //! \param value the value to be assigned to all members
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleReverseData(ValueT value) : Base(value), value(value) {}

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
                constexpr TupleReverseData(ValueT value, Tail... tail) : Base(tail...), value(value) {}

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
            //! \brief Definition of a tuple type with reverse order of the members (recursion anchor).
            //!
            //! Note: This definition has a single parameter in the list.
            //!
            //! \tparam ValueT the type of the current member
            //!
            template <typename ValueT>
            class TupleReverseData<ValueT>
            {
            public:
                //!
                //! \brief Standard constructor.
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleReverseData() : value{} {};

                //!
                //! \brief Constructor.
                //!
                //! Assign some `value` to the current member.
                //!
                //! \param value the value to be assigned to the current member
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleReverseData(ValueT value) : value(value) {}

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
            //! \brief Definition of a tuple type with no members.
            //!
            template <>
            class TupleReverseData<> 
            {
            public:
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleReverseData() {}
            };
           
            namespace
            {
                //!
                //! \brief A Helper type for the construction of a tuple type with non-reverse order of the members.
                //!
                //! The definition uses an index sequence for the parameter and argument inversion.
                //!
                //! \tparam IndexSequenceT an sequence of indices from 0 to the number of template parameters in the variadic list
                //! \tparam ValueT a variadic list of type parameters
                //! 
                template <typename IndexSequenceT, typename ...ValueT>
                struct TupleDataHelper;

                //!
                //! \brief Definition of the helper type for the construction of a tuple type with non-reverse order of the members.
                //!
                //! \tparam ValueT a variadic list of type parameters
                //! \tparam I a list of indices used for the parameter and argument inversion
                //! 
                template <typename ...ValueT, SizeT ...I>
                struct TupleDataHelper<::XXX_NAMESPACE::dataTypes::IndexSequence<I...>, ValueT...>
                {
                    static constexpr SizeT N = sizeof...(ValueT);

                    static_assert(N == sizeof...(I), "error: type parameter count does not match non-type parameter count.");

                    using Type = TupleReverseData<typename ::XXX_NAMESPACE::variadic::Pack<ValueT...>::template Type<(N - 1) - I>...>;

                    //!
                    //! \brief Argument inversion.
                    //!
                    //! This function inverts the order of the arguments and forwards them to a `TupleReverseData` type with
                    //! inverted template parameter list.
                    //!
                    //! \param values arguments to be forwarded to a `TupleReverseData` type with inverted template parameter list
                    //! \return a `TupleReverseData` type with inverted template parameter list
                    //!
                    HOST_VERSION
                    CUDA_DEVICE_VERSION
                    static inline constexpr auto Make(ValueT... values) -> Type
                    {
                        return {::XXX_NAMESPACE::variadic::Pack<ValueT...>::template Value<(N - 1) - I>(values...)...};
                    }
                };
            }

            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            //!
            //! \brief A tuple type with non-reverse order of the members.
            //!
            //! The order of the members in memory is NOT reversed: 
            //! `TupleData<int, float> {..}` is equivalent to `class {int member_1; float member_2;..}`.
            //! This tuple type inherits from a `TupleReverseData` type with inverted template parameter list.
            //!
            //! Note: This definition has at least two parameters in the list.
            //!
            //! \tparam ValueT a parameter list defining the member types
            //!
            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            template <typename ...ValueT>
            class TupleData : public TupleDataHelper<::XXX_NAMESPACE::dataTypes::IndexSequenceT<sizeof...(ValueT)>, ValueT...>::Type
            {
                using Base = typename TupleDataHelper<::XXX_NAMESPACE::dataTypes::IndexSequenceT<sizeof...(ValueT)>, ValueT...>::Type;

                // Needed for access to `Base` type.
                template <SizeT Index, typename ...T>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                friend constexpr auto& Get(TupleData<T...>&);

                template <SizeT Index, typename ...T>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                friend constexpr const auto& Get(const TupleData<T...>&);

            public:    
                //!
                //! \brief Standard constructor.
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleData() : Base() {}
                
                //!
                //! \brief Constructor.
                //!
                //! Assign the same `value` to all members.
                //!
                //! \param value the value to be assigned to all members
                //!
                template <typename T>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleData(T value) : Base(value)
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
                constexpr TupleData(ValueT... values) : Base(TupleDataHelper<::XXX_NAMESPACE::dataTypes::IndexSequenceT<sizeof...(ValueT)>, ValueT...>::Make(values...)) {}
            };

            //!
            //! \brief Definition of a tuple type with non-reverse order of the members.
            //!
            //! Note: This definition has a single parameter in the list.
            //!
            //! \tparam ValueT the type of the member
            //!
            template <typename ValueT>
            class TupleData<ValueT> : public TupleReverseData<ValueT>
            {
                using Base = TupleReverseData<ValueT>;
                
                // Needed for access to `Base` type.
                template <SizeT Index, typename ...T>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                friend constexpr auto& Get(TupleData<T...>&);

                template <SizeT Index, typename ...T>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                friend constexpr const auto& Get(const TupleData<T...>&);

            public:
                //!
                //! \brief Standard constructor.
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleData() : Base() {}
            
                //!
                //! \brief Constructor.
                //!
                //! Assign some `value` to the member.
                //!
                //! \param value the value to be assigned to the member
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleData(ValueT value) : Base(value) {}
            };

            //!
            //! \brief Definition of a tuple type with no members.
            //!
            template <>
            class TupleData<>
            {
            public:
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleData() {}
            };
        
            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            //!
            //! \brief A tuple type with non-reverse order of the members (base class).
            //!
            //! The order of the members in memory is NOT reversed: 
            //! `TupleData<int, float> {..}` is equivalent to `class {int member_1; float member_2;..}`.
            //! This tuple type inherits from a `TupleReverseData` type with inverted template parameter list.
            //!
            //! Note: This definition has at least two parameters in the list.
            //!
            //! \tparam ValueT a parameter list defining the member types
            //!
            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            template <typename ...ValueT>
            class TupleBase
            {   
                template <typename Proxy, SizeT ...I>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                static inline constexpr auto Dereference(const Proxy& proxy, ::XXX_NAMESPACE::dataTypes::IndexSequence<I...>)
                    -> TupleData<ValueT...>
                {
                    return {internal::Get<I>(proxy.data)...};
                }

            public:
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleBase() : data{} {};
                
                template <typename T>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleBase(T value) : data(value) 
                {
                    static_assert(::XXX_NAMESPACE::variadic::Pack<T, ValueT...>::IsConvertible(), "error: types are not convertible.");
                }
        
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleBase(ValueT... values) : data(values...) {}

                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleBase(const TupleBase& tuple) : data(tuple.data) {}

                template <typename ...T>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleBase(const TupleProxy<T...>& proxy) : data(Dereference(proxy, ::XXX_NAMESPACE::dataTypes::MakeIndexSequence<sizeof...(ValueT)>()))
                {
                    static_assert(sizeof...(T) == sizeof...(ValueT), "error: parameter lists have different size.");
                    static_assert(::XXX_NAMESPACE::variadic::Pack<T...>::template IsConvertible<ValueT...>(), "error: types are not convertible.");
                }
                
                TupleData<ValueT...> data;
            };

            template <typename ValueT_1, typename ValueT_2, typename ValueT_3, typename ValueT_4>
            class TupleBase<ValueT_1, ValueT_2, ValueT_3, ValueT_4>
            {
            public:
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

                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleBase(const TupleBase& tuple) : data(tuple.data) {}

                template <typename T_1, typename T_2, typename T_3, typename T_4>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleBase(const TupleProxy<T_1, T_2, T_3, T_4>& proxy) : data(proxy.x, proxy.y, proxy.z, proxy.w) {}
                
                union 
                {
                    struct { ValueT_1 x; ValueT_2 y; ValueT_3 z; ValueT_4 w; };
                    TupleData<ValueT_1, ValueT_2, ValueT_3, ValueT_4> data;
                };
            };

            template <typename ValueT_1, typename ValueT_2, typename ValueT_3>
            class TupleBase<ValueT_1, ValueT_2, ValueT_3>
            {
            public:
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

                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleBase(const TupleBase& tuple) : data(tuple.data) {}

                template <typename T_1, typename T_2, typename T_3>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleBase(const TupleProxy<T_1, T_2, T_3>& proxy) : data(proxy.x, proxy.y, proxy.z) {}
                
                union 
                {
                    struct { ValueT_1 x; ValueT_2 y; ValueT_3 z; };
                    TupleData<ValueT_1, ValueT_2, ValueT_3> data;
                };
            };

            template <typename ValueT_1, typename ValueT_2>
            class TupleBase<ValueT_1, ValueT_2>
            {
            public:
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

                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleBase(const TupleBase& tuple) : data(tuple.data) {}

                template <typename T_1, typename T_2>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleBase(const TupleProxy<T_1, T_2>& proxy) : data(proxy.x, proxy.y) {}
                
                union 
                {
                    struct { ValueT_1 x; ValueT_2 y; };
                    TupleData<ValueT_1, ValueT_2> data;
                };
            };

            template <typename ValueT>
            class TupleBase<ValueT>
            {
            public:
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleBase() : data{} {}

                template <typename T>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleBase(T value) : data(value) {}

                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleBase(const TupleBase& tuple) : data(tuple.data) {}

                template <typename T>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleBase(const TupleProxy<T>& proxy) : data(proxy.x) {}
                
                union 
                {
                    struct { ValueT x; };
                    TupleData<ValueT    > data;
                };
            };

            template <>
            class TupleBase<>
            {
            public:
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleBase() {}
            };
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //!
        //! \brief A tuple type with 3 elements.
        //!
        //! The implementation comprises simple arithmetic functions.
        //!
        //! \tparam T_1 some type
        //! \tparam T_2 some type
        //! \tparam T_3 some type
        //!
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////       
        template <typename ...ValueT>
        class Tuple : public internal::TupleBase<ValueT...>
        {
            using Base = internal::TupleBase<ValueT...>;
            
            static_assert(::XXX_NAMESPACE::variadic::Pack<ValueT...>::IsFundamental(), "error: fundamental parameter types assumed.");
            static_assert(!::XXX_NAMESPACE::variadic::Pack<ValueT...>::IsConst(), "error: non-const parameter types assumed.");
            static_assert(!::XXX_NAMESPACE::variadic::Pack<ValueT...>::IsVoid(), "error: non-void parameter types assumed.");
            static_assert(!::XXX_NAMESPACE::variadic::Pack<ValueT...>::IsVolatile(), "error: non-volatile parameter types assumed.");
            
          public:
            using Type = Tuple<ValueT...>;
            using Proxy = internal::TupleProxy<ValueT...>;

            HOST_VERSION
            CUDA_DEVICE_VERSION
            constexpr Tuple() = default;

            template <typename T>
            HOST_VERSION 
            CUDA_DEVICE_VERSION
            constexpr Tuple(T value) : Base(value) {}

            HOST_VERSION 
            CUDA_DEVICE_VERSION
            constexpr Tuple(ValueT... values) : Base(values...) {}

            template <typename ...T>
            HOST_VERSION 
            CUDA_DEVICE_VERSION
            constexpr Tuple(const Tuple<T...>& tuple) : Base(tuple) {}
            
            template <typename ...T>
            HOST_VERSION 
            CUDA_DEVICE_VERSION
            constexpr Tuple(const internal::TupleProxy<T...>& proxy) : Base(proxy) {}
          
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline auto operator-() const
            {
                Tuple tuple;

                ::XXX_NAMESPACE::compileTime::Loop<sizeof...(ValueT)>::Execute([&tuple, this] (const auto I) { internal::Get<I>(tuple.data) = -internal::Get<I>(Base::data); });

                return tuple;
            }
            
#define MACRO(OP, IN_T) \
    template <typename ...OtherT> \
    HOST_VERSION \
    CUDA_DEVICE_VERSION \
    inline auto operator OP(const IN_T<OtherT...>& other) -> Tuple& \
    { \
        ::XXX_NAMESPACE::compileTime::Loop<sizeof...(ValueT)>::Execute([&other, this] (const auto I) { internal::Get<I>(Base::data) OP internal::Get<I>(other.data); }); \
    \
        return *this; \
    }

            MACRO(=, Tuple)
            MACRO(+=, Tuple)
            MACRO(-=, Tuple)
            MACRO(*=, Tuple)
            MACRO(/=, Tuple)
            MACRO(=, internal::TupleProxy)
            MACRO(+=, internal::TupleProxy)
            MACRO(-=, internal::TupleProxy)
            MACRO(*=, internal::TupleProxy)
            MACRO(/=, internal::TupleProxy)
#undef MACRO

#define MACRO(OP) \
    template <typename OtherT> \
    HOST_VERSION \
    CUDA_DEVICE_VERSION \
    inline auto operator OP(const OtherT& value) -> Tuple& \
    { \
        ::XXX_NAMESPACE::compileTime::Loop<sizeof...(ValueT)>::Execute([&value, this] (const auto I) { internal::Get<I>(Base::data) OP value; }); \
    \
        return *this; \
    }

            MACRO(=)
            MACRO(+=)
            MACRO(-=)
            MACRO(*=)
            MACRO(/=)
#undef MACRO
            
        };

        template <>
        class Tuple<>
        {
        public:
            HOST_VERSION
            CUDA_DEVICE_VERSION
            constexpr Tuple() {};   
        };
     
        template <typename ...ValueT>
        std::ostream& operator<<(std::ostream& os, const Tuple<ValueT...>& tuple)
        {
            os << "( ";

            ::XXX_NAMESPACE::compileTime::Loop<sizeof...(ValueT)>::Execute([&tuple, &os] (const auto I) { os << Get<I>(tuple) << " "; });

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
        template <typename ...ValueT>
        struct ProvidesProxy<::XXX_NAMESPACE::dataTypes::Tuple<ValueT...>>
        {
            static constexpr bool value = true;
        };

        template <typename ...ValueT>
        struct ProvidesProxy<const ::XXX_NAMESPACE::dataTypes::Tuple<ValueT...>>
        {
            static constexpr bool value = true;
        };
    } // namespace internal
} // namespace XXX_NAMESPACE

#endif