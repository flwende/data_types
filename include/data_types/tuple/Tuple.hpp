// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(DATA_TYPES_TUPLE_TUPLE_HPP)
#define DATA_TYPES_TUPLE_TUPLE_HPP

#if !defined(OLD)

#include <type_traits>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#include <common/Math.hpp>
#include <auxiliary/Template.hpp>
#include <platform/Target.hpp>

namespace XXX_NAMESPACE
{
    namespace dataTypes
    {
        namespace internal
        {
            // Forward declaration.
            template <typename...>
            class TupleProxy;
        } // namespace internal

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
        template <typename T_1, typename T_2, typename T_3>
        class Tuple
        {
            static_assert(std::is_fundamental<T_1>::value, "error: T_1 is not a fundamental data type");
            static_assert(!std::is_void<T_1>::value, "error: T_1 is void -> not allowed");
            static_assert(!std::is_volatile<T_1>::value, "error: T_1 is volatile -> not allowed");
            static_assert(!std::is_const<T_1>::value, "error: T_1 is const -> not allowed");

            static_assert(std::is_fundamental<T_2>::value, "error: T_2 is not a fundamental data type");
            static_assert(!std::is_void<T_2>::value, "error: T_2 is void -> not allowed");
            static_assert(!std::is_volatile<T_2>::value, "error: T_2 is volatile -> not allowed");
            static_assert(!std::is_const<T_2>::value, "error: T_2 is const -> not allowed");

            static_assert(std::is_fundamental<T_3>::value, "error: T_3 is not a fundamental data type");
            static_assert(!std::is_void<T_3>::value, "error: T_3 is void -> not allowed");
            static_assert(!std::is_volatile<T_3>::value, "error: T_3 is volatile -> not allowed");
            static_assert(!std::is_const<T_3>::value, "error: T_3 is const -> not allowed");

          public:
            using Type = Tuple<T_1, T_2, T_3>;
            using Proxy = internal::TupleProxy<T_1, T_2, T_3>;
            using ValueT = typename ::XXX_NAMESPACE::dataTypes::Compare<T_1, typename ::XXX_NAMESPACE::dataTypes::Compare<T_2, T_3>::StrongerT>::StrongerT;

            HOST_VERSION
            CUDA_DEVICE_VERSION
            Tuple() = default;

            template <typename X>
            HOST_VERSION CUDA_DEVICE_VERSION Tuple(const X& xyz) : x(xyz), y(xyz), z(xyz)
            {
            }
            Tuple(const T_1& x, const T_2& y, const T_3& z) : x(x), y(y), z(z) {}

            template <typename X_1, typename X_2, typename X_3>
            HOST_VERSION CUDA_DEVICE_VERSION Tuple(const Tuple<X_1, X_2, X_3>& t) : x(t.x), y(t.y), z(t.z)
            {
            }

            template <typename X_1, typename X_2, typename X_3>
            HOST_VERSION CUDA_DEVICE_VERSION Tuple(const internal::TupleProxy<X_1, X_2, X_3>& tp) : x(tp.x), y(tp.y), z(tp.z)
            {
            }

            inline auto operator-() const { return Tuple(-x, -y, -z); }

#define MACRO(OP, IN_T)                                                                                                                                                                                                    \
    template <typename X_1, typename X_2, typename X_3>                                                                                                                                                                    \
    inline auto operator OP(const IN_T<X_1, X_2, X_3>& t)->Tuple&                                                                                                                                                          \
    {                                                                                                                                                                                                                      \
        x OP t.x;                                                                                                                                                                                                          \
        y OP t.y;                                                                                                                                                                                                          \
        z OP t.z;                                                                                                                                                                                                          \
                                                                                                                                                                                                                           \
        return *this;                                                                                                                                                                                                      \
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

#define MACRO(OP)                                                                                                                                                                                                          \
    template <typename X>                                                                                                                                                                                                  \
    inline auto operator OP(const X xyz)->Tuple&                                                                                                                                                                           \
    {                                                                                                                                                                                                                      \
        x OP xyz;                                                                                                                                                                                                          \
        y OP xyz;                                                                                                                                                                                                          \
        z OP xyz;                                                                                                                                                                                                          \
                                                                                                                                                                                                                           \
        return *this;                                                                                                                                                                                                      \
    }

            MACRO(=)
            MACRO(+=)
            MACRO(-=)
            MACRO(*=)
            MACRO(/=)
#undef MACRO

            T_1 x;
            T_2 y;
            T_3 z;
        };
        
        template <typename T_1, typename T_2, typename T_3>
        std::ostream& operator<<(std::ostream& os, const Tuple<T_1, T_2, T_3>& v)
        {
            os << "(" << v.x << "," << v.y << "," << v.z << ")";
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
        template <typename ...T>
        struct ProvidesProxy<::XXX_NAMESPACE::dataTypes::Tuple<T...>>
        {
            static constexpr bool value = true;
        };

        template <typename ...T>
        struct ProvidesProxy<const ::XXX_NAMESPACE::dataTypes::Tuple<T...>>
        {
            static constexpr bool value = true;
        };
    } // namespace internal

    namespace math
    {
        template <typename ...T>
        struct Func<::XXX_NAMESPACE::dataTypes::Tuple<T...>>
        {
            using Tuple = ::XXX_NAMESPACE::dataTypes::Tuple<T...>;
            using ValueT = typename std::remove_cv<typename Tuple::ValueT>::type;

            static constexpr ValueT One = Func<ValueT>::One;
            static constexpr ValueT MinusOne = Func<ValueT>::MinusOne;

            template <typename X>
            static auto sqrt(X x) -> Tuple
            {
                return ::XXX_NAMESPACE::math::sqrt(x);
            }

            template <typename X>
            static auto log(X x) -> Tuple
            {
                return ::XXX_NAMESPACE::math::log(x);
            }

            template <typename X>
            static auto exp(X x) -> Tuple
            {
                return ::XXX_NAMESPACE::math::exp(x);
            }
        };
    } // namespace math
} // namespace XXX_NAMESPACE

#else

#include <type_traits>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#include <common/Math.hpp>
#include <auxiliary/Template.hpp>
#include <platform/Target.hpp>

namespace XXX_NAMESPACE
{
    namespace dataTypes
    {
        namespace internal
        {
            // Forward declaration.
            template <typename...>
            class TupleProxy;
        } // namespace internal

        //template 

        namespace
        {
            template <SizeT>
            struct GetImplementation;
        }

        template <SizeT Index, typename TupleT>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        auto& Get(TupleT& tuple)
        {
            return GetImplementation<Index>{}(tuple);
        }

        template <SizeT Index, typename TupleT>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        const auto& Get(const TupleT& tuple)
        {
            return GetImplementation<Index>{}(tuple);
        }

        namespace
        {
#define MACRO(FUNC_NAME, RETURN_TYPE, MEMBER_NAME) \
            HOST_VERSION \
            CUDA_DEVICE_VERSION \
            auto FUNC_NAME() -> RETURN_TYPE& { return MEMBER_NAME; } \
\
            HOST_VERSION \
            CUDA_DEVICE_VERSION \
            auto FUNC_NAME() const -> const RETURN_TYPE& { return MEMBER_NAME; }

            template <typename ...T>
            class TupleBase;

            template <typename T_1>
            class TupleBase<T_1>
            {
                template <SizeT>
                friend class GetImplementation;

            protected:
                HOST_VERSION
                CUDA_DEVICE_VERSION
                TupleBase() = default;    

                template <typename X_1>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                TupleBase(const X_1& x) : x(x) {}

                template <typename X_1>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                TupleBase(const TupleBase<X_1>& other) : x(other.x) {}

                template <typename X_1>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                TupleBase(const internal::TupleProxy<X_1>& proxy) : x(proxy.x) {}

                MACRO(X, T_1, x)

            public:
                T_1 x;
            };

            template <typename T_1, typename T_2>
            class TupleBase<T_1, T_2>
            {
                template <SizeT>
                friend class GetImplementation;

            protected:
                HOST_VERSION
                CUDA_DEVICE_VERSION
                TupleBase() = default;    

                template <typename X>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                TupleBase(const X& xy) : x(xy), y(xy) {}

                template <typename X_1, typename X_2>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                TupleBase(const X_1& x, const X_2& y) : x(x), y(y) {}

                template <typename X_1, typename X_2>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                TupleBase(const TupleBase<X_1, X_2>& other) : x(other.x), y(other.y) {}

                template <typename X_1, typename X_2>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                TupleBase(const internal::TupleProxy<X_1, X_2>& proxy) : x(proxy.x), y(proxy.y) {}

                MACRO(X, T_1, x)
                MACRO(Y, T_2, y)

            public:
                T_1 x;
                T_2 y;
            };
            /*
            template <typename T_1, typename T_2, typename T_3>
            class TupleBase<T_1, T_2, T_3>
            {
                template <SizeT>
                friend class GetImplementation;

            protected:
                HOST_VERSION
                CUDA_DEVICE_VERSION
                TupleBase() = default;    

                template <typename X, typename Enable = std::enable_if_t<std::is_fundamental<X>::value>>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                TupleBase(const X xyz) : x(xyz), y(xyz), z(xyz) {}

                template <typename X_1, typename X_2, typename X_3>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                TupleBase(const X_1 x, const X_2 y, const X_3 z) : x(x), y(y), z(z) {}

                template <typename X_1, typename X_2, typename X_3>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                TupleBase(const TupleBase<X_1, X_2, X_3>& other) : x(other.x), y(other.y), z(other.z) {}
                
                template <typename X_1, typename X_2, typename X_3>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                TupleBase(const internal::TupleProxy<X_1, X_2, X_3>& proxy) : x(proxy.x), y(proxy.y), z(proxy.z) {}

                MACRO(X, T_1, x)
                MACRO(Y, T_2, y)
                MACRO(Z, T_3, z)

            public:
                T_1 x;
                T_2 y;
                T_3 z;
            };
            */
            template <typename T_1, typename T_2, typename T_3>
            class TupleBase<T_1, T_2, T_3>
            {
                template <SizeT>
                friend class GetImplementation;

            protected:
                HOST_VERSION
                CUDA_DEVICE_VERSION
                TupleBase() = default;    

                template <typename X, typename Enable = std::enable_if_t<std::is_fundamental<X>::value>>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                TupleBase(const X x) : x(x), y(x), z(x) {}

                template <typename ...X, typename EnableSize = std::enable_if_t<(sizeof...(X) > 0)>, typename EnableType = std::enable_if_t<::XXX_NAMESPACE::variadic::Pack<X...>::IsFundamental()>>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                TupleBase(const X... x) 
                    : 
                    x(::XXX_NAMESPACE::variadic::Pack<X...>::template Value<0>(x...)), 
                    y(::XXX_NAMESPACE::variadic::Pack<X...>::template Value<1>(x...)),
                    z(::XXX_NAMESPACE::variadic::Pack<X...>::template Value<2>(x...))
                {}

                /*
                template <typename X_1, typename X_2, typename X_3>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                TupleBase(const TupleBase<X_1, X_2, X_3>& other) : x(other.x), y(other.y), z(other.z) {}
                */
                /*
                template <typename X_1, typename X_2, typename X_3>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                TupleBase(const internal::TupleProxy<X_1, X_2, X_3>& proxy) : x(proxy.x), y(proxy.y), z(proxy.z) {}
                */
                template <typename ...X>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                TupleBase(const TupleBase<X...>& other) 
                    : 
                    x(::XXX_NAMESPACE::dataTypes::Get<0>(other)), y(::XXX_NAMESPACE::dataTypes::Get<1>(other)), z(::XXX_NAMESPACE::dataTypes::Get<2>(other)) 
                {}

                template <typename X_1, typename X_2, typename X_3>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                TupleBase(const internal::TupleProxy<X_1, X_2, X_3>& proxy) : x(proxy.x), y(proxy.y), z(proxy.z) {}

                MACRO(X, T_1, x)
                MACRO(Y, T_2, y)
                MACRO(Z, T_3, z)

            public:
                union {
                struct {
                T_1 x;
                T_2 y;
                T_3 z;
                };

                };
            };
            
            
            template <typename T_1, typename T_2, typename T_3, typename T_4>
            class TupleBase<T_1, T_2, T_3, T_4>
            {
                template <SizeT>
                friend class GetImplementation;

            protected:
                HOST_VERSION
                CUDA_DEVICE_VERSION
                TupleBase() = default;    

                template <typename X>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                TupleBase(const X& xyzw) : x(xyzw), y(xyzw), z(xyzw), w(xyzw) {}

                template <typename X_1, typename X_2, typename X_3, typename X_4>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                TupleBase(const X_1& x, const X_2& y, const X_3& z, const X_4& w) : x(x), y(y), z(z), w(w) {}

                template <typename X_1, typename X_2, typename X_3, typename X_4>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                TupleBase(const TupleBase<X_1, X_2, X_3, X_4>& other) : x(other.x), y(other.y), z(other.z), w(other.w) {}

                template <typename X_1, typename X_2, typename X_3, typename X_4>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                TupleBase(const internal::TupleProxy<X_1, X_2, X_3, X_4>& proxy) : x(proxy.x), y(proxy.y), z(proxy.z), w(proxy.w) {}

                MACRO(X, T_1, x)
                MACRO(Y, T_2, y)
                MACRO(Z, T_3, z)
                MACRO(W, T_4, w)

            public:
                T_1 x;
                T_2 y;
                T_3 z;
                T_4 w;
            };
#undef MACRO

#define MACRO(INDEX, MEMBER_NAME) \
            template <> \
            struct GetImplementation<INDEX> \
            { \
                template <typename TupleT> \
                HOST_VERSION \
                CUDA_DEVICE_VERSION \
                auto& operator()(TupleT& tuple) \
                { \
                    return tuple.MEMBER_NAME(); \
                } \
\
                template <typename TupleT> \
                HOST_VERSION \
                CUDA_DEVICE_VERSION \
                const auto& operator()(const TupleT& tuple) const \
                { \
                    return tuple.MEMBER_NAME(); \
                } \
            }; \

            MACRO(0, X);
            MACRO(1, Y);
            MACRO(2, Z);
            MACRO(3, W);
#undef MACRO
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
        /*
        template <typename T_1, typename T_2, typename T_3>
        class Tuple
        {
            //using Base = TupleBase<T_1, T_2, T_3>;

            static_assert(std::is_fundamental<T_1>::value, "error: T_1 is not a fundamental data type");
            static_assert(!std::is_void<T_1>::value, "error: T_1 is void -> not allowed");
            static_assert(!std::is_volatile<T_1>::value, "error: T_1 is volatile -> not allowed");
            static_assert(!std::is_const<T_1>::value, "error: T_1 is const -> not allowed");

            static_assert(std::is_fundamental<T_2>::value, "error: T_2 is not a fundamental data type");
            static_assert(!std::is_void<T_2>::value, "error: T_2 is void -> not allowed");
            static_assert(!std::is_volatile<T_2>::value, "error: T_2 is volatile -> not allowed");
            static_assert(!std::is_const<T_2>::value, "error: T_2 is const -> not allowed");

            static_assert(std::is_fundamental<T_3>::value, "error: T_3 is not a fundamental data type");
            static_assert(!std::is_void<T_3>::value, "error: T_3 is void -> not allowed");
            static_assert(!std::is_volatile<T_3>::value, "error: T_3 is volatile -> not allowed");
            static_assert(!std::is_const<T_3>::value, "error: T_3 is const -> not allowed");

          public:
            using Type = Tuple<T_1, T_2, T_3>;
            using Proxy = internal::TupleProxy<T_1, T_2, T_3>;
            using ValueT = typename ::XXX_NAMESPACE::dataTypes::Compare<T_1, typename ::XXX_NAMESPACE::dataTypes::Compare<T_2, T_3>::StrongerT>::StrongerT;

            HOST_VERSION
            CUDA_DEVICE_VERSION
            Tuple() = default;

            template <typename X>
            HOST_VERSION CUDA_DEVICE_VERSION Tuple(const X& xyz) : x(xyz), y(xyz), z(xyz)
            {
            }
            Tuple(const T_1& x, const T_2& y, const T_3& z) : x(x), y(y), z(z) {}

            template <typename X_1, typename X_2, typename X_3>
            HOST_VERSION CUDA_DEVICE_VERSION Tuple(const Tuple<X_1, X_2, X_3>& t) : x(t.x), y(t.y), z(t.z)
            {
            }

            template <typename X_1, typename X_2, typename X_3>
            HOST_VERSION CUDA_DEVICE_VERSION Tuple(const internal::TupleProxy<X_1, X_2, X_3>& tp) : x(tp.x), y(tp.y), z(tp.z)
            {
            }

            inline auto operator-() const { return Tuple(-x, -y, -z); }

#define MACRO(OP, IN_T)                                                                                                                                                                                                    \
    template <typename X_1, typename X_2, typename X_3>                                                                                                                                                                    \
    inline auto operator OP(const IN_T<X_1, X_2, X_3>& t)->Tuple&                                                                                                                                                          \
    {                                                                                                                                                                                                                      \
        x OP t.x;                                                                                                                                                                                                          \
        y OP t.y;                                                                                                                                                                                                          \
        z OP t.z;                                                                                                                                                                                                          \
                                                                                                                                                                                                                           \
        return *this;                                                                                                                                                                                                      \
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

#define MACRO(OP)                                                                                                                                                                                                          \
    template <typename X>                                                                                                                                                                                                  \
    inline auto operator OP(const X xyz)->Tuple&                                                                                                                                                                           \
    {                                                                                                                                                                                                                      \
        x OP xyz;                                                                                                                                                                                                          \
        y OP xyz;                                                                                                                                                                                                          \
        z OP xyz;                                                                                                                                                                                                          \
                                                                                                                                                                                                                           \
        return *this;                                                                                                                                                                                                      \
    }

            MACRO(=)
            MACRO(+=)
            MACRO(-=)
            MACRO(*=)
            MACRO(/=)
#undef MACRO

            T_1 x;
            T_2 y;
            T_3 z;
        };
        */
        /*
        template <typename T_1, typename T_2, typename T_3>
        class Tuple : public TupleBase<T_1, T_2, T_3>
        {
            using Base = TupleBase<T_1, T_2, T_3>;
            
            static_assert(::XXX_NAMESPACE::variadic::Pack<T_1, T_2, T_3>::IsFundamental(), "error: fundamental parameter types assumed.");
            static_assert(!::XXX_NAMESPACE::variadic::Pack<T_1, T_2, T_3>::IsConst(), "error: non-const parameter types assumed.");
            static_assert(!::XXX_NAMESPACE::variadic::Pack<T_1, T_2, T_3>::IsVoid(), "error: non-void parameter types assumed.");
            static_assert(!::XXX_NAMESPACE::variadic::Pack<T_1, T_2, T_3>::IsVolatile(), "error: non-volatile parameter types assumed.");

          public:
            using Type = Tuple<T_1, T_2, T_3>;
            using Proxy = internal::TupleProxy<T_1, T_2, T_3>;
            using ValueT = typename ::XXX_NAMESPACE::dataTypes::Compare<T_1, typename ::XXX_NAMESPACE::dataTypes::Compare<T_2, T_3>::StrongerT>::StrongerT;

            HOST_VERSION
            CUDA_DEVICE_VERSION
            Tuple() = default;

            template <typename X>
            HOST_VERSION CUDA_DEVICE_VERSION Tuple(const X& x) : Base(x)
            {
            }
            Tuple(const T_1& x, const T_2& y, const T_3& z) : Base(x, y, z) {}

            template <typename X_1, typename X_2, typename X_3>
            HOST_VERSION CUDA_DEVICE_VERSION Tuple(const Tuple<X_1, X_2, X_3>& t) : Base(t.x, t.y, t.z)
            {
            }

            template <typename X_1, typename X_2, typename X_3>
            HOST_VERSION CUDA_DEVICE_VERSION Tuple(const internal::TupleProxy<X_1, X_2, X_3>& tp) : Base(tp.x, tp.y, tp.z)
            {
            }

            inline auto operator-() const { return Tuple(-x, -y, -z); }

#define MACRO(OP, IN_T)                                                                                                                                                                                                    \
    template <typename X_1, typename X_2, typename X_3>                                                                                                                                                                    \
    inline auto operator OP(const IN_T<X_1, X_2, X_3>& t)->Tuple&                                                                                                                                                          \
    {                                                                                                                                                                                                                      \
        x OP t.x;                                                                                                                                                                                                          \
        y OP t.y;                                                                                                                                                                                                          \
        z OP t.z;                                                                                                                                                                                                          \
                                                                                                                                                                                                                           \
        return *this;                                                                                                                                                                                                      \
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

#define MACRO(OP)                                                                                                                                                                                                          \
    template <typename X>                                                                                                                                                                                                  \
    inline auto operator OP(const X xyz)->Tuple&                                                                                                                                                                           \
    {                                                                                                                                                                                                                      \
        x OP xyz;                                                                                                                                                                                                          \
        y OP xyz;                                                                                                                                                                                                          \
        z OP xyz;                                                                                                                                                                                                          \
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
        */
        
        template <typename ...T>
        class Tuple : public TupleBase<T...>
        {
            using Base = TupleBase<T...>;
            
            static_assert(::XXX_NAMESPACE::variadic::Pack<T...>::IsFundamental(), "error: fundamental parameter types assumed.");
            static_assert(!::XXX_NAMESPACE::variadic::Pack<T...>::IsConst(), "error: non-const parameter types assumed.");
            static_assert(!::XXX_NAMESPACE::variadic::Pack<T...>::IsVoid(), "error: non-void parameter types assumed.");
            static_assert(!::XXX_NAMESPACE::variadic::Pack<T...>::IsVolatile(), "error: non-volatile parameter types assumed.");

          public:
            using Type = Tuple<T...>;
            using Proxy = internal::TupleProxy<T...>;
            using ValueT = typename ::XXX_NAMESPACE::dataTypes::Compare<T...>::StrongerT;

            HOST_VERSION
            CUDA_DEVICE_VERSION
            Tuple() = default;

            template <typename X, typename Enable = std::enable_if_t<std::is_fundamental<X>::value>>
            HOST_VERSION CUDA_DEVICE_VERSION Tuple(const X x) : Base(x)
            {
            }

            //Tuple(const T... x) : Base(x...) {}
            Tuple(const T... x) : Base(
                ::XXX_NAMESPACE::variadic::Pack<T...>::template Value<0>(x...),
                ::XXX_NAMESPACE::variadic::Pack<T...>::template Value<1>(x...),
                ::XXX_NAMESPACE::variadic::Pack<T...>::template Value<2>(x...)) {}
            
            template <typename ...OtherT>
            HOST_VERSION CUDA_DEVICE_VERSION Tuple(const Tuple<OtherT...>& t) : Base(t)
            {
            }

            template <typename ...OtherT>
            HOST_VERSION CUDA_DEVICE_VERSION Tuple(const internal::TupleProxy<OtherT...>& tp) : Base(tp)
            {
            }

            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline auto operator-() const
            {
                Tuple tuple;

                ::XXX_NAMESPACE::compileTime::Loop<sizeof...(T)>::Execute([&tuple, this] (const auto I) { Get<I>(tuple) = -Get<I>(*this); });

                return tuple;
            }
            
#define MACRO(OP, IN_T)                                                                                                                                                                                                    \
    template <typename X_1, typename X_2, typename X_3>                                                                                                                                                                    \
    HOST_VERSION \
    CUDA_DEVICE_VERSION \
    inline auto operator OP(const IN_T<X_1, X_2, X_3>& tuple)->Tuple&                                                                                                                                                      \
    {        \
        ::XXX_NAMESPACE::compileTime::Loop<sizeof...(T)>::Execute([&tuple, this] (const auto I) { Get<I>(*this) OP Get<I>(tuple); }); \
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

#define MACRO(OP)                                                                                                                                                                                                          \
    template <typename X>                                                                                                                                                                                                  \
    HOST_VERSION \
    CUDA_DEVICE_VERSION \
    inline auto operator OP(const X x)->Tuple&                                                                                                                                                                           \
    {                                                                                                                                                                                                                      \
        ::XXX_NAMESPACE::compileTime::Loop<sizeof...(T)>::Execute([&x, this] (const auto I) { Get<I>(*this) OP x; }); \
        return *this;                                                                                                                                                                                                      \
    }

            MACRO(=)
            MACRO(+=)
            MACRO(-=)
            MACRO(*=)
            MACRO(/=)
#undef MACRO

        };
        
        template <typename T_1, typename T_2, typename T_3>
        std::ostream& operator<<(std::ostream& os, const Tuple<T_1, T_2, T_3>& v)
        {
            os << "(" << v.x << "," << v.y << "," << v.z << ")";
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
        template <typename ...T>
        struct ProvidesProxy<::XXX_NAMESPACE::dataTypes::Tuple<T...>>
        {
            static constexpr bool value = true;
        };

        template <typename ...T>
        struct ProvidesProxy<const ::XXX_NAMESPACE::dataTypes::Tuple<T...>>
        {
            static constexpr bool value = true;
        };
    } // namespace internal

    namespace math
    {
        template <typename ...T>
        struct Func<::XXX_NAMESPACE::dataTypes::Tuple<T...>>
        {
            using Tuple = ::XXX_NAMESPACE::dataTypes::Tuple<T...>;
            using ValueT = typename std::remove_cv<typename Tuple::ValueT>::type;

            static constexpr ValueT One = Func<ValueT>::One;
            static constexpr ValueT MinusOne = Func<ValueT>::MinusOne;

            template <typename X>
            static auto sqrt(X x) -> Tuple
            {
                return ::XXX_NAMESPACE::math::sqrt(x);
            }

            template <typename X>
            static auto log(X x) -> Tuple
            {
                return ::XXX_NAMESPACE::math::log(x);
            }

            template <typename X>
            static auto exp(X x) -> Tuple
            {
                return ::XXX_NAMESPACE::math::exp(x);
            }
        };
    } // namespace math
} // namespace XXX_NAMESPACE

#endif

#endif