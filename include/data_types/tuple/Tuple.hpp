// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(DATA_TYPES_TUPLE_TUPLE_HPP)
#define DATA_TYPES_TUPLE_TUPLE_HPP

#if defined(OLD)

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

#elif defined(NEW)

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

#else

#include <type_traits>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#include <common/Math.hpp>
#include <auxiliary/Template.hpp>
#include <data_types/integer_sequence/IntegerSequence.hpp>
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
        
            template <typename ...ValueT>
            class TupleReverseData;

            template <typename ValueT, typename ...Tail>
            class TupleReverseData<ValueT, Tail...> : public TupleReverseData<Tail...>
            {
                using Base = TupleReverseData<Tail...>;

            public:
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleReverseData() : value{} {};

                template <typename T>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleReverseData(T value) : Base(value), value(value) {}

                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleReverseData(ValueT value, Tail... tail) : Base(tail...), value(value) {}

                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr auto Get() -> ValueT& { return value; }

                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr auto Get() const -> const ValueT& { return value; }

            protected:
                ValueT value;
            };

            template <typename ValueT>
            class TupleReverseData<ValueT>
            {
            public:
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleReverseData() : value{} {};

                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleReverseData(ValueT value) : value(value) {}

                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr auto Get() -> ValueT& { return value; }

                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr auto Get() const -> const ValueT& { return value; }

            protected:
                ValueT value;
            
            };

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
                template <typename IntegerSequenceT, typename ...ValueT>
                struct TupleDataHelper;

                template <typename ...ValueT, SizeT ...I>
                struct TupleDataHelper<::XXX_NAMESPACE::dataTypes::IntegerSequence<SizeT, I...>, ValueT...>
                {
                    static constexpr SizeT N = sizeof...(ValueT);

                    static_assert(N == sizeof...(I), "error: type parameter count does not match non-type parameter count");

                    using Type = TupleReverseData<typename ::XXX_NAMESPACE::variadic::Pack<ValueT...>::template Type<(N - 1) - I>...>;

                    HOST_VERSION
                    CUDA_DEVICE_VERSION
                    static constexpr auto Make(ValueT... values)
                    {
                        return Type(::XXX_NAMESPACE::variadic::Pack<ValueT...>::template Value<(N - 1) - I>(values...)...);
                    }
                };
            }

            template <typename ...ValueT>
            class TupleData : public TupleDataHelper<::XXX_NAMESPACE::dataTypes::IntegerSequenceT<SizeT, sizeof...(ValueT)>, ValueT...>::Type
            {
            public:
                using Base = typename TupleDataHelper<::XXX_NAMESPACE::dataTypes::IntegerSequenceT<SizeT, sizeof...(ValueT)>, ValueT...>::Type;
                
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleData() : Base() {}
                
                template <typename T>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleData(T value) : Base(value) {}
                
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleData(ValueT... values) : Base(TupleDataHelper<::XXX_NAMESPACE::dataTypes::IntegerSequenceT<SizeT, sizeof...(ValueT)>, ValueT...>::Make(values...)) {}
            };

            template <typename ValueT>
            class TupleData<ValueT> : public TupleReverseData<ValueT>
            {
            public:
                using Base = TupleReverseData<ValueT>;

                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleData() : Base() {}
            
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleData(ValueT value) : Base(value) {}
            };

            template <>
            class TupleData<>
            {
            public:
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleData() {}
            };
        }

        namespace
        {
            template <SizeT Index, typename ...ValueT>
            struct GetImplementation;

            template <SizeT Index, typename ValueT, typename ...Tail>
            struct GetImplementation<Index, ValueT, Tail...>
            {
                HOST_VERSION
                CUDA_DEVICE_VERSION
                static constexpr auto& Value(::XXX_NAMESPACE::dataTypes::internal::TupleReverseData<ValueT, Tail...>& tuple)
                {
                    return GetImplementation<Index - 1, Tail...>::Value(tuple);
                }

                HOST_VERSION
                CUDA_DEVICE_VERSION
                static constexpr const auto& Value(const ::XXX_NAMESPACE::dataTypes::internal::TupleReverseData<ValueT, Tail...>& tuple)
                {
                    return GetImplementation<Index - 1, Tail...>::Value(tuple);
                }
            };

            template <typename ValueT, typename ...Tail>
            struct GetImplementation<0, ValueT, Tail...>
            {
                HOST_VERSION
                CUDA_DEVICE_VERSION
                static constexpr auto& Value(::XXX_NAMESPACE::dataTypes::internal::TupleReverseData<ValueT, Tail...>& tuple)
                {
                    return tuple.Get();
                }

                HOST_VERSION
                CUDA_DEVICE_VERSION
                static constexpr const auto& Value(const ::XXX_NAMESPACE::dataTypes::internal::TupleReverseData<ValueT, Tail...>& tuple)
                {
                    return tuple.Get();
                }
            };
        }

        template <SizeT Index, typename ...ValueT>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        constexpr auto& Get(internal::TupleReverseData<ValueT...>& tuple)
        {
            constexpr SizeT N = sizeof...(ValueT);

            static_assert(N > 0 && Index < N, "error: out of bounds data access.");

            return GetImplementation<Index, ValueT...>::Value(tuple);
        }

        template <SizeT Index, typename ...ValueT>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        constexpr const auto& Get(const internal::TupleReverseData<ValueT...>& tuple)
        {
            constexpr SizeT N = sizeof...(ValueT);

            static_assert(N > 0 && Index < N, "error: out of bounds data access.");

            return GetImplementation<Index, ValueT...>::Value(tuple);
        }

        template <SizeT Index, typename ...ValueT>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        constexpr auto& Get(internal::TupleData<ValueT...>& tuple)
        {
            constexpr SizeT N = sizeof...(ValueT);

            static_assert(N > 0 && Index < N, "error: out of bounds data access.");

            return Get<(N - 1) - Index>(static_cast<typename internal::TupleData<ValueT...>::Base&>(tuple));
        }

        template <SizeT Index, typename ...ValueT>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        constexpr const auto& Get(const internal::TupleData<ValueT...>& tuple)
        {
            constexpr SizeT N = sizeof...(ValueT);

            static_assert(N > 0 && Index < N, "error: out of bounds data access.");

            return Get<(N - 1) - Index>(static_cast<const typename internal::TupleData<ValueT...>::Base&>(tuple));
        }

        namespace internal
        {
            template <typename ...ValueT>
            class TupleBase
            {   
                template <typename ...OtherT, SizeT ...I>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                static constexpr auto Dereference(const TupleProxy<OtherT...>& proxy, ::XXX_NAMESPACE::dataTypes::IntegerSequence<SizeT, I...>)
                    -> TupleData<ValueT...>
                {
                    return {Get<I>(proxy.data)...};
                }

            public:
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleBase() : data{} {};
                
                template <typename T>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleBase(T value) : data(value) {}
        
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleBase(ValueT... values) : data(values...) {}

                template <typename ...T>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr TupleBase(const TupleProxy<T...>& proxy) : data(Dereference(proxy, ::XXX_NAMESPACE::dataTypes::MakeIntegerSequence<SizeT, sizeof...(ValueT)>())) {}
                
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
            using ScalarT = typename ::XXX_NAMESPACE::dataTypes::Compare<ValueT...>::StrongerT;

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

                ::XXX_NAMESPACE::compileTime::Loop<sizeof...(ValueT)>::Execute([&tuple, this] (const auto I) { Get<I>(tuple.data) = -Get<I>(Base::data); });

                return tuple;
            }
            
#define MACRO(OP, IN_T) \
    template <typename ...OtherT> \
    HOST_VERSION \
    CUDA_DEVICE_VERSION \
    inline auto operator OP(const IN_T<OtherT...>& other) -> Tuple& \
    { \
        ::XXX_NAMESPACE::compileTime::Loop<sizeof...(ValueT)>::Execute([&other, this] (const auto I) { Get<I>(Base::data) OP Get<I>(other.data); }); \
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
        ::XXX_NAMESPACE::compileTime::Loop<sizeof...(ValueT)>::Execute([&value, this] (const auto I) { Get<I>(Base::data) OP value; }); \
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

            ::XXX_NAMESPACE::compileTime::Loop<sizeof...(ValueT)>::Execute([&tuple, &os] (const auto I) { os << Get<I>(tuple.data) << " "; });
            
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

    namespace math
    {
        template <typename TupleT>
        struct Func;

        template <>
        struct Func<::XXX_NAMESPACE::dataTypes::Tuple<>> {};

        template <typename ...ValueT>
        struct Func<::XXX_NAMESPACE::dataTypes::Tuple<ValueT...>>
        {
            using Tuple = ::XXX_NAMESPACE::dataTypes::Tuple<ValueT...>;
            using ScalarT = typename std::remove_cv<typename Tuple::ScalarT>::type;

            static constexpr ScalarT One = Func<ScalarT>::One;
            static constexpr ScalarT MinusOne = Func<ScalarT>::MinusOne;

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