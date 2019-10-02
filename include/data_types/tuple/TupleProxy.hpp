// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(DATA_TYPES_TUPLE_TUPLE_PROXY_HPP)
#define DATA_TYPES_TUPLE_TUPLE_PROXY_HPP

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#if !defined(TUPLE_NAMESPACE)
#define TUPLE_NAMESPACE XXX_NAMESPACE
#endif

#include <common/DataLayout.hpp>
#include <data_types/DataTypes.hpp>
#include <common/Memory.hpp>
#include <auxiliary/Template.hpp>

// some forward declarations
namespace XXX_NAMESPACE
{
    namespace internal 
    {
        template <typename P, typename R>
        class iterator;
    }
}

namespace XXX_NAMESPACE
{
    namespace internal 
    {
        template <typename X, SizeType N, SizeType D, ::XXX_NAMESPACE::memory::DataLayout L>
        class Accessor;
    }
}

namespace TUPLE_NAMESPACE
{
    namespace internal
    {
        template <typename T_1, typename T_2, typename T_3>
        class tuple_proxy
        {
            static_assert(std::is_fundamental<T_1>::value, "error: T_1 is not a fundamental data type");
            static_assert(!std::is_void<T_1>::value, "error: T_1 is void -> not allowed");
            static_assert(!std::is_volatile<T_1>::value, "error: T_1 is volatile -> not allowed");

            static_assert(std::is_fundamental<T_2>::value, "error: T_2 is not a fundamental data type");
            static_assert(!std::is_void<T_2>::value, "error: T_2 is void -> not allowed");
            static_assert(!std::is_volatile<T_2>::value, "error: T_2 is volatile -> not allowed");

            static_assert(std::is_fundamental<T_3>::value, "error: T_3 is not a fundamental data type");
            static_assert(!std::is_void<T_3>::value, "error: T_3 is void -> not allowed");
            static_assert(!std::is_volatile<T_3>::value, "error: T_3 is volatile -> not allowed");

            template <typename, SizeType, SizeType, ::XXX_NAMESPACE::memory::DataLayout>
            friend class XXX_NAMESPACE::internal::Accessor;

            using T_1_unqualified = typename std::remove_cv<T_1>::type;
            using T_2_unqualified = typename std::remove_cv<T_2>::type;
            using T_3_unqualified = typename std::remove_cv<T_3>::type;

            static constexpr bool is_const_type = ::XXX_NAMESPACE::variadic::Pack<T_1, T_2, T_3>::IsConst();
            static constexpr bool is_homogeneous = ::XXX_NAMESPACE::variadic::Pack<T_1, T_2, T_3>::IsSame() || ::XXX_NAMESPACE::variadic::Pack<T_1, T_2, T_3>::SameSize();

        public:

            // some meta data: to be generated by the proxy generator
            using type = tuple_proxy<T_1, T_2, T_3>;
            using ConstType = tuple_proxy<const T_1, const T_2, const T_3>;
            using BasePointerType = typename std::conditional<is_homogeneous, XXX_NAMESPACE::pointer<T_1, T_2, T_3>, XXX_NAMESPACE::multi_pointer<T_1, T_2, T_3>>::type;
            using original_type = typename std::conditional<is_const_type, 
                const tuple<T_1_unqualified, T_2_unqualified, T_3_unqualified>,
                tuple<T_1_unqualified, T_2_unqualified, T_3_unqualified>>::type;

            T_1& x;
            T_2& y;
            T_3& z;
            
        private:

            HOST_VERSION
            CUDA_DEVICE_VERSION
            tuple_proxy(std::tuple<T_1&, T_2&, T_3&> t)
                :
                x(std::get<0>(t)),
                y(std::get<1>(t)),
                z(std::get<2>(t)) {}

        public:
            
        #define MACRO(OP, IN_T)                                                                 \
            inline tuple_proxy& operator OP (TUPLE_NAMESPACE::IN_T<T_1, T_2, T_3>& t)           \
            {                                                                                   \
                x OP t.x;                                                                       \
                y OP t.y;                                                                       \
                z OP t.z;                                                                       \
                return *this;                                                                   \
            }                                                                                   \
                                                                                                \
            inline tuple_proxy& operator OP (const TUPLE_NAMESPACE::IN_T<T_1, T_2, T_3>& t)     \
            {                                                                                   \
                x OP t.x;                                                                       \
                y OP t.y;                                                                       \
                z OP t.z;                                                                       \
                return *this;                                                                   \
            }                                                                                   \
                                                                                                \
            template <typename X_1, typename X_2, typename X_3>                                 \
            inline tuple_proxy& operator OP (TUPLE_NAMESPACE::IN_T<X_1, X_2, X_3>& t)           \
            {                                                                                   \
                x OP t.x;                                                                       \
                y OP t.y;                                                                       \
                z OP t.z;                                                                       \
                return *this;                                                                   \
            }                                                                                   \
                                                                                                \
            template <typename X_1, typename X_2, typename X_3>                                 \
            inline tuple_proxy& operator OP (const TUPLE_NAMESPACE::IN_T<X_1, X_2, X_3>& t)     \
            {                                                                                   \
                x OP t.x;                                                                       \
                y OP t.y;                                                                       \
                z OP t.z;                                                                       \
                return *this;                                                                   \
            }                                                                                   \

            MACRO(=, tuple)
            MACRO(+=, tuple)
            MACRO(-=, tuple)
            MACRO(*=, tuple)
            MACRO(/=, tuple)

            MACRO(=, internal::tuple_proxy)
            MACRO(+=, internal::tuple_proxy)
            MACRO(-=, internal::tuple_proxy)
            MACRO(*=, internal::tuple_proxy)
            MACRO(/=, internal::tuple_proxy)

        #undef MACRO

        #define MACRO(OP)                                                                       \
            template <typename X>                                                               \
            inline tuple_proxy& operator OP (const X xyz)                                       \
            {                                                                                   \
                x OP xyz;                                                                       \
                y OP xyz;                                                                       \
                z OP xyz;                                                                       \
                return *this;                                                                   \
            }                                                                                   \

            MACRO(=)
            MACRO(+=)
            MACRO(-=)
            MACRO(*=)
            MACRO(/=)

        #undef MACRO
        };

        template <typename T_1, typename T_2, typename T_3>
        std::ostream& operator<<(std::ostream& os, const tuple_proxy<T_1, T_2, T_3>& vp)
        {
            os << "(" << vp.x << "," << vp.y << "," << vp.z << ")";
            return os;
        }
    }
}

#endif