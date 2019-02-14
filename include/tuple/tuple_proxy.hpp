// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(TUPLE_TUPLE_PROXY_HPP)
#define TUPLE_TUPLE_PROXY_HPP

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#if !defined(TUPLE_NAMESPACE)
#define TUPLE_NAMESPACE XXX_NAMESPACE
#endif

#include "../common/data_layout.hpp"

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

            template <typename X, std::size_t D, data_layout L, typename Enabled>
            friend class accessor;

        public:

            // some meta data: to be generated by the proxy generator
            using type = tuple_proxy<T_1, T_2, T_3>;
            using const_type = tuple_proxy<const T_1, const T_2, const T_3>;
            using base_pointer = XXX_NAMESPACE::multi_pointer_inhomogeneous<T_1, T_2, T_3>;

            T_1& x;
            T_2& y;
            T_3& z;
            
        private:

            tuple_proxy(base_pointer m)
                :
                x(*(std::get<0>(m.base))),
                y(*(std::get<1>(m.base))),
                z(*(std::get<2>(m.base))) {}

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