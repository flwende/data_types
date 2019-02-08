// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(VEC_VEC_PROXY_HPP)
#define VEC_VEC_PROXY_HPP

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#if !defined(VEC_NAMESPACE)
#define VEC_NAMESPACE XXX_NAMESPACE
#endif

namespace VEC_NAMESPACE
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //! \brief A proxy data type for vec<T, D>
    //!
    //! This data type is returned by accessor<T, D, Layout>::operator[] if D = 1 and SoA data layout.
    //! It holds references to component(s) x [,y [and z]], so that data access via,
    //! e.g. obj[ ]..[ ].x, is possible.
    //!
    //! \tparam T data type
    //! \tparam D dimension
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    namespace internal
    {
        //! \brief D = 1 specialization with component x
        //!
        //! \tparam T data type
        template <typename T>
        class vec_proxy<T, 1>
        {
            static_assert(std::is_fundamental<T>::value, "error: T is not a fundamental data type");
            static_assert(!std::is_void<T>::value, "error: T is void -> not allowed");
            static_assert(!std::is_volatile<T>::value, "error: T is volatile -> not allowed");

        public:

            using type = vec_proxy<T, 1>;
            using const_type = vec_proxy<const T, 1>;
            using T_unqualified = typename std::remove_cv<T>::type;
            //! Remember the template type parameter T
            using element_type = T;
            //! Remember the template parameter D (=1)
            static constexpr std::size_t d = 1;
            using base_pointer = XXX_NAMESPACE::multi_pointer_n<T, 1>;

            T& x;

            vec_proxy(base_pointer m)
                :
                x(m.base) {}

            vec_proxy(const vec_proxy& vp)
                :
                x(vp.x) {}

            vec_proxy(VEC_NAMESPACE::vec<T_unqualified, 1>& v)
                :
                x(v.x) {}

            vec_proxy(const VEC_NAMESPACE::vec<T_unqualified, 1>& v)
                :
                x(v.x) {}

            inline VEC_NAMESPACE::vec<T_unqualified, 1>  operator-() const
            {
                return VEC_NAMESPACE::vec<T_unqualified, 1>(-x);
            }

        #define MACRO(OP, IN_T)                                                 \
            inline vec_proxy& operator OP (VEC_NAMESPACE::IN_T<T, 1>& v)        \
            {                                                                   \
                x OP v.x;                                                       \
                return *this;                                                   \
            }                                                                   \
                                                                                \
            inline vec_proxy& operator OP (const VEC_NAMESPACE::IN_T<T, 1>& v)  \
            {                                                                   \
                x OP v.x;                                                       \
                return *this;                                                   \
            }                                                                   \
                                                                                \
            template <typename X>                                               \
            inline vec_proxy& operator OP (VEC_NAMESPACE::IN_T<X, 1>& v)        \
            {                                                                   \
                x OP v.x;                                                       \
                return *this;                                                   \
            }                                                                   \
                                                                                \
            template <typename X>                                               \
            inline vec_proxy& operator OP (const VEC_NAMESPACE::IN_T<X, 1>& v)  \
            {                                                                   \
                x OP v.x;                                                       \
                return *this;                                                   \
            }                                                                   \

            MACRO(=, vec)
            MACRO(+=, vec)
            MACRO(-=, vec)
            MACRO(*=, vec)
            MACRO(/=, vec)

            MACRO(=, internal::vec_proxy)
            MACRO(+=, internal::vec_proxy)
            MACRO(-=, internal::vec_proxy)
            MACRO(*=, internal::vec_proxy)
            MACRO(/=, internal::vec_proxy)

        #undef MACRO

        #define MACRO(OP)                                                       \
            template <typename X>                                               \
            inline vec_proxy& operator OP (const X x)                           \
            {                                                                   \
                x OP x;                                                         \
                return *this;                                                   \
            }                                                                   \

            MACRO(=)
            MACRO(+=)
            MACRO(-=)
            MACRO(*=)
            MACRO(/=)

        #undef MACRO

            //! \brief Return the Euclidean norm of the vector
            //!
            //! \return Euclidean norm
            inline T length() const
            {
                return std::abs(x);
            }
        };

        //! \brief D = 2 specialization with component x
        //!
        //! \tparam T data type
        template <typename T>
        class vec_proxy<T, 2>
        {
            static_assert(std::is_fundamental<T>::value, "error: T is not a fundamental data type");
            static_assert(!std::is_void<T>::value, "error: T is void -> not allowed");
            static_assert(!std::is_volatile<T>::value, "error: T is volatile -> not allowed");

        public:

            using type = vec_proxy<T, 2>;
            using const_type = vec_proxy<const T, 2>;
            using T_unqualified = typename std::remove_cv<T>::type;
            //! Remember the template type parameter T
            using element_type = T;
            //! Remember the template parameter D (=2)
            static constexpr std::size_t d = 2;
            using base_pointer = XXX_NAMESPACE::multi_pointer_n<T, 2>;

            T& x;
            T& y;

            vec_proxy(base_pointer m)
                :
                x(m.base[0 * m.n_innermost]),
                y(m.base[1 * m.n_innermost]) {}

            vec_proxy(const vec_proxy& vp)
                :
                x(vp.x),
                y(vp.y) {}

            vec_proxy(VEC_NAMESPACE::vec<T_unqualified, 2>& v)
                :
                x(v.x),
                y(v.y) {}

            vec_proxy(const VEC_NAMESPACE::vec<T_unqualified, 2>& v)
                :
                x(v.x),
                y(v.y) {}

            inline VEC_NAMESPACE::vec<T_unqualified, 2>  operator-() const
            {
                return VEC_NAMESPACE::vec<T_unqualified, 2>(-x, -y);
            }

        #define MACRO(OP, IN_T)                                                 \
            inline vec_proxy& operator OP (VEC_NAMESPACE::IN_T<T, 2>& v)        \
            {                                                                   \
                x OP v.x;                                                       \
                y OP v.y;                                                       \
                return *this;                                                   \
            }                                                                   \
                                                                                \
            inline vec_proxy& operator OP (const VEC_NAMESPACE::IN_T<T, 2>& v)  \
            {                                                                   \
                x OP v.x;                                                       \
                y OP v.y;                                                       \
                return *this;                                                   \
            }                                                                   \
                                                                                \
            template <typename X>                                               \
            inline vec_proxy& operator OP (VEC_NAMESPACE::IN_T<X, 2>& v)        \
            {                                                                   \
                x OP v.x;                                                       \
                y OP v.y;                                                       \
                return *this;                                                   \
            }                                                                   \
                                                                                \
            template <typename X>                                               \
            inline vec_proxy& operator OP (const VEC_NAMESPACE::IN_T<X, 2>& v)  \
            {                                                                   \
                x OP v.x;                                                       \
                y OP v.y;                                                       \
                return *this;                                                   \
            }                                                                   \

            MACRO(=, vec)
            MACRO(+=, vec)
            MACRO(-=, vec)
            MACRO(*=, vec)
            MACRO(/=, vec)

            MACRO(=, internal::vec_proxy)
            MACRO(+=, internal::vec_proxy)
            MACRO(-=, internal::vec_proxy)
            MACRO(*=, internal::vec_proxy)
            MACRO(/=, internal::vec_proxy)

        #undef MACRO

        #define MACRO(OP)                                                       \
            template <typename X>                                               \
            inline vec_proxy& operator OP (const X xy)                          \
            {                                                                   \
                x OP xy;                                                        \
                y OP xy;                                                        \
                return *this;                                                   \
            }                                                                   \

            MACRO(=)
            MACRO(+=)
            MACRO(-=)
            MACRO(*=)
            MACRO(/=)

        #undef MACRO

            //! \brief Return the Euclidean norm of the vector
            //!
            //! \return Euclidean norm
            inline T length() const
            {
                return AUXILIARY_NAMESPACE::math<T>::sqrt(x * x + y * y);
            }
        };

        //! \brief D = 3 specialization with component x
        //!
        //! \tparam T data type
        template <typename T>
        class vec_proxy<T, 3>
        {
            static_assert(std::is_fundamental<T>::value, "error: T is not a fundamental data type");
            static_assert(!std::is_void<T>::value, "error: T is void -> not allowed");
            static_assert(!std::is_volatile<T>::value, "error: T is volatile -> not allowed");

        public:

            using type = vec_proxy<T, 3>;
            using const_type = vec_proxy<const T, 3>;
            using T_unqualified = typename std::remove_cv<T>::type;
            //! Remember the template type parameter T
            using element_type = T;
            //! Remember the template parameter D (=3)
            static constexpr std::size_t d = 3;
            using base_pointer = XXX_NAMESPACE::multi_pointer_n<T, 3>;

            vec_proxy(base_pointer m)
                :
                x(m.base[0 * m.n_innermost]),
                y(m.base[1 * m.n_innermost]),
                z(m.base[2 * m.n_innermost]) {}

            vec_proxy(const vec_proxy& vp)
                :
                x(vp.x),
                y(vp.y),
                z(vp.z) {}

            vec_proxy(VEC_NAMESPACE::vec<T_unqualified, 3>& v)
                :
                x(v.x),
                y(v.y),
                z(v.z) {}

            vec_proxy(const VEC_NAMESPACE::vec<T_unqualified, 3>& v)
                :
                x(v.x),
                y(v.y),
                z(v.z) {}

            inline VEC_NAMESPACE::vec<T_unqualified, 3>  operator-() const
            {
                return VEC_NAMESPACE::vec<T_unqualified, 3>(-x, -y, -z);
            }
            
        #define MACRO(OP, IN_T)                                                 \
            inline vec_proxy& operator OP (VEC_NAMESPACE::IN_T<T, 3>& v)        \
            {                                                                   \
                x OP v.x;                                                       \
                y OP v.y;                                                       \
                z OP v.z;                                                       \
                return *this;                                                   \
            }                                                                   \
                                                                                \
            inline vec_proxy& operator OP (const VEC_NAMESPACE::IN_T<T, 3>& v)  \
            {                                                                   \
                x OP v.x;                                                       \
                y OP v.y;                                                       \
                z OP v.z;                                                       \
                return *this;                                                   \
            }                                                                   \
                                                                                \
            template <typename X>                                               \
            inline vec_proxy& operator OP (VEC_NAMESPACE::IN_T<X, 3>& v)        \
            {                                                                   \
                x OP v.x;                                                       \
                y OP v.y;                                                       \
                z OP v.z;                                                       \
                return *this;                                                   \
            }                                                                   \
                                                                                \
            template <typename X>                                               \
            inline vec_proxy& operator OP (const VEC_NAMESPACE::IN_T<X, 3>& v)  \
            {                                                                   \
                x OP v.x;                                                       \
                y OP v.y;                                                       \
                z OP v.z;                                                       \
                return *this;                                                   \
            }                                                                   \

            MACRO(=, vec)
            MACRO(+=, vec)
            MACRO(-=, vec)
            MACRO(*=, vec)
            MACRO(/=, vec)

            MACRO(=, internal::vec_proxy)
            MACRO(+=, internal::vec_proxy)
            MACRO(-=, internal::vec_proxy)
            MACRO(*=, internal::vec_proxy)
            MACRO(/=, internal::vec_proxy)

        #undef MACRO

        #define MACRO(OP)                                                       \
            template <typename X>                                               \
            inline vec_proxy& operator OP (const X xyz)                         \
            {                                                                   \
                x OP xyz;                                                       \
                y OP xyz;                                                       \
                z OP xyz;                                                       \
                return *this;                                                   \
            }                                                                   \

            MACRO(=)
            MACRO(+=)
            MACRO(-=)
            MACRO(*=)
            MACRO(/=)

        #undef MACRO

            //! \brief Return the Euclidean norm of the vector
            //!
            //! \return Euclidean norm
            inline T length() const
            {
                return AUXILIARY_NAMESPACE::math<T>::sqrt(x * x + y * y + z * z);
            }

            T& x;
            T& y;
            T& z;
        };

        template <typename T>
        std::ostream& operator<<(std::ostream& os, const vec_proxy<T, 1>& vp)
        {
            os << "(" << vp.x << ")";
            return os;
        }

        template <typename T>
        std::ostream& operator<<(std::ostream& os, const vec_proxy<T, 2>& vp)
        {
            os << "(" << vp.x << "," << vp.y << ")";
            return os;
        }

        template <typename T>
        std::ostream& operator<<(std::ostream& os, const vec_proxy<T, 3>& vp)
        {
            os << "(" << vp.x << "," << vp.y << "," << vp.z << ")";
            return os;
        }
    }
}

#endif