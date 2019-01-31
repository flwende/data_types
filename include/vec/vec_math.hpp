// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(VEC_VEC_MATH_HPP)
#define VEC_VEC_MATH_HPP

#if defined(old)

#if !defined(VEC_NAMESPACE)
#if !defined(XXX_NAMESPACE)
#define VEC_NAMESPACE fw
#else
#define VEC_NAMESPACE XXX_NAMESPACE
#endif
#endif


namespace VEC_NAMESPACE
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // vec<T, D> math functions (applied element wise)
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    #if defined(HAVE_SYCL)
    #define MACRO(FUNC, IN_T, C)										\
    template <typename T, std::size_t D>									\
    inline vec<T, D> FUNC(const IN_T<C T, D>& v)								\
    {													\
        const vec<T, D> tmp(v);										\
        return cl::sycl::FUNC(tmp.sycl_vec);								\
    }													\
                                                        \
    template <typename T, std::size_t D>									\
    inline vec<T, D> FUNC(const IN_T<C T, D>&& v)								\
    {													\
        const vec<T, D> tmp(v);										\
        return cl::sycl::FUNC(tmp.sycl_vec);								\
    }
    #else
    #define MACRO(FUNC, IN_T, C)										\
    template <typename T>											\
    inline vec<T, 1> FUNC(const IN_T<C T, 1>& v)								\
    {													\
        return vec<T, 1>(MISC_NAMESPACE::math<T>::FUNC(v.x));						\
    }													\
                                                        \
    template <typename T>											\
    inline vec<T, 1> FUNC(const IN_T<C T, 1>&& v)								\
    {													\
        return vec<T, 1>(MISC_NAMESPACE::math<T>::FUNC(v.x));						\
    }													\
                                                        \
    template <typename T>											\
    inline vec<T, 2> FUNC(const IN_T<C T, 2>& v)								\
    {													\
        return vec<T, 2>(MISC_NAMESPACE::math<T>::FUNC(v.x), MISC_NAMESPACE::math<T>::FUNC(v.y));	\
    }													\
                                                        \
    template <typename T>											\
    inline vec<T, 2> FUNC(const IN_T<C T, 2>&& v)								\
    {													\
        return vec<T, 2>(MISC_NAMESPACE::math<T>::FUNC(v.x), MISC_NAMESPACE::math<T>::FUNC(v.y));	\
    }													\
                                                        \
    template <typename T>											\
    inline vec<T, 3> FUNC(const IN_T<C T, 3>& v)								\
    {													\
        return vec<T, 3>(MISC_NAMESPACE::math<T>::FUNC(v.x), MISC_NAMESPACE::math<T>::FUNC(v.y),	\
                        MISC_NAMESPACE::math<T>::FUNC(v.z));				\
    }													\
                                                        \
    template <typename T>											\
    inline vec<T, 3> FUNC(const IN_T<C T, 3>&& v)								\
    {													\
        return vec<T, 3>(MISC_NAMESPACE::math<T>::FUNC(v.x), MISC_NAMESPACE::math<T>::FUNC(v.y),	\
                        MISC_NAMESPACE::math<T>::FUNC(v.z));				\
    }
    #endif

    MACRO(sqrt, vec, )
    MACRO(log, vec, )
    MACRO(exp, vec, )

    #define MACRO_C1(FUNC, IN_T)										\
    MACRO(FUNC, IN_T,      )										\
    MACRO(FUNC, IN_T, const)										\

    MACRO_C1(sqrt, detail::vec_proxy)
    MACRO_C1(log, detail::vec_proxy)
    MACRO_C1(exp, detail::vec_proxy)

    #undef MACRO_C1
    #undef MACRO

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Operators
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    #if defined(HAVE_SYCL)
    #define MACRO(OP, IN_1_T, IN_2_T, C_1, C_2)								\
    template <typename T, std::size_t D>									\
    inline vec<T, D> operator OP (const IN_1_T<C_1 T, D>& v_1, const IN_2_T<C_2 T, D>& v_2)			\
    {													\
        return vec<T, D>(v_1).sycl_vec OP vec<T, D>(v_2).sycl_vec;					\
    }													\
                                                        \
    template <typename T, std::size_t D>									\
    inline vec<T, D> operator OP (const IN_1_T<C_1 T, D>&& v_1, const IN_2_T<C_2 T, D>& v_2)		\
    {													\
        return vec<T, D>(v_1).sycl_vec OP vec<T, D>(v_2).sycl_vec;					\
    }													\
                                                        \
    template <typename T, std::size_t D>									\
    inline vec<T, D> operator OP (const IN_1_T<C_1 T, D>& v_1, const IN_2_T<C_2 T, D>&& v_2)		\
    {													\
        return vec<T, D>(v_1).sycl_vec OP vec<T, D>(v_2).sycl_vec;					\
    }													\
                                                        \
    template <typename T, std::size_t D>									\
    inline vec<T, D> operator OP (const IN_1_T<C_1 T, D>&& v_1, const IN_2_T<C_2 T, D>&& v_2)		\
    {													\
        return vec<T, D>(v_1).sycl_vec OP vec<T, D>(v_2).sycl_vec;					\
    }
    #else
    #define MACRO(OP, IN_1_T, IN_2_T, C_1, C_2)								\
    template <typename T>											\
    inline vec<T, 1> operator OP (const IN_1_T<C_1 T, 1>& v_1, const IN_2_T<C_2 T, 1>& v_2)			\
    {													\
        return vec<T, 1>(v_1.x OP v_2.x);								\
    }													\
                                                        \
    template <typename T>											\
    inline vec<T, 1> operator OP (const IN_1_T<C_1 T, 1>&& v_1, const IN_2_T<C_2 T, 1>& v_2)		\
    {													\
        return vec<T, 1>(v_1.x OP v_2.x);								\
    }													\
                                                        \
    template <typename T>											\
    inline vec<T, 1> operator OP (const IN_1_T<C_1 T, 1>& v_1, const IN_2_T<C_2 T, 1>&& v_2)		\
    {													\
        return vec<T, 1>(v_1.x OP v_2.x);								\
    }													\
                                                        \
    template <typename T>											\
    inline vec<T, 1> operator OP (const IN_1_T<C_1 T, 1>&& v_1, const IN_2_T<C_2 T, 1>&& v_2)		\
    {													\
        return vec<T, 1>(v_1.x OP v_2.x);								\
    }													\
                                                        \
    template <typename T>											\
    inline vec<T, 2> operator OP (const IN_1_T<C_1 T, 2>& v_1, const IN_2_T<C_2 T, 2>& v_2)			\
    {													\
        return vec<T, 2>(v_1.x OP v_2.x, v_1.y OP v_2.y);						\
    }													\
                                                        \
    template <typename T>											\
    inline vec<T, 2> operator OP (const IN_1_T<C_1 T, 2>&& v_1, const IN_2_T<C_2 T, 2>& v_2)		\
    {													\
        return vec<T, 2>(v_1.x OP v_2.x, v_1.y OP v_2.y);						\
    }													\
                                                        \
    template <typename T>											\
    inline vec<T, 2> operator OP (const IN_1_T<C_1 T, 2>& v_1, const IN_2_T<C_2 T, 2>&& v_2)		\
    {													\
        return vec<T, 2>(v_1.x OP v_2.x, v_1.y OP v_2.y);						\
    }													\
                                                        \
    template <typename T>											\
    inline vec<T, 2> operator OP (const IN_1_T<C_1 T, 2>&& v_1, const IN_2_T<C_2 T, 2>&& v_2)		\
    {													\
        return vec<T, 2>(v_1.x OP v_2.x, v_1.y OP v_2.y);						\
    }													\
                                                        \
    template <typename T>											\
    inline vec<T, 3> operator OP (const IN_1_T<C_1 T, 3>& v_1, const IN_2_T<C_2 T, 3>& v_2)			\
    {													\
        return vec<T, 3>(v_1.x OP v_2.x, v_1.y OP v_2.y, v_1.z OP v_2.z);				\
    }													\
                                                        \
    template <typename T>											\
    inline vec<T, 3> operator OP (const IN_1_T<C_1 T, 3>&& v_1, const IN_2_T<C_2 T, 3>& v_2)		\
    {													\
        return vec<T, 3>(v_1.x OP v_2.x, v_1.y OP v_2.y, v_1.z OP v_2.z);				\
    }													\
                                                        \
    template <typename T>											\
    inline vec<T, 3> operator OP (const IN_1_T<C_1 T, 3>& v_1, const IN_2_T<C_2 T, 3>&& v_2)		\
    {													\
        return vec<T, 3>(v_1.x OP v_2.x, v_1.y OP v_2.y, v_1.z OP v_2.z);				\
    }													\
                                                        \
    template <typename T>											\
    inline vec<T, 3> operator OP (const IN_1_T<C_1 T, 3>&& v_1, const IN_2_T<C_2 T, 3>&& v_2)		\
    {													\
        return vec<T, 3>(v_1.x OP v_2.x, v_1.y OP v_2.y, v_1.z OP v_2.z);				\
    }
    #endif													\

    MACRO(*, vec, vec, , )
    MACRO(/, vec, vec, , )
    MACRO(+, vec, vec, , )
    MACRO(-, vec, vec, , )

    #define MACRO_C1(OP, IN_1_T, IN_2_T)									\
    MACRO(OP, IN_1_T, IN_2_T,      ,      )									\
    MACRO(OP, IN_1_T, IN_2_T, const,      )									\
    MACRO(OP, IN_2_T, IN_1_T,      ,      )									\
    MACRO(OP, IN_2_T, IN_1_T,      , const)									\
    
    #define MACRO_C2(OP, IN_1_T, IN_2_T)									\
    MACRO(OP, IN_1_T, IN_2_T,      ,      )									\
    MACRO(OP, IN_1_T, IN_2_T, const, const)									\

    MACRO_C1(*, detail::vec_proxy, vec)
    MACRO_C1(/, detail::vec_proxy, vec)
    MACRO_C1(+, detail::vec_proxy, vec)
    MACRO_C1(-, detail::vec_proxy, vec)

    MACRO_C2(*, detail::vec_proxy, detail::vec_proxy)
    MACRO_C2(/, detail::vec_proxy, detail::vec_proxy)
    MACRO_C2(+, detail::vec_proxy, detail::vec_proxy)
    MACRO_C2(-, detail::vec_proxy, detail::vec_proxy)

    #undef MACRO_C1
    #undef MACRO_C2
    #undef MACRO

    #if defined(HAVE_SYCL)
    #define MACRO(OP, IN_T, C)										\
    template <typename T, std::size_t D>									\
    inline vec<T, D> operator OP (const IN_T<C T, D>& v_1, const T x_2)					\
    {													\
        return vec<T, D>(v_1).sycl_vec OP x_2;								\
    }													\
                                                        \
    template <typename T, std::size_t D>									\
    inline vec<T, D> operator OP (const IN_T<C T, D>&& v_1, const T x_2)					\
    {													\
        return vec<T, D>(v_1).sycl_vec OP x_2;								\
    }													\
                                                        \
    template <typename T, std::size_t D>									\
    inline vec<T, D> operator OP (const T x_1, const IN_T<C T, D>& v_2)					\
    {													\
        return vec<T, D>(v_2).sycl_vec OP x_1;								\
    }													\
                                                        \
    template <typename T, std::size_t D>									\
    inline vec<T, D> operator OP (const T x_1, const IN_T<C T, D>&& v_2)					\
    {													\
        return vec<T, D>(v_2).sycl_vec OP x_1;								\
    }
    #else
    #define MACRO(OP, IN_T, C)										\
    template <typename T>											\
    inline vec<T, 1> operator OP (const IN_T<C T, 1>& v_1, const T x_2)					\
    {													\
        return vec<T, 1>(v_1.x OP x_2);									\
    }													\
                                                        \
    template <typename T>											\
    inline vec<T, 1> operator OP (const IN_T<C T, 1>&& v_1, const T x_2)					\
    {													\
        return vec<T, 1>(v_1.x OP x_2);									\
    }													\
                                                        \
    template <typename T>											\
    inline vec<T, 1> operator OP (const T x_1, const IN_T<C T, 1>& v_2)					\
    {													\
        return vec<T, 1>(x_1 OP v_2.x);									\
    }													\
                                                        \
    template <typename T>											\
    inline vec<T, 1> operator OP (const T x_1, const IN_T<C T, 1>&& v_2)					\
    {													\
        return vec<T, 1>(x_1 OP v_2.x);									\
    }													\
                                                        \
    template <typename T>											\
    inline vec<T, 2> operator OP (const IN_T<C T, 2>& v_1, const T x_2)					\
    {													\
        return vec<T, 2>(v_1.x OP x_2, v_1.y OP x_2);							\
    }													\
                                                        \
    template <typename T>											\
    inline vec<T, 2> operator OP (const IN_T<C T, 2>&& v_1, const T x_2)					\
    {													\
        return vec<T, 2>(v_1.x OP x_2, v_1.y OP x_2);							\
    }													\
                                                        \
    template <typename T>											\
    inline vec<T, 2> operator OP (const T x_1, const IN_T<C T, 2>& v_2)					\
    {													\
        return vec<T, 2>(x_1 OP v_2.x, x_1 OP v_2.y);							\
    }													\
                                                        \
    template <typename T>											\
    inline vec<T, 2> operator OP (const T x_1, const IN_T<C T, 2>&& v_2)					\
    {													\
        return vec<T, 2>(x_1 OP v_2.x, x_1 OP v_2.y);							\
    }													\
                                                        \
    template <typename T>											\
    inline vec<T, 3> operator OP (const IN_T<C T, 3>& v_1, const T x_2)					\
    {													\
        return vec<T, 3>(v_1.x OP x_2, v_1.y OP x_2, v_1.z OP x_2);					\
    }													\
                                                        \
    template <typename T>											\
    inline vec<T, 3> operator OP (const IN_T<C T, 3>&& v_1, const T x_2)					\
    {													\
        return vec<T, 3>(v_1.x OP x_2, v_1.y OP x_2, v_1.z OP x_2);					\
    }													\
                                                        \
    template <typename T>											\
    inline vec<T, 3> operator OP (const T x_1, const IN_T<C T, 3>& v_2)					\
    {													\
        return vec<T, 3>(x_1 OP v_2.x, x_1 OP v_2.y, x_1 OP v_2.z);					\
    }													\
                                                        \
    template <typename T>											\
    inline vec<T, 3> operator OP (const T x_1, const IN_T<C T, 3>&& v_2)					\
    {													\
        return vec<T, 3>(x_1 OP v_2.x, x_1 OP v_2.y, x_1 OP v_2.z);					\
    }
    #endif

    MACRO(*, vec, )
    MACRO(/, vec, )
    MACRO(+, vec, )
    MACRO(-, vec, )

    #define MACRO_C1(OP, IN_T)										\
    MACRO(OP, IN_T,      )											\
    MACRO(OP, IN_T, const)											\

    MACRO_C1(*, detail::vec_proxy)
    MACRO_C1(/, detail::vec_proxy)
    MACRO_C1(+, detail::vec_proxy)
    MACRO_C1(-, detail::vec_proxy)

    #undef MACRO_C1
    #undef MACRO

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Geometric operations
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    #define MACRO(IN_1_T, IN_2_T, C_1, C_2)									\
    template <typename T, std::size_t D>									\
    inline vec<T, D> hadamard_product(const IN_1_T<C_1 T, D>& v_1, const IN_2_T<C_2 T, D>& v_2)		\
    {													\
        return v_1 * v_2;										\
    }													\
                                                        \
    template <typename T, std::size_t D>									\
    inline vec<T, D> hadamard_product(const IN_1_T<C_1 T, D>&& v_1, const IN_2_T<C_2 T, D>& v_2)		\
    {													\
        return v_1 * v_2;										\
    }													\
                                                        \
    template <typename T, std::size_t D>									\
    inline vec<T, D> hadamard_product(const IN_1_T<C_1 T, D>& v_1, const IN_2_T<C_2 T, D>&& v_2)		\
    {													\
        return v_1 * v_2;										\
    }													\
                                                        \
    template <typename T, std::size_t D>									\
    inline vec<T, D> hadamard_product(const IN_1_T<C_1 T, D>&& v_1, const IN_2_T<C_2 T, D>&& v_2)		\
    {													\
        return v_1 * v_2;										\
    }													\

    MACRO(vec, vec, , )

    #define MACRO_C1(IN_1_T, IN_2_T)									\
    MACRO(IN_1_T, IN_2_T,      ,      )									\
    MACRO(IN_1_T, IN_2_T, const,      )									\
    MACRO(IN_2_T, IN_1_T,      ,      )									\
    MACRO(IN_2_T, IN_1_T,      , const)									\
    
    #define MACRO_C2(IN_1_T, IN_2_T)									\
    MACRO(IN_1_T, IN_2_T,      ,      )									\
    MACRO(IN_1_T, IN_2_T, const, const)									\

    MACRO_C1(detail::vec_proxy, vec)
    MACRO_C2(detail::vec_proxy, detail::vec_proxy)
    
    #undef MACRO_C1
    #undef MACRO_C2
    #undef MACRO

    #if defined(HAVE_SYCL)
    #define MACRO(IN_1_T, IN_2_T, C_1, C_2)									\
    template <typename T, std::size_t D>									\
    inline T dot_product(const IN_1_T<C_1 T, D>& v_1, const IN_2_T<C_2 T, D>& v_2)				\
    {													\
        return cl::sycl::dot(vec<T, D>(v_1).sycl_vec, vec<T, D>(v_2).sycl_vec);				\
    }													\
                                                        \
    template <typename T, std::size_t D>									\
    inline T dot_product(const IN_1_T<C_1 T, D>&& v_1, const IN_2_T<C_2 T, D>& v_2)				\
    {													\
        return cl::sycl::dot(vec<T, D>(v_1).sycl_vec, vec<T, D>(v_2).sycl_vec);				\
    }													\
                                                        \
    template <typename T, std::size_t D>									\
    inline T dot_product(const IN_1_T<C_1 T, D>& v_1, const IN_2_T<C_2 T, D>&& v_2)				\
    {													\
        return cl::sycl::dot(vec<T, D>(v_1).sycl_vec, vec<T, D>(v_2).sycl_vec);				\
    }													\
                                                        \
    template <typename T, std::size_t D>									\
    inline T dot_product(const IN_1_T<C_1 T, D>&& v_1, const IN_2_T<C_2 T, D>&& v_2)			\
    {													\
        return cl::sycl::dot(vec<T, D>(v_1).sycl_vec, vec<T, D>(v_2).sycl_vec);				\
    }													\
                                                        \
    template <typename T, std::size_t D>									\
    inline T dot(const IN_1_T<C_1 T, D>& v_1, const IN_2_T<C_2 T, D>& v_2)					\
    {													\
        return dot_product(v_1, v_2);									\
    }													\
                                                        \
    template <typename T, std::size_t D>									\
    inline T dot(const IN_1_T<C_1 T, D>&& v_1, const IN_2_T<C_2 T, D>& v_2)					\
    {													\
        return dot_product(v_1, v_2);									\
    }													\
                                                        \
    template <typename T, std::size_t D>									\
    inline T dot(const IN_1_T<C_1 T, D>& v_1, const IN_2_T<C_2 T, D>&& v_2)					\
    {													\
        return dot_product(v_1, v_2);									\
    }													\
                                                        \
    template <typename T, std::size_t D>									\
    inline T dot(const IN_1_T<C_1 T, D>&& v_1, const IN_2_T<C_2 T, D>&& v_2)				\
    {													\
        return dot_product(v_1, v_2);									\
    }
    #else
    #define MACRO(IN_1_T, IN_2_T, C_1, C_2)									\
    template <typename T>											\
    inline T dot_product(const IN_1_T<C_1 T, 1>& v_1, const IN_2_T<C_2 T, 1>& v_2)				\
    {													\
        return (v_1.x * v_2.x);										\
    }													\
                                                        \
    template <typename T>											\
    inline T dot_product(const IN_1_T<C_1 T, 1>&& v_1, const IN_2_T<C_2 T, 1>& v_2)				\
    {													\
        return (v_1.x * v_2.x);										\
    }													\
                                                        \
    template <typename T>											\
    inline T dot_product(const IN_1_T<C_1 T, 1>& v_1, const IN_2_T<C_2 T, 1>&& v_2)				\
    {													\
        return (v_1.x * v_2.x);										\
    }													\
                                                        \
    template <typename T>											\
    inline T dot_product(const IN_1_T<C_1 T, 1>&& v_1, const IN_2_T<C_2 T, 1>&& v_2)			\
    {													\
        return (v_1.x * v_2.x);										\
    }													\
                                                        \
    template <typename T>											\
    inline T dot_product(const IN_1_T<C_1 T, 2>& v_1, const IN_2_T<C_2 T, 2>& v_2)				\
    {													\
        return (v_1.x * v_2.x + v_1.y * v_2.y);								\
    }													\
                                                        \
    template <typename T>											\
    inline T dot_product(const IN_1_T<C_1 T, 2>&& v_1, const IN_2_T<C_2 T, 2>& v_2)				\
    {													\
        return (v_1.x * v_2.x + v_1.y * v_2.y);								\
    }													\
                                                        \
    template <typename T>											\
    inline T dot_product(const IN_1_T<C_1 T, 2>& v_1, const IN_2_T<C_2 T, 2>&& v_2)				\
    {													\
        return (v_1.x * v_2.x + v_1.y * v_2.y);								\
    }													\
                                                        \
    template <typename T>											\
    inline T dot_product(const IN_1_T<C_1 T, 2>&& v_1, const IN_2_T<C_2 T, 2>&& v_2)			\
    {													\
        return (v_1.x * v_2.x + v_1.y * v_2.y);								\
    }													\
                                                        \
    template <typename T>											\
    inline T dot_product(const IN_1_T<C_1 T, 3>& v_1, const IN_2_T<C_2 T, 3>& v_2)				\
    {													\
        return (v_1.x * v_2.x + v_1.y * v_2.y + v_1.z * v_2.z);						\
    }													\
                                                        \
    template <typename T>											\
    inline T dot_product(const IN_1_T<C_1 T, 3>&& v_1, const IN_2_T<C_2 T, 3>& v_2)				\
    {													\
        return (v_1.x * v_2.x + v_1.y * v_2.y + v_1.z * v_2.z);						\
    }													\
                                                        \
    template <typename T>											\
    inline T dot_product(const IN_1_T<C_1 T, 3>& v_1, const IN_2_T<C_2 T, 3>&& v_2)				\
    {													\
        return (v_1.x * v_2.x + v_1.y * v_2.y + v_1.z * v_2.z);						\
    }													\
                                                        \
    template <typename T>											\
    inline T dot_product(const IN_1_T<C_1 T, 3>&& v_1, const IN_2_T<C_2 T, 3>&& v_2)			\
    {													\
        return (v_1.x * v_2.x + v_1.y * v_2.y + v_1.z * v_2.z);						\
    }													\
                                                        \
    template <typename T, std::size_t D>									\
    inline T dot(const IN_1_T<C_1 T, D>& v_1, const IN_2_T<C_2 T, D>& v_2)					\
    {													\
        return dot_product(v_1, v_2);									\
    }													\
                                                        \
    template <typename T, std::size_t D>									\
    inline T dot(const IN_1_T<C_1 T, D>&& v_1, const IN_2_T<C_2 T, D>& v_2)					\
    {													\
        return dot_product(v_1, v_2);									\
    }													\
                                                        \
    template <typename T, std::size_t D>									\
    inline T dot(const IN_1_T<C_1 T, D>& v_1, const IN_2_T<C_2 T, D>&& v_2)					\
    {													\
        return dot_product(v_1, v_2);									\
    }													\
                                                        \
    template <typename T, std::size_t D>									\
    inline T dot(const IN_1_T<C_1 T, D>&& v_1, const IN_2_T<C_2 T, D>&& v_2)				\
    {													\
        return dot_product(v_1, v_2);									\
    }
    #endif													\

    MACRO(vec, vec, , )

    #define MACRO_C1(IN_1_T, IN_2_T)									\
    MACRO(IN_1_T, IN_2_T,      ,      )									\
    MACRO(IN_1_T, IN_2_T, const,      )									\
    MACRO(IN_2_T, IN_1_T,      ,      )									\
    MACRO(IN_2_T, IN_1_T,      , const)									\
    
    #define MACRO_C2(IN_1_T, IN_2_T)									\
    MACRO(IN_1_T, IN_2_T,      ,      )									\
    MACRO(IN_1_T, IN_2_T, const, const)									\

    MACRO_C1(detail::vec_proxy, vec)
    MACRO_C2(detail::vec_proxy, detail::vec_proxy)
    
    #undef MACRO_C1
    #undef MACRO_C2
    #undef MACRO

    #if defined(HAVE_SYCL)
    #define MACRO(IN_1_T, IN_2_T, C_1, C_2)									\
    template <typename T>											\
    inline vec<T, 3> cross_product(const IN_1_T<C_1 T, 3>& v_1, const IN_2_T<C_2 T, 3>& v_2)		\
    {													\
        return cl::sycl::cross(vec<T, 3>(v_1).sycl_vec, vec<T, 3>(v_2).sycl_vec);			\
    }													\
                                                        \
    template <typename T>											\
    inline vec<T, 3> cross_product(const IN_1_T<C_1 T, 3>&& v_1, const IN_2_T<C_2 T, 3>& v_2)		\
    {													\
        return cl::sycl::cross(vec<T, 3>(v_1).sycl_vec, vec<T, 3>(v_2).sycl_vec);			\
    }													\
                                                        \
    template <typename T>											\
    inline vec<T, 3> cross_product(const IN_1_T<C_1 T, 3>& v_1, const IN_2_T<C_2 T, 3>&& v_2)		\
    {													\
        return cl::sycl::cross(vec<T, 3>(v_1).sycl_vec, vec<T, 3>(v_2).sycl_vec);			\
    }													\
                                                        \
    template <typename T>											\
    inline vec<T, 3> cross_product(const IN_1_T<C_1 T, 3>&& v_1, const IN_2_T<C_2 T, 3>&& v_2)		\
    {													\
        return cl::sycl::cross(vec<T, 3>(v_1).sycl_vec, vec<T, 3>(v_2).sycl_vec);			\
    }													\
                                                        \
    template <typename T>											\
    inline vec<T, 3> cross(const IN_1_T<C_1 T, 3>& v_1, const IN_2_T<C_2 T, 3>& v_2)			\
    {													\
        return cross_product(v_1, v_2);									\
    }													\
                                                        \
    template <typename T>											\
    inline vec<T, 3> cross(const IN_1_T<C_1 T, 3>&& v_1, const IN_2_T<C_2 T, 3>& v_2)			\
    {													\
        return cross_product(v_1, v_2);									\
    }													\
                                                        \
    template <typename T>											\
    inline vec<T, 3> cross(const IN_1_T<C_1 T, 3>& v_1, const IN_2_T<C_2 T, 3>&& v_2)			\
    {													\
        return cross_product(v_1, v_2);									\
    }													\
                                                        \
    template <typename T>											\
    inline vec<T, 3> cross(const IN_1_T<C_1 T, 3>&& v_1, const IN_2_T<C_2 T, 3>&& v_2)			\
    {													\
        return cross_product(v_1, v_2);									\
    }
    #else
    #define MACRO(IN_1_T, IN_2_T, C_1, C_2)									\
    template <typename T>											\
    inline vec<T, 3> cross_product(const IN_1_T<C_1 T, 3>& v_1, const IN_2_T<C_2 T, 3>& v_2)		\
    {													\
        return vec<T, 3>(v_1.y * v_2.z - v_1.z * v_2.y,							\
                        v_1.z * v_2.x - v_1.x * v_2.z,					\
                        v_1.x * v_2.y - v_1.y * v_2.x);				\
    }													\
                                                        \
    template <typename T>											\
    inline vec<T, 3> cross_product(const IN_1_T<C_1 T, 3>&& v_1, const IN_2_T<C_2 T, 3>& v_2)		\
    {													\
        return vec<T, 3>(v_1.y * v_2.z - v_1.z * v_2.y,							\
                        v_1.z * v_2.x - v_1.x * v_2.z,					\
                        v_1.x * v_2.y - v_1.y * v_2.x);				\
    }													\
                                                        \
    template <typename T>											\
    inline vec<T, 3> cross_product(const IN_1_T<C_1 T, 3>& v_1, const IN_2_T<C_2 T, 3>&& v_2)		\
    {													\
        return vec<T, 3>(v_1.y * v_2.z - v_1.z * v_2.y,							\
                        v_1.z * v_2.x - v_1.x * v_2.z,					\
                        v_1.x * v_2.y - v_1.y * v_2.x);				\
    }													\
                                                        \
    template <typename T>											\
    inline vec<T, 3> cross_product(const IN_1_T<C_1 T, 3>&& v_1, const IN_2_T<C_2 T, 3>&& v_2)		\
    {													\
        return vec<T, 3>(v_1.y * v_2.z - v_1.z * v_2.y,							\
                        v_1.z * v_2.x - v_1.x * v_2.z,					\
                        v_1.x * v_2.y - v_1.y * v_2.x);				\
    }													\
                                                        \
    template <typename T>											\
    inline vec<T, 3> cross(const IN_1_T<C_1 T, 3>& v_1, const IN_2_T<C_2 T, 3>& v_2)			\
    {													\
        return cross_product(v_1, v_2);									\
    }													\
                                                        \
    template <typename T>											\
    inline vec<T, 3> cross(const IN_1_T<C_1 T, 3>&& v_1, const IN_2_T<C_2 T, 3>& v_2)			\
    {													\
        return cross_product(v_1, v_2);									\
    }													\
                                                        \
    template <typename T>											\
    inline vec<T, 3> cross(const IN_1_T<C_1 T, 3>& v_1, const IN_2_T<C_2 T, 3>&& v_2)			\
    {													\
        return cross_product(v_1, v_2);									\
    }													\
                                                        \
    template <typename T>											\
    inline vec<T, 3> cross(const IN_1_T<C_1 T, 3>&& v_1, const IN_2_T<C_2 T, 3>&& v_2)			\
    {													\
        return cross_product(v_1, v_2);									\
    }
    #endif													\

    MACRO(vec, vec, , )

    #define MACRO_C1(IN_1_T, IN_2_T)									\
    MACRO(IN_1_T, IN_2_T,      ,      )									\
    MACRO(IN_1_T, IN_2_T, const,      )									\
    MACRO(IN_2_T, IN_1_T,      ,      )									\
    MACRO(IN_2_T, IN_1_T,      , const)									\
    
    #define MACRO_C2(IN_1_T, IN_2_T)									\
    MACRO(IN_1_T, IN_2_T,      ,      )									\
    MACRO(IN_1_T, IN_2_T, const, const)									\

    MACRO_C1(detail::vec_proxy, vec)
    MACRO_C2(detail::vec_proxy, detail::vec_proxy)
    
    #undef MACRO_C1
    #undef MACRO_C2
    #undef MACRO
}
#else

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#if !defined(VEC_NAMESPACE)
#define VEC_NAMESPACE XXX_NAMESPACE
#endif

#include "../traits/traits.hpp"

namespace VEC_NAMESPACE
{

#define MACRO_UNQUALIFIED(OP, IN_T_1, IN_T_2)                                                                                               \
    template <typename T_1, typename T_2, std::size_t D, typename X = typename XXX_NAMESPACE::internal::compare<T_1, T_2>::stronger_type>   \
    vec<X, D> operator OP (IN_T_1<T_1, D>& x_1, IN_T_2<T_2, D>& x_2)                                                                        \
    {                                                                                                                                       \
        vec<X, D> y(x_1);                                                                                                                   \
        y OP ## = x_2;                                                                                                                      \
        return y;                                                                                                                           \
    }                                                                                                                                       \

#define MACRO_QUALIFIED(OP, IN_T_1, IN_T_2)                                                                                                 \
    MACRO_UNQUALIFIED(OP, IN_T_1, IN_T_2)                                                                                                   \
    MACRO_UNQUALIFIED(OP, const IN_T_1, IN_T_2)                                                                                             \
    MACRO_UNQUALIFIED(OP, IN_T_1, const IN_T_2)                                                                                             \
    MACRO_UNQUALIFIED(OP, const IN_T_1, const IN_T_2)                                                                                       \

#define MACRO(OP, IN_T_1, IN_T_2)                                                                                                           \
    MACRO_QUALIFIED(OP, IN_T_1, IN_T_1)                                                                                                     \
    MACRO_QUALIFIED(OP, IN_T_2, IN_T_2)                                                                                                     \
    MACRO_QUALIFIED(OP, IN_T_1, IN_T_2)                                                                                                     \
    MACRO_QUALIFIED(OP, IN_T_2, IN_T_1)                                                                                                     \

    MACRO(+, vec, internal::vec_proxy)
    MACRO(-, vec, internal::vec_proxy)
    MACRO(*, vec, internal::vec_proxy)
    MACRO(/, vec, internal::vec_proxy)

#undef MACRO
#undef MACRO_QUALIFIED
#undef MACRO_UNQUALIFIED

#define MACRO_UNQUALIFIED(OP, IN_T)                                                                                                         \
    template <typename T_1, typename T_2, std::size_t D, typename X = typename XXX_NAMESPACE::internal::compare<T_1, T_2>::stronger_type>   \
    vec<X, D> operator OP (IN_T<T_1, D>& x_1, const T_2 x_2)                                                                                \
    {                                                                                                                                       \
        vec<X, D> y(x_1);                                                                                                                   \
        y OP ## = x_2;                                                                                                                      \
        return y;                                                                                                                           \
    }                                                                                                                                       \
                                                                                                                                            \
    template <typename T_1, typename T_2, std::size_t D, typename X = typename XXX_NAMESPACE::internal::compare<T_1, T_2>::stronger_type>   \
    vec<X, D> operator OP (const T_1 x_1, IN_T<T_2, D>& x_2)                                                                                \
    {                                                                                                                                       \
        vec<X, D> y(x_1);                                                                                                                   \
        y OP ## = x_2;                                                                                                                      \
        return y;                                                                                                                           \
    }                                                                                                                                       \

#define MACRO_QUALIFIED(OP, IN_T)                                                                                                           \
    MACRO_UNQUALIFIED(OP, IN_T)                                                                                                             \
    MACRO_UNQUALIFIED(OP, const IN_T)                                                                                                       \

#define MACRO(OP, IN_T)                                                                                                                     \
    MACRO_QUALIFIED(OP, IN_T)                                                                                                               \

    MACRO(+, vec)
    MACRO(-, vec)
    MACRO(*, vec)
    MACRO(/, vec)

    MACRO(+, internal::vec_proxy)
    MACRO(-, internal::vec_proxy)
    MACRO(*, internal::vec_proxy)
    MACRO(/, internal::vec_proxy)

#undef MACRO
#undef MACRO_QUALIFIED
#undef MACRO_UNQUALIFIED

#define MACRO_UNQUALIFIED(OP, IN_T)                                                                                                         \
    template <typename T>                                                                                                                   \
    vec<T, 1> OP (IN_T<T, 1>& v)                                                                                                            \
    {                                                                                                                                       \
        return vec<T, 1>(MISC_NAMESPACE::math<T>:: OP (v.x));                                                                               \
    }                                                                                                                                       \
                                                                                                                                            \
    template <typename T>                                                                                                                   \
    vec<T, 2> OP (IN_T<T, 2>& v)                                                                                                            \
    {                                                                                                                                       \
        return vec<T, 2>(MISC_NAMESPACE::math<T>:: OP (v.x), MISC_NAMESPACE::math<T>:: OP (v.y));                                           \
    }                                                                                                                                       \
                                                                                                                                            \
    template <typename T>                                                                                                                   \
    vec<T, 3> OP (IN_T<T, 3>& v)                                                                                                            \
    {                                                                                                                                       \
        return vec<T, 3>(MISC_NAMESPACE::math<T>:: OP (v.x), MISC_NAMESPACE::math<T>:: OP (v.y), MISC_NAMESPACE::math<T>:: OP (v.z));       \
    }                                                                                                                                       \

#define MACRO_QUALIFIED(OP, IN_T)                                                                                                           \
    MACRO_UNQUALIFIED(OP, IN_T)                                                                                                             \
    MACRO_UNQUALIFIED(OP, const IN_T)                                                                                                       \

#define MACRO(OP, IN_T)                                                                                                                     \
    MACRO_QUALIFIED(OP, IN_T)                                                                                                               \

    MACRO(sqrt, vec)
    MACRO(log, vec)
    MACRO(exp, vec)

    MACRO(sqrt, internal::vec_proxy)
    MACRO(log, internal::vec_proxy)
    MACRO(exp, internal::vec_proxy)

#undef MACRO
#undef MACRO_QUALIFIED
#undef MACRO_UNQUALIFIED

#define MACRO_UNQUALIFIED(IN_T_1, IN_T_2)                                                                                                   \
    template <typename T_1, typename T_2, typename X = typename XXX_NAMESPACE::internal::compare<T_1, T_2>::stronger_type>                  \
    X dot_product(IN_T_1<T_1, 1>& x_1, IN_T_2<T_2, 1>& x_2)                                                                                 \
    {                                                                                                                                       \
        return (x_1.x * x_2.x);                                                                                                             \
    }                                                                                                                                       \
                                                                                                                                            \
    template <typename T_1, typename T_2, typename X = typename XXX_NAMESPACE::internal::compare<T_1, T_2>::stronger_type>                  \
    X dot_product(IN_T_1<T_1, 2>& x_1, IN_T_2<T_2, 2>& x_2)                                                                                 \
    {                                                                                                                                       \
        return (x_1.x * x_2.x + x_1.y * x_2.y);                                                                                             \
    }                                                                                                                                       \
                                                                                                                                            \
    template <typename T_1, typename T_2, typename X = typename XXX_NAMESPACE::internal::compare<T_1, T_2>::stronger_type>                  \
    X dot_product(IN_T_1<T_1, 3>& x_1, IN_T_2<T_2, 3>& x_2)                                                                                 \
    {                                                                                                                                       \
        return (x_1.x * x_2.x + x_1.y * x_2.y + x_1.z * x_2.z);                                                                             \
    }                                                                                                                                       \
                                                                                                                                            \
    template <typename T_1, typename T_2, std::size_t D, typename X = typename XXX_NAMESPACE::internal::compare<T_1, T_2>::stronger_type>   \
    X dot(IN_T_1<T_1, D>& x_1, IN_T_2<T_2, D>& x_2)                                                                                         \
    {                                                                                                                                       \
        return dot_product(x_1, x_2);                                                                                                       \
    }                                                                                                                                       \

#define MACRO_QUALIFIED(IN_T_1, IN_T_2)                                                                                                     \
    MACRO_UNQUALIFIED(IN_T_1, IN_T_2)                                                                                                       \
    MACRO_UNQUALIFIED(const IN_T_1, IN_T_2)                                                                                                 \
    MACRO_UNQUALIFIED(IN_T_1, const IN_T_2)                                                                                                 \
    MACRO_UNQUALIFIED(const IN_T_1, const IN_T_2)                                                                                           \

#define MACRO(IN_T_1, IN_T_2)                                                                                                               \
    MACRO_QUALIFIED(IN_T_1, IN_T_1)                                                                                                         \
    MACRO_QUALIFIED(IN_T_2, IN_T_2)                                                                                                         \
    MACRO_QUALIFIED(IN_T_1, IN_T_2)                                                                                                         \
    MACRO_QUALIFIED(IN_T_2, IN_T_1)                                                                                                         \

    MACRO(vec, internal::vec_proxy)

#undef MACRO
#undef MACRO_QUALIFIED
#undef MACRO_UNQUALIFIED

#define MACRO_UNQUALIFIED(IN_T)                                                                                                             \
    template <typename T_1, typename T_2, typename X = typename XXX_NAMESPACE::internal::compare<T_1, T_2>::stronger_type>                  \
    X dot_product(IN_T<T_1, 1>& x_1, const T_2 x_2)                                                                                         \
    {                                                                                                                                       \
        return (x_1.x * x_2);                                                                                                               \
    }                                                                                                                                       \
                                                                                                                                            \
    template <typename T_1, typename T_2, typename X = typename XXX_NAMESPACE::internal::compare<T_1, T_2>::stronger_type>                  \
    X dot_product(IN_T<T_1, 2>& x_1, const T_2 x_2)                                                                                         \
    {                                                                                                                                       \
        return (x_1.x + x_1.y) * x_2;                                                                                                       \
    }                                                                                                                                       \
                                                                                                                                            \
    template <typename T_1, typename T_2, typename X = typename XXX_NAMESPACE::internal::compare<T_1, T_2>::stronger_type>                  \
    X dot_product(IN_T<T_1, 3>& x_1, const T_2 x_2)                                                                                         \
    {                                                                                                                                       \
        return (x_1.x + x_1.y + x_1.z) * x_2;                                                                                               \
    }                                                                                                                                       \
                                                                                                                                            \
    template <typename T_1, typename T_2, typename X = typename XXX_NAMESPACE::internal::compare<T_1, T_2>::stronger_type>                  \
    X dot_product(const T_1 x_1, IN_T<T_2, 1>& x_2)                                                                                         \
    {                                                                                                                                       \
        return (x_1 * x_2.x);                                                                                                               \
    }                                                                                                                                       \
                                                                                                                                            \
    template <typename T_1, typename T_2, typename X = typename XXX_NAMESPACE::internal::compare<T_1, T_2>::stronger_type>                  \
    X dot_product(const T_1 x_1, IN_T<T_2, 2>& x_2)                                                                                         \
    {                                                                                                                                       \
        return x_1 * (x_2.x + x_2.y);                                                                                                       \
    }                                                                                                                                       \
                                                                                                                                            \
    template <typename T_1, typename T_2, typename X = typename XXX_NAMESPACE::internal::compare<T_1, T_2>::stronger_type>                  \
    X dot_product(const T_1 x_1, IN_T<T_2, 3>& x_2)                                                                                         \
    {                                                                                                                                       \
        return x_1 * (x_2.x + x_2.y + x_2.z);                                                                                               \
    }                                                                                                                                       \
                                                                                                                                            \
    template <typename T_1, typename T_2, std::size_t D, typename X = typename XXX_NAMESPACE::internal::compare<T_1, T_2>::stronger_type>   \
    X dot(IN_T<T_1, D>& x_1, const T_2 x_2)                                                                                                 \
    {                                                                                                                                       \
        return dot_product(x_1, x_2);                                                                                                       \
    }                                                                                                                                       \
                                                                                                                                            \
    template <typename T_1, typename T_2, std::size_t D, typename X = typename XXX_NAMESPACE::internal::compare<T_1, T_2>::stronger_type>   \
    X dot(const T_1 x_1, IN_T<T_2, D>& x_2)                                                                                                 \
    {                                                                                                                                       \
        return dot_product(x_1, x_2);                                                                                                       \
    }                                                                                                                                       \

#define MACRO_QUALIFIED(IN_T)                                                                                                               \
    MACRO_UNQUALIFIED(IN_T)                                                                                                                 \
    MACRO_UNQUALIFIED(const IN_T)                                                                                                           \
    
#define MACRO(IN_T)                                                                                                                         \
    MACRO_QUALIFIED(IN_T)                                                                                                                   \
    
    MACRO(vec)
    MACRO(internal::vec_proxy)

#undef MACRO
#undef MACRO_QUALIFIED
#undef MACRO_UNQUALIFIED

#define MACRO_UNQUALIFIED(IN_T_1, IN_T_2)                                                                                                   \
    template <typename T_1, typename T_2, typename X = typename XXX_NAMESPACE::internal::compare<T_1, T_2>::stronger_type>                  \
    vec<X, 3> cross_product(IN_T_1<T_1, 3>& x_1, IN_T_2<T_2, 3>& x_2)                                                                       \
    {                                                                                                                                       \
        return vec<X, 3>(x_1.y * x_2.z - x_1.z * x_2.y, x_1.z * x_2.x - x_1.x * x_2.z, x_1.x * x_2.y - x_1.y * x_2.x);                      \
    }                                                                                                                                       \
                                                                                                                                            \
    template <typename T_1, typename T_2, typename X = typename XXX_NAMESPACE::internal::compare<T_1, T_2>::stronger_type>                  \
    vec<X, 3> cross(IN_T_1<T_1, 3>& x_1, IN_T_2<T_2, 3>& x_2)                                                                               \
    {                                                                                                                                       \
        return cross_product(x_1, x_2);                                                                                                     \
    }                                                                                                                                       \

#define MACRO_QUALIFIED(IN_T_1, IN_T_2)                                                                                                     \
    MACRO_UNQUALIFIED(IN_T_1, IN_T_2)                                                                                                       \
    MACRO_UNQUALIFIED(const IN_T_1, IN_T_2)                                                                                                 \
    MACRO_UNQUALIFIED(IN_T_1, const IN_T_2)                                                                                                 \
    MACRO_UNQUALIFIED(const IN_T_1, const IN_T_2)                                                                                           \

#define MACRO(IN_T_1, IN_T_2)                                                                                                               \
    MACRO_QUALIFIED(IN_T_1, IN_T_1)                                                                                                         \
    MACRO_QUALIFIED(IN_T_2, IN_T_2)                                                                                                         \
    MACRO_QUALIFIED(IN_T_1, IN_T_2)                                                                                                         \
    MACRO_QUALIFIED(IN_T_2, IN_T_1)                                                                                                         \

    MACRO(vec, internal::vec_proxy)

#undef MACRO
#undef MACRO_QUALIFIED
#undef MACRO_UNQUALIFIED

#define MACRO_UNQUALIFIED(IN_T)                                                                                                             \
    template <typename T_1, typename T_2, typename X = typename XXX_NAMESPACE::internal::compare<T_1, T_2>::stronger_type>                  \
    vec<X, 3> cross_product(IN_T<T_1, 3>& x_1, const T_2 x_2)                                                                               \
    {                                                                                                                                       \
        return vec<X, 3>(x_1.y - x_1.z, x_1.z - x_1.x, x_1.x - x_1.y) * x_2;                                                                \
    }                                                                                                                                       \
                                                                                                                                            \
    template <typename T_1, typename T_2, typename X = typename XXX_NAMESPACE::internal::compare<T_1, T_2>::stronger_type>                  \
    vec<X, 3> cross_product(const T_1 x_1, IN_T<T_2, 3>& x_2)                                                                               \
    {                                                                                                                                       \
        return x_1 * vec<X, 3>(x_2.z - x_2.y, x_2.x - x_2.z, x_2.y - x_2.x);                                                                \
    }                                                                                                                                       \
                                                                                                                                            \
    template <typename T_1, typename T_2, typename X = typename XXX_NAMESPACE::internal::compare<T_1, T_2>::stronger_type>                  \
    vec<X, 3> cross(IN_T<T_1, 3>& x_1, const T_2 x_2)                                                                                       \
    {                                                                                                                                       \
        return cross_product(x_1, x_2);                                                                                                     \
    }                                                                                                                                       \
                                                                                                                                            \
    template <typename T_1, typename T_2, typename X = typename XXX_NAMESPACE::internal::compare<T_1, T_2>::stronger_type>                  \
    vec<X, 3> cross(const T_1 x_1, IN_T<T_2, 3>& x_2)                                                                                       \
    {                                                                                                                                       \
        return cross_product(x_1, x_2);                                                                                                     \
    }                                                                                                                                       \

#define MACRO_QUALIFIED(IN_T)                                                                                                               \
    MACRO_UNQUALIFIED(IN_T)                                                                                                                 \
    MACRO_UNQUALIFIED(const IN_T)                                                                                                           \
    
#define MACRO(IN_T)                                                                                                                         \
    MACRO_QUALIFIED(IN_T)                                                                                                                   \
    
    MACRO(vec)
    MACRO(internal::vec_proxy)

#undef MACRO
#undef MACRO_QUALIFIED
#undef MACRO_UNQUALIFIED
}

#endif

#endif
