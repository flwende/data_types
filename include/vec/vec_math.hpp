// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(VEC_MATH_HPP)
#define VEC_MATH_HPP

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
	}													\

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
	}													\

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
	}													\

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
	template <typename T>											\
	inline T dot_product(const IN_1_T<C_1 T, 1>& v_1, const IN_2_T<C_2 T, 1>& v_2)				\
	{													\
		const cl::sycl::vec<T, 1> _v_1(v_1.x);								\
		const cl::sycl::vec<T, 1> _v_2(v_2.x);								\
		return dot(_v_1, _v_2);										\
	}													\
														\
	template <typename T>											\
	inline T dot_product(const IN_1_T<C_1 T, 1>&& v_1, const IN_2_T<C_2 T, 1>& v_2)				\
	{													\
		const cl::sycl::vec<T, 1> _v_1(v_1.x);								\
		const cl::sycl::vec<T, 1> _v_2(v_2.x);								\
		return dot(_v_1, _v_2);										\
	}													\
														\
	template <typename T>											\
	inline T dot_product(const IN_1_T<C_1 T, 1>& v_1, const IN_2_T<C_2 T, 1>&& v_2)				\
	{													\
		const cl::sycl::vec<T, 1> _v_1(v_1.x);								\
		const cl::sycl::vec<T, 1> _v_2(v_2.x);								\
		return dot(_v_1, _v_2);										\
	}													\
														\
	template <typename T>											\
	inline T dot_product(const IN_1_T<C_1 T, 1>&& v_1, const IN_2_T<C_2 T, 1>&& v_2)			\
	{													\
		const cl::sycl::vec<T, 1> _v_1(v_1.x);								\
		const cl::sycl::vec<T, 1> _v_2(v_2.x);								\
		return dot(_v_1, _v_2);										\
	}													\
														\
	template <typename T>											\
	inline T dot_product(const IN_1_T<C_1 T, 2>& v_1, const IN_2_T<C_2 T, 2>& v_2)				\
	{													\
		const cl::sycl::vec<T, 2> _v_1(v_1.x, v_1.y);							\
		const cl::sycl::vec<T, 2> _v_2(v_2.x, v_2.y);							\
		return dot(_v_1, _v_2);										\
	}													\
														\
	template <typename T>											\
	inline T dot_product(const IN_1_T<C_1 T, 2>&& v_1, const IN_2_T<C_2 T, 2>& v_2)				\
	{													\
		const cl::sycl::vec<T, 2> _v_1(v_1.x, v_1.y);							\
		const cl::sycl::vec<T, 2> _v_2(v_2.x, v_2.y);							\
		return dot(_v_1, _v_2);										\
	}													\
														\
	template <typename T>											\
	inline T dot_product(const IN_1_T<C_1 T, 2>& v_1, const IN_2_T<C_2 T, 2>&& v_2)				\
	{													\
		const cl::sycl::vec<T, 2> _v_1(v_1.x, v_1.y);							\
		const cl::sycl::vec<T, 2> _v_2(v_2.x, v_2.y);							\
		return dot(_v_1, _v_2);										\
	}													\
														\
	template <typename T>											\
	inline T dot_product(const IN_1_T<C_1 T, 2>&& v_1, const IN_2_T<C_2 T, 2>&& v_2)			\
	{													\
		const cl::sycl::vec<T, 2> _v_1(v_1.x, v_1.y);							\
		const cl::sycl::vec<T, 2> _v_2(v_2.x, v_2.y);							\
		return dot(_v_1, _v_2);										\
	}													\
														\
	template <typename T>											\
	inline T dot_product(const IN_1_T<C_1 T, 3>& v_1, const IN_2_T<C_2 T, 3>& v_2)				\
	{													\
		const cl::sycl::vec<T, 3> _v_1(v_1.x, v_1.y, v_1.z);						\
		const cl::sycl::vec<T, 3> _v_2(v_2.x, v_2.y, v_2.z);						\
		return dot(_v_1, _v_2);										\
	}													\
														\
	template <typename T>											\
	inline T dot_product(const IN_1_T<C_1 T, 3>&& v_1, const IN_2_T<C_2 T, 3>& v_2)				\
	{													\
		const cl::sycl::vec<T, 3> _v_1(v_1.x, v_1.y, v_1.z);						\
		const cl::sycl::vec<T, 3> _v_2(v_2.x, v_2.y, v_2.z);						\
		return dot(_v_1, _v_2);										\
	}													\
														\
	template <typename T>											\
	inline T dot_product(const IN_1_T<C_1 T, 3>& v_1, const IN_2_T<C_2 T, 3>&& v_2)				\
	{													\
		const cl::sycl::vec<T, 3> _v_1(v_1.x, v_1.y, v_1.z);						\
		const cl::sycl::vec<T, 3> _v_2(v_2.x, v_2.y, v_2.z);						\
		return dot(_v_1, _v_2);										\
	}													\
														\
	template <typename T>											\
	inline T dot_product(const IN_1_T<C_1 T, 3>&& v_1, const IN_2_T<C_2 T, 3>&& v_2)			\
	{													\
		const cl::sycl::vec<T, 3> _v_1(v_1.x, v_1.y, v_1.z);						\
		const cl::sycl::vec<T, 3> _v_2(v_2.x, v_2.y, v_2.z);						\
		return dot(_v_1, _v_2);										\
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
	#endif

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
		const cl::sycl::vec<T, 3> _v_1(v_1.x, v_1.y, v_1.z);						\
		const cl::sycl::vec<T, 3> _v_2(v_2.x, v_2.y, v_2.z);						\
		const cl::sycl::vec<T, 3> y = cross(_v_1, _v_2);						\
		return vec<T, 3>(y.x(), y.y(), y.z());								\
	}													\
														\
	template <typename T>											\
	inline vec<T, 3> cross_product(const IN_1_T<C_1 T, 3>&& v_1, const IN_2_T<C_2 T, 3>& v_2)		\
	{													\
		const cl::sycl::vec<T, 3> _v_1(v_1.x, v_1.y, v_1.z);						\
		const cl::sycl::vec<T, 3> _v_2(v_2.x, v_2.y, v_2.z);						\
		const cl::sycl::vec<T, 3> y = cross(_v_1, _v_2);						\
		return vec<T, 3>(y.x(), y.y(), y.z());								\
	}													\
														\
	template <typename T>											\
	inline vec<T, 3> cross_product(const IN_1_T<C_1 T, 3>& v_1, const IN_2_T<C_2 T, 3>&& v_2)		\
	{													\
		const cl::sycl::vec<T, 3> _v_1(v_1.x, v_1.y, v_1.z);						\
		const cl::sycl::vec<T, 3> _v_2(v_2.x, v_2.y, v_2.z);						\
		const cl::sycl::vec<T, 3> y = cross(_v_1, _v_2);						\
		return vec<T, 3>(y.x(), y.y(), y.z());								\
	}													\
														\
	template <typename T>											\
	inline vec<T, 3> cross_product(const IN_1_T<C_1 T, 3>&& v_1, const IN_2_T<C_2 T, 3>&& v_2)		\
	{													\
		const cl::sycl::vec<T, 3> _v_1(v_1.x, v_1.y, v_1.z);						\
		const cl::sycl::vec<T, 3> _v_2(v_2.x, v_2.y, v_2.z);						\
		const cl::sycl::vec<T, 3> y = cross(_v_1, _v_2);						\
		return vec<T, 3>(y.x(), y.y(), y.z());								\
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
	#endif

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

#endif
