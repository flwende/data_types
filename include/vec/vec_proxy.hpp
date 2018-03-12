// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(VEC_PROXY_HPP)
#define VEC_PROXY_HPP

#if !defined(VEC_NAMESPACE)
#if !defined(XXX_NAMESPACE)
#define VEC_NAMESPACE fw
#else
#define VEC_NAMESPACE XXX_NAMESPACE
#endif
#endif

namespace VEC_NAMESPACE
{
	template <typename T>
	using math = MISC_NAMESPACE::math<T>;

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//! \brief A proxy data type for vec<T, D>
	//!
	//! This data type is returned by buffer<T, D, Layout, Alignment>::operator[]() if D = 1 and Layout=SoA.
	//! It holds references to component(s) x [,y [and z]] in main memory, so that data access via,
	//! e.g. obj[ ]..[ ].x, is possible.
	//!
	//! \tparam T data type
	//! \tparam D dimension
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	namespace detail
	{
		//! \brief D = 1 specialization with component x
		//!
		//! \tparam T data type
		template <typename T>
		class vec_proxy<T, 1>
		{
			using vec = VEC_NAMESPACE::vec<T, 1>;

		public:

			//! Remember the template type parameter T
			using fundamental_type = T;
			//! Remember the template parameter D (=1)
			static constexpr std::size_t dim = 1;

			//! Reference to x component of the corresponding vec<T, 1> object
			T& x;

			//! \brief Constructor taking the address of the x component of the vec<T, 1> object
			//!
			//! \param ptr pointer to x component
			//! \param n extent of the underlying field in dimension 1 (innermost)
			vec_proxy(T* ptr, const std::size_t n) : x(ptr[0 * n]) { ; }

			//! \brief Create a detail::vec_proxy<T, 1> object from a vec<T, 1> object
			//!
			//! \param v
			vec_proxy(vec& v) : x(v.x) { ; }

			//! \brief Create a detail::vec_proxy<T, 1> object from a vec<T, 1> object
			//!
			//! \param v
			vec_proxy(vec&& v) : x(v.x) { ; }

			//! \brief Copy constructor
			//!
			//! \param v
			vec_proxy(vec_proxy& v) : x(v.x) { ; }

			//! \brief Move constructor
			//!
			//! \param v
			vec_proxy(vec_proxy&& v) : x(v.x) { ; }

			//! \brief Create a detail::vec_proxy<T, 1> object from a vec<T, 1> object
			//!
			//! \param v
			//! \return a detail::proxy<T, 1> object
			inline vec_proxy& operator=(const vec& v)
			{
				x = v.x;

				return *this;
			}

			inline vec operator-()
			{
				constexpr T minus_one = math<T>::minus_one;
				return vec(minus_one * x);
			}

			#define MACRO(OP, IN_T)                 \
			inline void operator OP (const IN_T& v) \
			{                                       \
				x OP v.x;                           \
			}                                       \

			MACRO(+=, vec_proxy)
			MACRO(-=, vec_proxy)
			MACRO(*=, vec_proxy)
			MACRO(/=, vec_proxy)

			MACRO(+=, vec)
			MACRO(-=, vec)
			MACRO(*=, vec)
			MACRO(/=, vec)

			#undef MACRO

			#define MACRO(OP)                       \
			inline void operator OP (const T c)     \
			{                                       \
				x OP c;                             \
			}                                       \

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
				return x;
			}
		};

		//! \brief D = 2 specialization with components x and y
		//!
		//! \tparam T data type
		template <typename T>
		class vec_proxy<T, 2>
		{
			using vec = VEC_NAMESPACE::vec<T, 2>;

		public:

			//! Remember the template type parameter T
			using fundamental_type = T;
			//! Remember the template parameter D (=2)
			static constexpr std::size_t dim = 2;

			//! Reference to x component of the corresponding vec<T, 2> object
			T& x;
			//! Reference to y component of the corresponding vec<T, 2> object
			T& y;

			//! \brief Constructor taking the address of the x component of the vec<T, 2> object
			//!
			//! For D=2, the address of the y component is &ptr[1 * n].
			//!
			//! \param ptr pointer to x component
			//! \param n extent of the underlying field in dimension 1 (innermost)
			vec_proxy(T* ptr, const std::size_t n) : x(ptr[0 * n]), y(ptr[1 * n]) { ; }

			//! \brief Create a detail::vec_proxy<T, 2> object from a vec<T, 2> object
			//!
			//! \param v
			vec_proxy(vec& v) : x(v.x), y(v.y) { ; }

			//! \brief Create a detail::vec_proxy<T, 2> object from a vec<T, 2> object
			//!
			//! \param v
			vec_proxy(vec&& v) : x(v.x), y(v.y) { ; }

			//! \brief Copy constructor
			//!
			//! \param v
			vec_proxy(vec_proxy& v) : x(v.x), y(v.y) { ; }

			//! \brief Move constructor
			//!
			//! \param v
			vec_proxy(vec_proxy&& v) : x(v.x), y(v.y) { ; }

			//! \brief Create a detail::vec_proxy<T, 2> object from a vec<T, 2> object
			//!
			//! \param v
			//! \return a detail::proxy<T, 2> object
			inline vec_proxy& operator=(const vec& v)
			{
				x = v.x;
				y = v.y;

				return *this;
			}

			inline vec operator-()
			{
				constexpr T minus_one = math<T>::minus_one;
				return vec(minus_one * x, minus_one * y);
			}

			#define MACRO(OP, IN_T)                 \
			inline void operator OP (const IN_T& v) \
			{                                       \
				x OP v.x;                           \
				y OP v.y;                           \
			}                                       \

			MACRO(+=, vec_proxy)
			MACRO(-=, vec_proxy)
			MACRO(*=, vec_proxy)
			MACRO(/=, vec_proxy)

			MACRO(+=, vec)
			MACRO(-=, vec)
			MACRO(*=, vec)
			MACRO(/=, vec)

			#undef MACRO

			#define MACRO(OP)                       \
			inline void operator OP (const T c)     \
			{                                       \
				x OP c;                             \
				y OP c;                             \
			}                                       \

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
				return math<T>::sqrt(x * x + y * y);
			}
		};

		//! \brief D=3 specialization with components x, y and z
		//!
		//! \tparam T data type
		template <typename T>
		class vec_proxy<T, 3>
		{
			using vec = VEC_NAMESPACE::vec<T, 3>;

		public:

			//! Remember the template type parameter T
			using fundamental_type = T;
			//! Remember the template parameter D (=3)
			static constexpr std::size_t dim = 3;

			//! Reference to x component of the corresponding vec<T, 3> object
			T& x;
			//! Reference to y component of the corresponding vec<T, 3> object
			T& y;
			//! Reference to z component of the corresponding vec<T, 3> object
			T& z;

			//! \brief Constructor taking the address of the x component of the vec<T, 3> object
			//!
			//! For D=3, the address of the y component is &ptr[1 * n] and the address
			//! of the z componenent is &ptr[2 * n].
			//!
			//! \param ptr pointer to x component
			//! \param n extent of the underlying field in dimension 1 (innermost)
			vec_proxy(T* ptr, const std::size_t n) : x(ptr[0 * n]), y(ptr[1 * n]), z(ptr[2 * n]) { ; }

			//! \brief Create a detail::vec_proxy<T, 3> object from a vec<T, 3> object
			//!
			//! \param v
			vec_proxy(vec& v) : x(v.x), y(v.y), z(v.z) { ; }

			//! \brief Create a detail::vec_proxy<T, 3> object from a vec<T, 3> object
			//!
			//! \param v
			vec_proxy(vec&& v) : x(v.x), y(v.y), z(v.z) { ; }

			//! \brief Copy constructor
			//!
			//! \param v
			vec_proxy(vec_proxy& v) : x(v.x), y(v.y), z(v.z) { ; }

			//! \brief Move constructor
			//!
			//! \param v
			vec_proxy(vec_proxy&& v) : x(v.x), y(v.y), z(v.z) { ; }

			//! \brief Create a detail::vec_proxy<T, 3> object from a vec<T, 3> object
			//!
			//! \param v
			//! \return a detail::proxy<T, 3> object
			vec_proxy& operator=(const vec& v)
			{
				x = v.x;
				y = v.y;
				z = v.z;

				return *this;
			}

			inline vec operator-()
			{
				constexpr T minus_one = math<T>::minus_one;
				return vec(minus_one * x, minus_one * y, minus_one * z);
			}

			#define MACRO(OP, IN_T)                 \
			inline void operator OP (const IN_T& v) \
			{                                       \
				x OP v.x;                           \
				y OP v.y;                           \
				z OP v.z;                           \
			}                                       \

			MACRO(+=, vec_proxy)
			MACRO(-=, vec_proxy)
			MACRO(*=, vec_proxy)
			MACRO(/=, vec_proxy)

			MACRO(+=, vec)
			MACRO(-=, vec)
			MACRO(*=, vec)
			MACRO(/=, vec)

			#undef MACRO

			#define MACRO(OP)                       \
			inline void operator OP (const T c)     \
			{                                       \
				x OP c;                             \
				y OP c;                             \
				z OP c;                             \
			}                                       \

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
				return math<T>::sqrt(x * x + y * y + z * z);
			}
		};
	}
}

#endif
