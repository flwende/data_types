// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(DATA_TYPES_VEC_HPP)
#define DATA_TYPES_VEC_HPP

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#include "../misc/misc_math.hpp"

namespace XXX_NAMESPACE
{
	namespace detail
	{
		template <typename T, std::size_t D>
		class vec_proxy;
	}

	template <typename T>
	struct math;

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//! \brief A simple vector with D components
	//!
	//! \tparam T data type
	//! \tparam D dimension
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	template <typename T, std::size_t D>
	class vec;

	//! \brief D = 1 specialization with component x
	//!
	//! \tparam T data type
	template <typename T>
	class vec<T, 1>
	{
		using cvec = vec<const T, 1>;
		using vec_proxy = detail::vec_proxy<T, 1>;
		using proxy_cvec = detail::vec_proxy<const T, 1>;

	public:

		//! Remember the template type parameter T
		using fundamental_type = T;
		//! const version of this type
		using const_type = const cvec;
		//! Remember the template parameter D (=1)
		static constexpr std::size_t dim = 1;

		//! x component
		T x;

		//! \brief Standard constructor
		//!
		//! \param x
		vec(const T x = 0) : x(x) { ; }

		//! \brief Copy constructor
		//!
		//! \param v
		vec(const cvec& v) : x(v.x) { ; }

		//! \brief Create a vec<T, 1> object from a detail::vec_proxy<T, 1> object
		//!
		//! \param v
		vec(const vec_proxy& v) : x(v.x) { ; }

		//! \brief Create a vec<T, 1> object from a detail::vec_proxy<const T, 1> object
		//!
		//! \param v
		vec(proxy_cvec&& v) : x(v.x) { ; }

		inline vec operator-()
		{
			constexpr T minus_one = MATH_NAMESPACE::constants<T>::minus_one;
			return vec(minus_one * x);
		}

		#define MACRO(OP, IN_T)                 \
		inline void operator OP (const IN_T& v) \
		{                                       \
			x OP v.x;                           \
		}

		MACRO(+=, vec)
		MACRO(-=, vec)
		MACRO(*=, vec)
		MACRO(/=, vec)

		MACRO(+=, vec_proxy)
		MACRO(-=, vec_proxy)
		MACRO(*=, vec_proxy)
		MACRO(/=, vec_proxy)

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

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//! \brief D = 2 specialization with components x and y
	//!
	//! \tparam T data type
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	template <typename T>
	class vec<T, 2>
	{
		using cvec = vec<const T, 2>;
		using vec_proxy = detail::vec_proxy<T, 2>;
		using proxy_cvec = detail::vec_proxy<const T, 2>;

	public:

		//! Remember the template type parameter T
		using fundamental_type = T;
		//! const version of this type
		using const_type = const cvec;
		//! Remember the template parameter D (=2)
		static constexpr std::size_t dim = 2;

		//! x component
		T x;
		//! y component
		T y;

		//! \brief Standard constructor: all components are set to x (=0)
		//!
		//! \param x
		vec(const T x = 0) : x(x), y(x) { ; }

		//! \brief Constructor for component-wise initialization
		//!
		//! \param x
		//! \param y
		vec(const T x, const T y) : x(x), y(y) { ; }

		//! \brief Copy constructor
		//!
		//! \param v
		vec(const cvec& v) : x(v.x), y(v.y) { ; }

		//! \brief Create a vec<T, 2> object from a detail::vec_proxy<T, 2> object
		//!
		//! \param v
		vec(const vec_proxy& v) : x(v.x), y(v.y) { ; }

		//! \brief Create a vec<T, 2> object from a detail::vec_proxy<const T, 2> object
		//!
		//! \param v
		vec(proxy_cvec&& v) : x(v.x), y(v.y) { ; }

		inline vec operator-()
		{
			constexpr T minus_one = MATH_NAMESPACE::constants<T>::minus_one;
			return vec(minus_one * x, minus_one * y);
		}

		#define MACRO(OP, IN_T)                 \
		inline void operator OP (const IN_T& v) \
		{                                       \
			x OP v.x;                           \
			y OP v.y;                           \
		}                                       \

		MACRO(+=, vec)
		MACRO(-=, vec)
		MACRO(*=, vec)
		MACRO(/=, vec)

		MACRO(+=, vec_proxy)
		MACRO(-=, vec_proxy)
		MACRO(*=, vec_proxy)
		MACRO(/=, vec_proxy)

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
			return MATH_NAMESPACE::math<T>::sqrt(x * x + y * y);
		}
	};

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//! \brief D = 3 specialization with components x, y and z
	//!
	//! \tparam T data type
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	template <typename T>
	class vec<T, 3>
	{
		using cvec = vec<const T, 3>;
		using vec_proxy = detail::vec_proxy<T, 3>;
		using proxy_cvec = detail::vec_proxy<const T, 3>;

	public:

		//! Remember the template type parameter T
		using fundamental_type = T;
		//! const version of this type
		using const_type = const cvec;
		//! Remember the template parameter D (=3)
		static constexpr std::size_t dim = 3;

		//! x component
		T x;
		//! y component
		T y;
		//! z component
		T z;

		//! \brief Standard constructor: all components are set to x (=0)
		//!
		//! \param x
		vec(const T x = 0) : x(x), y(x), z(x) { ; }

		//! \brief Constructor for component-wise initialization
		//!
		//! \param x
		//! \param y
		//! \param z
		vec(const T x, const T y, const T z) : x(x), y(y), z(z) { ; }

		//! \brief Copy constructor
		//!
		//! \param v
		//vec(const vec& v) : x(v.x), y(v.y), z(v.z) { ; }

		vec(const cvec& v) : x(v.x), y(v.y), z(v.z) { ; }

		//! \brief Create a vec<T, 3> object from a detail::vec_proxy<T, 3> object
		//!
		//! \param v
		vec(const vec_proxy& v) : x(v.x), y(v.y), z(v.z) { ; }

		//! \brief Create a vec<T, 3> object from a detail::vec_proxy<const T, 3> object
		//!
		//! \param v
		vec(proxy_cvec&& v) : x(v.x), y(v.y), z(v.z) { ; }

		inline vec operator-()
		{
			constexpr T minus_one = MATH_NAMESPACE::constants<T>::minus_one;
			return vec(minus_one * x, minus_one * y, minus_one * z);
		}

		#define MACRO(OP, IN_T)                 \
		inline void operator OP (const IN_T& v) \
		{                                       \
			x OP v.x;                           \
			y OP v.y;                           \
			z OP v.z;                           \
		}                                       \

		MACRO(+=, vec)
		MACRO(-=, vec)
		MACRO(*=, vec)
		MACRO(/=, vec)

		MACRO(+=, vec_proxy)
		MACRO(-=, vec_proxy)
		MACRO(*=, vec_proxy)
		MACRO(/=, vec_proxy)

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
			return MATH_NAMESPACE::math<T>::sqrt(x * x + y * y + z * z);
		}
	};

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//! \brief Testing on whether T is of type vec<TT, DD> or not
	//!
	//! (default) T is not of type vec<TT, DD>.
	//!
	//! \tparam T
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	template <typename T>
	struct is_vec
	{
		static constexpr bool value = false;
	};

	#define MACRO(T)                                                                            \
	template <> struct is_vec<vec<T, 1>> { static constexpr bool value = true; };               \
	template <> struct is_vec<vec<T, 2>> { static constexpr bool value = true; };               \
	template <> struct is_vec<vec<T, 3>> { static constexpr bool value = true; };               \
	template <> struct is_vec<const vec<T, 1>> { static constexpr bool value = true; };         \
	template <> struct is_vec<const vec<T, 2>> { static constexpr bool value = true; };         \
	template <> struct is_vec<const vec<T, 3>> { static constexpr bool value = true; };         \

	MACRO(double)
	MACRO(float)
	MACRO(std::uint64_t)
	MACRO(std::int64_t)
	MACRO(std::uint32_t)
	MACRO(std::int32_t)

	MACRO(const double)
	MACRO(const float)
	MACRO(const std::uint64_t)
	MACRO(const std::int64_t)
	MACRO(const std::uint32_t)
	MACRO(const std::int32_t)

	#undef MACRO

	namespace detail
	{
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//! \brief Create a const version of data type T
		//!
		//! \tparam T data type
		//! \tparam Enabled needed for partial specialization with T = vec<TT, DD> and Layout = SoA
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		template <typename T, typename Enabled = void>
		struct make_const
		{
			//! const data type: if the const keyword appears multiple times, it is the same as if it appears just once
			using type = const T;
		};

		////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//! \brief Specialization with T = vec<TT, DD>
		//!
		//! \tparam T data type
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		template <typename T>
		struct make_const<T, typename std::enable_if<is_vec<T>::value>::type>
		{
			//! const data type: if the const keyword appears multiple times, it is the same as if it appears just once
			using type = typename T::const_type;
		};
	}
}

#include "vec_proxy.hpp"
#include "vec_math.hpp"

#endif
