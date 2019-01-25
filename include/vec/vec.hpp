// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(VEC_HPP)
#define VEC_HPP

#if defined(HAVE_SYCL)
#include <CL/sycl.hpp>
#endif

#include <cstdint>
#include <type_traits>

#if !defined(VEC_NAMESPACE)
#if !defined(XXX_NAMESPACE)
#define VEC_NAMESPACE fw
#else
#define VEC_NAMESPACE XXX_NAMESPACE
#endif
#endif

#if defined(OLD)
#include "../simd/simd.hpp"
#include "../misc/misc_math.hpp"

namespace VEC_NAMESPACE
{
	namespace detail
	{
		template <typename T, std::size_t D>
		class vec_proxy;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//! \brief A simple vector with D components
	//!
	//! \tparam T data type
	//! \tparam D dimension
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	template <typename T, std::size_t D>
	class vec;

	//! \brief D = 1 specialization with component x
	//!
	//! \tparam T data type
	template <typename T>
	class vec<T, 1>
	{
		static_assert(!std::is_const<T>::value, "error: vec<const T, 1> is not allowed");

		using vec_proxy = detail::vec_proxy<T, 1>;
		using vec_proxy_const = detail::vec_proxy<const T, 1>;

	public:

		//! Remember the template type parameter T
		using fundamental_type = T;
		//! Remember the template parameter D (=1)
		static constexpr std::size_t dim = 1;

		//! x component
		union
		{
			#if defined(HAVE_SYCL)
			cl::sycl::vec<T, 1> sycl_vec;
			#endif
			struct { T x; };
		};

		//! Constructors
		#if defined(HAVE_SYCL)
		vec(const T x = 0) : sycl_vec(x) { ; }
		vec(const cl::sycl::vec<T, 1>& v) : sycl_vec(v) { ; }
		vec(const vec& v) : sycl_vec(v.sycl_vec) { ; }
		vec(const vec_proxy& v) : sycl_vec(v.x) { ; }
		vec(const vec_proxy_const& v) : sycl_vec(v.x) { ; }

		vec operator=(const vec& v)
		{
			sycl_vec = v.sycl_vec;
			return *this;
		}

		vec operator=(const vec& v) const
		{
			sycl_vec = v.sycl_vec;
			return *this;
		}
		#else
		vec(const T x = 0) : x(x) { ; }
		vec(const vec& v) : x(v.x) { ; }
		vec(const vec_proxy& v) : x(v.x) { ; }
		vec(const vec_proxy_const& v) : x(v.x) { ; }
		#endif

		inline vec operator-()
		{
			constexpr T minus_one = MISC_NAMESPACE::math<T>::minus_one;
			#if defined(HAVE_SYCL)
			return vec(sycl_vec * minus_one);
			#else
			return vec(minus_one * x);
			#endif
		}

		#if defined(HAVE_SYCL)
		#define MACRO(OP, IN_T)				\
		inline void operator OP (const IN_T& v)		\
		{						\
			sycl_vec OP v.sycl_vec;			\
		}
		#else
		#define MACRO(OP, IN_T)				\
		inline void operator OP (const IN_T& v)		\
		{						\
			x OP v.x;				\
		}
		#endif

		MACRO(+=, vec)
		MACRO(-=, vec)
		MACRO(*=, vec)
		MACRO(/=, vec)

		MACRO(+=, vec_proxy)
		MACRO(-=, vec_proxy)
		MACRO(*=, vec_proxy)
		MACRO(/=, vec_proxy)

		#undef MACRO

		#if defined(HAVE_SYCL)
		#define MACRO(OP)				\
		inline void operator OP (const T c)		\
		{						\
			sycl_vec OP c;				\
		}
		#else
		#define MACRO(OP)				\
		inline void operator OP (const T c)		\
		{						\
			x OP c;					\
		}
		#endif						\

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

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//! \brief D = 2 specialization with components x and y
	//!
	//! \tparam T data type
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	template <typename T>
	class vec<T, 2>
	{
		static_assert(!std::is_const<T>::value, "error: vec<const T, 2> is not allowed");

		using vec_proxy = detail::vec_proxy<T, 2>;
		using vec_proxy_const = detail::vec_proxy<const T, 2>;

	public:

		//! Remember the template type parameter T
		using fundamental_type = T;
		//! Remember the template parameter D (=2)
		static constexpr std::size_t dim = 2;

		//! x, y component
		union
		{
			#if defined(HAVE_SYCL)
			cl::sycl::vec<T, 2> sycl_vec;
			#endif
			struct { T x; T y; };
		};

		//! Constructors
		#if defined(HAVE_SYCL)
		vec(const T x = 0) : sycl_vec(x) { ; }
		vec(const T x, const T y) : sycl_vec(x, y) { ; }
		vec(const cl::sycl::vec<T, 2>& v) : sycl_vec(v) { ; }
		vec(const vec& v) : sycl_vec(v.sycl_vec) { ; }
		vec(const vec_proxy& v) : sycl_vec(v.x, v.y) { ; }
		vec(const vec_proxy_const& v) : sycl_vec(v.x, v.y) { ; }

		vec operator=(const vec& v)
		{
			sycl_vec = v.sycl_vec;
			return *this;
		}

		vec operator=(const vec& v) const
		{
			sycl_vec = v.sycl_vec;
			return *this;
		}
		#else
		vec(const T x = 0) : x(x), y(x) { ; }
		vec(const T x, const T y) : x(x), y(y) { ; }
		vec(const vec& v) : x(v.x), y(v.y) { ; }
		vec(const vec_proxy& v) : x(v.x), y(v.y) { ; }
		vec(const vec_proxy_const& v) : x(v.x), y(v.y) { ; }
		#endif

		inline vec operator-()
		{
			constexpr T minus_one = MISC_NAMESPACE::math<T>::minus_one;
			#if defined(HAVE_SYCL)
			return vec(sycl_vec * minus_one);
			#else
			return vec(minus_one * x, minus_one * y);
			#endif
		}

		#if defined(HAVE_SYCL)
		#define MACRO(OP, IN_T)				\
		inline void operator OP (const IN_T& v)		\
		{						\
			sycl_vec OP v.sycl_vec;			\
		}
		#else
		#define MACRO(OP, IN_T)				\
		inline void operator OP (const IN_T& v)		\
		{						\
			x OP v.x;				\
			y OP v.y;				\
		}
		#endif

		MACRO(+=, vec)
		MACRO(-=, vec)
		MACRO(*=, vec)
		MACRO(/=, vec)

		MACRO(+=, vec_proxy)
		MACRO(-=, vec_proxy)
		MACRO(*=, vec_proxy)
		MACRO(/=, vec_proxy)

		#undef MACRO

		#if defined(HAVE_SYCL)
		#define MACRO(OP)				\
		inline void operator OP (const T c)		\
		{						\
			sycl_vec OP c;				\
		}
		#else
		#define MACRO(OP)				\
		inline void operator OP (const T c)		\
		{						\
			x OP c;					\
			y OP c;					\
		}
		#endif

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
			#if defined(HAVE_SYCL)
			return cl::sycl::length(sycl_vec);
			#else
			return MISC_NAMESPACE::math<T>::sqrt(x * x + y * y);
			#endif
			
		}
	};

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//! \brief D = 3 specialization with components x, y and z
	//!
	//! \tparam T data type
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	template <typename T>
	class vec<T, 3>
	{
		static_assert(!std::is_const<T>::value, "error: vec<const T, 3> is not allowed");

		using vec_proxy = detail::vec_proxy<T, 3>;
		using vec_proxy_const = detail::vec_proxy<const T, 3>;

	public:

		//! Remember the template type parameter T
		using fundamental_type = T;
		//! Remember the template parameter D (=3)
		static constexpr std::size_t dim = 3;

		//! x, y, z component
		union
		{
			#if defined(HAVE_SYCL)
			cl::sycl::vec<T, 3> sycl_vec;
			#endif
			struct { T x; T y; T z; };
		};
	       
		//! Constructors
		#if defined(HAVE_SYCL)
		vec(const T x = 0) : sycl_vec(x) { ; }
		vec(const T x, const T y, const T z) : sycl_vec(x, y, z) { ; }
		vec(const cl::sycl::vec<T, 3>& v) : sycl_vec(v) { ; }
		vec(const vec& v) : sycl_vec(v.sycl_vec) { ; }
		vec(const vec_proxy& v) : sycl_vec(v.x, v.y, v.z) { ; }
		vec(const vec_proxy_const& v) : sycl_vec(v.x, v.y, v.z) { ; }

		vec operator=(const vec& v)
		{
			sycl_vec = v.sycl_vec;
			return *this;
		}

		vec operator=(const vec& v) const
		{
			sycl_vec = v.sycl_vec;
			return *this;
		}
		#else
		vec(const T x = 0) : x(x), y(x), z(x) { ; }
		vec(const T x, const T y, const T z) : x(x), y(y), z(z) { ; }
		vec(const vec& v) : x(v.x), y(v.y), z(v.z) { ; }
		vec(const vec_proxy& v) : x(v.x), y(v.y), z(v.z) { ; }
		vec(const vec_proxy_const& v) : x(v.x), y(v.y), z(v.z) { ; }
		#endif

		inline vec operator-()
		{
			constexpr T minus_one = MISC_NAMESPACE::math<T>::minus_one;
			#if defined(HAVE_SYCL)
			return vec(sycl_vec * minus_one);
			#else
			return vec(minus_one * x, minus_one * y, minus_one * z);
			#endif
		}

		#if defined(HAVE_SYCL)
		#define MACRO(OP, IN_T)				\
		inline void operator OP (const IN_T& v)		\
		{						\
			sycl_vec OP v.sycl_vec;			\
		}
		#else
		#define MACRO(OP, IN_T)				\
		inline void operator OP (const IN_T& v)		\
		{						\
			x OP v.x;				\
			y OP v.y;				\
			z OP v.z;				\
		}
		#endif

		MACRO(+=, vec)
		MACRO(-=, vec)
		MACRO(*=, vec)
		MACRO(/=, vec)

		MACRO(+=, vec_proxy)
		MACRO(-=, vec_proxy)
		MACRO(*=, vec_proxy)
		MACRO(/=, vec_proxy)

		#undef MACRO

		#if defined(HAVE_SYCL)
		#define MACRO(OP)				\
		inline void operator OP (const T c)		\
		{						\
			sycl_vec OP c;				\
		}
		#else
		#define MACRO(OP)				\
		inline void operator OP (const T c)		\
		{						\
			x OP c;					\
			y OP c;					\
			z OP c;					\
		}
		#endif

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
			#if defined(HAVE_SYCL)
			return cl::sycl::length(sycl_vec);
			#else
			return MISC_NAMESPACE::math<T>::sqrt(x * x + y * y + z * z);
			#endif
		}
	};

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//! \brief Testing on whether T is of type vec<TT, DD> or not
	//!
	//! (default) T is not of type vec<TT, DD>.
	//!
	//! \tparam T
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	template <typename T>
	struct is_vec
	{
		static constexpr bool value = false;
	};

	#define MACRO(T)										\
	template <> struct is_vec<vec<T, 1>> { static constexpr bool value = true; };			\
	template <> struct is_vec<vec<T, 2>> { static constexpr bool value = true; };			\
	template <> struct is_vec<vec<T, 3>> { static constexpr bool value = true; };			\
	template <> struct is_vec<const vec<T, 1>> { static constexpr bool value = true; };		\
	template <> struct is_vec<const vec<T, 2>> { static constexpr bool value = true; };		\
	template <> struct is_vec<const vec<T, 3>> { static constexpr bool value = true; };		\

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
}

#include "vec_proxy.hpp"
#include "vec_misc.hpp"
#include "vec_math.hpp"
#else // OLD
namespace VEC_NAMESPACE
{
	template <typename T, std::uint32_t D>
	class vec;

    template <typename T_1, typename T_2, typename T_3>
    class A;

	namespace internal
	{
		template <typename T, std::uint32_t D>
		class vec_proxy;

        template <typename T_1, typename T_2, typename T_3>
        class A_proxy;

        template <typename T, typename Enabled = void>
		struct traits;
	}

	template <typename T>
	class vec<T, 3>
	{
		static_assert(std::is_fundamental<T>::value, "error: T is not a fundamental data type");
		static_assert(!std::is_void<T>::value, "error: T is void -> not allowed");
        static_assert(!std::is_volatile<T>::value, "error: T is volatile -> not allowed");
        static_assert(!std::is_const<T>::value, "error: T is const -> not allowed");

        template <typename TT, typename Enabled>
        friend struct internal::traits;

        // some meta data: to be generated by the proxy generator
        //using fundamental_type = T;
        //using unqualified_fundamental_type = T; // cv qualifiers not allowed
        using type = vec<T, 3>;
        //using unqualified_type = vec<T, 3>;
        using const_type = const vec<T, 3>;
        using proxy_type = typename internal::vec_proxy<T, 3>;
        using const_proxy_type = const typename internal::vec_proxy<const T, 3>;
		static constexpr std::uint32_t d = 3;
        static constexpr std::uint32_t sizeof_type = sizeof(T) + sizeof(T) + sizeof(T);

	public:

		vec() : x(0), y(0), z(0) {}
		vec(const T xyz) : x(xyz), y(xyz), z(xyz) {}
		vec(const T x, const T y, const T z) : x(x), y(y), z(z) {}

		template <typename X>
		vec(const vec<X, 3>& v) : x(v.x), y(v.y), z(v.z) {}

		template <typename X>
		vec(const internal::vec_proxy<X, 3>& vp) : x(vp.x), y(vp.y), z(vp.z) {}

		template <typename X>
		vec& operator+=(const vec<X, 3>& v)
		{
			x += v.x;
			y += v.y;
			z += v.z;
			return *this;	
		}

		template <typename X>
		vec& operator+=(const internal::vec_proxy<X, 3>& vp)
		{
			x += vp.x;
			y += vp.y;
			z += vp.z;
			return *this;	
		}

		template <typename X>
		vec& operator+=(const X xyz)
		{
			x += xyz;
			y += xyz;
			z += xyz;
			return *this;
		}

		T x;
		T y;
		T z;
	};

	template <typename T>
	std::ostream& operator<<(std::ostream& os, const vec<T, 3>& v)
	{
		os << "(" << v.x << "," << v.y << "," << v.z << ")";
		return os;
	}

	namespace internal
	{
		template <typename T>
		class vec_proxy<T, 3>
		{
			static_assert(std::is_fundamental<T>::value, "error: T is not a fundamental data type");
			static_assert(!std::is_void<T>::value, "error: T is void -> not allowed");
            static_assert(!std::is_volatile<T>::value, "error: T is volatile -> not allowed");

            template <typename TT, typename Enabled>
            friend struct traits;

            // some meta data: to be generated by the proxy generator
            //using fundamental_type = T;
            //using unqualified_fundamental_type = typename std::remove_const<fundamental_type>::type;
            using type = vec_proxy<T, 3>;
            //using unqualified_type = vec_proxy<unqualified_fundamental_type, 3>;
            //using const_type = const vec_proxy<const unqualified_fundamental_type, 3>;
            //using const_type = const vec_proxy<const typename std::remove_const<T>::type, 3>;
            using const_type = const vec_proxy<const T, 3>;
            static constexpr std::uint32_t d = 3;

			// some meta data: to be generated by the proxy generator
			using T_p1 = T;
			using T_p2 = T;
			using T_p3 = T;
			static constexpr bool is_homogeneous = (sizeof(T_p1) == sizeof(T_p2) && sizeof(T_p1) == sizeof(T_p3));
            using public_member_ptr_type = typename std::conditional<is_homogeneous, T, typename std::conditional<std::is_const<T>::value, const std::uint8_t, std::uint8_t>::type>::type;
            static constexpr std::uint32_t public_member_bytes = sizeof(T_p1) + sizeof(T_p2) + sizeof(T_p3);
            static constexpr std::uint32_t sizeof_type = sizeof(T) + sizeof(T) + sizeof(T);
			
		public:

            
            /*
			vec_proxy(public_member_ptr_type* ptr, const std::uint32_t n) 
                : 
                x(reinterpret_cast<T&>(ptr[0 * n])),
                y(reinterpret_cast<T&>(ptr[1 * n])),
                z(reinterpret_cast<T&>(ptr[2 * n])) {}
                */
               /*
            vec_proxy(public_member_ptr_type* ptr, const std::uint32_t n) 
                : 
                x(ptr[0 * n]),
                y(ptr[1 * n]),
                z(ptr[2 * n]) {}
                */
            
            vec_proxy(public_member_ptr_type* ptr, const std::uint32_t i, const std::uint32_t n) 
                : 
                x(reinterpret_cast<T&>(ptr[0 * n + i])),
                y(reinterpret_cast<T&>(ptr[1 * n + i])),
                z(reinterpret_cast<T&>(ptr[2 * n + i])) {}

			template <typename X, typename Y = typename std::enable_if<sizeof(T) == sizeof(X) && !(!std::is_const<T>::value && std::is_const<X>::value)>::type>
			vec_proxy(const vec_proxy<X, 3>& vp) 
                : 
                x(reinterpret_cast<T&>(vp.x)),
                y(reinterpret_cast<T&>(vp.y)),
                z(reinterpret_cast<T&>(vp.z)) {}

			template <typename X>
			vec_proxy& operator=(const X xyz)
			{
				x = xyz;
				y = xyz;
				z = xyz;
				return *this;
			}

			template <typename X>
			vec_proxy& operator+=(const vec<X, 3>& v)
			{
				x += v.x;
				y += v.y;
				z += v.z;
				return *this;
			}

			template <typename X>
			vec_proxy& operator+=(const vec_proxy<X, 3>& vp)
			{
				x += vp.x;
				y += vp.y;
				z += vp.z;
				return *this;
			}

			template <typename X>
			vec_proxy& operator+=(const X xyz)
			{
				x += xyz;
				y += xyz;
				z += xyz;
				return *this;
			}

			T& x;
			T& y;
			T& z;
		};

		template <typename T>
		std::ostream& operator<<(std::ostream& os, const vec_proxy<T, 3>& vp)
		{
			os << "(" << vp.x << "," << vp.y << "," << vp.z << ")";
			return os;
		}
	}

    template <typename T_1, typename T_2, typename T_3>
    class A
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

        template <typename TT, typename Enabled>
        friend struct internal::traits;

        // some meta data: to be generated by the proxy generator
        using type = A<T_1, T_2, T_3>;
        using const_type = const A<T_1, T_2, T_3>;
        using proxy_type = typename internal::A_proxy<T_1, T_2, T_3>;
        using const_proxy_type = const typename internal::A_proxy<const T_1, const T_2, const T_3>; 
        static constexpr std::uint32_t sizeof_type = sizeof(T_1) + sizeof(T_2) + sizeof(T_3);


    public:

        A() : u(0), v(0), w(0) {}
        template <typename X>
        A(const X x) : u(x), v(x), w(x) {}
        A(const T_1 u, const T_2 v, const T_3 w) : u(u), v(v), w(w) {}

        template <typename X_1, typename X_2, typename X_3>
        A(const A<X_1, X_2, X_3>& a) : u(a.u), v(a.v), w(a.w) {}

        T_1 u;
        T_2 v;
        T_3 w;
    };

    template <typename T_1, typename T_2, typename T_3>
    std::ostream& operator<<(std::ostream& os, const A<T_1, T_2, T_3>& a)
    {
        os << "(" << a.u << "," << a.v << "," << a.w << ")";
        return os;
    }

    namespace internal
    {
        template <typename T_1, typename T_2, typename T_3>
        class A_proxy
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

            template <typename TT, typename Enabled>
            friend struct traits;

            using type = A_proxy<T_1, T_2, T_3>;
            using const_type = const A_proxy<const T_1, const T_2, const T_3>;

            static constexpr bool is_homogeneous = (sizeof(T_1) == sizeof(T_2) && sizeof(T_1) == sizeof(T_3));
            using public_member_ptr_type = typename std::conditional<is_homogeneous, T_1, typename std::conditional<std::is_const<T_1>::value, const std::uint8_t, std::uint8_t>::type>::type;
            static constexpr std::uint32_t public_member_bytes = sizeof(T_1) + sizeof(T_2) + sizeof(T_3);
            static constexpr std::uint32_t offset_0 = 0;
            static constexpr std::uint32_t offset_1 = offset_0 + (is_homogeneous ? 1 : sizeof(T_1));
            static constexpr std::uint32_t offset_2 = offset_1 + (is_homogeneous ? 1 : sizeof(T_2));
            static constexpr std::uint32_t inc_0 = (is_homogeneous ? 1 : sizeof(T_1));
            static constexpr std::uint32_t inc_1 = (is_homogeneous ? 1 : sizeof(T_2));
            static constexpr std::uint32_t inc_2 = (is_homogeneous ? 1 : sizeof(T_3));
            static constexpr std::uint32_t sizeof_type = sizeof(T_1) + sizeof(T_2) + sizeof(T_3);

        public:
            /*
            A_proxy(public_member_ptr_type* ptr, const std::uint32_t n)
                :
                u(reinterpret_cast<T_1&>(ptr[offset_0 * n])),
                v(reinterpret_cast<T_2&>(ptr[offset_1 * n])),
                w(reinterpret_cast<T_3&>(ptr[offset_2 * n])) {}
            */
            A_proxy(public_member_ptr_type* ptr, const std::uint32_t i, const std::uint32_t n)
                :
                u(reinterpret_cast<T_1&>(ptr[offset_0 * n + i * inc_0])),
                v(reinterpret_cast<T_2&>(ptr[offset_1 * n + i * inc_1])),
                w(reinterpret_cast<T_3&>(ptr[offset_2 * n + i * inc_2])) {}

            template <typename X_1, typename X_2, typename X_3>
            A_proxy(const A_proxy<X_1, X_2, X_3>& ap)
                :
                u(reinterpret_cast<T_1&>(ap.u)),
                v(reinterpret_cast<T_2&>(ap.v)),
                w(reinterpret_cast<T_3&>(ap.w)) {}

            template <typename X>
            A_proxy& operator=(const X uvw)
            {
                u = uvw;
                v = uvw;
                w = uvw;
                return *this;
            }

            template <typename X_1, typename X_2, typename X_3>
            A_proxy& operator=(const A<X_1, X_2, X_3>& a)
            {
                u = a.u;
                v = a.v;
                w = a.w;
                return *this;
            }

            T_1& u;
            T_2& v;
            T_3& w;
        };

        template <typename T_1, typename T_2, typename T_3>
        std::ostream& operator<<(std::ostream& os, const A_proxy<T_1, T_2, T_3>& ap)
        {
            os << "(" << ap.u << "," << ap.v << "," << ap.w << ")";
            return os;
        }
    }
        
	template <typename T>
	struct provides_proxy_type
	{
		static constexpr bool value = false;
	};

	template <typename T, std::uint32_t D>
	struct provides_proxy_type<vec<T, D>>
	{
		static constexpr bool value = true;
	};

    template <typename T, std::uint32_t D>
	struct provides_proxy_type<const vec<T, D>>
	{
		static constexpr bool value = true;
	};

    template <typename T_1, typename T_2, typename T_3>
	struct provides_proxy_type<A<T_1, T_2, T_3>>
	{
		static constexpr bool value = true;
	};

    template <typename T_1, typename T_2, typename T_3>
	struct provides_proxy_type<const A<T_1, T_2, T_3>>
	{
		static constexpr bool value = true;
	};

    template <typename T>
	struct is_proxy_type
	{
		static constexpr bool value = false;
	};

    template <typename T, std::uint32_t D>
	struct is_proxy_type<internal::vec_proxy<T, D>>
	{
		static constexpr bool value = true;
	};

    template <typename T, std::uint32_t D>
	struct is_proxy_type<internal::vec_proxy<const T, D>>
	{
		static constexpr bool value = true;
	};

    template <typename T, std::uint32_t D>
	struct is_proxy_type<const internal::vec_proxy<const T, D>>
	{
		static constexpr bool value = true;
	};

    template <typename T_1, typename T_2, typename T_3>
    struct is_proxy_type<internal::A_proxy<T_1, T_2, T_3>>
	{
		static constexpr bool value = true;
	};

    template <typename T_1, typename T_2, typename T_3>
    struct is_proxy_type<internal::A_proxy<const T_1, const T_2, const T_3>>
	{
		static constexpr bool value = true;
	};

    template <typename T_1, typename T_2, typename T_3>
    struct is_proxy_type<const internal::A_proxy<const T_1, const T_2, const T_3>>
	{
		static constexpr bool value = true;
	};

	namespace internal
	{
        template <typename T, typename Enabled>
		struct traits
		{
            using type = T;
            //using unqualified_type = typename std::remove_cv<T>::type;
            using const_type = const T;//typename std::remove_cv<T>::type;

			//using fundamental_type = type;
            //using unqualified_fundamental_type = unqualified_type;
            //using const_fundamental_type = const_type;

            using proxy_type = T;

            using public_member_ptr_type = T;
            static constexpr std::uint32_t public_member_bytes = 1;
            static constexpr std::uint32_t sizeof_type = sizeof(T);
            /*
			static constexpr bool is_polymorphic = std::is_polymorphic<T>::value;
			static constexpr bool is_homogeneous = false;
			static constexpr std::size_t sizeof_public_members = 0;
			static constexpr std::size_t sizeof_rest = sizeof(T);
            */
		};

        template <typename T>
		struct traits<T, typename std::enable_if<provides_proxy_type<T>::value>::type>
		{
            using type = T;
            //using unqualified_type = typename T::unqualified_type;
            using const_type = typename T::const_type;

			//using fundamental_type = typename T::fundamental_type;
            //using unqualified_fundamental_type = typename T::unqualified_fundamental_type;
            //using const_fundamental_type = const unqualified_fundamental_type;

            using proxy_type = typename std::conditional<std::is_const<T>::value, typename T::const_proxy_type, typename T::proxy_type>::type;

            using public_member_ptr_type = typename traits<proxy_type>::public_member_ptr_type;
            static constexpr std::uint32_t public_member_bytes = traits<proxy_type>::public_member_bytes;
            static constexpr std::uint32_t sizeof_type = T::sizeof_type;
        };

        template <typename T>
		struct traits<T, typename std::enable_if<is_proxy_type<T>::value>::type>
		{
            using type = typename T::type;
            //using unqualified_type = typename T::unqualified_type;
            using const_type = typename T::const_type;

			//using fundamental_type = typename T::fundamental_type;
            //using unqualified_fundamental_type = typename T::unqualified_fundamental_type;
            //using const_fundamental_type = const unqualified_fundamental_type;

            using proxy_type = T;

            using public_member_ptr_type = typename T::public_member_ptr_type;
            static constexpr std::uint32_t public_member_bytes = T::public_member_bytes;
            static constexpr std::uint32_t sizeof_type = T::sizeof_type;
        };
	}
}
#endif

#endif
