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

#include "../misc/misc_memory.hpp"

namespace VEC_NAMESPACE
{
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//! \brief A proxy data type for vec<T, D>
	//!
	//! This data type is returned by buffer<T, D, Layout, Alignment>::operator[]() if D = 1 and Layout=SoA.
	//! It holds references to component(s) x [,y [and z]] in main memory, so that data access via,
	//! e.g. obj[ ]..[ ].x, is possible.
	//!
	//! \tparam T data type
	//! \tparam D dimension
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    namespace internal
    {
        template <typename T, std::size_t D>
        struct vec_memory
        {
            using T_unqualified = typename std::remove_cv<T>::type;
            static constexpr std::size_t num_members = D;
            static constexpr std::size_t record_size = num_members * sizeof(T);
            
            const std::size_t n_innermost;
            T* __restrict__ ptr;
            
            vec_memory(T* __restrict__ ptr, const std::size_t n_innermost) 
                :
                n_innermost(n_innermost),
                ptr(ptr) {}

            vec_memory(const typename vec_proxy<T_unqualified, D>::memory& m)
                :
                n_innermost(m.n_innermost),
                ptr(m.ptr) {}

            vec_memory(const typename vec_proxy<const T_unqualified, D>::memory& m)
                : 
                n_innermost(m.n_innermost),
                ptr(m.ptr) {}

            vec_memory(const vec_memory& m, const std::size_t slice_idx, const std::size_t idx)
                :
                n_innermost(m.n_innermost),
                ptr(&m.ptr[slice_idx * num_members * n_innermost + idx]) {}

            vec_memory at(const std::size_t slice_idx, const std::size_t idx)
            {
                return vec_memory(*this, slice_idx, idx);
            }

            vec_memory at(const std::size_t slice_idx, const std::size_t idx) const
            {
                return vec_memory(*this, slice_idx, idx);
            }

            // replace by internal alignment
            static std::size_t padding(const std::size_t n, const std::size_t alignment = SIMD_NAMESPACE::simd::alignment)
            {
                if (!MISC_NAMESPACE::is_power_of<2>(alignment))
                {
                    std::cerr << "warning: alignment is not a power of 2" << std::endl;
                    return n;
                }
                
                const std::size_t ratio = std::max(1UL, alignment / sizeof(T));

                return ((n + ratio - 1) / ratio) * ratio;
            }

            template <std::size_t DD>
            static T* allocate(const sarray<std::size_t, DD>& n, const std::size_t alignment = SIMD_NAMESPACE::simd::alignment)
            {
                if (n[0] != padding(n[0], alignment))
                {
                    std::cerr << "error in vec_proxy::vec_memory::allocate : n[0] does not match alignment" << std::endl;
                }

                return reinterpret_cast<T*>(_mm_malloc(n.reduce_mul() * record_size, alignment));
            }

            static void deallocate(vec_memory& m)
            {
                if (m.ptr)
                {
                    _mm_free(m.ptr);
                }
            }
        };

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
            using memory = typename internal::vec_memory<T, 1>;

            T& x;

            vec_proxy(memory m)
                :
                x(m.ptr) {}

            vec_proxy(const vec_proxy& vp)
                :
                x(vp.x) {}

            vec_proxy(VEC_NAMESPACE::vec<T_unqualified, 1>& v)
                :
                x(v.x) {}

            vec_proxy(const VEC_NAMESPACE::vec<T_unqualified, 1>& v)
                :
                x(v.x) {}

        #define MACRO(OP, IN_T)                                         \
            template <typename X>                                       \
            vec_proxy& operator OP (const VEC_NAMESPACE::IN_T<X, 1>& v) \
            {                                                           \
                x OP v.x;                                               \
				return *this;                                           \
            }                                                           \

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

        #define MACRO(OP)                                               \
			template <typename X>                                       \
			vec_proxy& operator OP (const X x)                          \
			{                                                           \
				x OP x;                                                 \
				return *this;                                           \
			}                                                           \

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
            using memory = typename internal::vec_memory<T, 2>;

            T& x;
			T& y;

            vec_proxy(memory m)
                :
                x(m.ptr[0 * m.n_innermost]),
                y(m.ptr[1 * m.n_innermost]) {}

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

        #define MACRO(OP, IN_T)                                         \
            template <typename X>                                       \
            vec_proxy& operator OP (const VEC_NAMESPACE::IN_T<X, 2>& v) \
            {                                                           \
                x OP v.x;                                               \
				y OP v.y;                                               \
				return *this;                                           \
            }                                                           \

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

        #define MACRO(OP)                                               \
			template <typename X>                                       \
			vec_proxy& operator OP (const X xy)                         \
			{                                                           \
				x OP xy;                                                \
				y OP xy;                                                \
				return *this;                                           \
			}                                                           \

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
                return MISC_NAMESPACE::math<T>::sqrt(x * x + y * y);
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
            using memory = typename internal::vec_memory<T, 3>;

            vec_proxy(memory m)
                :
                x(m.ptr[0 * m.n_innermost]),
                y(m.ptr[1 * m.n_innermost]),
                z(m.ptr[2 * m.n_innermost]) {}

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

        #define MACRO(OP, IN_T)                                         \
            template <typename X>                                       \
            vec_proxy& operator OP (VEC_NAMESPACE::IN_T<X, 3>& v)       \
            {                                                           \
                x OP v.x;                                               \
				y OP v.y;                                               \
				z OP v.z;                                               \
				return *this;                                           \
            }                                                           \
                                                                        \
            template <typename X>                                       \
            vec_proxy& operator OP (const VEC_NAMESPACE::IN_T<X, 3>& v) \
            {                                                           \
                x OP v.x;                                               \
				y OP v.y;                                               \
				z OP v.z;                                               \
				return *this;                                           \
            }                                                           \

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

        #define MACRO(OP)                                               \
			template <typename X>                                       \
			vec_proxy& operator OP (const X xyz)                        \
			{                                                           \
				x OP xyz;                                               \
				y OP xyz;                                               \
				z OP xyz;                                               \
				return *this;                                           \
			}                                                           \

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
                return MISC_NAMESPACE::math<T>::sqrt(x * x + y * y + z * z);
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