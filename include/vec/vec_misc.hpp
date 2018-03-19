// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(VEC_MISC_HPP)
#define VEC_MISC_HPP

#include <cstdint>

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
	//! \brief Create a const version of data type T
	//!
	//! \tparam T data type
	//! \tparam Enabled needed for partial specialization with T = vec<TT, DD>
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	template <typename T, typename Enabled = void>
	struct make_const
	{
		//! const type: if the const keyword appears multiple times, it is the same as if it appears just once
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
		//! const type: if the const keyword appears multiple times, it is the same as if it appears just once
		using type = typename T::const_type;
	};

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//! \brief Get fundamental data type and dimension of T
	//!
	//! \tparam T data type
	//! \tparam Enabled needed for partial specialization with T = vec<TT, DD>
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	template <typename T, XXX_NAMESPACE::data_layout Layout = AoS, typename Enabled = void>
	struct type_info
	{
		//! data type
		using mapped_type = T;

		//! dimension
		static constexpr std::size_t extra_dim = 1;

		//! get number of elements for padding
		static constexpr std::size_t get_n_padd(const std::size_t n_bytes)
		{
			return (n_bytes + sizeof(T) - 1) / sizeof(T);
		}
	};

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//! \brief Specialization with T = vec<TT, DD> and data layout SoA
	//!
	//! \tparam T data type
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	template <typename T>
	struct type_info<T, SoA, typename std::enable_if<is_vec<T>::value>::type>
	{
		//! data type
		using mapped_type = typename T::fundamental_type;

		//! dimension
		static constexpr std::size_t extra_dim = T::dim;

		//! get number of elements for padding
		static constexpr std::size_t get_n_padd(const std::size_t n_bytes)
		{
			return SIMD_NAMESPACE::simd::type<mapped_type>::width;
		}
	};

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//! \brief Specialization with T = vec<TT, DD> and data layout SoAoS
	//!
	//! \tparam T data type
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	template <typename T>
	struct type_info<T, SoAoS, typename std::enable_if<is_vec<T>::value>::type>
	{
		//! data type
		using mapped_type = typename T::fundamental_type;

		//! dimension
		static constexpr std::size_t extra_dim = T::dim;
		
		//! get number of elements for padding
		static constexpr std::size_t get_n_padd(const std::size_t n_bytes)
		{
			return SIMD_NAMESPACE::simd::type<mapped_type>::width;
		}
	};
}

#endif