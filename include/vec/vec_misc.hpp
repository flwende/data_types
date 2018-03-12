// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(VEC_MISC_HPP)
#define VEC_MISC_HPP

#if !defined(VEC_NAMESPACE)
#if !defined(XXX_NAMESPACE)
#define VEC_NAMESPACE fw
#else
#define VEC_NAMESPACE XXX_NAMESPACE
#endif
#endif

namespace VEC_NAMESPACE
{
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//! \brief Create a const version of data type T
	//!
	//! \tparam T data type
	//! \tparam Enabled needed for partial specialization with T = vec<TT, DD> and Layout = SoA
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	template <typename T, typename Enabled = void>
	struct make_const
	{
		//! const data type: if the const keyword appears multiple times, it is the same as if it appears just once
		using type = const T;
	};

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//! \brief Specialization with T = vec<TT, DD>
	//!
	//! \tparam T data type
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	template <typename T>
	struct make_const<T, typename std::enable_if<is_vec<T>::value>::type>
	{
		//! const data type: if the const keyword appears multiple times, it is the same as if it appears just once
		using type = typename T::const_type;
	};
}

#endif