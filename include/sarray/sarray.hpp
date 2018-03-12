// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(SARRAY_HPP)
#define SARRAY_HPP

#include <cstdint>
#include <utility>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

namespace XXX_NAMESPACE
{
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//! \brief A fixed sized array
	//!
	//! \tparam T data type
	//! \tparam D array size
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	template <typename T, std::size_t D>
	class sarray
	{
		T data[D];

	public:

		//! \brief Standard constructor
		sarray() : data{} { ; }

		//! \brief Constructor taking D (or less) arguments to initialize the array
		//!
		//! \tparam Args variadic template type parameter list
		//! \param args parameters
		template <typename ... Args>
		sarray(const Args ... args) : data{ static_cast<T>(std::move(args)) ... } { ; }

		//! \brief Array subscript operator
		//!
		//! \param idx element index
		//! \return reference to the element
		inline T& operator[](const std::size_t idx)
		{
			return data[idx];
		}

		//! \brief Array subscript operator
		//!
		//! \param idx element index
		//! \return const reference to the element
		inline const T& operator[](const std::size_t idx) const
		{
			return data[idx];
		}

		//! \brief Test whether two sarrays are the same
		//!
		//! \param rhs
		//! \return result of the element-wise comparison of the sarray contents
		inline bool operator==(const sarray& rhs)
		{
			bool is_same = true;
			for (std::size_t i = 0; i < D; ++i)
			{
				is_same &= (data[i] == rhs.data[i]);
			}
			return is_same;
		}

		//! \brief Create an sarray<T, X> object from the current one containing the leading X elements
		//!
		//! \tparam X size of the new sarray object
		//! \return sarray<T, X>
		template <std::size_t X>
		inline constexpr sarray<T, X> shrink() const
		{
			static_assert(X <= D, "error: sarray::shrink -> at most D elements available");

			sarray<T, X> tmp;
			for (std::size_t i = 0; i < X; ++i)
			{
				tmp[i] = data[i];
			}
			return tmp;
		}

		//! \brief Reduce across all entries
		//!
		//! \return reduction
		template <typename F>
		inline T reduce(F func, const T init)
		{
			T r = init;
			for (std::size_t i = 0; i < D; ++i)
			{
				r = func(r, data[i]);
			}
			return r;
		}
	};
}

#endif
