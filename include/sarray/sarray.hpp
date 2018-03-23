// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(SARRAY_HPP)
#define SARRAY_HPP

#include <cstdint>
#include <utility>

#if !defined(XXX_NAMESPACE)
#define MISC_NAMESPACE fw
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
		template <typename TT, std::size_t DD>
		friend class sarray;

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

		//! \brief Copy constructor
		//!
		//! \tparam DD array size
		//! \param x an sarray object of type T with size X
		template <std::size_t X>
		sarray(const sarray<T, X>& x)
		{
			constexpr std::size_t i_max = (X < D ? X : D);
			for (std::size_t i = 0; i < i_max; ++i)
			{
				data[i] = static_cast<T>(x.data[i]);
			}
			for (std::size_t i = i_max; i < D; ++i)
			{
				data[i] = static_cast<T>(0);
			}
		}

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

		//! \brief Reduce across all entries
		//!
		//! \return reduction
		template <typename F>
		inline T reduce(F func, const T init) const
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
