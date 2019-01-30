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

#if !defined(OLD)
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

		using element_type = T;
		static constexpr std::size_t d = D;

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
		//! \tparam X array size
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

		//! \brief Replace element at position 'idx'
		//!
		//! \param x replacement
		//! \param idx element index
		//! \return an sarray object
		sarray<T, D> replace(const T x, const std::size_t idx) const 
		{
			sarray<T, D> result = *this;

			if (idx < D) 
			{
				result[idx] = x;
			}

			return result;
		}

		//! \brief Insert element at position 'idx'
		//!
		//! the last element of the array will be removed
		//!
		//! \param x element to be inserted
		//! \param idx element index
		//! \return an sarray object
		sarray<T, D> insert_and_chop(const T x, const std::size_t idx = 0) const
		{
			if (idx >= D) return *this;

			sarray<T, D> result;

			// take everything before position 'idx'
			for (std::size_t i = 0; i < idx; ++i)
			{
				result.data[i] = data[i];
			}

			// insert element at position 'idx'
			result.data[idx] = x;

			// take everything behind position 'idx' 
			for (std::size_t i = idx; i < (D - 1); ++i)
			{
				result.data[i + 1] = data[i];
			}

			return result;
		}

		//! \brief Test whether two sarrays are the same
		//!
		//! \param rhs
		//! \return result of the element-wise comparison of the sarray contents
		inline bool operator==(const sarray& rhs) const
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
		//! \param func a lambda implementing the reduction
		//! \param r_0 start value for the reduction
		//! \return reduction
		template <typename F>
		inline T reduce(F func, const T r_0) const
		{
			T r = r_0;
			for (std::size_t i = 0; i < D; ++i)
			{
				r = func(r, data[i]);
			}
			return r;
		}

		inline T reduce_mul() const
		{
			return reduce([&](const std::size_t product, const std::size_t element) { return (product * element); }, 1);
		}

		std::size_t size() const
		{
			return D;
		}
	};
}
#else // OLD
namespace XXX_NAMESPACE
{
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//! \brief A fixed sized array
	//!
	//! \tparam T data type
	//! \tparam D array size
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	template <typename T, std::uint32_t D>
	class sarray
	{
		template <typename TT, std::uint32_t DD>
		friend class sarray;

		T data[D];

	public:

		//! \brief Standard constructor
		//sarray() : data{} { ; }

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
		template <std::uint32_t X>
		sarray(const sarray<T, X>& x)
		{
			constexpr std::uint32_t i_max = (X < D ? X : D);
			for (std::uint32_t i = 0; i < i_max; ++i)
			{
				data[i] = static_cast<T>(x.data[i]);
			}
			for (std::uint32_t i = i_max; i < D; ++i)
			{
				data[i] = static_cast<T>(0);
			}
		}

		//! \brief Size
		//!
		//! \return size of the array
		inline std::uint32_t size() const
		{
			return D;
		}

		//! \brief Array subscript operator
		//!
		//! \param idx element index
		//! \return reference to the element
		inline T& operator[](const std::uint32_t idx)
		{
			return data[idx];
		}

		//! \brief Array subscript operator
		//!
		//! \param idx element index
		//! \return const reference to the element
		inline const T& operator[](const std::uint32_t idx) const
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
			for (std::uint32_t i = 0; i < D; ++i)
			{
				is_same &= (data[i] == rhs.data[i]);
			}
			return is_same;
		}

		//! \brief Reduce across all entries
		//!
		//! \param func a lambda implementing the reduction
		//! \param r_0 start value for the reduction
		//! \return reduction
		template <typename F>
		inline T reduce(F func, const T r_0, const std::uint32_t d = D) const
		{
			T r = r_0;
			for (std::uint32_t i = 0, iMax = std::min(d, D); i < iMax; ++i)
			{
				r = func(r, data[i]);
			}
			return r;
		}
	};
}
#endif

#endif
