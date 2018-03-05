// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(TEMPLATE_STUFF_HPP)
#define TEMPLATE_STUFF_HPP

#include <cstdlib>
#include <cstdint>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

namespace XXX_NAMESPACE
{
	namespace variadic
	{
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//! \brief Get the number of arguments in the variadic list
		//!
		//! \tparam Args variadic argument list
		//! \return number of arguments
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		template <typename ... Args>
		constexpr std::size_t get_num_arguments()
		{
			return sizeof ... (Args);
		}

		////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//! \brief Get the value of the N-th argument and its type in the variadic list
		//!
		//! \tparam N element to access
		//! \tparam T data type (head)
		//! \tparam Args variadic argument list (tail)
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		template <std::int32_t N, typename T, typename ... Args>
		struct argument
		{
			//! Data type of the N-th argument
			using type = typename argument<N - 1, Args ...>::type;

			//! \brief Get the value of the N-th argument through recursion
			//!
			//! \param head frontmost argument
			//! \param tail remaining arguments
			//! \return argument list without the frontmost argument
			static constexpr T value(T head, Args ... tail)
			{
				return argument<N - 1, Args ...>::value(tail ...);
			}
		};

		////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//! \brief N = 0 Specialization (recursion ancher definition)
		//!
		//! \tparam T data type of the N-th element
		//! \tparam Args variadic argument list (tail)
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		template <typename T, typename ... Args>
		struct argument<0, T, Args ...>
		{
			//! Data type of the N-the element
			using type = T;

			//! \brief Value of the N-th argument
			//!
			//! \param head N-the argument
			//! \param tail remaining arguments
			//! \return N-the argument
			static constexpr T value(T head, Args ... tail)
			{
				return head;
			}
		}; 
	};
}

#endif
