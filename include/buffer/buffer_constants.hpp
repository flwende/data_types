// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(DATA_TYPES_CONSTANTS_HPP)
#define DATA_TYPES_CONSTANTS_HPP

namespace XXX_NAMESPACE
{
	namespace detail
	{
		//! \brief Definition of some constants for different FP types
		template <typename T>
		struct constants;

		//! \brief Specialization with T = double
		template <>
		struct constants<double>
		{
			static constexpr double one = 1.0;
			static constexpr double minus_one = -1.0;
		};

		//! \brief Specialization with T = float
		template <>
		struct constants<float>
		{
			static constexpr float one = 1.0F;
			static constexpr float minus_one = -1.0F;
		};
	}
}

#endif
