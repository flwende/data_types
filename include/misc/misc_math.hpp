// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(MISC_MATH_HPP)
#define MISC_MATH_HPP

#include <cmath>

namespace MISC_NAMESPACE
{
	//! \brief Definition of some math functions and constants for different FP types
	template <typename T>
	struct math;

	template <>
	struct math<double>
	{
		static constexpr double one = 1.0;
		static constexpr double minus_one = -1.0;

		static double sqrt(const double x)
		{
			return std::sqrt(x);
		}

		static double log(const double x)
		{
			return std::log(x);
		}

		static double exp(const double x)
		{
			return std::exp(x);
		}
	};

	//! \brief Specialization with T = float
	template <>
	struct math<float>
	{
		static constexpr float one = 1.0F;
		static constexpr float minus_one = -1.0F;

		static float sqrt(const float x)
		{
			return sqrtf(x);
		}

		static float log(const float x)
		{
			return logf(x);
		}

		static float exp(const float x)
		{
			return expf(x);
		}
	};
}

#endif
