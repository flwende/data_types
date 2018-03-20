// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <omp.h>

#include "kernel.hpp"

using namespace fw;

#if defined(__INTEL_SDLT)
	template <>
	template <>
	double kernel<real3_t>::exp<1>(buffer_type<real3_t, 1>& x)
	{
		double time = omp_get_wtime();
		
		auto a_x = x.access();
		#if defined(__INTEL_COMPILER)
		#pragma forceinline recursive
		#pragma omp simd
		#endif
		for (std::size_t i = 0; i < x.size(); ++i)
		{
			a_x[i] = ::exp(a_x[i]);
		}

		return (omp_get_wtime() - time);
	}

	template <>
	template <>
	double kernel<real3_t>::log<1>(buffer_type<real3_t, 1>& x)
	{
		double time = omp_get_wtime();
		
		auto a_x = x.access();
		#if defined(__INTEL_COMPILER)
		#pragma forceinline recursive
		#pragma omp simd
		#endif
		for (std::size_t i = 0; i < x.size(); ++i)
		{
			a_x[i] = ::log(a_x[i]);
		}

		return (omp_get_wtime() - time);
	}
#else
	template <>
	template <>
	double kernel<real3_t>::exp<1>(buffer_type<real3_t, 1>& x)
	{
		double time = omp_get_wtime();
		
		auto a_x = x.read_write();
		#if defined(USE_PLAIN_POINTER)	
		real_t* ptr_x = &(a_x[0].x);
		real_t* ptr_y = &(a_x[0].y);
		real_t* ptr_z = &(a_x[0].z);
		if (layout == fw::data_layout::SoA)
		{
			for (std::size_t i = 0; i < x.size[0]; ++i)
			{
				ptr_x[i] = fw::math<real_t>::exp(ptr_x[i]);
				ptr_y[i] = fw::math<real_t>::exp(ptr_y[i]);
				ptr_z[i] = fw::math<real_t>::exp(ptr_z[i]);
			}
		}
		else if (layout == fw::data_layout::AoS)
		{
			for (std::size_t i = 0; i < (3 * x.size[0]); i += 3)
			{
				ptr_x[i] = fw::math<real_t>::exp(ptr_x[i]);
				ptr_y[i] = fw::math<real_t>::exp(ptr_y[i]);
				ptr_z[i] = fw::math<real_t>::exp(ptr_z[i]);
			}
		}
		#else
		#if defined(__INTEL_COMPILER)
		#pragma forceinline recursive
		#pragma omp simd
		#endif
		for (std::size_t i = 0; i < x.size[0]; ++i)
		{
			a_x[i] = fw::exp(a_x[i]);
		}
		#endif

		return (omp_get_wtime() - time);
	}

	template <>
	template <>
	double kernel<real3_t>::exp<3>(buffer_type<real3_t, 3>& x)
	{
		double time = omp_get_wtime();

		auto a_x = x.read_write();	
		#if defined(USE_PLAIN_POINTER)
		for (std::size_t k = 0; k < x.size[2]; ++k)
		{
			for (std::size_t j = 0; j < x.size[1]; ++j)
			{
				real_t* ptr_x = &(a_x[k][j][0].x);
				real_t* ptr_y = &(a_x[k][j][0].y);
				real_t* ptr_z = &(a_x[k][j][0].z);
				if (layout == fw::data_layout::SoA)
				{
					for (std::size_t i = 0; i < x.size[0]; ++i)
					{
						ptr_x[i] = fw::math<real_t>::exp(ptr_x[i]);
						ptr_y[i] = fw::math<real_t>::exp(ptr_y[i]);
						ptr_z[i] = fw::math<real_t>::exp(ptr_z[i]);
					}
				}
				else if (layout == fw::data_layout::AoS)
				{
					for (std::size_t i = 0; i < (3 * x.size[0]); i += 3)
					{
						ptr_x[i] = fw::math<real_t>::exp(ptr_x[i]);
						ptr_y[i] = fw::math<real_t>::exp(ptr_y[i]);
						ptr_z[i] = fw::math<real_t>::exp(ptr_z[i]);
					}
				}
			}
		}
		#else
		for (std::size_t k = 0; k < x.size[2]; ++k)
		{
			for (std::size_t j = 0; j < x.size[1]; ++j)
			{
				#if defined(__INTEL_COMPILER)
				#pragma forceinline recursive
				#pragma omp simd
				#endif
				for (std::size_t i = 0; i < x.size[0]; ++i)
				{
					a_x[k][j][i] = fw::exp(a_x[k][j][i]);
				}
			}
		}
		#endif

		return (omp_get_wtime() - time);
	}

	template <>
	template <>
	double kernel<real3_t>::log<1>(buffer_type<real3_t, 1>& x)
	{
		double time = omp_get_wtime();
		
		auto a_x = x.read_write();
		#if defined(USE_PLAIN_POINTER)	
		real_t* ptr_x = &(a_x[0].x);
		real_t* ptr_y = &(a_x[0].y);
		real_t* ptr_z = &(a_x[0].z);
		if (layout == fw::data_layout::SoA)
		{
			for (std::size_t i = 0; i < x.size[0]; ++i)
			{
				ptr_x[i] = fw::math<real_t>::log(ptr_x[i]);
				ptr_y[i] = fw::math<real_t>::log(ptr_y[i]);
				ptr_z[i] = fw::math<real_t>::log(ptr_z[i]);
			}
		}
		else if (layout == fw::data_layout::AoS)
		{
			for (std::size_t i = 0; i < (3 * x.size[0]); i += 3)
			{
				ptr_x[i] = fw::math<real_t>::log(ptr_x[i]);
				ptr_y[i] = fw::math<real_t>::log(ptr_y[i]);
				ptr_z[i] = fw::math<real_t>::log(ptr_z[i]);
			}
		}
		#else
		#if defined(__INTEL_COMPILER)
		#pragma forceinline recursive
		#pragma omp simd
		#endif
		for (std::size_t i = 0; i < x.size[0]; ++i)
		{
			a_x[i] = fw::log(a_x[i]);
		}
		#endif

		return (omp_get_wtime() - time);
	}

	template <>
	template <>
	double kernel<real3_t>::log<3>(buffer_type<real3_t, 3>& x)
	{
		double time = omp_get_wtime();

		auto a_x = x.read_write();	
		#if defined(USE_PLAIN_POINTER)
		for (std::size_t k = 0; k < x.size[2]; ++k)
		{
			for (std::size_t j = 0; j < x.size[1]; ++j)
			{
				real_t* ptr_x = &(a_x[k][j][0].x);
				real_t* ptr_y = &(a_x[k][j][0].y);
				real_t* ptr_z = &(a_x[k][j][0].z);
				if (layout == fw::data_layout::SoA)
				{
					for (std::size_t i = 0; i < x.size[0]; ++i)
					{
						ptr_x[i] = fw::math<real_t>::log(ptr_x[i]);
						ptr_y[i] = fw::math<real_t>::log(ptr_y[i]);
						ptr_z[i] = fw::math<real_t>::log(ptr_z[i]);
					}
				}
				else if (layout == fw::data_layout::AoS)
				{
					for (std::size_t i = 0; i < (3 * x.size[0]); i += 3)
					{
						ptr_x[i] = fw::math<real_t>::log(ptr_x[i]);
						ptr_y[i] = fw::math<real_t>::log(ptr_y[i]);
						ptr_z[i] = fw::math<real_t>::log(ptr_z[i]);
					}
				}
			}
		}
		#else
		for (std::size_t k = 0; k < x.size[2]; ++k)
		{
			for (std::size_t j = 0; j < x.size[1]; ++j)
			{
				#if defined(__INTEL_COMPILER)
				#pragma forceinline recursive
				#pragma omp simd
				#endif
				for (std::size_t i = 0; i < x.size[0]; ++i)
				{
					a_x[k][j][i] = fw::log(a_x[k][j][i]);
				}
			}
		}
		#endif

		return (omp_get_wtime() - time);
	}
#endif