// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#include <iostream>
#include <cstdlib>
#include <cstdint>

#include "kernel.hpp"

using namespace fw;

//#define HAVE_SYCL

#if defined(HAVE_SYCL)
	template <std::size_t D>
	class my_kernel;

	template <>
	template <>
	double kernel<real3_t>::exp<2>(fw::buffer<real3_t, 2, fw::buffer_type::host_device, fw::data_layout::AoS>& x)
	{
		// queue
        	static cl::sycl::gpu_selector gpu;
        	static cl::sycl::queue gpu_q(gpu);

		// kernel
		static const cl::sycl::range<2> block(32, 4);
		const std::size_t nx = x.size[0];
		const std::size_t ny = x.size[1];
		const cl::sycl::range<2> grid(((nx + block[0] - 1) / block[0]) * block[0], ((ny + block[1] - 1) / block[1]) * block[1]);

		// info
		static bool first_call = true;
        	if (first_call)
		{
			try
			{
				std::cout << "device: " << gpu_q.get_device().get_info<cl::sycl::info::device::vendor>()
							<< gpu_q.get_device().get_info<cl::sycl::info::device::name>()
							<< std::endl;
			}
			catch (cl::sycl::exception)
			{
				std::cerr << "error: kernel -> no gpu device available -> STOP COMPUTATION" << std::endl;
				return 0.0;
			}

			std::cout << "block: " << block[0] << ", " << block[1] << std::endl;
			std::cout << "grid: " << grid[0] << ", " << grid[1] << std::endl;
		}

		// offload
		{
			gpu_q.submit([&] (cl::sycl::handler& h)
			{
				// accessors
				auto a_x = x.read_write<fw::target::device>(h);

				// execute
				h.parallel_for<class my_kernel<1>>(cl::sycl::nd_range<2>(grid, block), [=](cl::sycl::nd_item<2> item)
				{
					const std::size_t j = item.get_global(0);
					const std::size_t i = item.get_global(1);
					if (j < nx && i < ny)
					{
						a_x[i][j] += i * nx + j;
					}
				});
			});
		}

		return 0.0;
	}

	template <>
	template <>
	double kernel<real3_t>::exp<2>(fw::buffer<real3_t, 2, fw::buffer_type::host_device, fw::data_layout::SoA>& x)
	{
		// queue
        	static cl::sycl::gpu_selector gpu;
        	static cl::sycl::queue gpu_q(gpu);

		// kernel
		static const cl::sycl::range<2> block(32, 4);
		const std::size_t nx = x.size[0];
		const std::size_t ny = x.size[1];
		const cl::sycl::range<2> grid(((nx + block[0] - 1) / block[0]) * block[0], ((ny + block[1] - 1) / block[1]) * block[1]);

		// info
		static bool first_call = true;
        	if (first_call)
		{
			try
			{
				std::cout << "device: " << gpu_q.get_device().get_info<cl::sycl::info::device::vendor>()
							<< gpu_q.get_device().get_info<cl::sycl::info::device::name>()
							<< std::endl;
			}
			catch (cl::sycl::exception)
			{
				std::cerr << "error: kernel -> no gpu device available -> STOP COMPUTATION" << std::endl;
				return 0.0;
			}

			std::cout << "block: " << block[0] << ", " << block[1] << std::endl;
			std::cout << "grid: " << grid[0] << ", " << grid[1] << std::endl;
		}

		// offload
		{
			gpu_q.submit([&] (cl::sycl::handler& h)
			{
				// accessors
				auto a_x = x.read_write<fw::target::device>(h);

				// execute
				h.parallel_for<class my_kernel<2>>(cl::sycl::nd_range<2>(grid, block), [=](cl::sycl::nd_item<2> item)
				{
					const std::size_t j = item.get_global(0);
					const std::size_t i = item.get_global(1);
					if (j < nx && i < ny)
					{
						a_x[i][j] += i * nx + j;
					}
				});
			});
		}

		return 0.0;
	}
#elif defined(__INTEL_SDLT)
#include <omp.h>

	template <>
	template <>
	double kernel<real3_t>::exp<1>(sdlt::soa1d_container<real3_t>& x)
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
	double kernel<real3_t>::log<1>(sdlt::soa1d_container<real3_t>& x)
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
#include <omp.h>

	template <>
	template <>
	double kernel<real3_t>::exp<1>(fw::buffer<real3_t, 1, fw::buffer_type::host, fw::data_layout::AoS>& x)
	{
		double time = omp_get_wtime();
		
		auto a_x = x.read_write();
		#if defined(USE_PLAIN_POINTER)	
		real_t* ptr_x = &(a_x[0].x);
		real_t* ptr_y = &(a_x[0].y);
		real_t* ptr_z = &(a_x[0].z);
		for (std::size_t i = 0; i < (3 * x.size[0]); i += 3)
		{
			ptr_x[i] = fw::math<real_t>::exp(ptr_x[i]);
			ptr_y[i] = fw::math<real_t>::exp(ptr_y[i]);
			ptr_z[i] = fw::math<real_t>::exp(ptr_z[i]);
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
	double kernel<real3_t>::exp<1>(fw::buffer<real3_t, 1, fw::buffer_type::host, fw::data_layout::SoA>& x)
	{
		double time = omp_get_wtime();
		
		auto a_x = x.read_write();
		#if defined(USE_PLAIN_POINTER)	
		real_t* ptr_x = &(a_x[0].x);
		real_t* ptr_y = &(a_x[0].y);
		real_t* ptr_z = &(a_x[0].z);
		for (std::size_t i = 0; i < x.size[0]; ++i)
		{
			ptr_x[i] = fw::math<real_t>::exp(ptr_x[i]);
			ptr_y[i] = fw::math<real_t>::exp(ptr_y[i]);
			ptr_z[i] = fw::math<real_t>::exp(ptr_z[i]);
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
	double kernel<real3_t>::exp<3>(fw::buffer<real3_t, 3, fw::buffer_type::host, fw::data_layout::AoS>& x)
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
				for (std::size_t i = 0; i < (3 * x.size[0]); i += 3)
				{
					ptr_x[i] = fw::math<real_t>::exp(ptr_x[i]);
					ptr_y[i] = fw::math<real_t>::exp(ptr_y[i]);
					ptr_z[i] = fw::math<real_t>::exp(ptr_z[i]);
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
	double kernel<real3_t>::exp<3>(fw::buffer<real3_t, 3, fw::buffer_type::host, fw::data_layout::SoA>& x)
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
				for (std::size_t i = 0; i < x.size[0]; ++i)
				{
					ptr_x[i] = fw::math<real_t>::exp(ptr_x[i]);
					ptr_y[i] = fw::math<real_t>::exp(ptr_y[i]);
					ptr_z[i] = fw::math<real_t>::exp(ptr_z[i]);
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
	double kernel<real3_t>::log<1>(fw::buffer<real3_t, 1, fw::buffer_type::host, fw::data_layout::AoS>& x)
	{
		double time = omp_get_wtime();
		
		auto a_x = x.read_write();
		#if defined(USE_PLAIN_POINTER)	
		real_t* ptr_x = &(a_x[0].x);
		real_t* ptr_y = &(a_x[0].y);
		real_t* ptr_z = &(a_x[0].z);
		for (std::size_t i = 0; i < (3 * x.size[0]); i += 3)
		{
			ptr_x[i] = fw::math<real_t>::log(ptr_x[i]);
			ptr_y[i] = fw::math<real_t>::log(ptr_y[i]);
			ptr_z[i] = fw::math<real_t>::log(ptr_z[i]);
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
	double kernel<real3_t>::log<1>(fw::buffer<real3_t, 1, fw::buffer_type::host, fw::data_layout::SoA>& x)
	{
		double time = omp_get_wtime();
		
		auto a_x = x.read_write();
		#if defined(USE_PLAIN_POINTER)	
		real_t* ptr_x = &(a_x[0].x);
		real_t* ptr_y = &(a_x[0].y);
		real_t* ptr_z = &(a_x[0].z);
		for (std::size_t i = 0; i < x.size[0]; ++i)
		{
			ptr_x[i] = fw::math<real_t>::log(ptr_x[i]);
			ptr_y[i] = fw::math<real_t>::log(ptr_y[i]);
			ptr_z[i] = fw::math<real_t>::log(ptr_z[i]);
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
	double kernel<real3_t>::log<3>(fw::buffer<real3_t, 3, fw::buffer_type::host, fw::data_layout::AoS>& x)
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
				for (std::size_t i = 0; i < (3 * x.size[0]); i += 3)
				{
					ptr_x[i] = fw::math<real_t>::log(ptr_x[i]);
					ptr_y[i] = fw::math<real_t>::log(ptr_y[i]);
					ptr_z[i] = fw::math<real_t>::log(ptr_z[i]);
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

	template <>
	template <>
	double kernel<real3_t>::log<3>(fw::buffer<real3_t, 3, fw::buffer_type::host, fw::data_layout::SoA>& x)
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
				for (std::size_t i = 0; i < x.size[0]; ++i)
				{
					ptr_x[i] = fw::math<real_t>::log(ptr_x[i]);
					ptr_y[i] = fw::math<real_t>::log(ptr_y[i]);
					ptr_z[i] = fw::math<real_t>::log(ptr_z[i]);
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