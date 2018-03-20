// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <omp.h>

#include <kernel.hpp>

constexpr std::size_t NX_DEFAULT = 128;
constexpr std::size_t NY_DEFAULT = 128;
constexpr std::size_t NZ_DEFAULT = 128;

constexpr std::size_t WARMUP = 10;
constexpr std::size_t MEASUREMENT = 100;

constexpr double SPREAD = 0.2;
constexpr double OFFSET = 0.9;

int main(int argc, char** argv)
{
	// command line arguments
	std::size_t dim = 0;
	const std::size_t nx = (argc > 1 ? atoi(argv[++dim]) : NX_DEFAULT);
	const std::size_t ny = (argc > 2 ? atoi(argv[++dim]) : NY_DEFAULT);
	const std::size_t nz = (argc > 3 ? atoi(argv[++dim]) : NZ_DEFAULT);
	const std::size_t print_elem = (argc > 4 ? atoi(argv[4]) : std::min(nx, 12UL));

	// initialization
	srand48(nx);
	const real_t value = static_cast<real_t>(drand48() * 100.0);
	std::cout << "initial value = " << value << std::endl;

	// benchmark
	double time = 0.0;

	if (dim == 1)
	{
		buffer_type<real3_t, 1> buf(nx);
		buffer_type<real3_t, 1> buf_original(nx);
		#if defined(__INTEL_SDLT)
		auto a_buf = buf.access();
		auto a_buf_original = buf_original.access();
		#else
		auto a_buf = buf.read_write();
		auto a_buf_original = buf_original.read_write();
		#endif

		for (std::size_t x = 0; x < nx; ++x)
		{
			real_t s_1 = static_cast<real_t>(drand48() * SPREAD + OFFSET);
			real_t s_2 = static_cast<real_t>(drand48() * SPREAD + OFFSET);
			real_t s_3 = static_cast<real_t>(drand48() * SPREAD + OFFSET);
			a_buf[x] = {s_1 * value, s_2 * value, s_3 * value};
			a_buf_original[x] = a_buf[x];
		}

		for (std::size_t n = 0; n < WARMUP; ++n)
		{
			#if defined(__INTEL_SDLT)
			kernel<real3_t>::exp<1>(buf);
			kernel<real3_t>::log<1>(buf);
			#else
			kernel<real3_t>::exp<1>(buf);
			kernel<real3_t>::log<1>(buf);
			#endif
		}

		for (std::size_t n = 0; n < MEASUREMENT; ++n)
		{
			#if defined(__INTEL_SDLT)
			time += kernel<real3_t>::exp<1>(buf);
			time += kernel<real3_t>::log<1>(buf);
			#else
			time += kernel<real3_t>::exp<1>(buf);
			time += kernel<real3_t>::log<1>(buf);
			#endif
		}

		#if defined(CHECK_RESULTS)
		const double max_abs_error = static_cast<real_t>(1.0E-4);
		bool not_passed = false;
		for (std::size_t i = 0; i < nx; ++i)
		{
			#if defined(__INTEL_SDLT)
			real3_t _x_0 = a_buf[i];
			real3_t _x_1 = a_buf_original[i];
			const double x_0[3] = {_x_0.x, _x_0.y, _x_0.z};
			const double x_1[3] = {_x_1.x, _x_1.y, _x_1.z};
			#else
			const double x_0[3] = {a_buf[i].x, a_buf[i].y, a_buf[i].z};
			const double x_1[3] = {a_buf_original[i].x, a_buf_original[i].y, a_buf_original[i].z};
			#endif

			for (std::size_t ii = 0; ii < 3; ++ii)
			{
				const real_t abs_error = std::abs(x_1[ii] != static_cast<real_t>(0) ? (x_0[ii] - x_1[ii]) / x_1[ii] : x_0[ii] - x_1[ii]);
				if (abs_error > max_abs_error)
				{
					std::cout << "error: " << x_0[ii] << " vs " << x_1[ii] << " (" << abs_error << ")" << std::endl;
					not_passed = true;
					break;
				}
			}
			if (not_passed) break;
		}

		if (!not_passed)
		{
			std::cout << "success!" << std::endl;
		}
		#endif
	}
	#if !defined(__INTEL_SDLT)
	else if (dim == 3)
	{
		buffer_type<real3_t, 3> buf({nx, ny, nz});
		buffer_type<real3_t, 3> buf_original({nx, ny, nz});
		auto a_buf = buf.read_write();
		auto a_buf_original = buf_original.read_write();
		
		for (std::size_t z = 0; z < nz; ++z)
		{
			for (std::size_t y = 0; y < ny; ++y)
			{
				for (std::size_t x = 0; x < nx; ++x)
				{
					real_t s_1 = static_cast<real_t>(drand48() * SPREAD + OFFSET);
					real_t s_2 = static_cast<real_t>(drand48() * SPREAD + OFFSET);
					real_t s_3 = static_cast<real_t>(drand48() * SPREAD + OFFSET);
					a_buf[z][y][x] = {s_1 * value, s_2 * value, s_3 * value};
					a_buf_original[z][y][x] = a_buf[z][y][x];
				}
			}
		}

		for (std::size_t n = 0; n < WARMUP; ++n)
		{
			kernel<real3_t>::exp<3>(buf);
			kernel<real3_t>::log<3>(buf);
		}

		for (std::size_t n = 0; n < MEASUREMENT; ++n)
		{
			time += kernel<real3_t>::exp<3>(buf);
			time += kernel<real3_t>::log<3>(buf);
		}

		#if defined(CHECK_RESULTS)
		const double max_abs_error = static_cast<real_t>(1.0E-4);
		bool not_passed = false;
		for (std::size_t k = 0; k < nz; ++k)
		{
			for (std::size_t j = 0; j < ny; ++j)
			{
				for (std::size_t i = 0; i < nx; ++i)
				{
					const double x_0[3] = {a_buf[k][j][i].x, a_buf[k][j][i].y, a_buf[k][j][i].z};
					const double x_1[3] = {a_buf_original[k][j][i].x, a_buf_original[k][j][i].y, a_buf_original[k][j][i].z};

					for (std::size_t ii = 0; ii < 3; ++ii)
					{
						const real_t abs_error = std::abs(x_1[ii] != static_cast<real_t>(0) ? (x_0[ii] - x_1[ii]) / x_1[ii] : x_0[ii] - x_1[ii]);
						if (abs_error > max_abs_error)
						{
							std::cout << "error: " << x_0[ii] << " vs " << x_1[ii] << " (" << abs_error << ")" << std::endl;
							not_passed = true;
							break;
						}
					}
					if (not_passed) break;
				}
				if (not_passed) break;
			}
			if (not_passed) break;
		}

		if (!not_passed)
		{
			std::cout << "success!" << std::endl;
		}
		#endif
	}
	#endif

	std::cout << "elapsed time = " << (time / MEASUREMENT) * 1.0E3 << " ms" << std::endl;

	return 0;
}
