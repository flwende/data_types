#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <omp.h>
#include <buffer/buffer.hpp>

#define NX_DEFAULT 128
#define NY_DEFAULT 128
#define NZ_DEFAULT 128

#define WARMUP 10
#define MEASUREMENT 100

using namespace fw;

using real_t = float;
using real3_t = vec<real_t, 3>;

#if defined(__INTEL_SDLT)
#include <sdlt/sdlt.h>
typedef struct
{
	real_t x;
	real_t y;
	real_t z;
} real3_sdlt;
SDLT_PRIMITIVE(real3_sdlt, x, y, z)

inline real3_sdlt exp(const real3_sdlt& x)
{
	real3_sdlt y;
	y.x = exp(x.x);
	y.y = exp(x.y);
	y.z = exp(x.z);
	return y;
}

inline real3_sdlt log(const real3_sdlt& x)
{
	real3_sdlt y;
	y.x = log(x.x);
	y.y = log(x.y);
	y.z = log(x.z);
	return y;
}
#endif

int main(int argc, char** argv)
{

    std::size_t dim = 0;
    const std::size_t nx = (argc > 1 ? atoi(argv[++dim]) : NX_DEFAULT);
    const std::size_t ny = (argc > 2 ? atoi(argv[++dim]) : NY_DEFAULT);
    const std::size_t nz = (argc > 3 ? atoi(argv[++dim]) : NZ_DEFAULT);
    const std::size_t print_elem = (argc > 4 ? atoi(argv[4]) : std::min(nx, 12UL));

	double time = 0.0;

	if (dim == 1)
	{
		#if defined(__INTEL_SDLT)
		sdlt::soa1d_container<real3_sdlt> _buf(nx);
		auto buf = _buf.access();
		#else
		buffer<real3_t, 1, target::host, data_layout::SoA> _buf(nx);
		//buffer<real3_t, 1, target::host, data_layout::AoS> _buf(nx);
		auto buf = _buf.read_write();
		//std::vector<real3_t> buf(nx);
		#endif

		srand48(nx);
		const real_t value = static_cast<real_t>(drand48() * 100.0);
		std::cout << "initial value = " << value << std::endl;

		for (std::size_t x = 0; x < nx; ++x)
		{
			#if defined(__INTEL_SDLT)
			buf[x] = {value, value, value};
			#else
			buf[x] = value;
			#endif
		}

		for (std::size_t i = 0; i <WARMUP; ++i)
		{
			#if defined(__INTEL_COMPILER)
			#pragma forceinline recursive
			#pragma omp simd
			#endif
			for (std::size_t x = 0; x < nx; ++x)
			{
				buf[x] = exp(buf[x]);
			}
			#if defined(__INTEL_COMPILER)
			#pragma forceinline recursive
			#pragma omp simd
			#endif
			for (std::size_t x = 0; x < nx; ++x)
			{
				buf[x] = log(buf[x]);
			}
		}

		time = omp_get_wtime();
		for (std::size_t i = 0; i <MEASUREMENT; ++i)
		{
			#if defined(__INTEL_COMPILER)
			#pragma forceinline recursive
			#pragma omp simd
			#endif
			for (std::size_t x = 0; x < nx; ++x)
			{
				buf[x] = exp(buf[x]);
			}
			#if defined(__INTEL_COMPILER)
			#pragma forceinline recursive
			#pragma omp simd
			#endif
			for (std::size_t x = 0; x < nx; ++x)
			{
				buf[x] = log(buf[x]);
			}
		}
		time = omp_get_wtime() - time;

		#if defined(CHECK_RESULTS)
		const double max_abs_error = static_cast<real_t>(1.0E-4);
		bool not_passed = false;
		for (std::size_t x = 0; x < nx; ++x)
		{
			#if defined(__INTEL_SDLT)
			const real3_sdlt _x_0 = buf[x];
			const double x_0[3] = {_x_0.x, _x_0.y, _x_0.z};
			#else
			const double x_0[3] = {buf[x].x, buf[x].y, buf[x].z};
			#endif
			const double x_1[3] = {value, value, value};

			for (std::size_t i = 0; i < 3; ++i)
			{
				const real_t abs_error = std::abs(x_1[i] != static_cast<real_t>(0) ? (x_0[i] - x_1[i]) / x_1[i] : x_0[i] - x_1[i]);
				if (abs_error > max_abs_error)
				{
					std::cout << "error: " << x_0[i] << " vs " << x_1[i] << " (" << abs_error << ")" << std::endl;
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
	else if (dim == 2)
	{
		std::cerr << "not implemented for dim = 2" << std::endl;
	}

	else if (dim == 3)
	{
		buffer<real3_t, 3, target::host, data_layout::SoA> _buf({nx, ny, nz});
		//buffer<real3_t, 3, target::host, data_layout::AoS> _buf({nx, ny, nz});
		auto buf = _buf.read_write();
		//std::vector<std::vector<std::vector<real3_t>>> buf(nz, std::vector<std::vector<real3_t>>(ny, std::vector<real3_t>(nx)));

		srand48(nx);
		const real_t value = static_cast<real_t>(drand48() * 100.0);
		std::cout << "initial value = " << value << std::endl;

		for (std::size_t z = 0; z < nz; ++z)
		{
			for (std::size_t y = 0; y < ny; ++y)
			{
				for (std::size_t x = 0; x < nx; ++x)
				{
					buf[z][y][x] = value;
				}
			}
		}

		for (std::size_t i = 0; i <WARMUP; ++i)
		{
			for (std::size_t z = 0; z < nz; ++z)
			{
				for (std::size_t y = 0; y < ny; ++y)
				{
					#if defined(__INTEL_COMPILER)
					#pragma forceinline recursive
					#pragma omp simd
					#endif
					for (std::size_t x = 0; x < nx; ++x)
					{
						#if defined(COMPONENT_WISE)
						buf[z][y][x].x = MISC_NAMESPACE::math<real_t>::exp(buf[z][y][x].x);
						#else
						buf[z][y][x] = exp(buf[z][y][x]);
						#endif
					}
				}
			}

			for (std::size_t z = 0; z < nz; ++z)
			{
				for (std::size_t y = 0; y < ny; ++y)
				{
					#if defined(__INTEL_COMPILER)
					#pragma forceinline recursive
					#pragma omp simd
					#endif
					for (std::size_t x = 0; x < nx; ++x)
					{
						#if defined(COMPONENT_WISE)
						buf[z][y][x].x = MISC_NAMESPACE::math<real_t>::log(buf[z][y][x].x);
						#else
						buf[z][y][x] = log(buf[z][y][x]);
						#endif
					}
				}
			}
		}

		time = omp_get_wtime();
		for (std::size_t i = 0; i <MEASUREMENT; ++i)
		{
			for (std::size_t z = 0; z < nz; ++z)
			{
				for (std::size_t y = 0; y < ny; ++y)
				{
					#if defined(__INTEL_COMPILER)
					#pragma forceinline recursive
					#pragma omp simd
					#endif
					for (std::size_t x = 0; x < nx; ++x)
					{
						#if defined(COMPONENT_WISE)
						buf[z][y][x].x = MISC_NAMESPACE::math<real_t>::exp(buf[z][y][x].x);
						#else
						buf[z][y][x] = exp(buf[z][y][x]);
						#endif
					}
				}
			}

			for (std::size_t z = 0; z < nz; ++z)
			{
				for (std::size_t y = 0; y < ny; ++y)
				{
					#if defined(__INTEL_COMPILER)
					#pragma forceinline recursive
					#pragma omp simd
					#endif
					for (std::size_t x = 0; x < nx; ++x)
					{
						#if defined(COMPONENT_WISE)
						buf[z][y][x].x = MISC_NAMESPACE::math<real_t>::log(buf[z][y][x].x);
						#else
						buf[z][y][x] = log(buf[z][y][x]);
						#endif
					}
				}
			}
		}
		time = omp_get_wtime() - time;

		#if defined(CHECK_RESULTS)
		const double max_abs_error = static_cast<real_t>(1.0E-4);
		bool not_passed = false;
		for (std::size_t z = 0; z < nz; ++z)
		{
			for (std::size_t y = 0; y < ny; ++y)
			{
				for (std::size_t x = 0; x < nx; ++x)
				{
					const double x_0[3] = {buf[z][y][x].x, buf[z][y][x].y, buf[z][y][x].z};
					const double x_1[3] = {value, value, value};

					for (std::size_t i = 0; i < 3; ++i)
					{
						const real_t abs_error = std::abs(x_1[i] != static_cast<real_t>(0) ? (x_0[i] - x_1[i]) / x_1[i] : x_0[i] - x_1[i]);
						if (abs_error > max_abs_error)
						{
							std::cout << "error: " << x_0[i] << " vs " << x_1[i] << " (" << abs_error << ")" << std::endl;
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

    std::cout << "elapsed time = " << (time / MEASUREMENT) * 1.0E3 << " ms" << std::endl;

    return 0;
}
