// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#include <cstdint>
#include <iostream>
#include <kernel.hpp>

constexpr std::size_t NX_DEFAULT = 128;
constexpr std::size_t NY_DEFAULT = 128;
constexpr std::size_t NZ_DEFAULT = 128;

#if defined(CHECK_RESULTS)
constexpr std::size_t WARMUP = 0;
constexpr std::size_t MEASUREMENT = 1;
#else
constexpr std::size_t WARMUP = 10;
constexpr std::size_t MEASUREMENT = 100;
#endif

constexpr double SPREAD = 0.5;
constexpr double OFFSET = 1.0;

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
    const type value = static_cast<type>(drand48() * 10.0);
    std::cout << "initial value = " << value << std::endl;

    // benchmark
    double time = 0.0;

    if (dim == 1)
    {
        buffer_type<element_type, 1> buf(nx);
        buffer_type<element_type, 1> buf_original(nx);
        
        for (std::size_t x = 0; x < nx; ++x)
        {
            const type s_1 = static_cast<type>((2.0 * drand48() -1.0) * SPREAD + OFFSET);
            const type s_2 = static_cast<type>((2.0 * drand48() -1.0) * SPREAD + OFFSET);
            const type s_3 = static_cast<type>((2.0 * drand48() -1.0) * SPREAD + OFFSET);
            buf[x] = {static_cast<type_x>(s_1 * value), static_cast<type_y>(s_2 * value), static_cast<type_z>(s_3 * value)};
            buf_original[x] = buf[x];
        }

        for (std::size_t n = 0; n < WARMUP; ++n)
        {
            kernel<element_type>::exp<1>(buf);
            kernel<element_type>::log<1>(buf);
        }

        for (std::size_t n = 0; n < MEASUREMENT; ++n)
        {
            time += kernel<element_type>::exp<1>(buf);
            time += kernel<element_type>::log<1>(buf);
        }

        #if defined(CHECK_RESULTS)
        const double max_abs_error = static_cast<type>(1.0E-6);
        const std::size_t print_num_elements = 128;
        bool not_passed = false;

        for (std::size_t i = 0, e = 0; i < nx; ++i, ++e)
        {
            const double x_0[3] = {static_cast<double>(buf[i].x), static_cast<double>(buf[i].y), static_cast<double>(buf[i].z)};
            #if defined(ELEMENT_ACCESS)
            const double x_1[3] = {
                static_cast<double>(buf_original[i].x), 
                static_cast<double>(static_cast<type_y>(std::log(static_cast<type_y>(std::exp(buf_original[i].y))))),
                static_cast<double>(buf_original[i].z)};
            #else
            const double x_1[3] = {
                static_cast<double>(static_cast<type_x>(std::log(static_cast<type_x>(std::exp(buf_original[i].x))))), 
                static_cast<double>(static_cast<type_y>(std::log(static_cast<type_y>(std::exp(buf_original[i].y))))),
                static_cast<double>(static_cast<type_z>(std::log(static_cast<type_z>(std::exp(buf_original[i].z)))))};
            #endif

            for (std::size_t ii = 0; ii < 3; ++ii)
            {
                const type abs_error = std::abs(x_1[ii] != static_cast<type>(0) ? (x_0[ii] - x_1[ii]) / x_1[ii] : x_0[ii] - x_1[ii]);
                if (abs_error > max_abs_error)
                {
                    std::cout << "error: " << x_0[ii] << " vs " << x_1[ii] << " (" << abs_error << ")" << std::endl;
                    not_passed = true;
                    break;
                }

                if (e < print_num_elements)
                {
                    std::cout << buf[i] << std::endl;
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
    else if (dim == 3)
    {
        buffer_type<element_type, 3> buf({nx, ny, nz});
        buffer_type<element_type, 3> buf_original({nx, ny, nz});
        
        for (std::size_t z = 0; z < nz; ++z)
        {
            for (std::size_t y = 0; y < ny; ++y)
            {
                for (std::size_t x = 0; x < nx; ++x)
                {
                    const type s_1 = static_cast<type>((2.0 * drand48() -1.0) * SPREAD + OFFSET);
                    const type s_2 = static_cast<type>((2.0 * drand48() -1.0) * SPREAD + OFFSET);
                    const type s_3 = static_cast<type>((2.0 * drand48() -1.0) * SPREAD + OFFSET);
                    buf[z][y][x] = {static_cast<type_x>(s_1 * value), static_cast<type_y>(s_2 * value), static_cast<type_z>(s_3 * value)};
                    buf_original[z][y][x] = buf[z][y][x];
                }
            }
        }

        for (std::size_t n = 0; n < WARMUP; ++n)
        {
            kernel<element_type>::exp<3>(buf);
            kernel<element_type>::log<3>(buf);
        }

        for (std::size_t n = 0; n < MEASUREMENT; ++n)
        {
            time += kernel<element_type>::exp<3>(buf);
            time += kernel<element_type>::log<3>(buf);
        }

        #if defined(CHECK_RESULTS)
        const double max_abs_error = static_cast<type>(1.0E-6);
        const std::size_t print_num_elements = 128;
        bool not_passed = false;

        for (std::size_t k = 0, e = 0; k < nz; ++k)
        {
            for (std::size_t j = 0; j < ny; ++j)
            {
                for (std::size_t i = 0; i < nx; ++i, ++e)
                {
                    const double x_0[3] = {static_cast<double>(buf[k][j][i].x), static_cast<double>(buf[k][j][i].y), static_cast<double>(buf[k][j][i].z)};
                    #if defined(ELEMENT_ACCESS)
                    const double x_1[3] = {
                        static_cast<double>(buf_original[k][j][i].x), 
                        static_cast<double>(static_cast<type_y>(std::log(static_cast<type_y>(std::exp(buf_original[k][j][i].y))))),
                        static_cast<double>(buf_original[k][j][i].z)};
                    #else
                    const double x_1[3] = {
                        static_cast<double>(static_cast<type_x>(std::log(static_cast<type_x>(std::exp(buf_original[k][j][i].x))))), 
                        static_cast<double>(static_cast<type_y>(std::log(static_cast<type_y>(std::exp(buf_original[k][j][i].y))))),
                        static_cast<double>(static_cast<type_z>(std::log(static_cast<type_z>(std::exp(buf_original[k][j][i].z)))))};
                    #endif

                    for (std::size_t ii = 0; ii < 3; ++ii)
                    {
                        const type abs_error = std::abs(x_1[ii] != static_cast<type>(0) ? (x_0[ii] - x_1[ii]) / x_1[ii] : x_0[ii] - x_1[ii]);
                        if (abs_error > max_abs_error)
                        {
                            std::cout << "error: " << x_0[ii] << " vs " << x_1[ii] << " (" << abs_error << ")" << std::endl;
                            not_passed = true;
                            break;
                        }
                    }
                    if (not_passed) break;

                    if (e < print_num_elements)
                    {
                        std::cout << buf[k][j][i] << std::endl;
                    }
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