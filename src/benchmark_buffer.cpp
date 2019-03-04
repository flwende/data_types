// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#include <cstdint>
#include <iostream>
#include <vector>
#include <kernel.hpp>

constexpr std::size_t NX_DEFAULT = 128;
constexpr std::size_t NY_DEFAULT = 128;
constexpr std::size_t NZ_DEFAULT = 128;

#if defined(CHECK_RESULTS)
constexpr std::size_t WARMUP = 0;
constexpr std::size_t MEASUREMENT = 1;
#else
constexpr std::size_t WARMUP = 10;
constexpr std::size_t MEASUREMENT = 20;
#endif

constexpr double SPREAD = 1.0;
constexpr double OFFSET = 3.0;

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
    const type value = 1.0 + static_cast<type>(drand48());
    std::cout << "initial value = " << value << std::endl;

    // benchmark
    double time = 0.0;

    if (dim == 1)
    {
        std::vector<element_type> in_orig_1(nx);
        std::vector<element_type> in_orig_2(nx);
        buffer_type<element_type, 1> in_1, out_1, out_2;
        #if defined(VECTOR_PRODUCT)
        buffer_type<element_type, 1> in_2{{nx}};
        #endif
        
        in_1.resize({nx});
        out_1.resize({nx});
        out_2.resize({nx});
        
        for (std::size_t i = 0; i < nx; ++i)
        {
            const type s_1 = static_cast<type>((2.0 * drand48() -1.0) * SPREAD + OFFSET);
            const type s_2 = static_cast<type>((2.0 * drand48() -1.0) * SPREAD + OFFSET);
            const type s_3 = static_cast<type>((2.0 * drand48() -1.0) * SPREAD + OFFSET);
            in_orig_1[i] = {static_cast<type_x>(s_1 * value), static_cast<type_y>(s_2 * value), static_cast<type_z>(s_3 * value)};
            in_1[i] = in_orig_1[i];
            #if defined(INPLACE)
            out_2[i] = in_1[i];
            #endif

            #if defined(VECTOR_PRODUCT)
            {
                const type s_1 = static_cast<type>((2.0 * drand48() -1.0) * SPREAD + OFFSET);
                const type s_2 = static_cast<type>((2.0 * drand48() -1.0) * SPREAD + OFFSET);
                const type s_3 = static_cast<type>((2.0 * drand48() -1.0) * SPREAD + OFFSET);
                in_orig_2[i] = {static_cast<type_x>(s_1 * value), static_cast<type_y>(s_2 * value), static_cast<type_z>(s_3 * value)};
                in_2[i] = in_orig_2[i];
            }
            #endif
        }

        for (std::size_t n = 0; n < WARMUP; ++n)
        {
            #if defined(VECTOR_PRODUCT)
                kernel<element_type>::cross<1>(in_1, in_2, out_2);
            #else
                #if defined(INPLACE)
                kernel<element_type>::exp<1>(out_2);
                kernel<element_type>::log<1>(out_2);
                #else
                kernel<element_type>::exp<1>(in_1, out_1);
                kernel<element_type>::log<1>(out_1, out_2);
                #endif
            #endif
        }

        for (std::size_t n = 0; n < MEASUREMENT; ++n)
        {
            #if defined(VECTOR_PRODUCT)
                time += kernel<element_type>::cross<1>(in_1, in_2, out_2);
            #else
                #if defined(INPLACE)
                time += kernel<element_type>::exp<1>(out_2);
                time += kernel<element_type>::log<1>(out_2);
                #else
                time += kernel<element_type>::exp<1>(in_1, out_1);
                time += kernel<element_type>::log<1>(out_1, out_2);
                #endif
            #endif
        }

        #if defined(CHECK_RESULTS)
        const double max_abs_error = static_cast<type>(1.0E-6);
        const std::size_t print_num_elements = 128;
        bool not_passed = false;

        for (std::size_t i = 0, e = 0; i < nx; ++i, ++e)
        {
            const double tmp_1[3] = {static_cast<double>(out_2[i].x), static_cast<double>(out_2[i].y), static_cast<double>(out_2[i].z)};
            const element_type tmp_x = in_orig_1[i];
            #if defined(VECTOR_PRODUCT)
                const element_type tmp_y = in_orig_2[i];
                const element_type tmp_y_1 = 0.5 + fw::math<element_type>::log(1.0 + fw::cross(tmp_x, fw::math<element_type>::exp(tmp_y)));
                const double tmp_2[3] = {
                    static_cast<double>(tmp_y_1.x),
                    static_cast<double>(tmp_y_1.y),
                    static_cast<double>(tmp_y_1.z)};
            #else
                #if defined(ELEMENT_ACCESS)
                const double tmp_2[3] = {
                    static_cast<double>(tmp_1[0]), 
                    static_cast<double>(static_cast<type_y>(std::log(static_cast<type_y>(std::exp(tmp_x.y))))),
                    static_cast<double>(tmp_1[2])};
                #else
                const double tmp_2[3] = {
                    static_cast<double>(static_cast<type_x>(std::log(static_cast<type_x>(std::exp(tmp_x.x))))), 
                    static_cast<double>(static_cast<type_y>(std::log(static_cast<type_y>(std::exp(tmp_x.y))))),
                    static_cast<double>(static_cast<type_z>(std::log(static_cast<type_z>(std::exp(tmp_x.z)))))};
                #endif
            #endif

            for (std::size_t ii = 0; ii < 3; ++ii)
            {
                const type abs_error = std::abs(tmp_2[ii] != static_cast<type>(0) ? (tmp_1[ii] - tmp_2[ii]) / tmp_2[ii] : tmp_1[ii] - tmp_2[ii]);
                if (abs_error > max_abs_error)
                {
                    std::cout << "error: " << tmp_1[ii] << " vs " << tmp_2[ii] << " (" << abs_error << ")" << std::endl;
                    not_passed = true;
                    break;
                }
            }
            if (not_passed) break;

            if (e < print_num_elements)
            {
                std::cout << out_2[i] << std::endl;
            }
        }

        if (!not_passed)
        {
            std::cout << "success!" << std::endl;
        }
        #endif
    }
    else if (dim == 2)
    {
        std::vector<element_type> in_orig_1(nx * ny);
        std::vector<element_type> in_orig_2(nx * ny);
        buffer_type<element_type, 2> in_1, out_1, out_2;
        #if defined(VECTOR_PRODUCT)
        buffer_type<element_type, 2> in_2{{nx, ny}};
        #endif
    
        in_1.resize({nx, ny});
        out_1.resize({nx, ny});
        out_2.resize({nx, ny});
        
        for (std::size_t j = 0; j < ny; ++j)
        {
            for (std::size_t i = 0; i < nx; ++i)
            {
                const type s_1 = static_cast<type>((2.0 * drand48() -1.0) * SPREAD + OFFSET);
                const type s_2 = static_cast<type>((2.0 * drand48() -1.0) * SPREAD + OFFSET);
                const type s_3 = static_cast<type>((2.0 * drand48() -1.0) * SPREAD + OFFSET);
                in_orig_1[j * nx + i] = {static_cast<type_x>(s_1 * value), static_cast<type_y>(s_2 * value), static_cast<type_z>(s_3 * value)};
                in_1[j][i] = in_orig_1[j * nx + i];
                #if defined(INPLACE)
                out_2[j][i] = in_1[j][i];
                #endif

                #if defined(VECTOR_PRODUCT)
                {
                    const type s_1 = static_cast<type>((2.0 * drand48() -1.0) * SPREAD + OFFSET);
                    const type s_2 = static_cast<type>((2.0 * drand48() -1.0) * SPREAD + OFFSET);
                    const type s_3 = static_cast<type>((2.0 * drand48() -1.0) * SPREAD + OFFSET);
                    in_orig_2[j * nx + i] = {static_cast<type_x>(s_1 * value), static_cast<type_y>(s_2 * value), static_cast<type_z>(s_3 * value)};
                    in_2[j][i] = in_orig_2[j * nx + i];
                }
                #endif
            }
        }
        
        for (std::size_t n = 0; n < WARMUP; ++n)
        {
            #if defined(VECTOR_PRODUCT)
                kernel<element_type>::cross<2>(in_1, in_2, out_2);
            #else
                #if defined(INPLACE)
                kernel<element_type>::exp<2>(out_2);
                kernel<element_type>::log<2>(out_2);
                #else
                kernel<element_type>::exp<2>(in_1, out_1);
                kernel<element_type>::log<2>(out_1, out_2);
                #endif
            #endif
        }

        for (std::size_t n = 0; n < MEASUREMENT; ++n)
        {
            #if defined(VECTOR_PRODUCT)
                time += kernel<element_type>::cross<2>(in_1, in_2, out_2);
            #else
                #if defined(INPLACE)
                time += kernel<element_type>::exp<2>(out_2);
                time += kernel<element_type>::log<2>(out_2);
                #else
                time += kernel<element_type>::exp<2>(in_1, out_1);
                time += kernel<element_type>::log<2>(out_1, out_2);
                #endif
            #endif
        }

        #if defined(CHECK_RESULTS)
        const double max_abs_error = static_cast<type>(1.0E-6);
        const std::size_t print_num_elements = 128;
        bool not_passed = false;

        for (std::size_t j = 0, e = 0; j < ny; ++j)
        {
            for (std::size_t i = 0; i < nx; ++i, ++e)
            {
                const double tmp_1[3] = {static_cast<double>(out_2[j][i].x), static_cast<double>(out_2[j][i].y), static_cast<double>(out_2[j][i].z)};
                const element_type tmp_x = in_orig_1[j * nx + i];
                #if defined(VECTOR_PRODUCT)
                    const element_type tmp_y = in_orig_2[j * nx + i];
                    const element_type tmp_y_1 = 0.5 + fw::math<element_type>::log(1.0 + fw::cross(tmp_x, fw::math<element_type>::exp(tmp_y)));
                    const double tmp_2[3] = {
                        static_cast<double>(tmp_y_1.x),
                        static_cast<double>(tmp_y_1.y),
                        static_cast<double>(tmp_y_1.z)};
                #else
                    #if defined(ELEMENT_ACCESS)
                    const double tmp_2[3] = {
                        static_cast<double>(tmp_1[0]), 
                        static_cast<double>(static_cast<type_y>(std::log(static_cast<type_y>(std::exp(tmp_x.y))))),
                        static_cast<double>(tmp_1[2])};
                    #else
                    const double tmp_2[3] = {
                        static_cast<double>(static_cast<type_x>(std::log(static_cast<type_x>(std::exp(tmp_x.x))))), 
                        static_cast<double>(static_cast<type_y>(std::log(static_cast<type_y>(std::exp(tmp_x.y))))),
                        static_cast<double>(static_cast<type_z>(std::log(static_cast<type_z>(std::exp(tmp_x.z)))))};
                    #endif
                #endif

                for (std::size_t ii = 0; ii < 3; ++ii)
                {
                    const type abs_error = std::abs(tmp_2[ii] != static_cast<type>(0) ? (tmp_1[ii] - tmp_2[ii]) / tmp_2[ii] : tmp_1[ii] - tmp_2[ii]);
                    if (abs_error > max_abs_error)
                    {
                        std::cout << "error: " << tmp_1[ii] << " vs " << tmp_2[ii] << " (" << abs_error << ")" << std::endl;
                        not_passed = true;
                        break;
                    }
                }
                if (not_passed) break;

                if (e < print_num_elements)
                {
                    std::cout << out_2[j][i] << std::endl;
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
        std::vector<element_type> in_orig_1(nx * ny * nz);
        std::vector<element_type> in_orig_2(nx * ny * nz);
        buffer_type<element_type, 3> in_1, out_1, out_2;
        #if defined(VECTOR_PRODUCT)
        buffer_type<element_type, 3> in_2{{nx, ny, nz}};
        #endif
    
        in_1.resize({nx, ny, nz});
        out_1.resize({nx, ny, nz});
        out_2.resize({nx, ny, nz});
        
        for (std::size_t k = 0; k < nz; ++k)
        {
            for (std::size_t j = 0; j < ny; ++j)
            {
                for (std::size_t i = 0; i < nx; ++i)
                {
                    const type s_1 = static_cast<type>((2.0 * drand48() -1.0) * SPREAD + OFFSET);
                    const type s_2 = static_cast<type>((2.0 * drand48() -1.0) * SPREAD + OFFSET);
                    const type s_3 = static_cast<type>((2.0 * drand48() -1.0) * SPREAD + OFFSET);
                    in_orig_1[k * nx * ny + j * nx + i] = {static_cast<type_x>(s_1 * value), static_cast<type_y>(s_2 * value), static_cast<type_z>(s_3 * value)};
                    in_1[k][j][i] = in_orig_1[k * nx * ny + j * nx + i];
                    #if defined(INPLACE)
                    out_2[k][j][i] = in_1[k][j][i];
                    #endif

                    #if defined(VECTOR_PRODUCT)
                    {
                        const type s_1 = static_cast<type>((2.0 * drand48() -1.0) * SPREAD + OFFSET);
                        const type s_2 = static_cast<type>((2.0 * drand48() -1.0) * SPREAD + OFFSET);
                        const type s_3 = static_cast<type>((2.0 * drand48() -1.0) * SPREAD + OFFSET);
                        in_orig_2[k * nx * ny + j * nx + i] = {static_cast<type_x>(s_1 * value), static_cast<type_y>(s_2 * value), static_cast<type_z>(s_3 * value)};
                        in_2[k][j][i] = in_orig_2[k * nx * ny + j * nx + i];
                    }
                    #endif
                }
            }
        }

        for (std::size_t n = 0; n < WARMUP; ++n)
        {
            #if defined(VECTOR_PRODUCT)
                kernel<element_type>::cross<3>(in_1, in_2, out_2);
            #else
                #if defined(INPLACE)
                kernel<element_type>::exp<3>(out_2);
                kernel<element_type>::log<3>(out_2);
                #else
                kernel<element_type>::exp<3>(in_1, out_1);
                kernel<element_type>::log<3>(out_1, out_2);
                #endif
            #endif
        }

        for (std::size_t n = 0; n < MEASUREMENT; ++n)
        {
            #if defined(VECTOR_PRODUCT)
                time += kernel<element_type>::cross<3>(in_1, in_2, out_2);
            #else
                #if defined(INPLACE)
                time += kernel<element_type>::exp<3>(out_2);
                time += kernel<element_type>::log<3>(out_2);
                #else
                time += kernel<element_type>::exp<3>(in_1, out_1);
                time += kernel<element_type>::log<3>(out_1, out_2);
                #endif
            #endif
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
                    const double tmp_1[3] = {static_cast<double>(out_2[k][j][i].x), static_cast<double>(out_2[k][j][i].y), static_cast<double>(out_2[k][j][i].z)};
                    const element_type tmp_x = in_orig_1[k * nx * ny + j * nx + i];
                    #if defined(VECTOR_PRODUCT)
                        const element_type tmp_y = in_orig_2[k * nx * ny + j * nx + i];
                        const element_type tmp_y_1 = 0.5 + fw::math<element_type>::log(1.0 + fw::cross(tmp_x, fw::math<element_type>::exp(tmp_y)));
                        const double tmp_2[3] = {
                            static_cast<double>(tmp_y_1.x),
                            static_cast<double>(tmp_y_1.y),
                            static_cast<double>(tmp_y_1.z)};
                    #else
                        #if defined(ELEMENT_ACCESS)
                        const double tmp_2[3] = {
                            static_cast<double>(tmp_1[0]), 
                            static_cast<double>(static_cast<type_y>(std::log(static_cast<type_y>(std::exp(tmp_x.y))))),
                            static_cast<double>(tmp_1[2])};
                        #else
                        const double tmp_2[3] = {
                            static_cast<double>(static_cast<type_x>(std::log(static_cast<type_x>(std::exp(tmp_x.x))))), 
                            static_cast<double>(static_cast<type_y>(std::log(static_cast<type_y>(std::exp(tmp_x.y))))),
                            static_cast<double>(static_cast<type_z>(std::log(static_cast<type_z>(std::exp(tmp_x.z)))))};
                        #endif
                    #endif

                    for (std::size_t ii = 0; ii < 3; ++ii)
                    {
                        const type abs_error = std::abs(tmp_2[ii] != static_cast<type>(0) ? (tmp_1[ii] - tmp_2[ii]) / tmp_2[ii] : tmp_1[ii] - tmp_2[ii]);
                        if (abs_error > max_abs_error)
                        {
                            std::cout << "error: " << tmp_1[ii] << " vs " << tmp_2[ii] << " (" << abs_error << ")" << std::endl;
                            not_passed = true;
                            break;
                        }
                    }
                    if (not_passed) break;

                    if (e < print_num_elements)
                    {
                        std::cout << out_2[k][j][i] << std::endl;
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