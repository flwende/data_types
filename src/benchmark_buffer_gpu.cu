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

#if defined(__CUDACC__)
template <typename FieldT>
__global__ void foo(FieldT a)
{
    const std::size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const std::size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    
    const std::size_t nx = a.Size()[0];
    const std::size_t ny = a.Size()[1];
    const std::size_t nz = a.Size()[2];
    
    if (x < nx && y < ny && z < nz)
    {

        const std::size_t index = (z * ny + y) * nx + x;

#if defined(ELEMENT_ACCESS)
        a[z][y][x].x = 2.0 * index;
#else
        a[z][y][x].x = 1.0 * index;
        a[z][y][x].y = -2.0 * index;
        a[z][y][x].z = 3.0 * index;
#endif
    }
}
#endif

using namespace ::fw::math;

template <typename T>
class DEBUG;

int main(int argc, char** argv)
{
    // command line arguments
    std::size_t dim = 0;
    const std::size_t nx = (argc > 1 ? atoi(argv[++dim]) : NX_DEFAULT);
    const std::size_t ny = (argc > 2 ? atoi(argv[++dim]) : NY_DEFAULT);
    const std::size_t nz = (argc > 3 ? atoi(argv[++dim]) : NZ_DEFAULT);
    const std::size_t print_elem = (argc > 4 ? atoi(argv[4]) : std::min(nx, 12UL));
/*
    ::fw::dataTypes::Tuple<float, int, double> t_1{32.324, 1023, -34.43};
    //::fw::dataTypes::Tuple<float, int, double> t_1{32.324};
    std::cout << t_1 << std::endl;
    std::cout << -t_1 << std::endl;
    std::cout << -t_1 * 2.0 << std::endl;

    ::fw::dataTypes::internal::TupleBase<float, int, double> t{3.2f, 23, -2323.3453};
    std::cout << t.x << ", " << t.y << ", " << t.z << std::endl;
*/

    element_type x(1,2,3);
    //x = 1;
    //x = {1, 2, 3};

    // initialization
    srand48(nx);
    const type value = 1.0 + static_cast<type>(drand48());
    std::cout << "initial value = " << value << std::endl;
#if !defined(OLD)
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
        
        in_1.Resize({nx});
        out_1.Resize({nx});
        out_2.Resize({nx});
        
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
                kernel<element_type>::cross<1>(in_1, in_2, out_2, {nx});
            #else
                #if defined(INPLACE)
                kernel<element_type>::exp<1>(out_2, {nx});
                kernel<element_type>::log<1>(out_2, {nx});
                #else
                kernel<element_type>::exp<1>(in_1, out_1, {nx});
                kernel<element_type>::log<1>(out_1, out_2, {nx});
                #endif
            #endif
        }

        for (std::size_t n = 0; n < MEASUREMENT; ++n)
        {
            #if defined(VECTOR_PRODUCT)
                time += kernel<element_type>::cross<1>(in_1, in_2, out_2, {nx});
            #else
                #if defined(INPLACE)
                time += kernel<element_type>::exp<1>(out_2, {nx});
                time += kernel<element_type>::log<1>(out_2, {nx});
                #else
                time += kernel<element_type>::exp<1>(in_1, out_1, {nx});
                time += kernel<element_type>::log<1>(out_1, out_2, {nx});
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
    
        in_1.Resize({nx, ny});
        out_1.Resize({nx, ny});
        out_2.Resize({nx, ny});
        
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
                kernel<element_type>::cross<2>(in_1, in_2, out_2, {nx, ny});
            #else
                #if defined(INPLACE)
                kernel<element_type>::exp<2>(out_2, {nx, ny});
                kernel<element_type>::log<2>(out_2, {nx, ny});
                #else
                kernel<element_type>::exp<2>(in_1, out_1, {nx, ny});
                kernel<element_type>::log<2>(out_1, out_2, {nx, ny});
                #endif
            #endif
        }

        for (std::size_t n = 0; n < MEASUREMENT; ++n)
        {
            #if defined(VECTOR_PRODUCT)
                time += kernel<element_type>::cross<2>(in_1, in_2, out_2, {nx, ny});
            #else
                #if defined(INPLACE)
                time += kernel<element_type>::exp<2>(out_2, {nx, ny});
                time += kernel<element_type>::log<2>(out_2, {nx, ny});
                #else
                time += kernel<element_type>::exp<2>(in_1, out_1, {nx, ny});
                time += kernel<element_type>::log<2>(out_1, out_2, {nx, ny});
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
        #if defined(USE_1D_BUFFER)
            buffer_type<element_type, 1> in_1, out_1, out_2;
            #if defined(VECTOR_PRODUCT)
            buffer_type<element_type, 1> in_2{{nx * ny * nz}};
            #endif            
            in_1.Resize({nx * ny * nz});
            out_1.Resize({nx * ny * nz});
            out_2.Resize({nx * ny * nz}); 
        #else
            buffer_type<element_type, 3> in_1, out_1, out_2;
            #if defined(VECTOR_PRODUCT)
            buffer_type<element_type, 3> in_2{{nx, ny, nz}};
            #endif
            in_1.Resize({nx, ny, nz});
            out_1.Resize({nx, ny, nz});
            out_2.Resize({nx, ny, nz});
        #endif
        
        /*
        for (std::size_t k = 0; k < nz; ++k)
        {
            for (std::size_t j = 0; j < ny; ++j)
            {
                for (std::size_t i = 0; i < nx; ++i)
                {
                    const std::size_t index = k * nx * ny + j * nx + i;
                    const type s_1 = static_cast<type>((2.0 * drand48() -1.0) * SPREAD + OFFSET);
                    const type s_2 = static_cast<type>((2.0 * drand48() -1.0) * SPREAD + OFFSET);
                    const type s_3 = static_cast<type>((2.0 * drand48() -1.0) * SPREAD + OFFSET);
                    in_orig_1[index] = {static_cast<type_x>(s_1 * value), static_cast<type_y>(s_2 * value), static_cast<type_z>(s_3 * value)};
                    #if defined(USE_1D_BUFFER)
                        in_1[index] = in_orig_1[index];
                        #if defined(INPLACE)
                        out_2[index] = in_1[index];
                        #endif
                    #else
                        in_1[k][j][i] = in_orig_1[index];
                        #if defined(INPLACE)
                        out_2[k][j][i] = in_1[k][j][i];
                        #endif
                    #endif

                    #if defined(VECTOR_PRODUCT)
                    {
                        const type s_1 = static_cast<type>((2.0 * drand48() -1.0) * SPREAD + OFFSET);
                        const type s_2 = static_cast<type>((2.0 * drand48() -1.0) * SPREAD + OFFSET);
                        const type s_3 = static_cast<type>((2.0 * drand48() -1.0) * SPREAD + OFFSET);
                        in_orig_2[index] = {static_cast<type_x>(s_1 * value), static_cast<type_y>(s_2 * value), static_cast<type_z>(s_3 * value)};
                        #if defined(USE_1D_BUFFER)
                            in_2[index] = in_orig_2[index];
                        #else
                            in_2[k][j][i] = in_orig_2[index];
                        #endif
                    }
                    #endif
                }
            }
        }
       
        #if defined(USE_1D_BUFFER)
        constexpr std::size_t DD = 1;
        #else
        constexpr std::size_t DD = 3;
        #endif

        for (std::size_t n = 0; n < WARMUP; ++n)
        {
            #if defined(VECTOR_PRODUCT)
                kernel<element_type>::cross<3, DD>(in_1, in_2, out_2, {nx, ny, nz});
            #else
                #if defined(INPLACE)
                kernel<element_type>::exp<3, DD>(out_2, {nx, ny, nz});
                kernel<element_type>::log<3, DD>(out_2, {nx, ny, nz});
                #else
                kernel<element_type>::exp<3, DD>(in_1, out_1, {nx, ny, nz});
                kernel<element_type>::log<3, DD>(out_1, out_2, {nx, ny, nz});
                #endif
            #endif
        }

        for (std::size_t n = 0; n < MEASUREMENT; ++n)
        {
            #if defined(VECTOR_PRODUCT)
                time += kernel<element_type>::cross<3, DD>(in_1, in_2, out_2, {nx, ny, nz});
            #else
                #if defined(INPLACE)
                time += kernel<element_type>::exp<3, DD>(out_2, {nx, ny, nz});
                time += kernel<element_type>::log<3, DD>(out_2, {nx, ny, nz});
                #else
                time += kernel<element_type>::exp<3, DD>(in_1, out_1, {nx, ny, nz});
                time += kernel<element_type>::log<3, DD>(out_1, out_2, {nx, ny, nz});
                #endif
            #endif
        }
        */
        const dim3 block_size(128, 1, 1);
        const dim3 grid_size((nx + block_size.x - 1) / block_size.x, (ny + block_size.y - 1) / block_size.y, (nz + block_size.z - 1) / block_size.z);
	    //XXX_NAMESPACE::Container<element_type, 3, layout, XXX_NAMESPACE::target::GPU_CUDA> gpu_buffer{{nx, ny, nz}};
        
        for (std::size_t i = 0; i < 10; ++i)
        {
            foo<<<grid_size, block_size>>>(in_1.DeviceData());
	        //foo<<<grid_size, block_size>>>(gpu_buffer);
        }

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);
        for (std::size_t i = 0; i < 10; ++i)
        {
            foo<<<grid_size, block_size>>>(in_1.DeviceData());
	        //foo<<<grid_size, block_size>>>(gpu_buffer);
        }
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, start, stop);
        std::cout << "elapsed time: " << elapsed_time << "ms" << std::endl;

        if (argc > 4)
        {
            in_1.CopyDeviceToHost();

            //const float* ptr = reinterpret_cast<const float*>(&in_1[0][0][0].x);
            const std::int8_t* ptr = reinterpret_cast<const std::int8_t*>(&in_1[0][0][0].x);

            for (std::size_t z = 0; z < nz; ++z)
            {
                for (std::size_t y = 0; y < ny; ++y)
                {
                    for (std::size_t x = 0; x < nx; ++x)
                    {
                        //std::cout << ptr[(z * ny + y) * nx + x] << " ";
                        std::cout << static_cast<std::int32_t>(ptr[(z * ny + y) * nx + x]) << " ";
                    }
                    std::cout << std::endl;
                }
            }
        }

        cudaError_t error = cudaGetLastError();
        std::cout << cudaGetErrorString(error) << std::endl;

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
                    const std::size_t index = k * nx * ny + j * nx + i;
                    #if defined(USE_1D_BUFFER)
                        const double tmp_1[3] = {static_cast<double>(out_2[index].x), static_cast<double>(out_2[index].y), static_cast<double>(out_2[index].z)};
                    #else
                        const double tmp_1[3] = {static_cast<double>(out_2[k][j][i].x), static_cast<double>(out_2[k][j][i].y), static_cast<double>(out_2[k][j][i].z)};
                    #endif
                    const element_type tmp_x = in_orig_1[index];
                    #if defined(VECTOR_PRODUCT)
                        const element_type tmp_y = in_orig_2[index];
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
                    #if defined(USE_1D_BUFFER)
                        std::cout << out_2[index] << std::endl;
                    #else
                        std::cout << out_2[k][j][i] << std::endl;
                    #endif
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
#endif

    return 0;
}
