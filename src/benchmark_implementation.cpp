// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#include <benchmark.hpp>

constexpr double SPREAD = 1.0;
constexpr double OFFSET = 3.0;

#if defined(CHECK_RESULTS)
constexpr std::size_t WARMUP = 0;
constexpr std::size_t MEASUREMENT = 1;
#else
constexpr std::size_t WARMUP = 10;
constexpr std::size_t MEASUREMENT = 20;
#endif

#if defined(__CUDACC__)

template <typename ValueT, SizeT Dimension>
__global__
auto foo(DeviceField<ValueT, Dimension> a, DeviceField<ValueT, Dimension> b, DeviceField<ValueT, Dimension> c) -> std::enable_if_t<(Dimension == 1), void>
{
    using namespace ::fw::math;

    const SizeT thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id_x < a.Size()[0])
    {
        c[thread_id_x] = a[thread_id_x] * b[thread_id_x];
    }
}

template <typename ValueT, SizeT Dimension>
__global__
auto foo(DeviceField<ValueT, Dimension> a, DeviceField<ValueT, Dimension> b, DeviceField<ValueT, Dimension> c) -> std::enable_if_t<(Dimension == 2), void>
{
    using namespace ::fw::math;

    const SizeT thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;
    const SizeT thread_id_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (thread_id_x < a.Size()[0] && thread_id_y < a.Size()[1])
    {
        c[thread_id_y][thread_id_x] = a[thread_id_y][thread_id_x] * b[thread_id_y][thread_id_x];
    }
}

template <typename ValueT, SizeT Dimension>
__global__
auto foo(DeviceField<ValueT, Dimension> a, DeviceField<ValueT, Dimension> b, DeviceField<ValueT, Dimension> c) -> std::enable_if_t<(Dimension == 3), void>
{
    using namespace ::fw::math;

    const SizeT thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;
    const SizeT thread_id_y = blockIdx.y * blockDim.y + threadIdx.y;
    const SizeT thread_id_z = blockIdx.z * blockDim.z + threadIdx.z;

    if (thread_id_x < a.Size()[0] && thread_id_y < a.Size()[1] && thread_id_z < a.Size()[2])
    {
        c[thread_id_z][thread_id_y][thread_id_x].x = a[thread_id_z][thread_id_y][thread_id_x].y * b[thread_id_z][thread_id_y][thread_id_x].z;  
    } 
}

#endif

template <SizeT Dimension>
int benchmark(int argc, char** argv, const SizeArray<Dimension>& size)
{
    Field<ElementT, Dimension> field_1(size);
    Field<ElementT, Dimension> field_2(size);
    Field<ElementT, Dimension> field_3(size, true);

    // Field initialization.
    srand48(1);
    field_1.Set([] (const auto I) { return static_cast<RealT>((2.0 * drand48() -1.0) * SPREAD + OFFSET); });
    field_2.Set([] (const auto I) { return static_cast<RealT>((2.0 * drand48() -1.0) * SPREAD + OFFSET); });

    // Create device buffer and copy in the host data.
    field_1.CopyHostToDevice();
    field_2.CopyHostToDevice();
    field_3.CopyHostToDevice();

#if defined(__CUDACC__)
    const dim3 block{128, 1, 1};
    const dim3 grid = GetGridSize(size, block);

    std::cout << "block: " << block.x << ", " << block.y << ", " << block.z << std::endl;
    std::cout << "grid: " << grid.x << ", " << grid.y << ", " << grid.z << std::endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (SizeT i = 0; i < WARMUP; ++i)
    {
        foo<<<grid, block>>>(field_1.DeviceData(), field_2.DeviceData(), field_3.DeviceData());
    }

    cudaEventRecord(start, 0);

    for (SizeT i = 0; i < MEASUREMENT; ++i)
    {
        foo<<<grid, block>>>(field_1.DeviceData(), field_2.DeviceData(), field_3.DeviceData());
    }

    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaError_t error = cudaGetLastError();
    std::cout << cudaGetErrorString(error) << std::endl;

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    std::cout << "elapsed time: " << elapsed_time << "ms" << std::endl;
#endif

    auto data = field_3.Get(true);
    /*
    if (argc > (Dimension + 1))
    {
        if (Dimension == 3)
        {
            for (SizeT k = 0; k < size[2]; ++k)
                for (SizeT j = 0; j < size[1]; ++j)
                {
                    for (SizeT i = 0; i < size[0]; ++i)
                        std::cout << data[(k * size[1] + j) * size[0] + i] << " ";
                    std::cout << std::endl;
                }
        } 
        else if (Dimension == 2)
        {
            for (SizeT j = 0; j < size[1]; ++j)
            {
                for (SizeT i = 0; i < size[0]; ++i)
                    std::cout << data[j * size[0] + i] << " ";
                std::cout << std::endl;
            }
        }
        else if (Dimension == 1)
        {
            for (SizeT i = 0; i < size[0]; ++i)
                std::cout << data[i] << " ";
            std::cout << std::endl;
            
        }
    }
    */
    
    return 0;
}

template int benchmark<1>(int, char**, const SizeArray<1>&);
template int benchmark<2>(int, char**, const SizeArray<2>&);
template int benchmark<3>(int, char**, const SizeArray<3>&);
