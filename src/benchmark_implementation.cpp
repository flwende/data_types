// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#include <omp.h>
#include <benchmark.hpp>
#include <vector>

constexpr double SPREAD = 0.5;
constexpr double OFFSET = 2.0;

#if defined(CHECK_RESULTS)
constexpr std::size_t WARMUP = 0;
constexpr std::size_t MEASUREMENT = 1;
#else
constexpr std::size_t WARMUP = 10;
constexpr std::size_t MEASUREMENT = 20;
#endif

#if defined(__CUDACC__)
template <typename FuncT, typename Container>
CUDA_KERNEL
auto KernelImplementation(FuncT func, Container a, Container b, Container c) -> std::enable_if_t<(Container::TParam_Dimension == 1), void>
{
    using namespace ::fw::math;

    const SizeT x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < a.Size(0))
    {
        func(a[x], b[x], c[x]);
    }
}

template <typename FuncT, typename Container>
CUDA_KERNEL
auto KernelImplementation(FuncT func, Container a, Container b, Container c) -> std::enable_if_t<(Container::TParam_Dimension == 2), void>
{
    using namespace ::fw::math;

    const SizeT x = blockIdx.x * blockDim.x + threadIdx.x;
    const SizeT y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < a.Size(0) && y < a.Size(1))
    {
        func(a[y][x], b[y][x], c[y][x]);
    }
}

template <typename FuncT, typename Container>
CUDA_KERNEL
auto KernelImplementation(FuncT func, Container a, Container b, Container c) -> std::enable_if_t<(Container::TParam_Dimension == 3), void>
{
    using namespace ::fw::math;

    const SizeT x = blockIdx.x * blockDim.x + threadIdx.x;
    const SizeT y = blockIdx.y * blockDim.y + threadIdx.y;
    const SizeT z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < a.Size(0) && y < a.Size(1) && z < a.Size(2))
    {
        func(a[z][y][x], b[z][y][x], c[z][y][x]);
    } 
}

template <typename FuncT, typename Field>
auto Kernel(FuncT func, Field& a, Field& b, Field& c) -> void
{
    const dim3 block{128, 1, 1};
    const dim3 grid = GetGridSize(a.Size(), block);

    KernelImplementation<<<grid, block>>>(func, a.DeviceData(), b.DeviceData(), c.DeviceData());
}

#else
template <typename FuncT, typename Container>
auto KernelImplementation(FuncT func, Container& a, Container& b, Container& c) -> std::enable_if_t<(Container::TParam_Dimension == 1), void>
{
    using namespace ::fw::math;

    #pragma omp simd
    for (SizeT x = 0; x < a.Size(0); ++x)
    {
        func(a[x], b[x], c[x]);
    }
}

template <typename FuncT, typename Container>
auto KernelImplementation(FuncT func, Container& a, Container& b, Container& c) -> std::enable_if_t<(Container::TParam_Dimension == 2), void>
{
    using namespace ::fw::math;

    for (SizeT y = 0; y < a.Size(1); ++y)
    {
        #pragma omp simd
        for (SizeT x = 0; x < a.Size(0); ++x)
        {
            func(a[y][x], b[y][x], c[y][x]);
        }
    }
}

template <typename FuncT, typename Container>
auto KernelImplementation(FuncT func, Container& a, Container& b, Container& c) -> std::enable_if_t<(Container::TParam_Dimension == 3), void>
{
    using namespace ::fw::math;

    for (SizeT z = 0; z < a.Size(2); ++z)
    {
        for (SizeT y = 0; y < a.Size(1); ++y)
        {
            #pragma omp simd
            for (SizeT x = 0; x < a.Size(0); ++x)
            {
                func(a[z][y][x], b[z][y][x], c[z][y][x]);
            }
        }
    } 
}

template <typename FuncT, typename Field>
auto Kernel(FuncT func, Field& a, Field& b, Field& c) -> void
{
    KernelImplementation(func, a, b, c);
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

#if defined(CHECK_RESULTS)
    std::vector<ElementT> field_1_copy = field_1.Get();
    std::vector<ElementT> field_2_copy = field_2.Get();
#endif

#if defined(ELEMENT_ACCESS)
    auto kernel_1 = [] CUDA_DEVICE_VERSION (const auto& a, const auto& b, auto&& c) -> void { using namespace ::fw::math; c.z = Exp(a.x + b.y); };
    auto kernel_2 = [] CUDA_DEVICE_VERSION (const auto& a, const auto& b, auto&& c) -> void { using namespace ::fw::math; c.x = Log(a.z) - b.y; };
#else
    auto kernel_1 = [] CUDA_DEVICE_VERSION (const auto& a, const auto& b, auto&& c) -> void { using namespace ::fw::math; c = Exp(a + b); };
    auto kernel_2 = [] CUDA_DEVICE_VERSION (const auto& a, const auto& b, auto&& c) -> void { using namespace ::fw::math; c = Log(a) - b; };
#endif

#if defined(__CUDACC__)
    // Create device buffer and copy in the host data.
    field_1.CopyHostToDevice();
    field_2.CopyHostToDevice();
    field_3.CopyHostToDevice();
#endif

    for (SizeT i = 0; i < WARMUP; ++i)
    {
        Kernel(kernel_1, field_1, field_2, field_3);
        Kernel(kernel_2, field_3, field_2, field_1);
    }

#if defined(__CUDACC__)
    cudaDeviceSynchronize();
#endif

    double start_time = omp_get_wtime();

    for (SizeT i = 0; i < MEASUREMENT; ++i)
    {
        Kernel(kernel_1, field_1, field_2, field_3);
        Kernel(kernel_2, field_3, field_2, field_1);
    }

#if defined(__CUDACC__)
    cudaDeviceSynchronize();
#endif

    double stop_time = omp_get_wtime();

#if defined(__CUDACC__)
    cudaError_t error = cudaGetLastError();
    std::cout << cudaGetErrorString(error) << std::endl;
#endif

    std::cout << "elapsed time: " << (stop_time - start_time) * 1.0E3 << " ms" << std::endl;

#if defined(CHECK_RESULTS)
    {
        using namespace ::fw::math;

        const SizeT n = size.ReduceMul();
        std::vector<ElementT> reference(n);
        std::vector<ElementT> field_3_copy = field_3.Get(true);
    
        ElementT deviation;

        for (SizeT i = 0; i < n; ++i)
        {
#if defined(ELEMENT_ACCESS)
            reference[i].z = Exp(field_1_copy[i].x + field_2_copy[i].y);
#else
            reference[i] = Exp(field_1_copy[i] + field_2_copy[i]);
#endif
            ElementT rel_error = (reference[i] - field_3_copy[i]) / Max(ElementT(1.0E-9), reference[i]);
            deviation = Max(deviation, Abs(rel_error));

#if defined(ELEMENT_ACCESS)
            reference[i] = field_1_copy[i];
#endif
        }

        std::vector<ElementT> field_1_copy = field_1.Get(true);

        for (SizeT i = 0; i < n; ++i)
        {
#if defined(ELEMENT_ACCESS)
            reference[i].x = Log(field_3_copy[i].z) - field_2_copy[i].y;
#else
            reference[i] = Log(field_3_copy[i]) - field_2_copy[i];
#endif
            ElementT rel_error = (reference[i] - field_1_copy[i]) / Max(ElementT(1.0E-9), reference[i]);
            deviation = Max(deviation, Abs(rel_error));
        }

        std::cout << "deviation: " << deviation << std::endl;
    }
#endif
    
    return 0;
}

template int benchmark<1>(int, char**, const SizeArray<1>&);
template int benchmark<2>(int, char**, const SizeArray<2>&);
template int benchmark<3>(int, char**, const SizeArray<3>&);
