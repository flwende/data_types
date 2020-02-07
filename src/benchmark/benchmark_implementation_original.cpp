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
template <typename FuncT, SizeT Dimension>
CUDA_KERNEL
auto KernelImplementation(FuncT func, ElementT* a, ElementT* b, ElementT* c, SizeArray<Dimension> size) -> std::enable_if_t<(Dimension == 1), void>
{
    using namespace ::fw::math;

    const SizeT x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < size[0])
    {
        const SizeT i = x;

        func(&a[i], &b[i], &c[i]);
    }
}

template <typename FuncT, SizeT Dimension>
CUDA_KERNEL
auto KernelImplementation(FuncT func, ElementT* a, ElementT* b, ElementT* c, SizeArray<Dimension> size) -> std::enable_if_t<(Dimension == 2), void>
{
    using namespace ::fw::math;

    const SizeT x = blockIdx.x * blockDim.x + threadIdx.x;
    const SizeT y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < size[0] && y < size[1])
    {
        const SizeT i = y * size[0] + x;

        func(&a[i], &b[i], &c[i]);
    }
}

template <typename FuncT, SizeT Dimension>
CUDA_KERNEL
auto KernelImplementation(FuncT func, ElementT* a, ElementT* b, ElementT* c, SizeArray<Dimension> size) -> std::enable_if_t<(Dimension == 3), void>
{
    using namespace ::fw::math;

    const SizeT x = blockIdx.x * blockDim.x + threadIdx.x;
    const SizeT y = blockIdx.y * blockDim.y + threadIdx.y;
    const SizeT z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < size[0] && y < size[1] && z < size[2])
    {
        const SizeT i = (z * size[1] + y) * size[0] + x;

        func(&a[i], &b[i], &c[i]);
    } 
}

template <typename FuncT, SizeT Dimension>
auto Kernel(FuncT func, ElementT* a, ElementT* b, ElementT* c, const SizeArray<Dimension>& size) -> void
{
    const dim3 block{128, 1, 1};
    const dim3 grid = GetGridSize(size, block);

    KernelImplementation<<<grid, block>>>(func, a, b, c, size);
}
#else
template <typename FuncT, SizeT Dimension>
auto KernelImplementation(FuncT func, ElementT* a, ElementT* b, ElementT* c, SizeArray<Dimension> size) -> std::enable_if_t<(Dimension == 1), void>
{
    using namespace ::fw::math;

    #pragma omp simd
    for (SizeT i = 0; i < size[0]; ++i)
    {
        func(&a[i], &b[i], &c[i]);
    }
}

template <typename FuncT, SizeT Dimension>
auto KernelImplementation(FuncT func, ElementT* a, ElementT* b, ElementT* c, SizeArray<Dimension> size) -> std::enable_if_t<(Dimension == 2), void>
{
    using namespace ::fw::math;

    #pragma omp simd
    for (SizeT i = 0; i < (size[0] * size[1]); ++i)
    {
        func(&a[i], &b[i], &c[i]);
    }
}

template <typename FuncT, SizeT Dimension>
auto KernelImplementation(FuncT func, ElementT* a, ElementT* b, ElementT* c, SizeArray<Dimension> size) -> std::enable_if_t<(Dimension == 3), void>
{
    using namespace ::fw::math;

    #pragma omp simd
    for (SizeT i = 0; i < (size[0] * size[1] * size[2]); ++i)
    {
        func(&a[i], &b[i], &c[i]);
    }
}

template <typename FuncT, SizeT Dimension>
auto Kernel(FuncT func, ElementT* a, ElementT* b, ElementT* c, const SizeArray<Dimension>& size) -> void
{
    KernelImplementation(func, a, b, c, size);
}
#endif

template <SizeT Dimension>
int benchmark(int argc, char** argv, const SizeArray<Dimension>& size)
{
    const SizeT n = size.ReduceMul();

    ElementT* data_1 = reinterpret_cast<ElementT*>(_mm_malloc(size.ReduceMul() * sizeof(ElementT), ::fw::simd::alignment));
    ElementT* data_2 = reinterpret_cast<ElementT*>(_mm_malloc(size.ReduceMul() * sizeof(ElementT), ::fw::simd::alignment));
    ElementT* data_3 = reinterpret_cast<ElementT*>(_mm_malloc(size.ReduceMul() * sizeof(ElementT), ::fw::simd::alignment));
    
    // Field initialization.
    srand48(1);
    for (SizeT i = 0; i < n; ++i)
    {
        data_1[i] = static_cast<RealT>((2.0 * drand48() -1.0) * SPREAD + OFFSET);
    }
    for (SizeT i = 0; i < n; ++i)
    {
        data_2[i] = static_cast<RealT>((2.0 * drand48() -1.0) * SPREAD + OFFSET);
    }

#if defined(__CUDACC__)
    // Create device buffer and copy in the host data.
    ElementT* d_data_1;
    ElementT* d_data_2;
    ElementT* d_data_3;

    cudaMalloc((void**)&d_data_1, n * sizeof(ElementT));
    cudaMalloc((void**)&d_data_2, n * sizeof(ElementT));
    cudaMalloc((void**)&d_data_3, n * sizeof(ElementT));

    cudaMemcpy((void*)d_data_1, (const void*)data_1, n * sizeof(ElementT), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_data_2, (const void*)data_2, n * sizeof(ElementT), cudaMemcpyHostToDevice);

    ElementT* field_1 = d_data_1;
    ElementT* field_2 = d_data_2;
    ElementT* field_3 = d_data_3;
#else
    ElementT* field_1 = data_1;
    ElementT* field_2 = data_2;
    ElementT* field_3 = data_3;
#endif

#if defined(SAXPY_KERNEL)
#if defined(ELEMENT_ACCESS)
    auto kernel_1 = [] CUDA_DEVICE_VERSION (const auto* a, const auto* b, auto* c) -> void { using namespace ::fw::math; c->z = a->x * 3.2 + b->y; };
    auto kernel_2 = [] CUDA_DEVICE_VERSION (const auto* a, const auto* b, auto* c) -> void { using namespace ::fw::math; c->x = a->z * 3.2 + b->y; };
#else
    auto kernel_1 = [] CUDA_DEVICE_VERSION (const auto* a, const auto* b, auto* c) -> void { using namespace ::fw::math; (*c) = (*a) * 3.2 + (*b); };
    auto kernel_2 = kernel_1;
#endif
#else
#if defined(ELEMENT_ACCESS)
    auto kernel_1 = [] CUDA_DEVICE_VERSION (const auto* a, const auto* b, auto* c) -> void { using namespace ::fw::math; c->z = Exp(a->x + b->y); };
    auto kernel_2 = [] CUDA_DEVICE_VERSION (const auto* a, const auto* b, auto* c) -> void { using namespace ::fw::math; c->x = Log(a->z) - b->y; };
#else
    auto kernel_1 = [] CUDA_DEVICE_VERSION (const auto* a, const auto* b, auto* c) -> void { using namespace ::fw::math; *c = Exp((*a) + (*b)); };
    auto kernel_2 = [] CUDA_DEVICE_VERSION (const auto* a, const auto* b, auto* c) -> void { using namespace ::fw::math; *c = Log(*a) - (*b); };
#endif
#endif

    for (SizeT i = 0; i < WARMUP; ++i)
    {
        Kernel(kernel_1, field_1, field_2, field_3, size);
        Kernel(kernel_2, field_3, field_2, field_1, size);
    }

#if defined(__CUDACC__)
    cudaDeviceSynchronize();
#endif

    double start_time = omp_get_wtime();

    for (SizeT i = 0; i < MEASUREMENT; ++i)
    {
        Kernel(kernel_1, field_1, field_2, field_3, size);
        Kernel(kernel_2, field_3, field_2, field_1, size);
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
    
    _mm_free(data_1);
    _mm_free(data_2);
    _mm_free(data_3);

    return 0;
}

template int benchmark<1>(int, char**, const SizeArray<1>&);
template int benchmark<2>(int, char**, const SizeArray<2>&);
template int benchmark<3>(int, char**, const SizeArray<3>&);
