// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#include <cstdint>
#include <cstdlib>
#include <vector>
#include <type_traits>
#include <omp.h>
#include <benchmark.hpp>

constexpr double SPREAD = 0.5;
constexpr double OFFSET = 2.0;

#if defined(SAXPY_KERNEL)
#define NUM_ACCESSES_2
#else
#define NUM_ACCESSES_3
#endif

#if defined(__CUDACC__)
template <typename FuncT, SizeT Dimension, typename ValueT>
CUDA_KERNEL
auto KernelImplementation(FuncT func, [[maybe_unused]] ValueT* a, [[maybe_unused]] ValueT* b, ValueT* c, SizeArray<Dimension> size, [[maybe_unused]] const std::int32_t* index = nullptr) -> std::enable_if_t<(Dimension == 1), void>
{
    using namespace ::fw::math;

    const SizeT x = blockIdx.x * blockDim.x + threadIdx.x;

#if defined(DIFFUSION)
    if (index)
    {
        if (x < size[0])
        {        
            const SizeT index_1d = index[x];

#if defined(SOA_LAYOUT)
#if defined(NUM_ACCESSES_1)
            func(&c[index_1d], size[0]);
#elif defined(NUM_ACCESSES_2)
            func(&a[index_1d], &c[index_1d], size[0]);
#elif defined(NUM_ACCESSES_3)
            func(&a[index_1d], &b[index_1d], &c[index_1d], size[0]);
#endif
#else
#if defined(NUM_ACCESSES_1)
            func(c[index_1d]);
#elif defined(NUM_ACCESSES_2)
            func(a[index_1d], c[index_1d]);
#elif defined(NUM_ACCESSES_3)
            func(a[index_1d], b[index_1d], c[index_1d]);
#endif
#endif
        }
    }
    else
#endif
    {
        if (x < size[0])
        {        
            const SizeT index_1d = x;

#if defined(SOA_LAYOUT)
#if defined(NUM_ACCESSES_1)
            func(&c[index_1d], size[0]);
#elif defined(NUM_ACCESSES_2)
            func(&a[index_1d], &c[index_1d], size[0]);
#elif defined(NUM_ACCESSES_3)
            func(&a[index_1d], &b[index_1d], &c[index_1d], size[0]);
#endif
#else
#if defined(NUM_ACCESSES_1)
            func(c[index_1d]);
#elif defined(NUM_ACCESSES_2)
            func(a[index_1d], c[index_1d]);
#elif defined(NUM_ACCESSES_3)
            func(a[index_1d], b[index_1d], c[index_1d]);
#endif
#endif
        }
    }
}

template <typename FuncT, SizeT Dimension, typename ValueT>
CUDA_KERNEL
auto KernelImplementation(FuncT func, [[maybe_unused]] ValueT* a, [[maybe_unused]] ValueT* b, ValueT* c, SizeArray<Dimension> size, [[maybe_unused]] const std::int32_t* index = nullptr) -> std::enable_if_t<(Dimension == 2), void>
{
    using namespace ::fw::math;

    const SizeT x = blockIdx.x * blockDim.x + threadIdx.x;
    const SizeT y = blockIdx.y * blockDim.y + threadIdx.y;

#if defined(SOA_LAYOUT)
    const SizeT n = size.ReduceMul();
#endif

    if (x < size[0] && y < size[1])
    {
        const SizeT index_1d = y * size[0] + x;

#if defined(SOA_LAYOUT)
#if defined(NUM_ACCESSES_1)
        func(&c[index_1d], n);
#elif defined(NUM_ACCESSES_2)
        func(&a[index_1d], &c[index_1d], n);
#elif defined(NUM_ACCESSES_3)
        func(&a[index_1d], &b[index_1d], &c[index_1d], n);
#endif
#else
#if defined(NUM_ACCESSES_1)
        func(c[index_1d]);
#elif defined(NUM_ACCESSES_2)
        func(a[index_1d], c[index_1d]);
#elif defined(NUM_ACCESSES_3)
        func(a[index_1d], b[index_1d], c[index_1d]);
#endif
#endif
    }
}

template <typename FuncT, SizeT Dimension, typename ValueT>
CUDA_KERNEL
auto KernelImplementation(FuncT func, [[maybe_unused]] ValueT* a, [[maybe_unused]] ValueT* b, ValueT* c, SizeArray<Dimension> size, [[maybe_unused]] const std::int32_t* index = nullptr) -> std::enable_if_t<(Dimension == 3), void>
{
    using namespace ::fw::math;

    const SizeT x = blockIdx.x * blockDim.x + threadIdx.x;
    const SizeT y = blockIdx.y * blockDim.y + threadIdx.y;
    const SizeT z = blockIdx.z * blockDim.z + threadIdx.z;

#if defined(SOA_LAYOUT)
    const SizeT n = size.ReduceMul();
#endif

    if (x < size[0] && y < size[1] && z < size[2])
    {
        const SizeT index_1d = (z * size[1] + y) * size[0] + x;

#if defined(SOA_LAYOUT)
#if defined(NUM_ACCESSES_1)
        func(&c[index_1d], n);
#elif defined(NUM_ACCESSES_2)
        func(&a[index_1d], &c[index_1d], n);
#elif defined(NUM_ACCESSES_3)
        func(&a[index_1d], &b[index_1d], &c[index_1d], n);
#endif
#else
#if defined(NUM_ACCESSES_1)
        func(c[index_1d]);
#elif defined(NUM_ACCESSES_2)
        func(a[index_1d], c[index_1d]);
#elif defined(NUM_ACCESSES_3)
        func(a[index_1d], b[index_1d], c[index_1d]);
#endif
#endif
    } 
}

template <typename FuncT, SizeT Dimension, typename ValueT>
auto Kernel(FuncT func, ValueT* a, ValueT* b, ValueT* c, const SizeArray<Dimension>& size, const std::int32_t* index = nullptr) -> void
{
    const dim3 block{128, 1, 1};
    const dim3 grid = GetGridSize(size, block);

    KernelImplementation<<<grid, block>>>(func, a, b, c, size, index);
}

#else // __CUDACC__

template <typename FuncT, SizeT Dimension, typename ValueT>
auto KernelImplementation(FuncT func, [[maybe_unused]] ValueT* a, [[maybe_unused]] ValueT* b, ValueT* c, SizeArray<Dimension> size, [[maybe_unused]] const std::int32_t* index = nullptr)
    -> std::enable_if_t<Dimension == 1, void>
{
    using namespace ::fw::math;

    const SizeT n = size.ReduceMul();

#if defined(DIFFUSION)
    if (Dimension == 1 && index)
    {
        #pragma omp simd
        for (SizeT i = 0; i < size[0]; ++i)
        {
            const SizeT index_1d = index[i];

#if defined(SOA_LAYOUT)
#if defined(NUM_ACCESSES_1)
            func(&c[index_1d], n);
#elif defined(NUM_ACCESSES_2)
            func(&a[index_1d], &c[index_1d], n);
#elif defined(NUM_ACCESSES_3)
            func(&a[index_1d], &b[index_1d], &c[index_1d], n);
#endif
#else
#if defined(NUM_ACCESSES_1)
            func(c[index_1d]);
#elif defined(NUM_ACCESSES_2)
            func(a[index_1d], c[index_1d]);
#elif defined(NUM_ACCESSES_3)
            func(a[index_1d], b[index_1d], c[index_1d]);
#endif
#endif
        }
    }
    else
#endif
    {
        #pragma omp simd
        for (SizeT i = 0; i < size[0]; ++i)
        {
            const SizeT index_1d = i;

#if defined(SOA_LAYOUT)
#if defined(NUM_ACCESSES_1)
            func(&c[index_1d], n);
#elif defined(NUM_ACCESSES_2)
            func(&a[index_1d], &c[index_1d], n);
#elif defined(NUM_ACCESSES_3)
            func(&a[index_1d], &b[index_1d], &c[index_1d], n);
#endif
#else
#if defined(NUM_ACCESSES_1)
            func(c[index_1d]);
#elif defined(NUM_ACCESSES_2)
            func(a[index_1d], c[index_1d]);
#elif defined(NUM_ACCESSES_3)
            func(a[index_1d], b[index_1d], c[index_1d]);
#endif
#endif
        }
    }
}

template <typename FuncT, SizeT Dimension, typename ValueT>
auto KernelImplementation(FuncT func, [[maybe_unused]] ValueT* a, [[maybe_unused]] ValueT* b, ValueT* c, SizeArray<Dimension> size, [[maybe_unused]] const std::int32_t* index = nullptr)
    -> std::enable_if_t<Dimension == 2, void>
{
    using namespace ::fw::math;

    const SizeT n = size.ReduceMul();

    for (SizeT j = 0; j < size[1]; ++j)
    {
        #pragma omp simd
        for (SizeT i = 0; i < size[0]; ++i)
        {
            const SizeT index_1d = j * size[0] + i;

#if defined(SOA_LAYOUT)
#if defined(NUM_ACCESSES_1)
            func(&c[index_1d], n);
#elif defined(NUM_ACCESSES_2)
            func(&a[index_1d], &c[index_1d], n);
#elif defined(NUM_ACCESSES_3)
            func(&a[index_1d], &b[index_1d], &c[index_1d], n);
#endif
#else
#if defined(NUM_ACCESSES_1)
            func(c[index_1d]);
#elif defined(NUM_ACCESSES_2)
            func(a[index_1d], c[index_1d]);
#elif defined(NUM_ACCESSES_3)
            func(a[index_1d], b[index_1d], c[index_1d]);
#endif
#endif
        }
    }
}

template <typename FuncT, SizeT Dimension, typename ValueT>
auto KernelImplementation(FuncT func, [[maybe_unused]] ValueT* a, [[maybe_unused]] ValueT* b, ValueT* c, SizeArray<Dimension> size, [[maybe_unused]] const std::int32_t* index = nullptr)
    -> std::enable_if_t<Dimension == 3, void>
{
    using namespace ::fw::math;

    const SizeT n = size.ReduceMul();

    for (SizeT k = 0; k < size[2]; ++k)
    {
        for (SizeT j = 0; j < size[1]; ++j)
        {
            #pragma omp simd
            for (SizeT i = 0; i < size[0]; ++i)
            {
                const SizeT index_1d = (k * size[1] + j) * size[0] + i;

#if defined(SOA_LAYOUT)
#if defined(NUM_ACCESSES_1)
                func(&c[index_1d], n);
#elif defined(NUM_ACCESSES_2)
                func(&a[index_1d], &c[index_1d], n);
#elif defined(NUM_ACCESSES_3)
                func(&a[index_1d], &b[index_1d], &c[index_1d], n);
#endif
#else
#if defined(NUM_ACCESSES_1)
                func(c[index_1d]);
#elif defined(NUM_ACCESSES_2)
                func(a[index_1d], c[index_1d]);
#elif defined(NUM_ACCESSES_3)
                func(a[index_1d], b[index_1d], c[index_1d]);
#endif
#endif
            }
        }
    }
}

template <typename FuncT, SizeT Dimension, typename ValueT>
auto Kernel(FuncT func, ValueT* a, ValueT* b, ValueT* c, const SizeArray<Dimension>& size, const std::int32_t* index = nullptr) -> void
{
    KernelImplementation(func, a, b, c, size, index);
}
#endif

#if defined(SOA_LAYOUT)
template <SizeT Dimension>
int benchmark(int argc, char** argv, const SizeArray<Dimension>& size)
{
    static_assert(std::is_same<ElementT, ::fw::dataTypes::Vec<RealT, 3>>::value);

    const SizeT n = size.ReduceMul();
#if defined(CHECK_RESULTS)
    const SizeT WARMUP = 0;
    const SizeT MEASUREMENT = 1;
#else
    const SizeT WARMUP = (n > 1048576 ? 10 : 100);
    const SizeT MEASUREMENT = (n > 1048576 ? 20 : 1000);
#endif

    RealT* data_1 = reinterpret_cast<RealT*>(_mm_malloc(n * 3 * sizeof(RealT), ::fw::simd::alignment));
    RealT* data_2 = reinterpret_cast<RealT*>(_mm_malloc(n * 3 * sizeof(RealT), ::fw::simd::alignment));
    RealT* data_3 = reinterpret_cast<RealT*>(_mm_malloc(n * 3 * sizeof(RealT), ::fw::simd::alignment));
    
    // Field initialization.
    srand48(1);
    for (SizeT i = 0; i < n; ++i)
    {
        const RealT value = static_cast<RealT>((2.0 * drand48() -1.0) * SPREAD + OFFSET);
        data_1[i + 0 * n] = value;
        data_1[i + 1 * n] = value;
        data_1[i + 2 * n] = value;
    }
    for (SizeT i = 0; i < n; ++i)
    {
        const RealT value = static_cast<RealT>((2.0 * drand48() -1.0) * SPREAD + OFFSET);
        data_2[i + 0 * n] = value;
        data_2[i + 1 * n] = value;
        data_2[i + 2 * n] = value;
    }

#if defined(__CUDACC__)
    // Create device buffer and copy in the host data.
    RealT* d_data_1;
    RealT* d_data_2;
    RealT* d_data_3;

    cudaMalloc((void**)&d_data_1, n * 3 * sizeof(RealT));
    cudaMalloc((void**)&d_data_2, n * 3 * sizeof(RealT));
    cudaMalloc((void**)&d_data_3, n * 3 * sizeof(RealT));

    cudaMemcpy((void*)d_data_1, (const void*)data_1, n * 3 * sizeof(RealT), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_data_2, (const void*)data_2, n * 3 * sizeof(RealT), cudaMemcpyHostToDevice);

    RealT* field_1 = d_data_1;
    RealT* field_2 = d_data_2;
    RealT* field_3 = d_data_3;
#else
    RealT* field_1 = data_1;
    RealT* field_2 = data_2;
    RealT* field_3 = data_3;
#endif

#if defined(DIFFUSION)
    std::vector<std::int32_t> h_index(n);
    double diffusion = 0.0;
    if (const char* env_string = std::getenv("DIFFUSION_FACTOR"))
    {
        diffusion = std::atof(env_string);
    }

    srand48(1);
    for (SizeT i = 0; i < n; ++i)
    {
        h_index[i] = static_cast<std::int32_t>(i + (n / 2) * drand48() * diffusion) % n;
    }

#if defined(__CUDACC__)
    std::int32_t* d_index = nullptr;
    cudaMalloc((void**)&d_index, n * sizeof(std::int32_t));
    cudaMemcpy((void*)d_index, (const void*)h_index.data(), n * sizeof(std::int32_t), cudaMemcpyHostToDevice);
    const std::int32_t* index = d_index;
#else
    const std::int32_t* index = h_index.data();
#endif
#else // DIFFUSION
    const std::int32_t* index = nullptr;
#endif

#if defined(SAXPY_KERNEL)
#if defined(ELEMENT_ACCESS)
    auto kernel_1 = [] CUDA_DEVICE_VERSION (const auto* a, auto* b, const SizeT n) -> void { using namespace ::fw::math; b[1 * n] = a[0 * n] * static_cast<TypeX>(3.2) + b[1 * n]; };
    auto kernel_2 = [] CUDA_DEVICE_VERSION (const auto* a, auto* b, const SizeT n) -> void { using namespace ::fw::math; b[1 * n] = a[0 * n] * static_cast<TypeX>(3.2) + b[1 * n]; };
#else
    auto kernel_1 = [] CUDA_DEVICE_VERSION (const auto* a, auto* b, const SizeT n) -> void 
    { 
        using namespace ::fw::math; 

        b[0 * n] = a[0 * n] * static_cast<TypeX>(3.2) + b[0 * n];
        b[1 * n] = a[1 * n] * static_cast<TypeY>(3.2) + b[1 * n];
        b[2 * n] = a[2 * n] * static_cast<TypeZ>(3.2) + b[2 * n];
    };
    auto kernel_2 = kernel_1;
#endif
#elif defined(NUM_ACCESSES_1)
    auto kernel_1 = [] CUDA_DEVICE_VERSION (auto* a, const SizeT n) -> void 
    { 
        using namespace ::fw::math; 

        a[0 * n] = a[0 * n] * static_cast<TypeX>(3.2);
        a[1 * n] = a[1 * n] * static_cast<TypeY>(3.2);
        a[2 * n] = a[2 * n] * static_cast<TypeZ>(3.2);
    };
    auto kernel_2 = kernel_1;
#elif defined(NUM_ACCESSES_3)
    auto kernel_1 = [] CUDA_DEVICE_VERSION (const auto* a, const auto* b, auto* c, const SizeT n) -> void 
    { 
        using namespace ::fw::math; 

        c[0 * n] = a[0 * n] * static_cast<TypeX>(3.2) + b[0 * n];
        c[1 * n] = a[1 * n] * static_cast<TypeY>(3.2) + b[1 * n];
        c[2 * n] = a[2 * n] * static_cast<TypeZ>(3.2) + b[2 * n];
    };
    auto kernel_2 = kernel_1;
#else
#if defined(ELEMENT_ACCESS)
    auto kernel_1 = [] CUDA_DEVICE_VERSION (const auto* a, const auto* b, auto* c, const SizeT n) -> void { using namespace ::fw::math; c[2 * n] = Exp(a[0 * n] + b[1 * n]); };
    auto kernel_2 = [] CUDA_DEVICE_VERSION (const auto* a, const auto* b, auto* c, const SizeT n) -> void { using namespace ::fw::math; c[0 * n] = Log(a[2 * n]) - b[1 * n]; };
#else
    auto kernel_1 = [] CUDA_DEVICE_VERSION (const auto* a, const auto* b, auto* c, const SizeT n) -> void 
    { 
        using namespace ::fw::math; 
        
        c[0 * n] = Exp(a[0 * n] + b[0 * n]);
        c[1 * n] = Exp(a[1 * n] + b[1 * n]);
        c[2 * n] = Exp(a[2 * n] + b[2 * n]);
    };
    auto kernel_2 = [] CUDA_DEVICE_VERSION (const auto* a, const auto* b, auto* c, const SizeT n) -> void
    { 
        using namespace ::fw::math; 
        
        c[0 * n] = Log(a[0 * n]) - b[0 * n];
        c[1 * n] = Log(a[1 * n]) - b[1 * n];
        c[2 * n] = Log(a[2 * n]) - b[2 * n];
    };
#endif
#endif

    for (SizeT i = 0; i < WARMUP; ++i)
    {
        Kernel(kernel_1, field_1, field_2, field_3, size, index);
        Kernel(kernel_2, field_3, field_2, field_1, size, index);
    }

#if defined(__CUDACC__)
    cudaDeviceSynchronize();
#endif

    double start_time = omp_get_wtime();

    for (SizeT i = 0; i < MEASUREMENT; ++i)
    {
        Kernel(kernel_1, field_1, field_2, field_3, size, index);
        Kernel(kernel_2, field_3, field_2, field_1, size, index);
    }

#if defined(__CUDACC__)
    cudaDeviceSynchronize();
#endif

    double stop_time = omp_get_wtime();

#if defined(__CUDACC__)
    cudaError_t error = cudaGetLastError();
    std::cout << "# CUDA: " << cudaGetErrorString(error) << std::endl;
#endif

#if defined(DIFFUSION)
    //std::cout << "# elapsed time in ms" << "\t" << "diffusion factor" << std::endl;
    //std::cout << (stop_time - start_time) * 1.0E3 << "\t" << diffusion << std::endl;
    std::cout << "# time per loop iteration in ns" << "\t" << "diffusion factor" << std::endl;
    std::cout << (stop_time - start_time) * 1.0E9 / (2 * MEASUREMENT * n) << "\t" << diffusion << std::endl;
#else
    //std::cout << "# elapsed time in ms" << std::endl;
    //std::cout << (stop_time - start_time) * 1.0E3 << std::endl;
    std::cout << "#time per loop iteration in ns" << std::endl;
    std::cout << (stop_time - start_time) * 1.0E9 / (2 * MEASUREMENT * n) << std::endl;
#endif
    
    _mm_free(data_1);
    _mm_free(data_2);
    _mm_free(data_3);

    return 0;
}

#else // SOA_LAYOUT

template <SizeT Dimension>
int benchmark(int argc, char** argv, const SizeArray<Dimension>& size)
{
    const SizeT n = size.ReduceMul();
#if defined(CHECK_RESULTS)
    const SizeT WARMUP = 0;
    const SizeT MEASUREMENT = 1;
#else
    const SizeT WARMUP = (n > 1048576 ? 10 : 100);
    const SizeT MEASUREMENT = (n > 1048576 ? 20 : 1000);
#endif

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

#if defined(DIFFUSION)
    std::vector<std::int32_t> h_index(n);
    double diffusion = 0.0;
    if (const char* env_string = std::getenv("DIFFUSION_FACTOR"))
    {
        diffusion = std::atof(env_string);
    }

    srand48(1);
    for (SizeT i = 0; i < n; ++i)
    {
        h_index[i] = static_cast<std::int32_t>(i + (n / 2) * drand48() * diffusion) % n;
    }

#if defined(__CUDACC__)
    std::int32_t* d_index = nullptr;
    cudaMalloc((void**)&d_index, n * sizeof(std::int32_t));
    cudaMemcpy((void*)d_index, (const void*)h_index.data(), n * sizeof(std::int32_t), cudaMemcpyHostToDevice);
    const std::int32_t* index = d_index;
#else
    const std::int32_t* index = h_index.data();
#endif
#else // DIFFUSION
    const std::int32_t* index = nullptr;
#endif

#if defined(SAXPY_KERNEL)
#if defined(ELEMENT_ACCESS)
    auto kernel_1 = [] CUDA_DEVICE_VERSION (const auto& a, auto& b) -> void { using namespace ::fw::math; b.y = a.x * static_cast<TypeX>(3.2) + b.y; };
    auto kernel_2 = [] CUDA_DEVICE_VERSION (const auto& a, auto& b) -> void { using namespace ::fw::math; b.y = a.x * static_cast<TypeX>(3.2) + b.y; };
#else
    auto kernel_1 = [] CUDA_DEVICE_VERSION (const auto& a, auto& b) -> void { using namespace ::fw::math; b = a * 3.2 + b; };
    auto kernel_2 = kernel_1;
#endif
#elif defined(NUM_ACCESSES_1)
    auto kernel_1 = [] CUDA_DEVICE_VERSION (auto& a) -> void { using namespace ::fw::math; a = a * 3.2; };
    auto kernel_2 = kernel_1;
#elif defined(NUM_ACCESSES_3)
    auto kernel_1 = [] CUDA_DEVICE_VERSION (const auto& a, const auto& b, auto& c) -> void { using namespace ::fw::math; c = a * 3.2 + b; };
    auto kernel_2 = kernel_1;
#else
#if defined(ELEMENT_ACCESS)
    auto kernel_1 = [] CUDA_DEVICE_VERSION (const auto& a, const auto& b, auto& c) -> void { using namespace ::fw::math; c.z = Exp(a.x + b.y); };
    auto kernel_2 = [] CUDA_DEVICE_VERSION (const auto& a, const auto& b, auto& c) -> void { using namespace ::fw::math; c.x = Log(a.z) - b.y; };
#else
    auto kernel_1 = [] CUDA_DEVICE_VERSION (const auto& a, const auto& b, auto& c) -> void { using namespace ::fw::math; c = Exp(a + b); };
    auto kernel_2 = [] CUDA_DEVICE_VERSION (const auto& a, const auto& b, auto& c) -> void { using namespace ::fw::math; c = Log(a) - b; };
#endif
#endif

    for (SizeT i = 0; i < WARMUP; ++i)
    {
        Kernel(kernel_1, field_1, field_2, field_3, size, index);
        Kernel(kernel_2, field_3, field_2, field_1, size, index);
    }

#if defined(__CUDACC__)
    cudaDeviceSynchronize();
#endif

    double start_time = omp_get_wtime();

    for (SizeT i = 0; i < MEASUREMENT; ++i)
    {
        Kernel(kernel_1, field_1, field_2, field_3, size, index);
        Kernel(kernel_2, field_3, field_2, field_1, size, index);
    }

#if defined(__CUDACC__)
    cudaDeviceSynchronize();
#endif

    double stop_time = omp_get_wtime();

#if defined(__CUDACC__)
    cudaError_t error = cudaGetLastError();
    std::cout << "# CUDA: " << cudaGetErrorString(error) << std::endl;
#endif

#if defined(DIFFUSION)
    //std::cout << "# elapsed time in ms" << "\t" << "diffusion factor" << std::endl;
    //std::cout << (stop_time - start_time) * 1.0E3 << "\t" << diffusion << std::endl;
    std::cout << "# time per loop iteration in ns" << "\t" << "diffusion factor" << std::endl;
    std::cout << (stop_time - start_time) * 1.0E9 / (2 * MEASUREMENT * n) << "\t" << diffusion << std::endl;
#else
    //std::cout << "# elapsed time in ms" << std::endl;
    //std::cout << (stop_time - start_time) * 1.0E3 << std::endl;
    std::cout << "#time per loop iteration in ns" << std::endl;
    std::cout << (stop_time - start_time) * 1.0E9 / (2 * MEASUREMENT * n) << std::endl;
#endif
    
    _mm_free(data_1);
    _mm_free(data_2);
    _mm_free(data_3);

    return 0;
}
#endif

template int benchmark<1>(int, char**, const SizeArray<1>&);
template int benchmark<2>(int, char**, const SizeArray<2>&);
template int benchmark<3>(int, char**, const SizeArray<3>&);
