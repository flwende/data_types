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

#include <vec/SimdVec.hpp>
#include <auxiliary/Function.hpp>
#include <auxiliary/Pack.hpp>

using ::XXX_NAMESPACE::memory::DataLayout;
using ::XXX_NAMESPACE::dataTypes::internal::Get;
using ::XXX_NAMESPACE::dataTypes::SimdVec;
using ::XXX_NAMESPACE::dataTypes::SimdVecRef;

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
auto KernelImplementation(FuncT func, Container a, Container b, Container c, [[maybe_unused]] const std::int32_t* index = nullptr) -> std::enable_if_t<(Container::TParam_Dimension == 1), void>
{
    using namespace ::fw::math;

    const SizeT x = blockIdx.x * blockDim.x + threadIdx.x;

#if defined(DIFFUSION)
    if (index)
    {
        if (x < a.Size(0))
        {
            const std::int32_t i = index[x];

            func(a[i], b[i], c[i]);
        }
    }
    else
#endif
    {
        if (x < a.Size(0))
        {
            func(a[x], b[x], c[x]);
        }
    }
}

template <typename FuncT, typename Container>
CUDA_KERNEL
auto KernelImplementation(FuncT func, Container a, Container b, Container c, [[maybe_unused]] const std::int32_t* index = nullptr) -> std::enable_if_t<(Container::TParam_Dimension == 2), void>
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
auto KernelImplementation(FuncT func, Container a, Container b, Container c, [[maybe_unused]] const std::int32_t* index = nullptr) -> std::enable_if_t<(Container::TParam_Dimension == 3), void>
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
void Kernel(FuncT func, Field& a, Field& b, Field& c, const std::int32_t* index = nullptr)
{
    const dim3 block{128, 1, 1};
    const dim3 grid = GetGridSize(a.Size(), block);

    KernelImplementation<<<grid, block>>>(func, a.DeviceData(), b.DeviceData(), c.DeviceData(), index);
}

#else // __CUDACC__

template <typename FuncT, typename Container>
auto KernelImplementation(FuncT func, const Container& a, const Container& b, Container& c, [[maybe_unused]] const std::int32_t* index = nullptr) 
    -> std::enable_if_t<(Container::TParam_Dimension == 1 && Container::TParam_Layout != DataLayout::AoSoA), void>
{
    using namespace ::fw::math;
       
#if defined(DIFFUSION)
    if (index)
    {
        #pragma omp simd
        for (SizeT x = 0; x < a.Size(0); ++x)
        {
            const std::int32_t i = index[x];
            func(a[i], b[i], c[i]);
        }
    }
    else
#endif
    {
        #pragma omp simd
        for (SizeT x = 0; x < a.Size(0); ++x)
        {
            func(a[x], b[x], c[x]);
        }
    }
}

template <typename FuncT, typename Container>
auto KernelImplementation(FuncT func, const Container& a, const Container& b, Container& c, [[maybe_unused]] const std::int32_t* index = nullptr) 
    -> std::enable_if_t<(Container::TParam_Dimension == 1 && Container::TParam_Layout == DataLayout::AoSoA), void>
{
    using namespace ::fw::math;

#if defined(USE_HIGHLEVEL_VECTOR_REF)
    constexpr SizeT CHUNK_SIZE = Container::GetInnerArraySize();

    for (SizeT x = 0; x < a.Size(0); x += CHUNK_SIZE)
    {
        const SizeT i_max = std::min(CHUNK_SIZE, a.Size(0) - x);
        const SimdVecRef ta(a.At(x), i_max);
        const SimdVecRef tb(b.At(x), i_max);
        SimdVecRef tc(c.At(x), i_max);

        #pragma omp simd
        for (SizeT i = 0; i < i_max; ++i)
        {
            func(ta[i], tb[i], tc[i]);
        }
    }
#else 
    #pragma omp simd
    for (SizeT x = 0; x < a.Size(0); ++x)
    {    
        func(a[x], b[x], c[x]);
    }
#endif
}

template <typename FuncT, typename Container>
auto KernelImplementation(FuncT func, const Container& a, const Container& b, Container& c, [[maybe_unused]] const std::int32_t* index = nullptr) 
    -> std::enable_if_t<(Container::TParam_Dimension == 2 && Container::TParam_Layout != DataLayout::AoSoA), void>
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
auto KernelImplementation(FuncT func, const Container& a, const Container& b, Container& c, [[maybe_unused]] const std::int32_t* index = nullptr) 
    -> std::enable_if_t<(Container::TParam_Dimension == 2 && Container::TParam_Layout == DataLayout::AoSoA), void>
{
    using namespace ::fw::math;
    
    for (SizeT y = 0; y < a.Size(1); ++y)
    {
#if defined(USE_HIGHLEVEL_VECTOR_REF)
        constexpr SizeT CHUNK_SIZE = Container::GetInnerArraySize();

        for (SizeT x = 0; x < a.Size(0); x += CHUNK_SIZE)
        {
            const SizeT i_max = std::min(CHUNK_SIZE, a.Size(0) - x);
            const SimdVecRef ta(a[y].At(x), i_max);
            const SimdVecRef tb(b[y].At(x), i_max);
            SimdVecRef tc(c[y].At(x), i_max);

            #pragma omp simd
            for (SizeT i = 0; i < i_max; ++i)
            {
                func(ta[i], tb[i], tc[i]);
            }
        }
#else       
        #pragma omp simd
        for (SizeT x = 0; x < a.Size(0); ++x)
        {    
            func(a[y][x], b[y][x], c[y][x]);
        }
#endif
    }

}

template <typename FuncT, typename Container>
auto KernelImplementation(FuncT func, const Container& a, const Container& b, Container& c, [[maybe_unused]] const std::int32_t* index = nullptr) 
    -> std::enable_if_t<(Container::TParam_Dimension == 3 && Container::TParam_Layout != DataLayout::AoSoA), void>
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

template <typename FuncT, typename Container>
auto KernelImplementation(FuncT func, const Container& a, const Container& b, Container& c, [[maybe_unused]] const std::int32_t* index = nullptr) 
    -> std::enable_if_t<(Container::TParam_Dimension == 3 && Container::TParam_Layout == DataLayout::AoSoA), void>
{
    using namespace ::fw::math;

    for (SizeT z = 0; z < a.Size(2); ++z)
    {
        for (SizeT y = 0; y < a.Size(1); ++y)
        {
#if defined(USE_HIGHLEVEL_VECTOR_REF)
            constexpr SizeT CHUNK_SIZE = Container::GetInnerArraySize();

            for (SizeT x = 0; x < a.Size(0); x += CHUNK_SIZE)
            {
                const SizeT i_max = std::min(CHUNK_SIZE, a.Size(0) - x);
                const SimdVecRef ta(a[z][y].At(x), i_max);
                const SimdVecRef tb(b[z][y].At(x), i_max);
                SimdVecRef tc(c[z][y].At(x), i_max);

                #pragma omp simd
                for (SizeT i = 0; i < i_max; ++i)
                {
                    func(ta[i], tb[i], tc[i]);
                }
            }
#else
            #pragma omp simd
            for (SizeT x = 0; x < a.Size(0); ++x)
            {
                func(a[z][y][x], b[z][y][x], c[z][y][x]);
            }
#endif
        }
    } 
}

template <typename FuncT, typename Field>
void Kernel(FuncT func, const Field& a, const Field& b, Field& c, const std::int32_t* index = nullptr)
{
    KernelImplementation(func, a, b, c, index);
}
#endif

template <SizeT Dimension>
int benchmark(int argc, char** argv, const SizeArray<Dimension>& size)
{
    using namespace ::fw::math;

    const SizeT n = size.ReduceMul();
    Field<ElementT, Dimension> field_1(size);
    Field<ElementT, Dimension> field_2(size);
    Field<ElementT, Dimension> field_3(size, true);

    // Field initialization.
    srand48(1);
    field_1.Set([] (const auto I) { return static_cast<RealT>((2.0 * drand48() -1.0) * SPREAD + OFFSET); });
    field_2.Set([] (const auto I) { return static_cast<RealT>((2.0 * drand48() -1.0) * SPREAD + OFFSET); });

#if defined(__CUDACC__)
    // Create device buffer and copy in the host data.
    field_1.CopyHostToDevice();
    field_2.CopyHostToDevice();
    field_3.CopyHostToDevice();
#endif

#if defined(CHECK_RESULTS)
    std::vector<ElementT> field_1_copy = field_1.Get();
    std::vector<ElementT> field_2_copy = field_2.Get();
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
    auto kernel_1 = [] CUDA_DEVICE_VERSION (const auto& a, const auto& b, auto&& c) -> void { using namespace ::fw::math; c.z =  a.x * 3.2 + b.y; };
    auto kernel_1 = [] CUDA_DEVICE_VERSION (const auto& a, const auto& b, auto&& c) -> void { using namespace ::fw::math; c.x =  a.z * 3.2 + b.y; };
#else
    auto kernel_1 = [] CUDA_DEVICE_VERSION (const auto& a, const auto& b, auto&& c) -> void { using namespace ::fw::math; c =  a * 3.2 + b; };
    auto kernel_2 = kernel_1;
#endif
#elif defined(COMPUTE_KERNEL)
#if defined(ELEMENT_ACCESS)
    auto kernel_1 = [] CUDA_DEVICE_VERSION (const auto& a, const auto& b, auto&& c) -> void { 
        using namespace ::fw::math; 
        const RealT zero = 0;
        RealT tmp = c.z;
        for (SizeT i = 0; i < 10; ++i) tmp += Exp(Min(zero, tmp + a.x) + b.y);
        c.z = tmp;
    };
    auto kernel_2 = [] CUDA_DEVICE_VERSION (const auto& a, const auto& b, auto&& c) -> void { 
        using namespace ::fw::math; 
        const RealT one = 1;
        const RealT two = 2;
        RealT tmp = c.x;
        for (SizeT i = 0; i < 10; ++i) tmp += Log(Max(one, static_cast<RealT>(Min(two, tmp + a.z))) - b.y);
        c.x = tmp;
    };
#else
    auto kernel_1 = [] CUDA_DEVICE_VERSION (const auto& a, const auto& b, auto&& c) -> void { 
        using namespace ::fw::math; 
        const ElementT zero(0);
        ElementT tmp = c;
        for (SizeT i = 0; i < 10; ++i) tmp += Exp(Min(zero, tmp + a) + b);
        c = tmp;
    };
    auto kernel_2 = [] CUDA_DEVICE_VERSION (const auto& a, const auto& b, auto&& c) -> void { 
        using namespace ::fw::math; 
        const ElementT one(1);
        const ElementT two(2);
        ElementT tmp = c;
        for (SizeT i = 0; i < 10; ++i) tmp += Log(Max(one, Min(two, tmp + a)) - b);
        c = tmp;
    };
#endif
#else
#if defined(ELEMENT_ACCESS)
    auto kernel_1 = [] CUDA_DEVICE_VERSION (const auto& a, const auto& b, auto&& c) -> void { using namespace ::fw::math; c.z = Exp(a.x + b.y); };
    auto kernel_2 = [] CUDA_DEVICE_VERSION (const auto& a, const auto& b, auto&& c) -> void { using namespace ::fw::math; c.x = Log(a.z) - b.y; };
#else
    auto kernel_1 = [] CUDA_DEVICE_VERSION (const auto& a, const auto& b, auto&& c) -> void { using namespace ::fw::math; c = Exp(a + b); };
    auto kernel_2 = [] CUDA_DEVICE_VERSION (const auto& a, const auto& b, auto&& c) -> void { using namespace ::fw::math; c = Log(a) - b; };
#endif
#endif

    for (SizeT i = 0; i < WARMUP; ++i)
    {
        Kernel(kernel_1, field_1, field_2, field_3, index);
        Kernel(kernel_2, field_3, field_2, field_1, index);
    }

#if defined(__CUDACC__)
    cudaDeviceSynchronize();
#endif

    double start_time = omp_get_wtime();

    for (SizeT i = 0; i < MEASUREMENT; ++i)
    {
        Kernel(kernel_1, field_1, field_2, field_3, index);
        Kernel(kernel_2, field_3, field_2, field_1, index);
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
    std::cout << "# elapsed time in ms" << "\t" << "diffusion factor" << std::endl;
    std::cout << (stop_time - start_time) * 1.0E3 << "\t" << diffusion std::endl;
#else
    std::cout << "# elapsed time in ms" << std::endl;
    std::cout << (stop_time - start_time) * 1.0E3 << diffusion std::endl;
#endif

#if defined(CHECK_RESULTS)
    {
        using namespace ::fw::math;

        std::vector<ElementT> reference(n);
        std::vector<ElementT> field_3_result = field_3.Get(true);
    
        ElementT deviation;

        for (SizeT i = 0; i < n; ++i)
        {
#if defined(ELEMENT_ACCESS)
            reference[i].z = Exp(field_1_copy[i].x + field_2_copy[i].y);
#else
            reference[i] = Exp(field_1_copy[i] + field_2_copy[i]);
#endif
            ElementT rel_error = (reference[i] - field_3_result[i]) / Max(ElementT(1.0E-9), reference[i]);
            deviation = Max(deviation, Abs(rel_error));

#if defined(ELEMENT_ACCESS)
            reference[i] = field_1_copy[i];
#endif
        }

        std::vector<ElementT> field_1_result = field_1.Get(true);

        for (SizeT i = 0; i < n; ++i)
        {
#if defined(ELEMENT_ACCESS)
            reference[i].x = Log(field_3_result[i].z) - field_2_copy[i].y;
#else
            reference[i] = Log(field_3_result[i]) - field_2_copy[i];
#endif
            ElementT rel_error = (reference[i] - field_1_result[i]) / Max(ElementT(1.0E-9), reference[i]);
            deviation = Max(deviation, Abs(rel_error));
        }

        std::cout << "# deviation:" << deviation << std::endl;
    }
#endif
    
    return 0;
}

template int benchmark<1>(int, char**, const SizeArray<1>&);
template int benchmark<2>(int, char**, const SizeArray<2>&);
template int benchmark<3>(int, char**, const SizeArray<3>&);
