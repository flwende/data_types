// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#include <omp.h>
#include <benchmark.hpp>
#include <vector>

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
//constexpr std::size_t WARMUP = 10;
//constexpr std::size_t MEASUREMENT = 20;
constexpr std::size_t WARMUP = 0;
constexpr std::size_t MEASUREMENT = 1;
#endif

//#define FW_USE_HIGHLEVEL_VECTOR_REF
//#define FW_USE_HIGHLEVEL_VECTOR

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

template <typename FuncT, typename Field, typename Filter_A, typename Filter_B, typename Filter_C>
auto Kernel(FuncT func, Field& a, Field& b, Field& c, Filter_A, Filter_B, Filter_C) -> void
{
    const dim3 block{128, 1, 1};
    const dim3 grid = GetGridSize(a.Size(), block);

    KernelImplementation<<<grid, block>>>(func, a.DeviceData(), b.DeviceData(), c.DeviceData());
}
#else
template <typename FuncT, typename Container, typename Filter_A, typename Filter_B, typename Filter_C>
auto KernelImplementation(FuncT func, const Container& a, const Container& b, Container& c, Filter_A fa, Filter_B fb, Filter_C fc) 
    -> std::enable_if_t<(Container::TParam_Dimension == 1 && Container::TParam_Layout != DataLayout::AoSoA), void>
{
    using namespace ::fw::math;

    constexpr SizeT CHUNK_SIZE = 32;
    [[maybe_unused]] SimdVec<ElementT, CHUNK_SIZE, DataLayout::SoA> ta, tb, tc;
    
#if defined(FW_USE_HIGHLEVEL_VECTOR)
    for (SizeT x = 0; x < a.Size(0); x += CHUNK_SIZE)
    {
        const SizeT i_max = std::min(CHUNK_SIZE, a.Size(0) - x);

        ta.Load(a.At(x), fa, i_max);
        tb.Load(b.At(x), fb, i_max);

        #pragma omp simd
        for (SizeT i = 0; i < CHUNK_SIZE; ++i)
        {
            func(ta[i], tb[i], tc[i]);
        }
        
        tc.Store(c.At(x), fc, i_max);
    }
#else        
    #pragma omp simd
    for (SizeT x = 0; x < a.Size(0); ++x)
    {
        func(a[x], b[x], c[x]);
    }
#endif
}

template <typename FuncT, typename Container, typename Filter_A, typename Filter_B, typename Filter_C>
auto KernelImplementation(FuncT func, const Container& a, const Container& b, Container& c, Filter_A fa, Filter_B fb, Filter_C fc) 
    -> std::enable_if_t<(Container::TParam_Dimension == 1 && Container::TParam_Layout == DataLayout::AoSoA), void>
{
    using namespace ::fw::math;

    constexpr SizeT CHUNK_SIZE = Container::GetInnerArraySize();
    [[maybe_unused]] SimdVec<ElementT, CHUNK_SIZE, DataLayout::SoA> ta, tb, tc;

#if defined(FW_USE_HIGHLEVEL_VECTOR_REF)
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
#elif defined(FW_USE_HIGHLEVEL_VECTOR)
    for (SizeT x = 0; x < a.Size(0); x += CHUNK_SIZE)
    {
        const SizeT i_max = std::min(CHUNK_SIZE, a.Size(0) - x);

        ta.Load(a.At(x), fa, i_max);
        tb.Load(b.At(x), fb, i_max);

        #pragma omp simd
        for (SizeT i = 0; i < CHUNK_SIZE; ++i)
        {
            func(ta[i], tb[i], tc[i]);
        }
        
        tc.Store(c.At(x), fc, i_max);
    }
#else 
    #pragma omp simd
    for (SizeT x = 0; x < a.Size(0); ++x)
    {    
        func(a[x], b[x], c[x]);
    }
#endif
}

template <typename FuncT, typename Container, typename Filter_A, typename Filter_B, typename Filter_C>
auto KernelImplementation(FuncT func, const Container& a, const Container& b, Container& c, Filter_A fa, Filter_B fb, Filter_C fc) 
    -> std::enable_if_t<(Container::TParam_Dimension == 2 && Container::TParam_Layout != DataLayout::AoSoA), void>
{
    using namespace ::fw::math;

    constexpr SizeT CHUNK_SIZE = 32;
    [[maybe_unused]] SimdVec<ElementT, CHUNK_SIZE, DataLayout::SoA> ta, tb, tc;
    
    for (SizeT y = 0; y < a.Size(1); ++y)
    {
#if defined(FW_USE_HIGHLEVEL_VECTOR)
        for (SizeT x = 0; x < a.Size(0); x += CHUNK_SIZE)
        {
            const SizeT i_max = std::min(CHUNK_SIZE, a.Size(0) - x);

            ta.Load(a[y].At(x), fa, i_max);
            tb.Load(b[y].At(x), fb, i_max);

            #pragma omp simd
            for (SizeT i = 0; i < CHUNK_SIZE; ++i)
            {
                func(ta[i], tb[i], tc[i]);
            }
            
            tc.Store(c[y].At(x), fc, i_max);
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

template <typename FuncT, typename Container, typename Filter_A, typename Filter_B, typename Filter_C>
auto KernelImplementation(FuncT func, const Container& a, const Container& b, Container& c, Filter_A fa, Filter_B fb, Filter_C fc) 
    -> std::enable_if_t<(Container::TParam_Dimension == 2 && Container::TParam_Layout == DataLayout::AoSoA), void>
{
    using namespace ::fw::math;

    constexpr SizeT CHUNK_SIZE = Container::GetInnerArraySize();
    [[maybe_unused]] SimdVec<ElementT, CHUNK_SIZE, DataLayout::SoA> ta, tb, tc;
    
    for (SizeT y = 0; y < a.Size(1); ++y)
    {
#if defined(FW_USE_HIGHLEVEL_VECTOR_REF)
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
#elif defined(FW_USE_HIGHLEVEL_VECTOR)
        for (SizeT x = 0; x < a.Size(0); x += CHUNK_SIZE)
        {
            const SizeT i_max = std::min(CHUNK_SIZE, a.Size(0) - x);

            ta.Load(a[y].At(x), fa, i_max);
            tb.Load(b[y].At(x), fb, i_max);

            #pragma omp simd
            for (SizeT i = 0; i < CHUNK_SIZE; ++i)
            {
                func(ta[i], tb[i], tc[i]);
            }
            
            tc.Store(c[y].At(x), fc, i_max);
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

template <typename FuncT, typename Container, typename Filter_A, typename Filter_B, typename Filter_C>
auto KernelImplementation(FuncT func, const Container& a, const Container& b, Container& c, Filter_A fa, Filter_B fb, Filter_C fc) 
    -> std::enable_if_t<(Container::TParam_Dimension == 3 && Container::TParam_Layout != DataLayout::AoSoA), void>
{
    using namespace ::fw::math;

    constexpr SizeT CHUNK_SIZE = 32;
    [[maybe_unused]] SimdVec<ElementT, CHUNK_SIZE, DataLayout::SoA> ta, tb, tc;

    for (SizeT z = 0; z < a.Size(2); ++z)
    {
        for (SizeT y = 0; y < a.Size(1); ++y)
        {
#if defined(FW_USE_HIGHLEVEL_VECTOR)
            for (SizeT x = 0; x < a.Size(0); x += CHUNK_SIZE)
            {
                const SizeT i_max = std::min(CHUNK_SIZE, a.Size(0) - x);

                ta.Load(a[z][y].At(x), fa, i_max);
                tb.Load(b[z][y].At(x), fb, i_max);

                #pragma omp simd
                for (SizeT i = 0; i < CHUNK_SIZE; ++i)
                {
                    func(ta[i], tb[i], tc[i]);
                }
                
                tc.Store(c[z][y].At(x), fc, i_max);
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

template <typename FuncT, typename Container, typename Filter_A, typename Filter_B, typename Filter_C>
auto KernelImplementation(FuncT func, const Container& a, const Container& b, Container& c, Filter_A fa, Filter_B fb, Filter_C fc) 
    -> std::enable_if_t<(Container::TParam_Dimension == 3 && Container::TParam_Layout == DataLayout::AoSoA), void>
{
    using namespace ::fw::math;

    constexpr SizeT CHUNK_SIZE = Container::GetInnerArraySize();
    [[maybe_unused]] SimdVec<ElementT, CHUNK_SIZE, DataLayout::SoA> ta, tb, tc;

    for (SizeT z = 0; z < a.Size(2); ++z)
    {
        for (SizeT y = 0; y < a.Size(1); ++y)
        {
#if defined(FW_USE_HIGHLEVEL_VECTOR_REF)
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
#elif defined(FW_USE_HIGHLEVEL_VECTOR)
            for (SizeT x = 0; x < a.Size(0); x += CHUNK_SIZE)
            {
                const SizeT i_max = std::min(CHUNK_SIZE, a.Size(0) - x);

                ta.Load(a[z][y].At(x), fa, i_max);
                tb.Load(b[z][y].At(x), fb, i_max);

                #pragma omp simd
                for (SizeT i = 0; i < CHUNK_SIZE; ++i)
                {
                    func(ta[i], tb[i], tc[i]);
                }
                
                tc.Store(c[z][y].At(x), fc, i_max);
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

template <typename FuncT, typename Field, typename Filter_A, typename Filter_B, typename Filter_C>
auto Kernel(FuncT func, const Field& a, const Field& b, Field& c, Filter_A fa, Filter_B fb, Filter_C fc) -> void
{
    KernelImplementation(func, a, b, c, fa, fb, fc);
}
#endif

#define TESTING_1

#if defined(TESTING_1)
#if defined(__CUDACC__)
template <typename Container>
CUDA_KERNEL
auto TestKernelImplemenation(Container a, Container b) -> std::enable_if_t<(Container::TParam_Dimension != 3), void>
{
    return 0;
}

template <typename Container>
CUDA_KERNEL
auto TestKernelImplemenation(Container a, Container b) -> std::enable_if_t<(Container::TParam_Dimension == 3), void>
{
    using namespace ::fw::math;

    const SizeT x = blockIdx.x * blockDim.x + threadIdx.x;
    const SizeT y = blockIdx.y * blockDim.y + threadIdx.y;
    const SizeT z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < a.Size(0) && y < a.Size(1) && z < a.Size(2))
    {
        b[z][y][x] = 2.0 * a[z][y][x];
    } 
}

template <SizeT Dimension, typename T>
auto TestKernel(T& a, T& b) -> SizeT
{
    const dim3 block{128, 1, 1};
    const dim3 grid = GetGridSize(a.Size(), block);

    TestKernelImplemenation<<<grid, block>>>(a.DeviceData(), b.DeviceData());

    return 0;
}
#else
template <SizeT Dimension, typename T>
auto TestKernelImplemenation(const T& a, const T& b) -> std::enable_if_t<Dimension != 3, SizeT> {}

template <SizeT Dimension, typename T>
auto TestKernelImplemenation(const T& a, const T& b) -> std::enable_if_t<Dimension == 3, SizeT>
{
    using namespace ::fw::math;

    SizeT sum = 0;

    for (SizeT z = 0; z < a.Size(2); ++z)
    {
        for (SizeT y = 0; y < a.Size(1); ++y)
        {
            #pragma omp simd reduction(+:sum)
            for (SizeT x = 0; x < a.Size(0); ++x)
            {
                //sum += Dot(a[z][y][x], b[z][y][x]);                                                    
                sum += a[z][y][x].z * b[z][y][x].z;
            }
        }
    }

    return sum;
}

template <SizeT Dimension, typename T>
auto TestKernel(const T& a, const T& b) -> SizeT
{
    return TestKernelImplemenation<Dimension>(a, b);
}
#endif
#endif

#if defined(TESTING_2)
template <SizeT Dimension, typename T>
void Print(const T& field)
{
    if constexpr (Dimension == 3)
    {
        for (SizeT z = 0; z < field.Size(2); ++z)
        {
            for (SizeT y = 0; y < field.Size(1); ++y)
            {
                for (SizeT x = 0; x < field.Size(0); ++x)
                {
                    std::cout << field[z][y][x] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }
}
#endif

template <SizeT Dimension>
int benchmark(int argc, char** argv, const SizeArray<Dimension>& size)
{
    using namespace ::fw::math;

    Field<ElementT, Dimension> field_1(size);
    Field<ElementT, Dimension> field_2(size);

#if defined(TESTING_1)
    field_1.Set([] (const auto&) { return 0; });
    field_2.Set([] (const auto&) { return 0; });

    SizeT sum = 0;
    for (SizeT i = 0; i < (argc > 4 ? std::atoi(argv[4]) : 1); ++i)
    {
        sum += TestKernel<Dimension>(field_1, field_2);
    }

    if (sum == 0)
    {
        std::cout << "Success!" << std::endl;
    }

    return 0;
#endif

    // Field initialization.
    srand48(1);
    field_1.Set([] (const auto I) { return static_cast<RealT>((2.0 * drand48() -1.0) * SPREAD + OFFSET); });
    field_2.Set([] (const auto I) { return static_cast<RealT>((2.0 * drand48() -1.0) * SPREAD + OFFSET); });

    Field<ElementT, Dimension> field_3(size, true);

#if defined(TESTING_2)
    //Print<Dimension>(field_1);

    Kernel([] CUDA_DEVICE_VERSION (const auto& a, const auto& b, auto&& c) -> void { using namespace ::fw::math; c = Exp(a + b); }
        , field_1, field_2, field_3, ::fw::auxiliary::AssignAll, ::fw::auxiliary::AssignAll, ::fw::auxiliary::AssignAll);

    if constexpr (Dimension == 3)
    {
        for (SizeT z = 0; z < field_1.Size(2); ++z)
        {
            for (SizeT y = 0; y < field_1.Size(1); ++y)
            {
                for (SizeT x = 0; x < field_1.Size(0); ++x)
                {
                    const auto& result = Exp(field_1[z][y][x] + field_2[z][y][x]);
                    if (result != field_3[z][y][x]) { std::cout << "error: " << result << " vs. " << field_3[z][y][x] << std::endl; return 1; }
                }
            }
        }
    }

    return 0;
#endif

#if defined(CHECK_RESULTS)
    std::vector<ElementT> field_1_copy = field_1.Get();
    std::vector<ElementT> field_2_copy = field_2.Get();
#endif

#if defined(ELEMENT_ACCESS)
    auto kernel_1 = [] CUDA_DEVICE_VERSION (const auto& a, const auto& b, auto&& c) -> void { using namespace ::fw::math; c.z = Exp(a.x + b.y); };
    auto kernel_2 = [] CUDA_DEVICE_VERSION (const auto& a, const auto& b, auto&& c) -> void { using namespace ::fw::math; c.x = Log(a.z) - b.y; };
    auto load_1_a = [] CUDA_DEVICE_VERSION (const auto& in, auto&& out) -> void { out.x = in.x; out.y = 0; out.z = 0; };
    auto load_1_b = [] CUDA_DEVICE_VERSION (const auto& in, auto&& out) -> void { out.x = 0; out.y = in.y; out.z = 0; };
    auto store_1_c = [] CUDA_DEVICE_VERSION (const auto& in, auto&& out) -> void { out.z = in.z; };
    auto load_2_a = [] CUDA_DEVICE_VERSION (const auto& in, auto&& out) -> void { out.x = 1; out.y = 1; out.z = in.z; };
    auto load_2_b = [] CUDA_DEVICE_VERSION (const auto& in, auto&& out) -> void { out.x = 0; out.y = in.y; out.z = 0; };
    auto store_2_c = [] CUDA_DEVICE_VERSION (const auto& in, auto&& out) -> void { out.x = in.x; };
#else
    auto kernel_1 = [] CUDA_DEVICE_VERSION (const auto& a, const auto& b, auto&& c) -> void { using namespace ::fw::math; c = Exp(a + b); };
    auto kernel_2 = [] CUDA_DEVICE_VERSION (const auto& a, const auto& b, auto&& c) -> void { using namespace ::fw::math; c = Log(a) - b; };
    auto load_1_a = ::fw::auxiliary::AssignAll;
    auto load_1_b = ::fw::auxiliary::AssignAll;
    auto store_1_c = ::fw::auxiliary::AssignAll;
    auto load_2_a = ::fw::auxiliary::AssignAll;
    auto load_2_b = ::fw::auxiliary::AssignAll;
    auto store_2_c = ::fw::auxiliary::AssignAll;
#endif

#if defined(__CUDACC__)
    // Create device buffer and copy in the host data.
    field_1.CopyHostToDevice();
    field_2.CopyHostToDevice();
    field_3.CopyHostToDevice();
#endif

    for (SizeT i = 0; i < WARMUP; ++i)
    {
        Kernel(kernel_1, field_1, field_2, field_3, load_1_a, load_1_b, store_1_c);
        Kernel(kernel_2, field_3, field_2, field_1, load_2_a, load_2_b, store_2_c);
    }

#if defined(__CUDACC__)
    cudaDeviceSynchronize();
#endif

    double start_time = omp_get_wtime();

    for (SizeT i = 0; i < MEASUREMENT; ++i)
    {
        Kernel(kernel_1, field_1, field_2, field_3, load_1_a, load_1_b, store_1_c);
        Kernel(kernel_2, field_3, field_2, field_1, load_2_a, load_2_b, store_2_c);
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

        std::cout << "deviation: " << deviation << std::endl;
    }
#endif
    
    return 0;
}

template int benchmark<1>(int, char**, const SizeArray<1>&);
template int benchmark<2>(int, char**, const SizeArray<2>&);
template int benchmark<3>(int, char**, const SizeArray<3>&);
