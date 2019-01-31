// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(MISC_MISC_MEMORY_HPP)
#define MISC_MISC_MEMORY_HPP

#include <cstdint>
#include <immintrin.h>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#if !defined(MISC_NAMESPACE)
#define MISC_NAMESPACE XXX_NAMESPACE
#endif

#include "../sarray/sarray.hpp"
#include "../simd/simd.hpp"

namespace MISC_NAMESPACE
{
    template <typename T>
    struct memory
    {
        using type = T;

        const std::size_t n_innermost;
        T* __restrict__ ptr;

        memory(T* __restrict__ ptr, const std::size_t n_innermost)
            :
            n_innermost(n_innermost),
            ptr(ptr)
        { ; }

        template <typename TT>
        memory(const memory<TT>& m)
            :
            n_innermost(m.n_innermost),
            ptr(reinterpret_cast<T*>(m.ptr))
        { ; }

        T& at(const std::size_t slice_idx, const std::size_t idx)
        {
            return ptr[n_innermost * slice_idx + idx];
        }

        const T& at(const std::size_t slice_idx, const std::size_t idx) const
        {
            return ptr[n_innermost * slice_idx + idx];
        }

        // replace by internal alignment
        static std::size_t padding(const std::size_t n, const std::size_t alignment = SIMD_NAMESPACE::simd::alignment)
        {
            // FIX
            return n;
        }

        template <std::size_t D>
        static T* allocate(const sarray<std::size_t, D>& n, const std::size_t alignment = SIMD_NAMESPACE::simd::alignment)
        {
            if (n[0] != padding(n[0], alignment))
            {
                std::cerr << "error in memory::allocate : n[0] does not match alignment" << std::endl;
            }

            return reinterpret_cast<T*>(_mm_malloc(n.reduce_mul() * sizeof(T), alignment));
        }

        static void deallocate(memory& m)
        {
            if (m.ptr)
            {
                _mm_free(m.ptr);
            }
        }
    };
}

#endif