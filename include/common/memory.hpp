// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(COMMON_MEMORY_HPP)
#define COMMON_MEMORY_HPP

#include <cstdint>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#include "../auxiliary/math.hpp"
#include "../sarray/sarray.hpp"
#include "../simd/simd.hpp"

namespace XXX_NAMESPACE
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    //! \brief A simple memory manager
    //!
    //! Note: it is not a manager in the sense of explicitly providing memory, but more a wrapper type
    //! with some additional functionality regarding memory (de)allocation and accessing it in the multi-
    //! dimensional case with different data layouts: here AoS!
    //!
    //! THE POINTER MANAGED BY THIS CLASS IS EXTERNAL!
    //! 
    //! \tparam T type of the data being managed
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
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

        static std::size_t padding(const std::size_t n, const std::size_t alignment = SIMD_NAMESPACE::simd::alignment)
        {
            if (!AUXILIARY_NAMESPACE::is_power_of<2>(alignment))
            {
                std::cerr << "warning: alignment is not a power of 2" << std::endl;
                return n;
            }

            const std::size_t ratio = AUXILIARY_NAMESPACE::least_common_multiple(alignment, sizeof(T)) / sizeof(T);

            return ((n + ratio - 1) / ratio) * ratio;
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