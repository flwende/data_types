// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(COMMON_MEMORY_HPP)
#define COMMON_MEMORY_HPP

#include <cstdint>
#include <memory>
#include <tuple>

#if defined(__CUDACC__)
#include <cuda_runtime.h>

#define HOST_VERSION __host__
#define CUDA_DEVICE_VERSION __device__
#define CUDA_KERNEL __global__
#else
#define HOST_VERSION 
#define CUDA_DEVICE_VERSION 
#define CUDA_KERNEL 
#endif

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#include "../auxiliary/math.hpp"
#include "../auxiliary/variadic.hpp"
#include "../sarray/sarray.hpp"
#include "../simd/simd.hpp"

namespace XXX_NAMESPACE
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    //! \brief A pointer wrapper for homogeneous structured types and SoA data layout
    //!
    //! It provides some functionality regarding memory (de)allocation and accessing it in the multi-
    //! dimensional case with different data layouts: here SoA!
    //!
    //! Idea: Multidimensional fields are contiguous sequences of stabs (innermost dimension n_0).
    //! Stabs are separated by 'N x n_0' elements of type T, with N being the number of members.
    //! All elements of the field can be accessed through jumping into the stab using a stab index and the
    //! base pointer to the 1st member of the 0th element of the field, and within the stab by adding
    //! a multiple of n_0 according to what member should be accessed.
    //! The resulting pointer then is shifted by an intra-stab index to access the actual data member.
    //! 
    //! THE POINTER MANAGED BY THIS CLASS IS EXTERNAL!
    //! 
    //! \tparam T... types (one for each data member; all the same)
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <typename ...T>
    class pointer
    {
        template <typename ...>
        friend class pointer;

        // number of data members
        static constexpr std::size_t N = AUXILIARY_NAMESPACE::variadic::pack<T...>::length;
        static_assert(N > 0, "error: empty parameter pack");

        // all members have the same type: get this type
        using value_type = typename AUXILIARY_NAMESPACE::variadic::pack<T...>::template type<0>;

        // check if all types are the same
        static constexpr bool is_homogeneous = AUXILIARY_NAMESPACE::variadic::pack<T...>::is_same() || AUXILIARY_NAMESPACE::variadic::pack<T...>::has_same_size();
        static_assert(is_homogeneous, "error: use the inhomogeneous multi_pointer instead");

        // create tuple from the base pointer
        template <std::size_t ...I>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto get_values(std::index_sequence<I...>)
            -> std::tuple<T&...>
        {
            return {reinterpret_cast<T&>(ptr[I * n_0])...};
        }

        template <std::size_t ...I>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto get_values(std::index_sequence<I...>) const
            -> std::tuple<const T&...>
        {
            return {reinterpret_cast<const T&>(ptr[I * n_0])...};
        }

        template <std::size_t ...I>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto get_values(const std::size_t stab_idx, const std::size_t idx, std::index_sequence<I...>)
            -> std::tuple<T&...>
        {
            return {reinterpret_cast<T&>(ptr[(stab_idx * N + I) * n_0 + idx])...};
        }

        template <std::size_t ...I>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto get_values(const std::size_t stab_idx, const std::size_t idx, std::index_sequence<I...>) const
            -> std::tuple<const T&...>
        {
            return {reinterpret_cast<const T&>(ptr[(stab_idx * N + I) * n_0 + idx])...};
        }

        // extent of the innermost dimension (w.r.t. a multidimensional field declaration)
        const std::size_t n_0;
        // base pointer
        value_type* ptr;

    public:

        // constructor: from external base pointer and innermist dimension
        pointer(value_type* ptr, const std::size_t n_0)
            :
            n_0(n_0),
            ptr(ptr) 
        {}

        // constructor: from an existing pointer and a stab index (stab_idx) and an intra-stab index (idx)
        HOST_VERSION
        CUDA_DEVICE_VERSION
        pointer(const pointer& mp, const std::size_t stab_idx, const std::size_t idx)
            :
            n_0(mp.n_0),
            ptr(&mp.ptr[stab_idx * N * n_0 + idx]) 
        {}

        // copy /conversion constructors
        pointer(const pointer<typename std::remove_cv<T>::type...>& mp)
            :
            n_0(mp.n_0),
            ptr(reinterpret_cast<value_type*>(mp.ptr)) 
        {}

        pointer(const pointer<const typename std::remove_cv<T>::type...>& mp)
            :
            n_0(mp.n_0),
            ptr(reinterpret_cast<value_type*>(mp.ptr)) 
        {}

        // get a new pointer shifted by stab_idx and idx
        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto at(const std::size_t idx)
            -> pointer
        {
            return {*this, 0, idx};
        }

        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto at(const std::size_t idx) const
            -> pointer<const T...>
        {
            return {*this, 0, idx};
        }

        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto at(const std::size_t stab_idx, const std::size_t idx)
            -> pointer
        {
            return {*this, stab_idx, idx};
        }

        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto at(const std::size_t stab_idx, const std::size_t idx) const
            -> pointer<const T...>
        {
            return {*this, stab_idx, idx};
        }

        // dereference / access
        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto operator*()
        {
            return get_values(std::make_index_sequence<N>{});
        }

        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto operator*() const
        {
            return get_values(std::make_index_sequence<N>{});
        }

        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto access(const std::size_t idx)
        {
            return get_values(0, idx, std::make_index_sequence<N>{});
        }

        template <bool Enable = true>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto access(const std::size_t idx) const
        {
            return get_values(0, idx, std::make_index_sequence<N>{});
        }

        template <bool Enable = true>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto access(const std::size_t stab_idx, const std::size_t idx)
        {
            return get_values(stab_idx, idx, std::make_index_sequence<N>{});
        }

        template <bool Enable = true>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto access(const std::size_t stab_idx, const std::size_t idx) const
        {
            return get_values(stab_idx, idx, std::make_index_sequence<N>{});
        }

        // get base pointer
        inline auto get_pointer()
            -> value_type*
        {
            return ptr;
        }

        inline auto get_pointer() const
            -> const value_type*
        {
            return ptr;
        }

        // pointer increment
        inline auto operator++()
            -> pointer&
        {
            ++ptr;
            return *this;
        }

        inline auto operator++(int)
            -> pointer
        {
            pointer mp(*this);
            ++ptr;
            return mp;
        }

        inline auto operator+=(const std::size_t n)
            -> pointer&
        {
            ptr += n;
            return *this;
        }

        // comparison
        inline auto operator==(const pointer& p) const
            -> bool
        {
            return (ptr == p.ptr);
        }

        inline auto operator!=(const pointer& p) const
            -> bool
        {
            return (ptr != p.ptr);
        }

        // allocator class
        class allocator
        {           
        protected:

            static constexpr std::size_t default_alignment = SIMD_NAMESPACE::simd::alignment;

        public:

            static auto padding(const std::size_t n, const std::size_t alignment = default_alignment)
                -> std::size_t
            {
                if (!MATH_NAMESPACE::is_power_of<2>(alignment))
                {
                    std::cerr << "warning: alignment is not a power of 2" << std::endl;
                    return n;
                }

                const std::size_t ratio = MATH_NAMESPACE::least_common_multiple(alignment, sizeof(value_type)) / sizeof(value_type);

                return ((n + ratio - 1) / ratio) * ratio;
            }

            static auto allocate(std::size_t n, const std::size_t alignment = default_alignment)
                -> value_type*
            {
                return reinterpret_cast<value_type*>(_mm_malloc(n * sizeof(value_type), alignment));
            }

            template <std::size_t D>
            static auto allocate(const sarray<std::size_t, D>& n, const std::size_t alignment = default_alignment)
                -> value_type*
            {
                const std::size_t num_elements = padding(n[0], alignment) * N * n.reduce_mul(1);

                return allocate(num_elements, alignment);
            }

            static auto deallocate(value_type* ptr, std::size_t n = 0)
                -> void
            {
                if (ptr)
                {
                    _mm_free(ptr);
                }
            }
        };
    };

    // define N-dimensional homogeneous structured type
    namespace
    {
        // defines 'template <T, N> struct type_gen {..};'
        MACRO_VARIADIC_TYPE_GEN(XXX_NAMESPACE::pointer);
    }

    template <typename T, std::size_t N>
    using pointer_n = typename type_gen<T, N>::type;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    //! \brief A pointer wrapper for inhomogeneous structured types and SoA data layout
    //!
    //! It provides some functionality regarding memory (de)allocation and accessing it in the multi-
    //! dimensional case with different data layouts: here SoA!
    //!
    //! Idea: Similar to the homogeneous pointer, but a bit more complicated to implement as
    //! multiple base pointers need to be managed internally, one for each data member of the inhomogeneous
    //! structured type.
    //! 
    //! THE POINTER MANAGED BY THIS CLASS IS EXTERNAL!
    //! 
    //! \tparam T... types (one for each data member; can be all different)
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <typename ...T>
    class multi_pointer
    {
        template <typename ...X>
        friend class multi_pointer;

        // number of data members
        static constexpr std::size_t N = AUXILIARY_NAMESPACE::variadic::pack<T...>::length;
        static_assert(N > 0, "error: empty parameter pack");
        
        // check if all types are the same: we don't want that here
        static constexpr bool is_homogeneous = AUXILIARY_NAMESPACE::variadic::pack<T...>::is_same() || AUXILIARY_NAMESPACE::variadic::pack<T...>::has_same_size();
        static_assert(!is_homogeneous, "error: use the homogeneous pointer instead");

        // find out the byte-size of the largest type
        static constexpr std::size_t size_largest_type = AUXILIARY_NAMESPACE::variadic::pack<T...>::size_of_largest_type();

        // determine the total byte-size of all data members that have a size different (smaller) than the largest type
        static constexpr std::size_t size_rest = AUXILIARY_NAMESPACE::variadic::pack<T...>::size_of_pack_excluding_largest_type();

        // size of the inhomogeneous structured type
        static constexpr std::size_t record_size = AUXILIARY_NAMESPACE::variadic::pack<T...>::size_of_pack();

        // determine the number of elements of the structured type that is needed so that their overall size
        // is an integral multiple of each data member type
        static constexpr std::size_t record_padding_factor = std::max(1UL, MATH_NAMESPACE::least_common_multiple(size_largest_type, size_rest) / std::max(1UL, size_rest));

        // determine the scaling factor of each member-type-size w.r.t. to the largest type
        static constexpr XXX_NAMESPACE::sarray<std::size_t, N> size_scaling_factor{size_largest_type / sizeof(T)...};

        // (exclusive) prefix sum over the byte-sizes of the template arguments
        static constexpr XXX_NAMESPACE::sarray<std::size_t, N> offset = MATH_NAMESPACE::prefix_sum(XXX_NAMESPACE::sarray<std::size_t, N>{sizeof(T)...});
    
        // create a pointer tuple from a base pointer and the 'offset's for a field with extent of the innermost dimension 'n_0'
        template <std::size_t ...I>
        inline constexpr auto make_pointer_tuple(std::uint8_t* __restrict__ ptr, const std::size_t n_0, std::index_sequence<I...>)
            -> std::tuple<T*...>
        {
            return {reinterpret_cast<T*>(&ptr[offset[I] * n_0])...};
        }

        // create a pointer tuple from an existing pointer tuple, a stab index (stab_idx) and an intra-stab index (idx)
        template <std::size_t ...I>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline constexpr auto make_pointer_tuple(const std::tuple<T*...>& ptr, const std::size_t stab_idx, const std::size_t idx, std::index_sequence<I...>)
            -> std::tuple<T*...>
        {
            return {std::get<I>(ptr) + stab_idx * num_units * size_scaling_factor[I] + idx...};
        }

        // increment the pointer tuple
        inline constexpr auto increment_pointer_tuple(const std::size_t inc = 1)
            -> void
        {
            AUXILIARY_NAMESPACE::variadic::loop<N>::execute([inc, this] (auto& I) {std::get<I.value>(ptr) += inc;});
        }

        // create tuple from the base pointer
        template <std::size_t ...I>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto get_values(std::index_sequence<I...>)
            -> std::tuple<T&...>
        {
            return {*(std::get<I>(ptr))...};
        }

        template <std::size_t ...I>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto get_values(std::index_sequence<I...>) const
            -> std::tuple<const T&...>
        {
            return {*(std::get<I>(ptr))...};
        }

        template <std::size_t ...I>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto get_values(const std::size_t stab_idx, const std::size_t idx, std::index_sequence<I...>)
            -> std::tuple<T&...>
        {
            return {*(std::get<I>(ptr) + stab_idx * num_units * size_scaling_factor[I] + idx)...};
        }

        template <std::size_t ...I>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto get_values(const std::size_t stab_idx, const std::size_t idx, std::index_sequence<I...>) const
            -> std::tuple<const T&...>
        {
            return {*(std::get<I>(ptr) + stab_idx * num_units * size_scaling_factor[I] + idx)...};
        }

        // all members have different type: use std::uint8_t for all of them
        using value_type = std::uint8_t;

        // extent of the innermost dimension of the filed in units of largest type
        const std::size_t num_units;
        // base pointers (of different type) are managed internally by using a tuple
        std::tuple<T*...> ptr;

    public:

        // constructor: from external base pointer and innermist dimension
        multi_pointer(std::uint8_t* __restrict__ ptr, const std::size_t n_0)
            :
            num_units((n_0 * record_size) / size_largest_type),
            ptr(make_pointer_tuple(ptr, n_0, std::make_index_sequence<N>{})) 
        {}

        // constructor: from an existing multi_pointer and a stab index (stab_idx) and an intra-stab index (idx)
        HOST_VERSION
        CUDA_DEVICE_VERSION
        multi_pointer(const multi_pointer& mp, const std::size_t stab_idx, const std::size_t idx)
            :
            num_units(mp.num_units),
            ptr(make_pointer_tuple(mp.ptr, stab_idx, idx, std::make_index_sequence<N>{})) 
        {}

        // copy / conversion constructors
        multi_pointer(const multi_pointer<typename std::remove_cv<T>::type...>& mp)
            :
            num_units(mp.num_units),
            ptr(mp.ptr) 
        {}

        multi_pointer(const multi_pointer<const typename std::remove_cv<T>::type...>& mp)
            :
            num_units(mp.num_units),
            ptr(mp.ptr) 
        {}

        // get a new multi_pointer shifted by [stab_idx and] 
        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto at(const std::size_t idx)
            -> multi_pointer
        {
            return {*this, 0, idx};
        }

        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto at(const std::size_t idx) const
            -> multi_pointer<const T...>
        {
            return {*this, 0, idx};
        }

        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto at(const std::size_t stab_idx, const std::size_t idx)
            -> multi_pointer
        {
            return {*this, stab_idx, idx};
        }

        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto at(const std::size_t stab_idx, const std::size_t idx) const
            -> multi_pointer<const T...>
        {
            return {*this, stab_idx, idx};
        }

        // dereference / access
        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto operator*()
        {
            return get_values(std::make_index_sequence<N>{});
        }

        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto operator*() const
        {
            return get_values(std::make_index_sequence<N>{});
        }


        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto access(const std::size_t idx)
        {
            return get_values(0, idx, std::make_index_sequence<N>{});
        }

        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto access(const std::size_t idx) const
        {
            return get_values(0, idx, std::make_index_sequence<N>{});
        }

        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto access(const std::size_t stab_idx, const std::size_t idx)
        {
            return get_values(stab_idx, idx, std::make_index_sequence<N>{});
        }

        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto access(const std::size_t stab_idx, const std::size_t idx) const
        {
            return get_values(stab_idx, idx, std::make_index_sequence<N>{});
        }
        
        // get base pointer
        inline auto get_pointer()
            -> value_type*
        {
            return reinterpret_cast<value_type*>(std::get<0>(ptr));
        }

        inline auto get_pointer() const
            -> const value_type*
        {
            return reinterpret_cast<const value_type*>(std::get<0>(ptr));
        }

        // pointer increment
        inline auto operator++()
            -> multi_pointer&
        {
            increment_pointer_tuple();
            return *this;
        }

        inline auto operator++(int)
            -> multi_pointer
        {
            multi_pointer mp(*this);
            increment_pointer_tuple();
            return mp;
        }

        inline auto operator+=(const std::size_t n)
            -> multi_pointer&
        {
            increment_pointer_tuple(n);
            return *this;
        }

        // comparison
        inline auto operator==(const multi_pointer& p) const
            -> bool
        {
            return (std::get<0>(ptr) == std::get<0>(p.ptr));
        }

        inline auto operator!=(const multi_pointer& p) const
            -> bool
        {
            return (std::get<0>(ptr) != std::get<0>(p.ptr));
        }

        // allocator class
        class allocator
        {           
        protected:

            static constexpr std::size_t default_alignment = SIMD_NAMESPACE::simd::alignment;

        public:

            static auto padding(const std::size_t n, const std::size_t alignment = default_alignment)
                -> std::size_t
            {
                if (!MATH_NAMESPACE::is_power_of<2>(alignment))
                {
                    std::cerr << "warning: alignment is not a power of 2" << std::endl;
                    return n;
                }

                const std::size_t byte_padding_factor = MATH_NAMESPACE::least_common_multiple(alignment, size_largest_type) / size_largest_type;
                const std::size_t ratio = MATH_NAMESPACE::least_common_multiple(record_padding_factor, byte_padding_factor);

                return ((n + ratio - 1) / ratio) * ratio;
            }

            static auto allocate(std::size_t n, const std::size_t alignment = default_alignment)
                -> value_type*
            {
                return reinterpret_cast<value_type*>(_mm_malloc(n * sizeof(value_type), alignment));
            }

            template <std::size_t D>
            static auto allocate(const sarray<std::size_t, D>& n, const std::size_t alignment = default_alignment)
                -> value_type*
            {
                const std::size_t num_elements = padding(n[0], alignment) * n.reduce_mul(1) * record_size;

                return allocate(num_elements, alignment);
            }

            static auto deallocate(value_type* ptr, std::size_t n = 0)
                -> void
            {
                if (ptr)
                {
                    _mm_free(ptr);
                }
            }
        };
    };
}

#endif