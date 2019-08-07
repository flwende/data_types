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
    //! \brief A simple pointer wrapper
    //!
    //! It provides some functionality regarding memory (de)allocation and accessing it in the multi-
    //! dimensional case with different data layouts: here AoS!
    //!
    //! THE POINTER MANAGED BY THIS CLASS IS EXTERNAL!
    //! 
    //! \tparam T type of the pointer being managed
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <typename T>
    class pointer
    {
    public:

        using value_type = T;

        // base pointer and extent of the innermost dimension (w.r.t. a multidimensional field declaration)
        const std::size_t n_0;
        T* __restrict__ ptr;

        // constructor: from external base pointer and innermist dimension
        pointer(T* __restrict__ ptr, const std::size_t n_0)
            :
            n_0(n_0),
            ptr(ptr) {}

        // constructor: from an existing pointer and a stab index (stab_idx) and an intra-stab index (idx)
        HOST_VERSION
        CUDA_DEVICE_VERSION
        pointer(const pointer& p, const std::size_t stab_idx, const std::size_t idx)
            :
            n_0(p.n_0),
            ptr(&p.ptr[stab_idx * n_0 + idx]) {}    

        // copy / conversion constructors
        template <typename TT>
        pointer(const pointer<TT>& p)
            :
            n_0(p.n_0),
            ptr(reinterpret_cast<T*>(p.ptr)) {}

        // get a new pointer shifted by [stab_idx and] idx
        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline pointer at(const std::size_t idx)
        {
            return pointer(*this, 0, idx);
        }

        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline const pointer at(const std::size_t idx) const
        {
            return pointer(*this, 0, idx);
        }

        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline pointer at(const std::size_t stab_idx, const std::size_t idx)
        {
            return pointer(*this, stab_idx, idx);
        }

        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline const pointer at(const std::size_t stab_idx, const std::size_t idx) const
        {
            return pointer(*this, stab_idx, idx);
        }

        // dereference
        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline T& operator*()
        {
            return *ptr;
        }

        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline const T& operator*() const
        {
            return *ptr;
        }

        // get base pointer
        inline T* get_pointer()
        {
            return ptr;
        }

        inline const T* get_pointer() const
        {
            return ptr;
        }

        // pointer increment
        inline pointer& operator++()
        {
            ++ptr;
            return *this;
        }

        inline pointer operator++(int)
        {
            pointer p(*this);
            ++ptr;
            return p;
        }

        inline pointer& operator+=(const std::size_t n)
        {
            ptr += n;
            return *this;
        }

        // comparison
        inline bool operator==(const pointer& p) const
        {
            return (ptr == p.ptr);
        }

        inline bool operator!=(const pointer& p) const
        {
            return (ptr != p.ptr);
        }

        // allocator class
        class allocator : public std::allocator<T>
        {           
        protected:

            static constexpr std::size_t default_alignment = SIMD_NAMESPACE::simd::alignment;

        public:

            using pointer = typename std::allocator<T>::pointer;

            static std::size_t padding(const std::size_t n, const std::size_t alignment = default_alignment)
            {
                if (!MATH_NAMESPACE::is_power_of<2>(alignment))
                {
                    std::cerr << "warning: alignment is not a power of 2" << std::endl;
                    return n;
                }

                const std::size_t ratio = MATH_NAMESPACE::least_common_multiple(alignment, sizeof(T)) / sizeof(T);

                return ((n + ratio - 1) / ratio) * ratio;
            }

            static pointer allocate(std::size_t n, const std::size_t alignment = default_alignment)
            {
                return reinterpret_cast<pointer>(_mm_malloc(n * sizeof(T), alignment));
            }

            template <std::size_t D>
            static pointer allocate(const sarray<std::size_t, D>& n, const std::size_t alignment = default_alignment)
            {
                const std::size_t num_elements = padding(n[0], alignment) * n.reduce_mul(1);

                return allocate(num_elements, alignment);
            }

            static void deallocate(pointer ptr, std::size_t n = 0)
            {
                if (ptr)
                {
                    _mm_free(ptr);
                }
            }
        };
    };

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
    class multi_pointer
    {
        template <typename ...X>
        friend class multi_pointer;

        // number of data members
        static constexpr std::size_t N = sizeof...(T);
        static_assert(N > 0, "error: no template arguments specified");

        // all members have the same type: get this type
        using head_type = typename AUXILIARY_NAMESPACE::variadic::pack<T...>::template type<0>;

        // check if all types are the same
        static constexpr bool is_homogeneous = AUXILIARY_NAMESPACE::variadic::pack<T...>::is_same();
        static_assert(is_homogeneous, "error: use the inhomogeneous multi pointer instead");

        // size of the homogeneous structured type
        static constexpr std::size_t record_size = N * sizeof(head_type);

        // create tuple from the base pointer
        template <std::size_t ...I>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline std::tuple<T&...> get_values(std::index_sequence<I...>)
        {
            return {ptr[I * n_0]...};
        }

    public:

        using value_type = head_type;

        // base pointer and extent of the innermost dimension (w.r.t. a multidimensional field declaration)
        const std::size_t n_0;
        value_type* __restrict__ ptr;

        // constructor: from external base pointer and innermist dimension
        multi_pointer(value_type* __restrict__ ptr, const std::size_t n_0)
            :
            n_0(n_0),
            ptr(ptr) {}

        // constructor: from an existing multi_pointer and a stab index (stab_idx) and an intra-stab index (idx)
        HOST_VERSION
        CUDA_DEVICE_VERSION
        multi_pointer(const multi_pointer& mp, const std::size_t stab_idx, const std::size_t idx)
            :
            n_0(mp.n_0),
            ptr(&mp.ptr[stab_idx * N * n_0 + idx]) {}

        // copy /conversion constructors
        multi_pointer(const multi_pointer<typename std::remove_cv<T>::type...>& mp)
            :
            n_0(mp.n_0),
            ptr(reinterpret_cast<value_type*>(mp.ptr)) {}

        multi_pointer(const multi_pointer<const typename std::remove_cv<T>::type...>& mp)
            :
            n_0(mp.n_0),
            ptr(reinterpret_cast<value_type*>(mp.ptr)) {}

        // get a new multi_pointer shifted by stab_idx and idx
        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline multi_pointer at(const std::size_t idx)
        {
            return multi_pointer(*this, 0, idx);
        }

        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline multi_pointer<const T...> at(const std::size_t idx) const
        {
            return multi_pointer<const T...>(*this, 0, idx);
        }

        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline multi_pointer at(const std::size_t stab_idx, const std::size_t idx)
        {
            return multi_pointer(*this, stab_idx, idx);
        }

        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline multi_pointer<const T...> at(const std::size_t stab_idx, const std::size_t idx) const
        {
            return multi_pointer<const T...>(*this, stab_idx, idx);
        }

        // dereference
        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline std::tuple<T&...> operator*()
        {
            return get_values(std::make_index_sequence<N>{});
        }

        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline std::tuple<const T&...> operator*() const
        {
            return get_values(std::make_index_sequence<N>{});
        }

        // get base pointer
        inline value_type* get_pointer()
        {
            return ptr;
        }

        inline const value_type* get_pointer() const
        {
            return ptr;
        }

        // pointer increment
        inline multi_pointer& operator++()
        {
            ++ptr;
            return *this;
        }

        inline multi_pointer operator++(int)
        {
            multi_pointer mp(*this);
            ++ptr;
            return mp;
        }

        inline multi_pointer& operator+=(const std::size_t n)
        {
            ptr += n;
            return *this;
        }

        // comparison
        inline bool operator==(const multi_pointer& p) const
        {
            return (ptr == p.ptr);
        }

        inline bool operator!=(const multi_pointer& p) const
        {
            return (ptr != p.ptr);
        }

        // allocator class
        class allocator : public XXX_NAMESPACE::pointer<value_type>::allocator
        {
            using base = typename XXX_NAMESPACE::pointer<value_type>::allocator;
            
        public:

            template <std::size_t D>
            static typename base::pointer allocate(const sarray<std::size_t, D>& n, const std::size_t alignment = base::default_alignment)
            {
                const std::size_t num_elements = (base::padding(n[0], alignment) * N) * n.reduce_mul(1);

                return base::allocate(num_elements, alignment);
            }
        };
    };

    // define N-dimensional homogeneous structured type
    namespace
    {
        // defines 'template <T, N> struct type_gen {..};'
        MACRO_VARIADIC_TYPE_GEN(XXX_NAMESPACE::multi_pointer);
    }

    template <typename T, std::size_t N>
    using multi_pointer_n = typename type_gen<T, N>::type;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    //! \brief A pointer wrapper for inhomogeneous structured types and SoA data layout
    //!
    //! It provides some functionality regarding memory (de)allocation and accessing it in the multi-
    //! dimensional case with different data layouts: here SoA!
    //!
    //! Idea: Similar to the homogeneous multi_pointer, but a bit more complicated to implement as
    //! multiple base pointers need to be managed internally, one for each data member of the inhomogeneous
    //! structured type.
    //! 
    //! THE POINTER MANAGED BY THIS CLASS IS EXTERNAL!
    //! 
    //! \tparam T... types (one for each data member; can be all different)
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <typename ...T>
    class multi_pointer_inhomogeneous
    {
        template <typename ...X>
        friend class multi_pointer_inhomogeneous;

        // number of data members
        static constexpr std::size_t N = sizeof...(T);
        static_assert(N > 0, "error: no template arguments specified");
        
        // get the type of the first template argument
        using head_type = typename AUXILIARY_NAMESPACE::variadic::pack<T...>::template type<0>;

        // check if all types are the same: we don't want that here
        static constexpr bool is_homogeneous = AUXILIARY_NAMESPACE::variadic::pack<T...>::is_same();
        static_assert(!is_homogeneous, "error: use the homogeneous multi pointer instead");

        using size_array = XXX_NAMESPACE::sarray<std::size_t, N>;
        // base pointers (of different type) are managed internally by using a tuple
        using pointer_tuple = std::tuple<T* __restrict__...>;
        
        // find out the byte-size of the largest type
        static constexpr std::size_t size_largest_type = AUXILIARY_NAMESPACE::variadic::pack<T...>::size_of_largest_type();

        // determine the total byte-size of all data members that have a size different (smaller) than the largest type
        static constexpr std::size_t size_rest = AUXILIARY_NAMESPACE::variadic::pack<T...>::size_of_pack_excluding_largest_type();

        // size of the inhomogeneous structured type
        static constexpr std::size_t record_size = AUXILIARY_NAMESPACE::variadic::pack<T...>::size_of_pack();

        // determine the number of elements of the structured type that is needed so that their overall size
        // is an integral multiple of each data member type
        static constexpr std::size_t record_padding_factor = MATH_NAMESPACE::least_common_multiple(size_largest_type, size_rest) / std::max(1UL, size_rest);

        // determine the scaling factor of each member-type-size w.r.t. to the largest type
        static constexpr size_array size_scaling_factor{size_largest_type / sizeof(T)...};

        // (exclusive) prefix sum over the byte-sizes of the template arguments
        static constexpr size_array offset = MATH_NAMESPACE::prefix_sum(size_array{sizeof(T)...});
    
        // create a pointer tuple from a base pointer and the 'offset's for a field with extent of the innermost dimension 'n_0'
        template <std::size_t ...I>
        inline constexpr pointer_tuple make_pointer_tuple(std::uint8_t* __restrict__ ptr, const std::size_t n_0, std::index_sequence<I...>)
        {
            return {reinterpret_cast<T*>(&ptr[offset[I] * n_0])...};
        }

        // create a pointer tuple from an existing pointer tuple, a stab index (stab_idx) and an intra-stab index (idx)
        template <std::size_t ...I>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline constexpr pointer_tuple make_pointer_tuple(const pointer_tuple& ptr, const std::size_t stab_idx, const std::size_t idx, std::index_sequence<I...>)
        {
            return {std::get<I>(ptr) + stab_idx * num_units * size_scaling_factor[I] + idx...};
        }

        // increment the pointer tuple
        inline constexpr void increment_pointer_tuple(const std::size_t inc = 1)
        {
            AUXILIARY_NAMESPACE::variadic::loop<N>::execute([inc, this] (auto& I) {std::get<I.value>(ptr) += inc;});
        }

        // create tuple from the base pointer
        template <std::size_t ...I>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline std::tuple<T&...> get_values(std::index_sequence<I...>)
        {
            return {*(std::get<I>(ptr))...};
        }

        // extent of the innermost dimension of the filed in units of largest type
        const std::size_t num_units;

    public:

        // all members have different type: use std::uint8_t for all of them
        using value_type = std::uint8_t;

        // multiple base pointers
        pointer_tuple ptr;

        // constructor: from external base pointer and innermist dimension
        multi_pointer_inhomogeneous(std::uint8_t* __restrict__ ptr, const std::size_t n_0)
            :
            num_units((n_0 * record_size) / size_largest_type),
            ptr(make_pointer_tuple(ptr, n_0, std::make_index_sequence<N>{})) {}

        // constructor: from an existing multi_pointer and a stab index (stab_idx) and an intra-stab index (idx)
        HOST_VERSION
        CUDA_DEVICE_VERSION
        multi_pointer_inhomogeneous(const multi_pointer_inhomogeneous& mp, const std::size_t stab_idx, const std::size_t idx)
            :
            num_units(mp.num_units),
            ptr(make_pointer_tuple(mp.ptr, stab_idx, idx, std::make_index_sequence<N>{})) {}

        // copy / conversion constructors
        multi_pointer_inhomogeneous(const multi_pointer_inhomogeneous<typename std::remove_cv<T>::type...>& mp)
            :
            num_units(mp.num_units),
            ptr(mp.ptr) {}

        multi_pointer_inhomogeneous(const multi_pointer_inhomogeneous<const typename std::remove_cv<T>::type...>& mp)
            :
            num_units(mp.num_units),
            ptr(mp.ptr) {}

        // get a new multi_pointer shifted by [stab_idx and] 
        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline multi_pointer_inhomogeneous at(const std::size_t idx)
        {
            return multi_pointer_inhomogeneous(*this, 0, idx);
        }

        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline multi_pointer_inhomogeneous<const T...> at(const std::size_t idx) const
        {
            return multi_pointer_inhomogeneous<const T...>(*this, 0, idx);
        }

        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline multi_pointer_inhomogeneous at(const std::size_t stab_idx, const std::size_t idx)
        {
            return multi_pointer_inhomogeneous(*this, stab_idx, idx);
        }

        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline multi_pointer_inhomogeneous<const T...> at(const std::size_t stab_idx, const std::size_t idx) const
        {
            return multi_pointer_inhomogeneous<const T...>(*this, stab_idx, idx);
        }

        // dereference
        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline std::tuple<T&...> operator*()
        {
            return get_values(std::make_index_sequence<N>{});
        }

        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline std::tuple<const T&...> operator*() const
        {
            return get_values(std::make_index_sequence<N>{});
        }
        
        // get base pointer
        inline value_type* get_pointer()
        {
            return reinterpret_cast<value_type*>(std::get<0>(ptr));
        }

        inline const value_type* get_pointer() const
        {
            return reinterpret_cast<const value_type*>(std::get<0>(ptr));
        }

        // pointer increment
        inline multi_pointer_inhomogeneous& operator++()
        {
            increment_pointer_tuple();
            return *this;
        }

        inline multi_pointer_inhomogeneous operator++(int)
        {
            multi_pointer_inhomogeneous mp(*this);
            increment_pointer_tuple();
            return mp;
        }

        inline multi_pointer_inhomogeneous& operator+=(const std::size_t n)
        {
            increment_pointer_tuple(n);
            return *this;
        }

        // comparison
        inline bool operator==(const multi_pointer_inhomogeneous& p) const
        {
            return (std::get<0>(ptr) == std::get<0>(p.ptr));
        }

        inline bool operator!=(const multi_pointer_inhomogeneous& p) const
        {
            return (std::get<0>(ptr) != std::get<0>(p.ptr));
        }

        // allocator class
        class allocator : public XXX_NAMESPACE::pointer<value_type>::allocator
        {
            using base = typename XXX_NAMESPACE::pointer<value_type>::allocator;
            
        public:

            static std::size_t padding(const std::size_t n, const std::size_t alignment = base::default_alignment)
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

            template <std::size_t D>
            static typename base::pointer allocate(const sarray<std::size_t, D>& n, const std::size_t alignment = base::default_alignment)
            {
                const std::size_t num_elements = base::padding(n[0], alignment) * n.reduce_mul(1) * record_size;

                return base::allocate(num_elements, alignment);
            }
        };
    };
}

#endif