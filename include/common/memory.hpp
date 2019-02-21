// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(COMMON_pointer_HPP)
#define COMMON_pointer_HPP

#include <cstdint>
#include <tuple>

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
    struct pointer
    {
        using type = T;

        const std::size_t n_0;
        T* __restrict__ ptr;

        pointer(T* __restrict__ ptr, const std::size_t n_0)
            :
            n_0(n_0),
            ptr(ptr) {}

        template <typename TT>
        pointer(const pointer<TT>& p)
            :
            n_0(p.n_0),
            ptr(reinterpret_cast<T*>(p.ptr)) {}

        pointer at(const std::size_t stab_idx)
        {
            return pointer(&ptr[n_0 * stab_idx], n_0);
        }

        pointer<const T> at(const std::size_t stab_idx) const
        {
            return pointer<const T>(&ptr[n_0 * stab_idx], n_0);
        }

        T& at(const std::size_t stab_idx, const std::size_t idx)
        {
            return ptr[n_0 * stab_idx + idx];
        }

        const T& at(const std::size_t stab_idx, const std::size_t idx) const
        {
            return ptr[n_0 * stab_idx + idx];
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
                std::cerr << "error in pointer::allocate : n[0] does not match alignment" << std::endl;
            }

            return reinterpret_cast<T*>(_mm_malloc(n.reduce_mul() * sizeof(T), alignment));
        }

        static void deallocate(pointer& p)
        {
            if (p.ptr)
            {
                _mm_free(p.ptr);
            }
        }
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
    template <typename ... T>
    class multi_pointer
    {
        template <typename ... X>
        friend class multi_pointer;

#if defined(__INTEL_COMPILER)
        template <typename T_head, typename ... T_tail>
        struct have_same_size
        {
            static constexpr bool value = (sizeof(T_head) == have_same_size<T_tail ...>::size);
            static constexpr std::size_t size = (value ? sizeof(T_head) : 0);
        };

        template <typename T_head>
        struct have_same_size<T_head>
        {
            static constexpr bool value = true;
            static constexpr std::size_t size = sizeof(T_head);
        };
#endif

        // number of data members
        static constexpr std::size_t N = sizeof ... (T);
        static_assert(N > 0, "error: no template arguments specified");

        // all members have the same type: get this type
        using ptr_type = typename AUXILIARY_NAMESPACE::variadic::argument<0, T ...>::type;

        // check if all types are the same
#if defined(__INTEL_COMPILER)
        static constexpr bool is_homogeneous = have_same_size<T ...>::value;
#else
        static constexpr bool is_homogeneous = AUXILIARY_NAMESPACE::variadic::fold([](const bool result, const bool is_same) constexpr { return result && is_same; }, true, std::is_same<ptr_type, T>::value ...);
#endif
        static_assert(is_homogeneous, "error: use the inhomogeneous multi pointer instead");

        // size of the homogeneous structured type
        static constexpr std::size_t record_size = N * sizeof(ptr_type);

    public:

        // base pointer and extent of the innermost dimension (w.r.t. a multidimensional field declaration)
        const std::size_t n_0;
        ptr_type* __restrict__ ptr;

        // constructor: from external base pointer and innermist dimension
        multi_pointer(ptr_type* __restrict__ ptr, const std::size_t n_0)
            :
            n_0(n_0),
            ptr(ptr) {}

        // constructor: from an existing multi_pointer and a stab index (stab_idx) and an intra-stab index (idx)
        multi_pointer(const multi_pointer& mp, const std::size_t stab_idx, const std::size_t idx)
            :
            n_0(mp.n_0),
            ptr(&mp.ptr[stab_idx * N * n_0 + idx]) {}

        // copy constructors
        multi_pointer(const multi_pointer<typename std::remove_cv<T>::type ...>& mp)
            :
            n_0(mp.n_0),
            ptr(reinterpret_cast<ptr_type*>(mp.ptr)) {}

        multi_pointer(const multi_pointer<const typename std::remove_cv<T>::type ...>& mp)
            :
            n_0(mp.n_0),
            ptr(reinterpret_cast<ptr_type*>(mp.ptr)) {}

        // get a new multi_pointer shifted by stab_idx and idx
        multi_pointer at(const std::size_t stab_idx, const std::size_t idx = 0)
        {
            return multi_pointer(*this, stab_idx, idx);
        }

        multi_pointer<const T ...> at(const std::size_t stab_idx, const std::size_t idx = 0) const
        {
            return multi_pointer<const T ...>(*this, stab_idx, idx);
        }

        // this function provides the padded value for a given 'n' taking into account the desired alignment
        static std::size_t padding(const std::size_t n, const std::size_t alignment = SIMD_NAMESPACE::simd::alignment)
        {
            if (!AUXILIARY_NAMESPACE::is_power_of<2>(alignment))
            {
                std::cerr << "warning: alignment is not a power of 2" << std::endl;
                return n;
            }

            const std::size_t ratio = AUXILIARY_NAMESPACE::least_common_multiple(alignment, sizeof(ptr_type)) / sizeof(ptr_type);
            
            return ((n + ratio - 1) / ratio) * ratio;
        }

        // allocate a contigous chunk of memory for a field of size 'n' (d-dimensional)
        template <std::size_t D>
        static ptr_type* allocate(const sarray<std::size_t, D>& n, const std::size_t alignment = SIMD_NAMESPACE::simd::alignment)
        {
            if (n[0] != padding(n[0], alignment))
            {
                std::cerr << "error in vec_proxy::pointer::allocate : n[0] does not match alignment" << std::endl;
            }

            return reinterpret_cast<ptr_type*>(_mm_malloc(n.reduce_mul() * record_size, alignment));
        }

        // deallocate memory
        static void deallocate(multi_pointer& mp)
        {
            if (mp.ptr)
            {
                _mm_free(mp.ptr);
            }
        }
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
    template <typename ... T>
    class multi_pointer_inhomogeneous
    {
        template <typename ... X>
        friend class multi_pointer_inhomogeneous;
        
#if defined(__INTEL_COMPILER)
        template <typename T_head, typename ... T_tail>
        struct size_info
        {
            static constexpr bool equal = (sizeof(T_head) == size_info<T_tail ...>::size);
            static constexpr std::size_t size = (equal ? sizeof(T_head) : 0);
            static constexpr std::size_t size_all = sizeof(T_head) + size_info<T_tail ...>::size_all;
            static constexpr std::size_t max_size = std::max(sizeof(T_head), size_info<T_tail ...>::max_size);

            static constexpr std::size_t get_size_rest(const std::size_t max_size, const std::size_t size_rest = 0)
            {
                return size_info<T_tail ...>::get_size_rest(max_size, size_rest + (sizeof(T_head) == max_size ? 0 : sizeof(T_head)));
            }
        };

        template <typename T_head>
        struct size_info<T_head>
        {
            static constexpr bool equal = true;
            static constexpr std::size_t size = sizeof(T_head);
            static constexpr std::size_t size_all = sizeof(T_head);
            static constexpr std::size_t max_size = sizeof(T_head);

            static constexpr std::size_t get_size_rest(const std::size_t max_size, const std::size_t size_rest = 0)
            {
                return (size_rest + (sizeof(T_head) == max_size ? 0 : sizeof(T_head)));
            }
        };
#endif

        // number of data members
        static constexpr std::size_t N = sizeof ... (T);
        static_assert(N > 0, "error: no template arguments specified");

        // get the type of the first template argument
        using head_type = typename AUXILIARY_NAMESPACE::variadic::argument<0, T ...>::type;

        // check if all types are the same: we don't want that here
#if defined(__INTEL_COMPILER)
        static constexpr bool is_homogeneous = size_info<T ...>::equal;
#else
        static constexpr bool is_homogeneous = AUXILIARY_NAMESPACE::variadic::fold([](const bool result, const bool is_same) constexpr { return result && is_same; }, true, std::is_same<head_type, T>::value ...);
#endif
        static_assert(!is_homogeneous, "error: use the homogeneous multi pointer instead");

        using size_array = XXX_NAMESPACE::sarray<std::size_t, N>;
        // base pointers (of different type) are managed internally by using a tuple
        using pointer_tuple = std::tuple<T* __restrict__ ...>;
        
        // find out the byte-size of the largest type
#if defined(__INTEL_COMPILER)
        static constexpr std::size_t size_largest_type = size_info<T ...>::max_size;
#else
        static constexpr std::size_t size_largest_type = AUXILIARY_NAMESPACE::variadic::fold(
            [](const std::size_t max_size, const std::size_t argument_size) constexpr { return std::max(max_size, argument_size); }, 
            0, 
            sizeof(T) ...);
#endif

        // determine the total byte-size of all data members that have a size different (smaller) than the largest type
#if defined(__INTEL_COMPILER)
        static constexpr std::size_t size_rest = size_info<T ...>::get_size_rest(size_largest_type);
#else
        static constexpr std::size_t size_rest = AUXILIARY_NAMESPACE::variadic::fold(
            [](const std::size_t size, const std::size_t argument_size) constexpr { return size + argument_size; }, 
            0, 
            (size_largest_type == sizeof(T) ? 0 : sizeof(T)) ...);
#endif

        // size of the inhomogeneous structured type
#if defined(__INTEL_COMPILER)
	static constexpr std::size_t record_size = size_info<T ...>::size_all;
#else
        static constexpr std::size_t record_size = AUXILIARY_NAMESPACE::variadic::fold(
            [](const std::size_t size, const std::size_t argument_size) constexpr { return size + argument_size; }, 
            0, 
            sizeof(T) ...);
#endif

        // determine the number of elements of the structured type that is needed so that their overall size
        // is an integral multiple of each data member type
        static constexpr std::size_t record_padding_factor = AUXILIARY_NAMESPACE::least_common_multiple(size_largest_type, size_rest) / std::max(1UL, size_rest);

        // determine the scaling factor of each member-type-size w.r.t. to the largest type
        static constexpr size_array size_scaling_factor{size_largest_type / sizeof(T) ...};

        // (exclusive) prefix sum over the byte-sizes of the template arguments
        static constexpr size_array offset = AUXILIARY_NAMESPACE::prefix_sum(size_array{sizeof(T) ...});
    
        // create a pointer tuple from a base pointer and the 'offset's for a field with extent of the innermost dimension 'n_0'
        template <std::size_t ... I>
        inline pointer_tuple make_pointer_tuple(std::uint8_t* __restrict__ ptr, const std::size_t n_0, std::index_sequence<I ...>)
        {
            return {reinterpret_cast<T* __restrict__>(&ptr[offset[I] * n_0]) ...};
        }

        // create a pointer tuple from an existing pointer tuple, a stab index (stab_idx) and an intra-stab index (idx)
        template <std::size_t ...I>
        inline pointer_tuple make_pointer_tuple(const pointer_tuple& ptr, const std::size_t stab_idx, const std::size_t idx, std::index_sequence<I ...>)
        {
            return {reinterpret_cast<T* __restrict__>(std::get<I>(ptr)) + stab_idx * num_units * size_scaling_factor[I] + idx ...};
        }

        // extent of the innermost dimension of the filed in units of largest type
        const std::size_t num_units;

    public:

        // multiple base pointers
        pointer_tuple ptr;

        // constructor: from external base pointer and innermist dimension
        multi_pointer_inhomogeneous(std::uint8_t* __restrict__ ptr, const std::size_t n_0)
            :
            num_units((n_0 * record_size) / size_largest_type),
            ptr(make_pointer_tuple(ptr, n_0, std::make_index_sequence<N>{})) {}

        // constructor: from an existing multi_pointer and a stab index (stab_idx) and an intra-stab index (idx)
        multi_pointer_inhomogeneous(const multi_pointer_inhomogeneous& mp, const std::size_t stab_idx, const std::size_t idx)
            :
            num_units(mp.num_units),
            ptr(make_pointer_tuple(mp.ptr, stab_idx, idx, std::make_index_sequence<N>{})) {}

        // copy constructors
        multi_pointer_inhomogeneous(const multi_pointer_inhomogeneous<typename std::remove_cv<T>::type ...>& mp)
            :
            num_units(mp.num_units),
            ptr(mp.ptr) {}

        multi_pointer_inhomogeneous(const multi_pointer_inhomogeneous<const typename std::remove_cv<T>::type ...>& mp)
            :
            num_units(mp.num_units),
            ptr(mp.ptr) {}

        // get a new multi_pointer shifted by stab_idx and idx
        multi_pointer_inhomogeneous at(const std::size_t stab_idx, const std::size_t idx = 0)
        {
            return multi_pointer_inhomogeneous(*this, stab_idx, idx);
        }

        multi_pointer_inhomogeneous<const T ...> at(const std::size_t stab_idx, const std::size_t idx = 0) const
        {
            return multi_pointer_inhomogeneous<const T ...>(*this, stab_idx, idx);
        }

        // this function provides the padded value for a given 'n' taking into account the desired alignment
        // and the internal record padding
        static std::size_t padding(const std::size_t n, const std::size_t alignment = SIMD_NAMESPACE::simd::alignment)
        {
            if (!AUXILIARY_NAMESPACE::is_power_of<2>(alignment))
            {
                std::cerr << "warning: alignment is not a power of 2" << std::endl;
                return n;
            }

            const std::size_t byte_padding_factor = AUXILIARY_NAMESPACE::least_common_multiple(alignment, size_largest_type) / size_largest_type;
            const std::size_t ratio = AUXILIARY_NAMESPACE::least_common_multiple(record_padding_factor, byte_padding_factor);

            return ((n + ratio - 1) / ratio) * ratio;
        }

        // allocate a contigous chunk of memory for a field of size 'n' (d-dimensional)
        template <std::size_t D>
        static std::uint8_t* allocate(const sarray<std::size_t, D>& n, const std::size_t alignment = SIMD_NAMESPACE::simd::alignment)
        {
            if (n[0] != padding(n[0], alignment))
            {
                std::cerr << "error in vec_proxy::pointer::allocate : n[0] does not match alignment" << std::endl;
            }

            return reinterpret_cast<std::uint8_t*>(_mm_malloc(n.reduce_mul() * record_size, alignment));
        }

        // deallocate memory
        static void deallocate(multi_pointer_inhomogeneous& mp)
        {
            if (std::get<0>(mp.ptr))
            {
                _mm_free(reinterpret_cast<std::uint8_t*>(std::get<0>(mp.ptr)));
            }
        }
    };
}

#endif