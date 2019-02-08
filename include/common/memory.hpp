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

        const std::size_t n_innermost;
        T* __restrict__ base;

        pointer(T* __restrict__ base, const std::size_t n_innermost)
            :
            n_innermost(n_innermost),
            base(base) {}

        template <typename TT>
        pointer(const pointer<TT>& m)
            :
            n_innermost(m.n_innermost),
            base(reinterpret_cast<T*>(m.base)) {}

        T& at(const std::size_t stab_idx, const std::size_t idx)
        {
            return base[n_innermost * stab_idx + idx];
        }

        const T& at(const std::size_t stab_idx, const std::size_t idx) const
        {
            return base[n_innermost * stab_idx + idx];
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

        static void deallocate(pointer& m)
        {
            if (m.base)
            {
                _mm_free(m.base);
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

        // number of data members
        static constexpr std::size_t N = sizeof ... (T);
        static_assert(N > 0, "error: no template arguments specified");

        // all members have the same type: get this type
        using base_type = typename AUXILIARY_NAMESPACE::variadic::argument<0, T ...>::type;

        // check if all types are the same
        static constexpr bool is_homogeneous = AUXILIARY_NAMESPACE::variadic::fold([](const bool result, const bool is_same) constexpr { return result && is_same; }, true, std::is_same<base_type, T>::value ...);
        static_assert(is_homogeneous, "error: use the inhomogeneous multi pointer instead");

        // size of the homogeneous structured type
        static constexpr std::size_t record_size = N * sizeof(base_type);

    public:

        // base pointer and extent of the innermost dimension (w.r.t. a multidimensional field declaration)
        const std::size_t n_innermost;
        base_type* __restrict__ base;

        // constructor: from external base pointer and innermist dimension
        multi_pointer(base_type* __restrict__ base, const std::size_t n_innermost)
            :
            n_innermost(n_innermost),
            base(base) {}

        // constructor: from an existing multi_pointer and a stab index (stab_idx) and an intra-stab index (idx)
        multi_pointer(const multi_pointer& m, const std::size_t stab_idx, const std::size_t idx)
            :
            n_innermost(m.n_innermost),
            base(&m.base[stab_idx * N * n_innermost + idx]) {}

        // copy constructors
        multi_pointer(const multi_pointer<typename std::remove_cv<T>::type ...>& m)
            :
            n_innermost(m.n_innermost),
            base(reinterpret_cast<base_type*>(m.base)) {}

        multi_pointer(const multi_pointer<const typename std::remove_cv<T>::type ...>& m)
            :
            n_innermost(m.n_innermost),
            base(reinterpret_cast<base_type*>(m.base)) {}

        // get a new multi_pointer shifted by stab_idx and idx
        multi_pointer at(const std::size_t stab_idx, const std::size_t idx)
        {
            return multi_pointer(*this, stab_idx, idx);
        }

        multi_pointer at(const std::size_t stab_idx, const std::size_t idx) const
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

            const std::size_t ratio = AUXILIARY_NAMESPACE::least_common_multiple(alignment, sizeof(base_type)) / sizeof(base_type);
            
            return ((n + ratio - 1) / ratio) * ratio;
        }

        // allocate a contigous chunk of memory for a field of size 'n' (d-dimensional)
        template <std::size_t D>
        static base_type* allocate(const sarray<std::size_t, D>& n, const std::size_t alignment = SIMD_NAMESPACE::simd::alignment)
        {
            if (n[0] != padding(n[0], alignment))
            {
                std::cerr << "error in vec_proxy::pointer::allocate : n[0] does not match alignment" << std::endl;
            }

            return reinterpret_cast<base_type*>(_mm_malloc(n.reduce_mul() * record_size, alignment));
        }

        // deallocate memory
        static void deallocate(multi_pointer& m)
        {
            if (m.base)
            {
                _mm_free(m.base);
            }
        }
    };

    // define N-dimensional homogeneous structured type
    namespace
    {
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
        
        // number of data members
        static constexpr std::size_t N = sizeof ... (T);
        static_assert(N > 0, "error: no template arguments specified");

        // get the type of the first template argument
        using head_type = typename AUXILIARY_NAMESPACE::variadic::argument<0, T ...>::type;

        // check if all types are the same: we don't want that here
        static constexpr bool is_homogeneous = AUXILIARY_NAMESPACE::variadic::fold([](const bool result, const bool is_same) constexpr { return result && is_same; }, true, std::is_same<head_type, T>::value ...);
        static_assert(!is_homogeneous, "error: use the homogeneous multi pointer instead");

        using size_array = XXX_NAMESPACE::sarray<std::size_t, N>;
        // base pointers (of different type) are managed internally by using a tuple
        using pointer_tuple = std::tuple<T* __restrict__ ...>;
        
        // find out the byte-size of the largest type
        static constexpr std::size_t size_largest_type = AUXILIARY_NAMESPACE::variadic::fold(
            [](const std::size_t max_size, const std::size_t argument_size) constexpr { return std::max(max_size, argument_size); }, 
            0, 
            sizeof(T) ...);

        // determine the total byte-size of all data members that have a size different (smaller) than the largest type
        static constexpr std::size_t size_rest = AUXILIARY_NAMESPACE::variadic::fold(
            [](const std::size_t size, const std::size_t argument_size) constexpr { return size + argument_size; }, 
            0, 
            (size_largest_type == sizeof(T) ? 0 : sizeof(T)) ...);

        // size of the inhomogeneous structured type
        static constexpr std::size_t record_size = AUXILIARY_NAMESPACE::variadic::fold(
            [](const std::size_t size, const std::size_t argument_size) constexpr { return size + argument_size; }, 
            0, 
            sizeof(T) ...);

        // determine the number of elements of the structured type that is needed so that their overall size
        // is an integral multiple of each data member type
        static constexpr std::size_t record_padding_factor = AUXILIARY_NAMESPACE::least_common_multiple(size_largest_type, size_rest) / std::max(1UL, size_rest);

        // determine the scaling factor of each member-type-size w.r.t. to the largest type
        static constexpr size_array size_scaling_factor{size_largest_type / sizeof(T) ...};

        // (exclusive) prefix sum over the byte-sizes of the template arguments
        static constexpr size_array offset = AUXILIARY_NAMESPACE::prefix_sum(size_array{sizeof(T) ...});
    
        // create a pointer tuple from a base pointer and the 'offset's for a field with extent of the innermost dimension 'n_innermost'
        template <std::size_t ... I>
        inline pointer_tuple make_pointer_tuple(std::uint8_t* __restrict__ base, const std::size_t n_innermost, std::index_sequence<I ...>)
        {
            return {reinterpret_cast<T* __restrict__>(&base[offset[I] * n_innermost]) ...};
        }

        // create a pointer tuple from an existing pointer tuple, a stab index (stab_idx) and an intra-stab index (idx)
        template <std::size_t ...I>
        inline pointer_tuple make_pointer_tuple(const pointer_tuple& base, const std::size_t stab_idx, const std::size_t idx, std::index_sequence<I ...>)
        {
            return {reinterpret_cast<T* __restrict__>(std::get<I>(base)) + stab_idx * num_units * size_scaling_factor[I] + idx ...};
        }

        // extent of the innermost dimension of the filed in units of largest type
        const std::size_t num_units;

    public:

        // multiple base pointers
        pointer_tuple base;

        // constructor: from external base pointer and innermist dimension
        multi_pointer_inhomogeneous(std::uint8_t* __restrict__ base, const std::size_t n_innermost)
            :
            num_units((n_innermost * record_size) / size_largest_type),
            base(make_pointer_tuple(base, n_innermost, std::make_index_sequence<N>{})) {}

        // constructor: from an existing multi_pointer and a stab index (stab_idx) and an intra-stab index (idx)
        multi_pointer_inhomogeneous(const multi_pointer_inhomogeneous& m, const std::size_t stab_idx, const std::size_t idx)
            :
            num_units(m.num_units),
            base(make_pointer_tuple(m.base, stab_idx, idx, std::make_index_sequence<N>{})) {}

        // copy constructors
        multi_pointer_inhomogeneous(const multi_pointer_inhomogeneous<typename std::remove_cv<T>::type ...>& m)
            :
            num_units(m.num_units),
            base(m.base) {}

        multi_pointer_inhomogeneous(const multi_pointer_inhomogeneous<const typename std::remove_cv<T>::type ...>& m)
            :
            num_units(m.num_units),
            base(m.base) {}

        // get a new multi_pointer shifted by stab_idx and idx
        multi_pointer_inhomogeneous at(const std::size_t stab_idx, const std::size_t idx)
        {
            return multi_pointer_inhomogeneous(*this, stab_idx, idx);
        }

        multi_pointer_inhomogeneous at(const std::size_t stab_idx, const std::size_t idx) const
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
        static void deallocate(multi_pointer_inhomogeneous& m)
        {
            if (std::get<0>(m.base))
            {
                _mm_free(reinterpret_cast<std::uint8_t*>(std::get<0>(m.base)));
            }
        }
    };
}

#endif