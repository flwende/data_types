// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(COMMON_MEMORY_HPP)
#define COMMON_MEMORY_HPP

#include <cstdint>
#include <memory>
#include <tuple>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#include <common/Math.hpp>
#include <auxiliary/Template.hpp>
#include <common/DataLayout.hpp>
#include <data_types/DataTypes.hpp>
#include <platform/Target.hpp>
#include <data_types/array/Array.hpp>
#include <platform/simd/Simd.hpp>

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
        static constexpr SizeType N = ::XXX_NAMESPACE::variadic::Pack<T...>::Size;
        static_assert(N > 0, "error: empty parameter pack");

        // all members have the same type: get this type
        using value_type = typename ::XXX_NAMESPACE::variadic::Pack<T...>::template Type<0>;

        // check if all types are the same
        static constexpr bool is_homogeneous = ::XXX_NAMESPACE::variadic::Pack<T...>::IsSame() || ::XXX_NAMESPACE::variadic::Pack<T...>::SameSize();
        static_assert(is_homogeneous, "error: use the inhomogeneous multi_pointer instead");

        // create tuple from the base pointer
        template <SizeType ...I>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto get_values(std::integer_sequence<SizeType, I...>)
            -> std::tuple<T&...>
        {
            return {reinterpret_cast<T&>(ptr[I * n_0])...};
        }

        template <SizeType ...I>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto get_values(std::integer_sequence<SizeType, I...>) const
            -> std::tuple<const T&...>
        {
            return {reinterpret_cast<const T&>(ptr[I * n_0])...};
        }

        template <SizeType ...I>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto get_values(const SizeType stab_idx, const SizeType idx, std::integer_sequence<SizeType, I...>)
            -> std::tuple<T&...>
        {
            return {reinterpret_cast<T&>(ptr[(stab_idx * N + I) * n_0 + idx])...};
        }

        template <SizeType ...I>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto get_values(const SizeType stab_idx, const SizeType idx, std::integer_sequence<SizeType, I...>) const
            -> std::tuple<const T&...>
        {
            return {reinterpret_cast<const T&>(ptr[(stab_idx * N + I) * n_0 + idx])...};
        }

        // extent of the innermost dimension (w.r.t. a multidimensional field declaration)
        SizeType n_0;
        // base pointer
        value_type* ptr;

    public:

        pointer()
            :
            n_0(0),
            ptr(nullptr)
        {}

        // constructor: from external base pointer and innermist dimension
        pointer(value_type* ptr, const SizeType n_0)
            :
            n_0(n_0),
            ptr(ptr) 
        {}

        // constructor: from an existing pointer and a stab index (stab_idx) and an intra-stab index (idx)
        HOST_VERSION
        CUDA_DEVICE_VERSION
        pointer(const pointer& p, const SizeType stab_idx, const SizeType idx)
            :
            n_0(p.n_0),
            ptr(&p.ptr[stab_idx * N * n_0 + idx]) 
        {}

        // copy /conversion constructors
        template <typename ...OtherT>
        pointer(const pointer<OtherT...>& p)
            :
            n_0(p.n_0),
            ptr(reinterpret_cast<value_type*>(p.ptr)) 
        {
            static_assert(::XXX_NAMESPACE::variadic::Pack<value_type, OtherT...>::IsConvertible(), "error: types are not convertible");
        }

        auto operator=(const pointer& p)
            -> pointer&
        {
            n_0 = p.n_0;
            ptr = reinterpret_cast<value_type*>(p.ptr);

            return *this;
        }

        // swap
        inline auto swap(pointer& p)
            -> pointer&
        {
            SizeType this_n_0 = n_0;
            n_0 = p.n_0;
            p.n_0 = this_n_0;

            value_type* this_ptr = ptr;
            ptr = p.ptr;
            p.ptr = this_ptr;

            return *this;
        }

        // get a new pointer shifted by stab_idx and idx
        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto at(const SizeType idx)
            -> pointer
        {
            return {*this, 0, idx};
        }

        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto at(const SizeType idx) const
            -> pointer<const T...>
        {
            return {*this, 0, idx};
        }

        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto at(const SizeType stab_idx, const SizeType idx)
            -> pointer
        {
            return {*this, stab_idx, idx};
        }

        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto at(const SizeType stab_idx, const SizeType idx) const
            -> pointer<const T...>
        {
            return {*this, stab_idx, idx};
        }

        // dereference / access
        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto operator*()
        {
            return get_values(std::make_integer_sequence<SizeType, N>{});
        }

        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto operator*() const
        {
            return get_values(std::make_integer_sequence<SizeType, N>{});
        }

        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto access(const SizeType idx)
        {
            return get_values(0, idx, std::make_integer_sequence<SizeType, N>{});
        }

        template <bool Enable = true>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto access(const SizeType idx) const
        {
            return get_values(0, idx, std::make_integer_sequence<SizeType, N>{});
        }

        template <bool Enable = true>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto access(const SizeType stab_idx, const SizeType idx)
        {
            return get_values(stab_idx, idx, std::make_integer_sequence<SizeType, N>{});
        }

        template <bool Enable = true>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto access(const SizeType stab_idx, const SizeType idx) const
        {
            return get_values(stab_idx, idx, std::make_integer_sequence<SizeType, N>{});
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
            pointer p(*this);
            ++ptr;
            return p;
        }

        inline auto operator+=(const SizeType n)
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

            static constexpr SizeType default_alignment = SIMD_NAMESPACE::simd::alignment;

            static auto padding(const SizeType n, const SizeType alignment = default_alignment)
                -> SizeType
            {
                if (!::XXX_NAMESPACE::math::IsPowerOf<2>(alignment))
                {
                    std::cerr << "warning: alignment is not a power of 2" << std::endl;
                    return n;
                }

                const SizeType ratio = ::XXX_NAMESPACE::math::LeastCommonMultiple(alignment, static_cast<SizeType>(sizeof(value_type))) / static_cast<SizeType>(sizeof(value_type));

                return ((n + ratio - 1) / ratio) * ratio;
            }

        public:

            template <::XXX_NAMESPACE::memory::DataLayout L, SizeType D, bool Enable = true>
            static auto get_allocation_shape(const sarray<SizeType, D>& n, const SizeType alignment = default_alignment)
                -> typename std::enable_if<(L != ::XXX_NAMESPACE::memory::DataLayout::SoA && Enable), std::pair<SizeType, SizeType>>::type
            {
                return {padding(n[0], alignment), N * n.reduce_mul(1)};
            }

            template <::XXX_NAMESPACE::memory::DataLayout L, SizeType D, bool Enable = true>
            static auto get_allocation_shape(const sarray<SizeType, D>& n, const SizeType alignment = default_alignment)
                -> typename std::enable_if<(L == ::XXX_NAMESPACE::memory::DataLayout::SoA && Enable), std::pair<SizeType, SizeType>>::type
            {
                return {padding(n.reduce_mul(), alignment), N};
            }

            static auto get_byte_size(const std::pair<SizeType, SizeType>& allocation_shape)
                -> SizeType
            {
                return allocation_shape.first * allocation_shape.second * sizeof(value_type);
            }

            template <XXX_NAMESPACE::target Target, bool Enable = true>
            static auto allocate(const std::pair<SizeType, SizeType>& allocation_shape, const SizeType alignment = default_alignment)
                -> typename std::enable_if<(Target == XXX_NAMESPACE::target::Host && Enable), value_type*>::type
            {
                // NOTE: aligned_alloc results in a segfault here -> use _mm_malloc
                return reinterpret_cast<value_type*>(_mm_malloc(get_byte_size(allocation_shape), alignment));
            }

            template <XXX_NAMESPACE::target Target, bool Enable = true>
            static auto deallocate(pointer& p)
                -> typename std::enable_if<(Target == XXX_NAMESPACE::target::Host && Enable), void>::type
            {
                if (p.get_pointer())
                {
                    _mm_free(p.get_pointer());
                }
            }

            #if defined(__CUDACC__)
            template <XXX_NAMESPACE::target Target, bool Enable = true>
            static auto allocate(const std::pair<SizeType, SizeType>& allocation_shape, const SizeType alignment = default_alignment)
                -> typename std::enable_if<(Target == XXX_NAMESPACE::target::GPU_CUDA && Enable), value_type*>::type
            {
                const SizeType num_elements = allocation_shape.first * allocation_shape.second;
                value_type* d_ptr;

                cudaMalloc((void**)&d_ptr, num_elements * sizeof(value_type));

                return d_ptr;
            }

            template <XXX_NAMESPACE::target Target, bool Enable = true>
            static auto deallocate(pointer& p)
                -> typename std::enable_if<(Target == XXX_NAMESPACE::target::GPU_CUDA && Enable), void>::type
            {
                if (p.get_pointer())
                {
                    cudaFree(p.get_pointer());
                }
            }
            #endif
        };
    };

    // define N-dimensional homogeneous structured type
    namespace
    {
        // defines 'template <T, N> struct TypeGen {..};'
        MACRO_TYPE_GEN(XXX_NAMESPACE::pointer);
    }

    template <typename T, SizeType C_N>
    using pointer_n = typename TypeGen<T, C_N>::Type;

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

        static constexpr SizeType one = static_cast<SizeType>(1);

        // number of data members
        static constexpr SizeType N = ::XXX_NAMESPACE::variadic::Pack<T...>::length;
        static_assert(N > 0, "error: empty parameter pack");
        
        // check if all types are the same: we don't want that here
        static constexpr bool is_homogeneous = ::XXX_NAMESPACE::variadic::Pack<T...>::IsSame() || ::XXX_NAMESPACE::variadic::Pack<T...>::SameSize();
        static_assert(!is_homogeneous, "error: use the homogeneous pointer instead");

        // find out the byte-size of the largest type
        static constexpr SizeType size_largest_type = ::XXX_NAMESPACE::variadic::Pack<T...>::size_of_largest_type();

        // determine the total byte-size of all data members that have a size different (smaller) than the largest type
        static constexpr SizeType size_rest = ::XXX_NAMESPACE::variadic::Pack<T...>::size_of_pack_excluding_largest_type();

        // size of the inhomogeneous structured type
        static constexpr SizeType record_size = ::XXX_NAMESPACE::variadic::Pack<T...>::size_of_pack();

        // determine the number of elements of the structured type that is needed so that their overall size
        // is an integral multiple of each data member type
        static constexpr SizeType record_padding_factor = std::max(one, ::XXX_NAMESPACE::math::LeastCommonMultiple(size_largest_type, size_rest) / std::max(one, size_rest));

        // determine the scaling factor of each member-type-size w.r.t. to the largest type
        static constexpr XXX_NAMESPACE::sarray<SizeType, N> size_scaling_factor{size_largest_type / static_cast<SizeType>(sizeof(T))...};

        // (exclusive) prefix sum over the byte-sizes of the template arguments
        static constexpr XXX_NAMESPACE::sarray<SizeType, N> offset = ::XXX_NAMESPACE::math::PrefixSum(XXX_NAMESPACE::sarray<SizeType, N>{sizeof(T)...});
    
        // create a pointer tuple from a base pointer and the 'offset's for a field with extent of the innermost dimension 'n_0'
        template <SizeType ...I>
        inline constexpr auto make_pointer_tuple(std::uint8_t* __restrict__ ptr, const SizeType n_0, std::integer_sequence<SizeType, I...>)
            -> std::tuple<T*...>
        {
            return {reinterpret_cast<T*>(&ptr[offset[I] * n_0])...};
        }

        // create a pointer tuple from an existing pointer tuple, a stab index (stab_idx) and an intra-stab index (idx)
        template <SizeType ...I>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline constexpr auto make_pointer_tuple(const std::tuple<T*...>& ptr, const SizeType stab_idx, const SizeType idx, std::integer_sequence<SizeType, I...>)
            -> std::tuple<T*...>
        {
            return {std::get<I>(ptr) + stab_idx * num_units * size_scaling_factor[I] + idx...};
        }

        // increment the pointer tuple
        inline constexpr auto increment_pointer_tuple(const SizeType inc = 1)
            -> void
        {
            ::XXX_NAMESPACE::compileTime::Loop<N>::Execute([inc, this] (auto I) {std::get<I>(ptr) += inc;});
        }

        // create tuple from the base pointer
        template <SizeType ...I>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto get_values(std::integer_sequence<SizeType, I...>)
            -> std::tuple<T&...>
        {
            return {*(std::get<I>(ptr))...};
        }

        template <SizeType ...I>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto get_values(std::integer_sequence<SizeType, I...>) const
            -> std::tuple<const T&...>
        {
            return {*(std::get<I>(ptr))...};
        }

        template <SizeType ...I>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto get_values(const SizeType stab_idx, const SizeType idx, std::integer_sequence<SizeType, I...>)
            -> std::tuple<T&...>
        {
            return {*(std::get<I>(ptr) + stab_idx * num_units * size_scaling_factor[I] + idx)...};
        }

        template <SizeType ...I>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto get_values(const SizeType stab_idx, const SizeType idx, std::integer_sequence<SizeType, I...>) const
            -> std::tuple<const T&...>
        {
            return {*(std::get<I>(ptr) + stab_idx * num_units * size_scaling_factor[I] + idx)...};
        }

        // all members have different type: use std::uint8_t for all of them
        using value_type = std::uint8_t;

        // extent of the innermost dimension of the filed in units of largest type
        SizeType num_units;
        // base pointers (of different type) are managed internally by using a tuple
        std::tuple<T*...> ptr;

    public:

        multi_pointer()
            :
            num_units(0),
            ptr{}
        {}

        // constructor: from external base pointer and innermist dimension
        multi_pointer(std::uint8_t* __restrict__ ptr, const SizeType n_0)
            :
            num_units((n_0 * record_size) / size_largest_type),
            ptr(make_pointer_tuple(ptr, n_0, std::make_integer_sequence<SizeType, N>{})) 
        {}

        // constructor: from an existing multi_pointer and a stab index (stab_idx) and an intra-stab index (idx)
        HOST_VERSION
        CUDA_DEVICE_VERSION
        multi_pointer(const multi_pointer& mp, const SizeType stab_idx, const SizeType idx)
            :
            num_units(mp.num_units),
            ptr(make_pointer_tuple(mp.ptr, stab_idx, idx, std::make_integer_sequence<SizeType, N>{})) 
        {}

        // copy / conversion constructors
        template <typename ...OtherT>
        multi_pointer(const multi_pointer<OtherT...>& mp)
            :
            num_units(mp.num_units),
            ptr(mp.ptr) 
        {
            static_assert(::XXX_NAMESPACE::variadic::Pack<value_type, OtherT...>::IsConvertible(), "error: types are not convertible");
        }

        auto operator=(const multi_pointer& mp)
            -> multi_pointer&
        {
            num_units = mp.num_units;
            ptr = mp.ptr;

            return *this;
        }

        // swap
        inline auto swap(multi_pointer& mp)
            -> multi_pointer&
        {
            SizeType this_num_units = num_units;
            num_units = mp.num_units;
            mp.num_units = this_num_units;

            std::tuple<T*...> this_ptr = ptr;
            ptr = mp.ptr;
            mp.ptr = this_ptr;

            return *this;
        }

        // get a new multi_pointer shifted by [stab_idx and] 
        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto at(const SizeType idx)
            -> multi_pointer
        {
            return {*this, 0, idx};
        }

        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto at(const SizeType idx) const
            -> multi_pointer<const T...>
        {
            return {*this, 0, idx};
        }

        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto at(const SizeType stab_idx, const SizeType idx)
            -> multi_pointer
        {
            return {*this, stab_idx, idx};
        }

        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto at(const SizeType stab_idx, const SizeType idx) const
            -> multi_pointer<const T...>
        {
            return {*this, stab_idx, idx};
        }

        // dereference / access
        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto operator*()
        {
            return get_values(std::make_integer_sequence<SizeType, N>{});
        }

        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto operator*() const
        {
            return get_values(std::make_integer_sequence<SizeType, N>{});
        }


        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto access(const SizeType idx)
        {
            return get_values(0, idx, std::make_integer_sequence<SizeType, N>{});
        }

        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto access(const SizeType idx) const
        {
            return get_values(0, idx, std::make_integer_sequence<SizeType, N>{});
        }

        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto access(const SizeType stab_idx, const SizeType idx)
        {
            return get_values(stab_idx, idx, std::make_integer_sequence<SizeType, N>{});
        }

        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto access(const SizeType stab_idx, const SizeType idx) const
        {
            return get_values(stab_idx, idx, std::make_integer_sequence<SizeType, N>{});
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

        inline auto operator+=(const SizeType n)
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

            static constexpr SizeType default_alignment = SIMD_NAMESPACE::simd::alignment;

            static auto padding(const SizeType n, const SizeType alignment = default_alignment)
                -> SizeType
            {
                if (!::XXX_NAMESPACE::math::IsPowerOf<2>(alignment))
                {
                    std::cerr << "warning: alignment is not a power of 2" << std::endl;
                    return n;
                }

                const SizeType byte_padding_factor = ::XXX_NAMESPACE::math::LeastCommonMultiple(alignment, size_largest_type) / size_largest_type;
                const SizeType ratio = ::XXX_NAMESPACE::math::LeastCommonMultiple(record_padding_factor, byte_padding_factor);

                return ((n + ratio - 1) / ratio) * ratio;
            }

        public:

            template <::XXX_NAMESPACE::memory::DataLayout L, SizeType D, bool Enable = true>
            static auto get_allocation_shape(const sarray<SizeType, D>& n, const SizeType alignment = default_alignment)
                -> typename std::enable_if<(L != ::XXX_NAMESPACE::memory::DataLayout::SoA && Enable), std::pair<SizeType, SizeType>>::type
            {
                return {padding(n[0], alignment), n.reduce_mul(1)};
            }

            template <::XXX_NAMESPACE::memory::DataLayout L, SizeType D, bool Enable = true>
            static auto get_allocation_shape(const sarray<SizeType, D>& n, const SizeType alignment = default_alignment)
                -> typename std::enable_if<(L == ::XXX_NAMESPACE::memory::DataLayout::SoA && Enable), std::pair<SizeType, SizeType>>::type
            {
                return {padding(n.reduce_mul(), alignment), 1};
            }

            static auto get_byte_size(const std::pair<SizeType, SizeType>& allocation_shape)
            {
                return allocation_shape.first * allocation_shape.second * record_size;
            }

            template <XXX_NAMESPACE::target Target, bool Enable = true>
            static auto allocate(const std::pair<SizeType, SizeType>& allocation_shape, const SizeType alignment = default_alignment)
                -> typename std::enable_if<(Target == XXX_NAMESPACE::target::Host && Enable), value_type*>::type
            {
                // NOTE: aligned_alloc results in a segfault here -> use _mm_malloc
                return reinterpret_cast<value_type*>(_mm_malloc(get_byte_size(allocation_shape), alignment));
            }

            template <XXX_NAMESPACE::target Target, bool Enable = true>
            static auto deallocate(multi_pointer& mp)
                -> typename std::enable_if<(Target == XXX_NAMESPACE::target::Host && Enable), void>::type
            {
                if (mp.get_pointer())
                {
                    _mm_free(mp.get_pointer());
                }
            }

            #if defined(__CUDACC__)
            template <XXX_NAMESPACE::target Target, bool Enable = true>
            static auto allocate(const std::pair<SizeType, SizeType>& allocation_shape, const SizeType alignment = default_alignment)
                -> typename std::enable_if<(Target == XXX_NAMESPACE::target::GPU_CUDA && Enable), value_type*>::type
            {
                const SizeType num_elements = allocation_shape.first * allocation_shape.second;
                value_type* d_ptr;

                cudaMalloc((void**)&d_ptr, num_elements * record_size);

                return d_ptr;
            }

            template <XXX_NAMESPACE::target Target, bool Enable = true>
            static auto deallocate(multi_pointer& mp)
                -> typename std::enable_if<(Target == XXX_NAMESPACE::target::GPU_CUDA && Enable), void>::type
            {
                if (mp.get_pointer())
                {
                    cudaFree(mp.get_pointer());
                }
            }
            #endif
        };
    };
}

#endif