// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(FIELD_FIELD_HPP)
#define FIELD_FIELD_HPP

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#include <common/data_layout.hpp>
#include <common/memory.hpp>
#include <common/traits.hpp>
#include <platform/target.hpp>
#include <sarray/sarray.hpp>

namespace XXX_NAMESPACE
{
    namespace internal
    {
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //! \brief Accessor type implementing array subscript operator chaining [][]..[]
        //!
        //! This data type basically collects all array indices and determines the final memory reference recursively.
        //! 
        //! Idea: all memory is allocated as a contiguous set of stabs (innermost dimension n[0] with padding).
        //! For D > 1, the number of stabs is determined as the reduction over all indices, but without accounting for
        //! the innermost dimension: [k][j][i] -> '(k * n[1] + j) * n[0] + i' -> 'stab_idx = k * n[1] + j'
        //! The 'memory' type holds a base-pointer-like memory reference to [0][0]..[0] and can deduce the final
        //! memory reference from 'stab_idx', 'n[0]' and 'i'.
        //! The result (recursion anchor, D=1) of the array subscript operator chaining is either a reference of type 
        //! 'T' in case of the AoS data layout or if there is no proxy type available with the SoA data layout, or
        //! a proxy type that is initialized through the final memory reference type in case of SoA.
        //!
        //! \tparam T data type
        //! \tparam N recursion level
        //! \tparam D dimension
        //! \tparam Data_layout any of SoA (struct of arrays) and AoS (array of structs)
        //! \tparam Enabled needed for partial specialization for data types providing a proxy type
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        template <typename T, std::size_t N, std::size_t D, data_layout L>
        class accessor
        {
            //using base_pointer = typename internal::traits<T, L>::base_pointer;
            using base_pointer = typename std::conditional<std::is_const<T>::value, const typename internal::traits<T, L>::base_pointer, typename internal::traits<T, L>::base_pointer>::type;
            base_pointer& data;
            const sarray<std::size_t, D>& n;
            const std::size_t stab_idx;

        public:

            //! \brief Standard constructor
            //!
            //! \param data pointer-like memory reference
            //! \param n extent of the D-dimensional array
            //! \param stab_idx the offset in units of 'innermost dimension n[0]'
            HOST_VERSION
            CUDA_DEVICE_VERSION
            accessor(base_pointer& data, const sarray<std::size_t, D>& n, const std::size_t stab_idx = 0) 
                : 
                data(data), 
                n(n), 
                stab_idx(stab_idx) {}

            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline auto operator[] (const std::size_t idx)
                -> accessor<T, N - 1, D, L>
            {
                std::size_t delta = idx;

                for (std::size_t i = 1; i < (N - 1); ++i)
                {
                    delta *= n[i];
                }               

                return {data, n, stab_idx + delta};
            }

            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline auto operator[] (const std::size_t idx) const
                -> accessor<T, N - 1, D, L>
            {
                std::size_t delta = idx;

                for (std::size_t i = 1; i < (N - 1); ++i)
                {
                    delta *= n[i];
                }               

                return {data, n, stab_idx + delta};
            }
        };
        
        template <typename T, std::size_t D, data_layout L>
        class accessor<T, 1, D, L>
        {
            //using base_pointer = typename internal::traits<T, L>::base_pointer;
            //using const_base_pointer = typename internal::traits<const T, L>::base_pointer;
            using base_pointer = typename std::conditional<std::is_const<T>::value, const typename internal::traits<T, L>::base_pointer, typename internal::traits<T, L>::base_pointer>::type;
            static constexpr bool UseProxyType = (L != data_layout::AoS && internal::provides_proxy_type<T>::value);
            using value_type = typename std::conditional<UseProxyType, typename internal::traits<T, L>::proxy_type, T&>::type;
            using const_value_type = typename std::conditional<UseProxyType, const typename internal::traits<const T, L>::proxy_type, const T&>::type;
            
            base_pointer& data;
            const sarray<std::size_t, D>& n;
            const std::size_t stab_idx;

        public:

            //! \brief Standard constructor
            //!
            //! \param ptr base pointer
            //! \param n extent of the D-dimensional array
            HOST_VERSION
            CUDA_DEVICE_VERSION
            accessor(base_pointer& data, const sarray<std::size_t, D>& n, const std::size_t stab_idx = 0) 
                : 
                data(data), 
                n(n), 
                stab_idx(stab_idx) {}
            
            template <bool Enable = true>
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline auto operator[] (const std::size_t idx)
                -> typename std::enable_if<(!UseProxyType && Enable), value_type>::type
            {
                return std::get<0>(data.access(0, stab_idx * n[0] + idx));
            }

            template <bool Enable = true>
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline auto operator[] (const std::size_t idx) const
                -> typename std::enable_if<(!UseProxyType && Enable), const_value_type>::type
            {
                return std::get<0>(data.access(0, stab_idx * n[0] + idx));
            }

            template <bool Enable = true, data_layout Layout = L>
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline auto operator[] (const std::size_t idx)
                -> typename std::enable_if<(UseProxyType && Layout == data_layout::SoAi && Enable), value_type>::type
            {
                return data.access(stab_idx, idx);
            }

            template <bool Enable = true, data_layout Layout = L>
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline auto operator[] (const std::size_t idx) const
                -> typename std::enable_if<(UseProxyType && Layout == data_layout::SoAi && Enable), value_type>::type
            {
                return data.access(stab_idx, idx);
            }

            template <bool Enable = true, data_layout Layout = L>
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline auto operator[] (const std::size_t idx)
                -> typename std::enable_if<(UseProxyType && Layout == data_layout::SoA && Enable), value_type>::type
            {
                return data.access(0, stab_idx * n[0] + idx);
            }

            template <bool Enable = true, data_layout Layout = L>
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline auto operator[] (const std::size_t idx) const
                -> typename std::enable_if<(UseProxyType && Layout == data_layout::SoA && Enable), value_type>::type
            {
                return data.access(0, stab_idx * n[0] + idx);
            }            
        };
    }
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //! \brief Multi-dimensional field data type
    //!
    //! This field data type uses a plain C-pointer to dynamically adapt to the needed memory requirement.
    //! In case of D > 1, its is determined as the product of all dimensions with the
    //! innermost dimension padded according to T, the data layout and the (default) data alignment.
    //! All memory is contiguous and can be moved as a whole (important e.g. for data transfers).
    //! The field, however, allows access to the data using array subscript operator chaining together with proxy objects.
    //! \n\n
    //! In case of Data_layout = AoS, data is stored with all elements placed in main memory one after the other.
    //! \n
    //! In case of Data_layout = SoA (applied only if T is a record type) the individual members
    //! are placed one after the other along the innermost dimension, e.g. for
    //! field<vec<double, 3>, 2, SoA>({3, 2}) the memory layout would be the following one:
    //! <pre>
    //!     [0][0].x
    //!     [0][1].x
    //!     [0][2].x
    //!     ######## (padding)
    //!     [0][0].y
    //!     [0][1].y
    //!     [0][2].y
    //!     ######## (padding)
    //!     [1][0].x
    //!     [1][1].x
    //!     [1][2].x
    //!     ######## (padding)
    //!     [1][0].y
    //!     [1][1].y
    //!     [1][2].y
    //! </pre>
    //! You can access the individual components of field<vec<double, 3>, 2, SoA> b({3,2})
    //! as usual, e.g., b[1][0].x = ...
    //! \n
    //! GNU and Clang/LLVM seem to optimize the proxies away.
    //!
    //! \tparam T data type
    //! \tparam D dimension
    //! \tparam Data_layout any of SoA (struct of arrays) and AoS (array of structs)
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <typename T, std::size_t D, data_layout L = data_layout::AoS>
    class field
    {
        static_assert(!std::is_const<T>::value, "error: field with const elements is not allowed");

    public:

        using element_type = T;
        static constexpr std::size_t dimension = D;
        static constexpr data_layout layout = L;

    private:

        using const_element_type = typename internal::traits<element_type, L>::const_type;
        template <typename X>
        using base_pointer = typename internal::traits<X, L>::base_pointer;
        using allocator_type = typename base_pointer<element_type>::allocator;
        
        sarray<std::size_t, D> n;
        std::pair<std::size_t, std::size_t> allocation_shape;
        base_pointer<element_type> data;
        base_pointer<const_element_type> const_data;
        bool release_memory;
        #if defined(__CUDACC__)
        field* d_this; 
        #endif
/*
        auto set_data(const element_type& value)
            -> void
        {
            if (!data.get()) return;

            const std::size_t n_stabs = n.reduce_mul(1);
            
            for (std::size_t i_s = 0; i_s < n_stabs; ++i_s)
            {
                // get base_pointer to this stab, and use a 1d-accessor to access the elements in it
                base_pointer<element_type> data_stab = data->at(i_s, 0);
                internal::accessor<element_type, 1, D, L> stab(data_stab, n);
                
                for (std::size_t i = 0; i < n[0]; ++i)
                {
                    stab[i] = value;
                }
            }
        }
        */

    public:

        field()
            :
            n{},
            allocation_shape{0, 0},
            data{},
            const_data{},
            release_memory(false)
        {}
            
        field(const sarray<std::size_t, D>& n, const bool initialize_to_zero = false)
            :
            n(n),
            allocation_shape(allocator_type::template get_allocation_shape<L>(n)),
            data(allocator_type::template allocate<XXX_NAMESPACE::target::Host>(allocation_shape), allocation_shape.first),
            const_data(*data),
            release_memory(true)
        {
            if (initialize_to_zero)
            {
                //set_data({});
            }
        }
        
        ~field()
        {
            if (release_memory)
            {
                allocator_type::template deallocate<XXX_NAMESPACE::target::Host>(data);
            }
        }

        auto size() const
            -> const sarray<std::size_t, D>&
        {
            return n;   
        }

        auto resize(const sarray<std::size_t, D>& n, const bool initialize_to_zero = false)
            -> void
        {
            allocator_type::template deallocate<XXX_NAMESPACE::target::Host>(data);

            this->n = n;
            allocation_shape = allocator_type::template get_allocation_shape<L>(n);
            data = base_pointer<element_type>(allocator_type::template allocate<XXX_NAMESPACE::target::Host>(allocation_shape), allocation_shape.first);
            const_data = base_pointer<const_element_type>(data);
            
            if (initialize_to_zero)
            {
               // set_data({});
            }
        }

        auto swap(field& b)
            -> void
        {
            /*
            if (n != b.n)
            {
                std::cerr << "error: field swapping not possible because of different extents" << std::endl;
                return;
            }

            n.swap(b.n);

            std::pair<std::size_t, std::size_t> this_allocation_shape = allocation_shape;
            allocation_shape = b.allocation_shape;
            b.allocation_shape = this_allocation_shape;

            data.swap(b.data);
            const_data.swap(b.const_data);
            */
        }

        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto operator[](const std::size_t idx)
        {
            return internal::accessor<element_type, D, D, L>(data, n)[idx];
        }

        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto operator[](const std::size_t idx) const
        {
            return internal::accessor<const_element_type, D, D, L>(const_data, n)[idx];
        }

        inline auto at(std::size_t idx)
        {
            return internal::accessor<element_type, D, D, L>(data, n)[idx];
        }

        inline auto at(std::size_t idx) const
        {
            return internal::accessor<const_element_type, D, D, L>(const_data, n)[idx];
        }
    };
}

#endif