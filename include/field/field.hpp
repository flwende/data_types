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
    template <typename T, std::size_t D, data_layout L>
    class device_field;

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
        template <typename X>
        using shared_base_pointer = std::shared_ptr<base_pointer<X>>;
        using allocator = typename base_pointer<element_type>::allocator;
        
        template <XXX_NAMESPACE::target Target>
        struct deleter
        {
            template <bool Enable = true>
            auto operator()(base_pointer<element_type>* ptr) const
                -> typename std::enable_if<(Target == XXX_NAMESPACE::target::Host && Enable), void>::type
            {
                allocator::template deallocate<Target>(*ptr);
            }

        #if defined(__CUDACC__)
            template <bool Enable = true>
            auto operator()(device_field<T, D, L>* ptr) const
                -> typename std::enable_if<(Target == XXX_NAMESPACE::target::GPU_CUDA && Enable), void>::type
            {
                std::cout << "dev" << std::endl;
                allocator::template deallocate<XXX_NAMESPACE::target::GPU_CUDA>(ptr->data);
            }
        #endif    
        };

    public:

        field() = default;
            
        field(const sarray<std::size_t, D>& n, const bool initialize_to_zero = false)
            :
            n(n),
            allocation_shape(allocator::template get_allocation_shape<L>(n)),
            data(make_shared_base_pointer<element_type, XXX_NAMESPACE::target::Host>(allocation_shape)),
            const_data(new base_pointer<const_element_type>(*data), [] (auto p) {}),
            d_this{}
        {
            if (initialize_to_zero)
            {
                set_data({});
            }
        }
        
        auto resize(const sarray<std::size_t, D>& new_n, const bool initialize_to_zero = false)
            -> void
        {
            if (n != new_n)
            {
                n = new_n;
                allocation_shape = allocator::template get_allocation_shape<L>(n);
                data = make_shared_base_pointer<element_type, XXX_NAMESPACE::target::Host>(allocation_shape);
                const_data = shared_base_pointer<const_element_type>(new base_pointer<const_element_type>(*data), [] (auto p) {});
                
                if (initialize_to_zero)
                {
                    set_data({});
                }

            #if defined(__CUDACC__)
                if (d_this.get())
                {
                    resize_device_data(initialize_to_zero);
                }
            #endif
            }
        }

        auto swap(field& b)
            -> void
        {
            // TODO: implementation; if (owns_data) {..}
        }

        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto operator[](const std::size_t idx)
        {
            return internal::accessor<element_type, D, D, L>(*data, n)[idx];
        }

        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto operator[](const std::size_t idx) const
        {
            return internal::accessor<const_element_type, D, D, L>(*const_data, n)[idx];
        }

        HOST_VERSION
        CUDA_DEVICE_VERSION    
        auto size() const
            -> const sarray<std::size_t, D>&
        {
            return n;   
        }

    #if defined(__CUDACC__)
        
        auto device(const bool sync_with_host = false)
            -> device_field<T, D, L>&
        {
            if (!d_this.get())
            {
                resize_device_data(sync_with_host);
            }

            return *d_this;
        }
        
        auto copy_device_to_host()
            -> void
        {
            if (d_this.get())
            {
                cudaMemcpy((void*)data->get_pointer(), (const void*)d_this->data.get_pointer(), allocator::get_byte_size(allocation_shape), cudaMemcpyDeviceToHost);
            }
        }

        auto copy_host_to_device()
            -> void
        {
        
            if (d_this.get())
            {
                cudaMemcpy((void*)d_this->data.get_pointer(), (const void*)data->get_pointer(), allocator::get_byte_size(allocation_shape), cudaMemcpyHostToDevice);
            }
        }
    #endif

    private:

        field(const sarray<std::size_t, D>& n, const std::pair<std::size_t, std::size_t>& allocation_shape, const shared_base_pointer<element_type>& data)
            :
            n(n),
            allocation_shape(allocation_shape),
            data(data),
            const_data(new base_pointer<const_element_type>(*data), [] (auto p) {}),
            d_this{}
        {}

        template <typename X, XXX_NAMESPACE::target Target>
        auto make_base_pointer(const std::pair<std::size_t, std::size_t>& allocation_shape)
            -> base_pointer<X>
        {
            return {allocator::template allocate<Target>(allocation_shape), allocation_shape.first};
        }

        template <typename X, XXX_NAMESPACE::target Target>
        auto make_shared_base_pointer(const std::pair<std::size_t, std::size_t>& allocation_shape)
            -> shared_base_pointer<X>
        {
            return {new base_pointer<X>(allocator::template allocate<Target>(allocation_shape), allocation_shape.first), deleter<Target>()};
        }

        auto set_data(const element_type& value)
            -> void
        {   
            for (std::size_t stab_idx = 0; stab_idx < n.reduce_mul(1); ++stab_idx)
            {
                // get base_pointer to this stab, and use a 1d-accessor to access the elements in it
                base_pointer<element_type> stab_pointer(*data, stab_idx, 0);
                internal::accessor<element_type, 1, D, L> stab(stab_pointer, n);

                for (std::size_t i = 0; i < n[0]; ++i)
                {
                    stab[i] = value;
                }
            }
        }

    #if defined(__CUDACC__)
        auto resize_device_data(const bool sync_with_host = false)
            -> void
        {
            d_this = std::shared_ptr<device_field<T, D, L>>(new device_field<T, D, L>(n, allocation_shape, make_base_pointer<element_type, XXX_NAMESPACE::target::GPU_CUDA>(allocation_shape)), deleter<XXX_NAMESPACE::target::GPU_CUDA>()); 
        
            if (sync_with_host)
            {
                cudaMemcpy((void*)d_this->data.get_pointer(), (const void*)data->get_pointer(), allocator::get_byte_size(allocation_shape), cudaMemcpyHostToDevice);
            }
        }
    #endif

        sarray<std::size_t, D> n;
        std::pair<std::size_t, std::size_t> allocation_shape;
        shared_base_pointer<element_type> data;
        shared_base_pointer<const_element_type> const_data;
        std::shared_ptr<device_field<T, D, L>> d_this;
    };

#if defined(__CUDACC__)
    template <typename T, std::size_t D, data_layout L>
    class device_field
    {
        static_assert(!std::is_const<T>::value, "error: field with const elements is not allowed");

        template <typename, std::size_t, data_layout>
        friend class field;

    public:

        using element_type = T;
        static constexpr std::size_t dimension = D;
        static constexpr data_layout layout = L;

    //private:

        using const_element_type = typename internal::traits<element_type, L>::const_type;
        template <typename X>
        using base_pointer = typename internal::traits<X, L>::base_pointer;

        device_field(const sarray<std::size_t, D>& n, const std::pair<std::size_t, std::size_t>& allocation_shape, const base_pointer<element_type>& data)
            :
            n(n),
            allocation_shape(allocation_shape),
            data(data),
            const_data(data)
        {}

    public:

        CUDA_DEVICE_VERSION
        inline auto operator[](const std::size_t idx)
        {
            return internal::accessor<element_type, D, D, L>(data, n)[idx];
        }

        CUDA_DEVICE_VERSION
        inline auto operator[](const std::size_t idx) const
        {
            return internal::accessor<const_element_type, D, D, L>(const_data, n)[idx];
        }

        CUDA_DEVICE_VERSION    
        auto size() const
            -> const sarray<std::size_t, D>&
        {
            return n;   
        }

    private:

        sarray<std::size_t, D> n;
        std::pair<std::size_t, std::size_t> allocation_shape;
        base_pointer<element_type> data;
        base_pointer<const_element_type> const_data;
    };
#endif
}

#endif