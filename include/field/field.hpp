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
#include <type_traits>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#include <common/data_layout.hpp>
#include <common/data_types.hpp>
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
        //! For C_D > 1, the number of stabs is determined as the reduction over all indices, but without accounting for
        //! the innermost dimension: [k][j][i] -> '(k * n[1] + j) * n[0] + i' -> 'stab_idx = k * n[1] + j'
        //! The 'memory' type holds a base-pointer-like memory reference to [0][0]..[0] and can deduce the final
        //! memory reference from 'stab_idx', 'n[0]' and 'i'.
        //! The result (recursion anchor, C_D=1) of the array subscript operator chaining is either a reference of type 
        //! 'ElementT' in case of the AoS data layout or if there is no proxy type available with the SoA data layout, or
        //! a proxy type that is initialized through the final memory reference type in case of SoA.
        //!
        //! \tparam ElementT data type
        //! \tparam C_N recursion level
        //! \tparam C_D dimension
        //! \tparam Data_layout any of SoA (struct of arrays) and AoS (array of structs)
        //! \tparam Enabled needed for partial specialization for data types providing a proxy type
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        template <typename ElementT, SizeType C_N, SizeType C_D, data_layout C_Layout>
        class Accessor
        {
            using BasePointerType = typename std::conditional<std::is_const<ElementT>::value, const typename internal::traits<ElementT, C_Layout>::BasePointerType, typename internal::traits<ElementT, C_Layout>::BasePointerType>::type;
            BasePointerType& data;
            const SizeArray<C_D>& n;
            const SizeType stab_idx;

        public:
	    
            //! \brief Standard constructor
            //!
            //! \param data pointer-like memory reference
            //! \param n extent of the C_D-dimensional array
            //! \param stab_idx the offset in units of 'innermost dimension n[0]'
            HOST_VERSION
            CUDA_DEVICE_VERSION
            Accessor(BasePointerType& data, const SizeArray<C_D>& n, const SizeType stab_idx = 0) 
                : 
                data(data), 
                n(n), 
                stab_idx(stab_idx) {}
	    
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline auto operator[] (const SizeType idx)
                -> Accessor<ElementT, C_N - 1, C_D, C_Layout>
            {
                return {data, n, stab_idx + idx * n.reduce_mul(1, C_N - 1)};
            }
	    
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline auto operator[] (const SizeType idx) const
                -> Accessor<ElementT, C_N - 1, C_D, C_Layout>
            {
                return {data, n, stab_idx + idx * n.reduce_mul(1, C_N - 1)};
            }
        };
        
        template <typename ElementT, SizeType C_D, data_layout C_Layout>
        class Accessor<ElementT, 1, C_D, C_Layout>
        {
            using BasePointerType = typename std::conditional<std::is_const<ElementT>::value, const typename internal::traits<ElementT, C_Layout>::BasePointerType, typename internal::traits<ElementT, C_Layout>::BasePointerType>::type;
            static constexpr bool UseProxyType = (C_Layout != data_layout::AoS && internal::provides_proxy_type<ElementT>::value);
            using value_type = typename std::conditional<UseProxyType, typename internal::traits<ElementT, C_Layout>::ProxyType, ElementT&>::type;
            using const_value_type = typename std::conditional<UseProxyType, const typename internal::traits<const ElementT, C_Layout>::ProxyType, const ElementT&>::type;
            
            BasePointerType& data;
            const SizeArray<C_D>& n;
            const SizeType stab_idx;

        public:

            //! \brief Standard constructor
            //!
            //! \param ptr base pointer
            //! \param n extent of the C_D-dimensional array
            HOST_VERSION
            CUDA_DEVICE_VERSION
            Accessor(BasePointerType& data, const SizeArray<C_D>& n, const SizeType stab_idx = 0) 
                : 
                data(data), 
                n(n), 
                stab_idx(stab_idx) {}
            
            template <bool Enable = true>
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline auto operator[] (const SizeType idx)
                -> typename std::enable_if<(!UseProxyType && Enable), value_type>::type
            {
                return std::get<0>(data.access(stab_idx, idx));
            }

            template <bool Enable = true>
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline auto operator[] (const SizeType idx) const
                -> typename std::enable_if<(!UseProxyType && Enable), const_value_type>::type
            {
                return std::get<0>(data.access(stab_idx, idx));
            }

            template <bool Enable = true, data_layout Layout = C_Layout>
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline auto operator[] (const SizeType idx)
                -> typename std::enable_if<(UseProxyType && Layout == data_layout::SoAi && Enable), value_type>::type
            {
                return data.access(stab_idx, idx);
            }

            template <bool Enable = true, data_layout Layout = C_Layout>
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline auto operator[] (const SizeType idx) const
                -> typename std::enable_if<(UseProxyType && Layout == data_layout::SoAi && Enable), value_type>::type
            {
                return data.access(stab_idx, idx);
            }

            template <bool Enable = true, data_layout Layout = C_Layout>
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline auto operator[] (const SizeType idx)
                -> typename std::enable_if<(UseProxyType && Layout == data_layout::SoA && Enable), value_type>::type
            {
                return data.access(0, stab_idx * n[0] + idx);
            }

            template <bool Enable = true, data_layout Layout = C_Layout>
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline auto operator[] (const SizeType idx) const
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
    //! In case of C_D > 1, its is determined as the product of all dimensions with the
    //! innermost dimension padded according to ElementT, the data layout and the (default) data alignment.
    //! All memory is contiguous and can be moved as a whole (important e.g. for data transfers).
    //! The field, however, allows access to the data using array subscript operator chaining together with proxy objects.
    //! \n\n
    //! In case of Data_layout = AoS, data is stored with all elements placed in main memory one after the other.
    //! \n
    //! In case of Data_layout = SoA (applied only if ElementT is a record type) the individual members
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
    //! \tparam ElementT data type
    //! \tparam C_D dimension
    //! \tparam C_Layout any of SoA (struct of arrays) and AoS (array of structs)
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <typename T, SizeType C_D, data_layout C_Layout = data_layout::AoS>
    class Field;

    template <typename ElementT, SizeType C_Dimension, data_layout C_Layout, target C_Target>
    class Container
    {
        using Self = Container<ElementT, C_Dimension, C_Layout, C_Target>;

        friend class Field<ElementT, C_Dimension, C_Layout>;

    public:

        using ElementType = ElementT;
        static constexpr SizeType Dimension = C_Dimension;
        static constexpr data_layout Layout = C_Layout;
        static constexpr target Target = C_Target;

    private:

        using ConstElementT = typename internal::traits<ElementT, C_Layout>::ConstType;
        template <typename T>
        using BasePointerType = typename internal::traits<T, C_Layout>::BasePointerType;
        using AllocatorT = typename BasePointerType<ElementT>::allocator;

        struct Deleter
        {
            auto operator()(BasePointerType<ElementT>* ptr) const
                -> void
            {
                AllocatorT::template deallocate<C_Target>(*ptr);
            }
        };

        Container()
            :
            n{},
            allocation_shape{}
        {}

        Container(const SizeArray<C_Dimension>& n)
            :
            n(n),
            allocation_shape(AllocatorT::template get_allocation_shape<C_Layout>(n)),
            data(new BasePointerType<ElementT>(AllocatorT::template allocate<Target>(allocation_shape), allocation_shape.first), Deleter()),
	        ptr(*data),
            const_ptr(*data)
        {}

        static constexpr bool UseProxyType = (C_Layout != data_layout::AoS && internal::provides_proxy_type<ElementT>::value);
        using return_type = std::conditional_t<(C_Dimension == 1), std::conditional_t<UseProxyType, typename internal::traits<ElementT, C_Layout>::ProxyType, ElementT&>, internal::Accessor<ElementT, C_Dimension - 1, C_Dimension, C_Layout>>;
        using const_return_type = std::conditional_t<(C_Dimension == 1), std::conditional_t<UseProxyType, const typename internal::traits<const ElementT, C_Layout>::ProxyType, const ElementT&>, internal::Accessor<ConstElementT, C_Dimension - 1, C_Dimension, C_Layout>>;

    public:

        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto operator[](const SizeType idx)
            -> return_type
        {
            return internal::Accessor<ElementT, C_Dimension, C_Dimension, C_Layout>(ptr, n)[idx];
        }

        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto operator[](const SizeType idx) const
            -> const_return_type
        {
            return internal::Accessor<ConstElementT, C_Dimension, C_Dimension, C_Layout>(const_ptr, n)[idx];
        }

        template <typename FuncT>
        auto Set(FuncT func)
        {
            const SizeType pitch = allocation_shape.first;
            const SizeType num_stabs = allocation_shape.second;

            if (data.get())
            {   
                for (size_t stab_index = 0; stab_index < num_stabs; ++stab_index)
                {
                    for (size_t i = 0; i < n[0]; ++i)
                    {
                        std::get<0>(data.get()->access(stab_index, i)) = func();
                    }
                }
                
            }
        }

        HOST_VERSION
        CUDA_DEVICE_VERSION    
        const auto& Size() const
        {
            return n;   
        }

        auto IsEmpty() const
        {
            return (data.get() == nullptr);
        }

    private:

        auto GetByteSize() const
        {
            return AllocatorT::get_byte_size(allocation_shape);
        }

        auto GetBasePointer()
        {
            return data.get()->get_pointer();
        }

        SizeArray<C_Dimension> n;
        std::pair<SizeType, SizeType> allocation_shape;
	    std::shared_ptr<BasePointerType<ElementT>> data;
	    BasePointerType<ElementT> ptr;
        BasePointerType<ConstElementT> const_ptr;
    };

    template <typename T, SizeType C_D, data_layout C_Layout>
    class Field
    {
        static_assert(!std::is_const<T>::value, "error: field with const elements is not allowed");

    public:

        using element_type = T;
        static constexpr SizeType dimension = C_D;
        static constexpr data_layout layout = C_Layout;

    private:

        using const_element_type = typename internal::traits<element_type, C_Layout>::ConstType;
        
    public:

        Field() = default;
            
        Field(const SizeArray<C_D>& n, const bool initialize_to_zero = false)
            :
            n(n)
        {
            Resize(n, initialize_to_zero);
        }

        auto Resize(const SizeArray<C_D>& new_n, const bool initialize_to_zero = false)
            -> void
        {
            if (n != new_n)
            {
                n = new_n;
                data = Container<T, C_D, C_Layout, XXX_NAMESPACE::target::Host>(n);
                
                if (initialize_to_zero)
                {
                    data.Set([] () { return 0; });
                }

            #if defined(__CUDACC__)
                device_data = Container<T, C_D, C_Layout, target::GPU_CUDA>(n);

                if (initialize_to_zero)
                {
                    CopyHostToDevice();
                }
            #endif
            }
        }
        
        template <typename FuncT>
        auto Set(FuncT func)
        {
            data.Set(func);
        }
        
        auto Swap(Field& b)
            -> void
        {
            // TODO: implementation; if (owns_data) {..}
        }

        static constexpr bool UseProxyType = (C_Layout != data_layout::AoS && internal::provides_proxy_type<T>::value);
        using value_type = std::conditional_t<(C_D == 1), std::conditional_t<UseProxyType, typename internal::traits<T, C_Layout>::ProxyType, T&>, internal::Accessor<element_type, C_D - 1, C_D, C_Layout>>;
        using const_value_type = std::conditional_t<(C_D == 1), std::conditional_t<UseProxyType, const typename internal::traits<const T, C_Layout>::ProxyType, const T&>, internal::Accessor<const_element_type, C_D - 1, C_D, C_Layout>>;

        inline auto operator[](const SizeType idx)
            -> value_type
        {
            return data[idx];
        }

        inline auto operator[](const SizeType idx) const
            -> const_value_type
        {
            return data[idx];
        }

        HOST_VERSION
        CUDA_DEVICE_VERSION    
        inline auto size() const
            -> const SizeArray<C_D>&
        {
            return n;   
        }

    #if defined(__CUDACC__)
        auto GetDeviceAccess(const bool sync_with_host = false)
            -> Container<T, C_D, C_Layout, target::GPU_CUDA>&
        {
            if (device_data.IsEmpty())
            {
                device_data = Container<T, C_D, C_Layout, target::GPU_CUDA>(n);
            }

            if (sync_with_host)
            {
                CopyHostToDevice();
            }

            return device_data;
        }

        auto CopyDeviceToHost()
            -> void
        {
            if (!device_data.IsEmpty())
            {
                cudaMemcpy((void*)data.GetBasePointer(), (const void*)device_data.GetBasePointer(), data.GetByteSize(), cudaMemcpyDeviceToHost);
            }
        }

        auto CopyHostToDevice()
            -> void
        {
            if (!device_data.IsEmpty())
            {
                cudaMemcpy((void*)device_data.GetBasePointer(), (const void*)data.GetBasePointer(), data.GetByteSize(), cudaMemcpyHostToDevice);
            }
        }
    #endif

        SizeArray<C_D> n;
        Container<T, C_D, C_Layout, target::Host> data;
    #if defined(__CUDACC__)
        Container<T, C_D, C_Layout, target::GPU_CUDA> device_data;
    #endif
    };
}

#endif
