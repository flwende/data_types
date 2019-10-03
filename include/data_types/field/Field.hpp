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

#include <common/DataLayout.hpp>
#include <data_types/DataTypes.hpp>
#include <common/Memory.hpp>
#include <common/Traits.hpp>
#include <platform/Target.hpp>

namespace XXX_NAMESPACE
{
    namespace dataTypes
    {
        namespace internal
        {
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            //!
            //! \brief Accessor type for array subscript operator chaining [][]..[].
            //!
            //! This data type basically collects all array indices and determines the final memory reference recursively.
            //! 
            //! Idea: all memory is allocated as a contiguous set of stabs (innermost dimension n[0] with padding).
            //! For C_Dimension > 1, the number of stabs is determined as the reduction over all indices, but without accounting for
            //! the innermost dimension: [k][j][i] -> '(k * n[1] + j) * n[0] + i' -> 'stab_index = k * n[1] + j'
            //! The 'memory' type holds a base-pointer-like memory reference to [0][0]..[0] and can deduce the final
            //! memory reference from 'stab_index', 'n[0]' and 'i'.
            //! The result (recursion anchor, C_Dimension=1) of the array subscript operator chaining is either a reference of type 
            //! 'ValueT' in case of the AoS data layout or if there is no proxy type available with the SoA data layout, or
            //! a proxy type that is initialized through the final memory reference type in case of SoA.
            //!
            //! \tparam ValueT element type
            //! \tparam C_R recursion level
            //! \tparam C_Dimension the dimension of the field
            //! \tparam C_Layout any of AoS, SoAi, SoA
            //!
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            template <typename ValueT, SizeType C_R, SizeType C_Dimension, ::XXX_NAMESPACE::memory::DataLayout C_Layout>
            class Accessor
            {
                using BasePointerType = typename ::XXX_NAMESPACE::internal::Traits<ValueT, C_Layout>::BasePointerType;
                using PointerType = std::conditional_t<std::is_const<ValueT>::value, const BasePointerType, BasePointerType>;
                using DataLayout = ::XXX_NAMESPACE::memory::DataLayout;
                using SizeArray = ::XXX_NAMESPACE::dataTypes::SizeArray<C_Dimension>;

            public:
                // Template parameters.
                using ValueType = ValueT;
                static constexpr SizeType R = C_R;
                static constexpr SizeType Dimension = C_Dimension;
                static constexpr DataLayout Layout = C_Layout;
            
                //!
                //! \brief Constructor.
                //!
                //! \param ptr base-pointer-like memory reference
                //! \param n extent of the C_Dimension-dimensional field
                //! \param stab_index the offset in units of 'innermost dimension n[0]'
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                Accessor(PointerType& ptr, const SizeArray& n, const SizeType stab_index = 0) 
                    : 
                    ptr(ptr), 
                    n(n), 
                    stab_index(stab_index) 
                {}
            
                //!
                //! \brief Array subscript operator.
                //!
                //! This function returns a lower-dimensional accessor type with the `stab_index` shifted 
                //! by the number of stabs in the (C_Dimension-1)-dimension sub-volume according to the `index` value.
                //!
                //! \param index element index
                //! \return a lower-dimensional accessor type with a shifted stab_index
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline auto operator[] (const SizeType index)
                    -> Accessor<ValueT, C_R - 1, C_Dimension, C_Layout>
                {
                    return {ptr, n, stab_index + index * n.ReduceMul(1, C_R - 1)};
                }
            
                //!
                //! \brief Array subscript operator.
                //!
                //! This function returns a lower-dimensional accessor type with the `stab_index` shifted 
                //! by the number of stabs in the (C_Dimension-1)-dimension sub-volume according to the `index` value.
                //!
                //! \param index element index
                //! \return a lower-dimensional accessor type with a shifted stab_index
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline auto operator[] (const SizeType index) const
                    -> Accessor<ValueT, C_R - 1, C_Dimension, C_Layout>
                {
                    return {ptr, n, stab_index + index * n.ReduceMul(1, C_R - 1)};
                }

            private:
                PointerType& ptr;
                const SizeArray& n;
                const SizeType stab_index;
            };
            
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            //!
            //! \brief Accessor type for array subscript operator chaining [][]..[] (recursion anchor).
            //!
            //! This is the recursion anchor (`C_R=1`). Depending on the element type and the data layout, either a reference
            //! to an element or a proxy type is returned by the array subscript operator.
            //!
            //! \tparam ValueT element type
            //! \tparam C_Dimension the dimension of the field
            //! \tparam C_Layout any of AoS, SoAi, SoA
            //!
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            template <typename ValueT, SizeType C_Dimension, ::XXX_NAMESPACE::memory::DataLayout C_Layout>
            class Accessor<ValueT, 1, C_Dimension, C_Layout>
            {
                using BasePointerType = typename ::XXX_NAMESPACE::internal::Traits<ValueT, C_Layout>::BasePointerType;
                using PointerType = std::conditional_t<std::is_const<ValueT>::value, const BasePointerType, BasePointerType>;
                using ProxyType = typename ::XXX_NAMESPACE::internal::Traits<ValueT, C_Layout>::ProxyType;
                using ConstProxyType = const typename ::XXX_NAMESPACE::internal::Traits<const ValueT, C_Layout>::ProxyType;
                using SizeArray = ::XXX_NAMESPACE::dataTypes::SizeArray<C_Dimension>;
                using DataLayout = ::XXX_NAMESPACE::memory::DataLayout;

            public:

                //!
                //! \brief Constructor.
                //!
                //! \param ptr base pointer
                //! \param n extent of the field
                //! \param stab_index the stab_index
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                Accessor(PointerType& ptr, const SizeArray& n, const SizeType stab_index = 0) 
                    : 
                    ptr(ptr), 
                    n(n), 
                    stab_index(stab_index) 
                {}

                //!
                //! \brief Array subscript operator (AoS data layout).
                //!
                //! The return value of `At(..)` is a tuple with a single reference to some variable of type `ValueT`.
                //! Get the reference through `std::get<0>`.
                //!
                //! \tparam Enable used for multi-versioning depending on the data layout
                //! \param index the intra-stab index
                //! \return a reference to a variable of type `ValueT`
                //!
                template <DataLayout Enable = C_Layout>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline auto operator[] (const SizeType index)
                    -> std::enable_if_t<Enable == DataLayout::AoS, ValueT&>
                {
                    return std::get<0>(ptr.At(stab_index, index));
                }

                //!
                //! \brief Array subscript operator (AoS data layout).
                //!
                //! The return value of `At(..)` is a tuple with a single reference to some variable of type `ValueT`.
                //! Get the reference through `std::get<0>`.
                //!
                //! \tparam Enable used for multi-versioning depending on the data layout
                //! \param index the intra-stab index
                //! \return a const reference to a variable of type `ValueT`
                //!
                template <DataLayout Enable = C_Layout>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline auto operator[] (const SizeType index) const
                    -> std::enable_if_t<Enable == DataLayout::AoS, const ValueT&>
                {
                    return std::get<0>(ptr.At(stab_index, index));
                }

                //!
                //! \brief Array subscript operator (SoAi data layout).
                //!
                //! The return value of `At(..)` is a tuple of references that is used for the proxy type construction.
                //!
                //! \tparam Enable used for multi-versioning depending on the data layout
                //! \param index the intra-stab index
                //! \return a proxy type
                //!
                template <DataLayout Enable = C_Layout>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline auto operator[] (const SizeType index)
                    -> std::enable_if_t<Enable == DataLayout::SoAi, ProxyType>
                {
                    return {ptr.At(stab_index, index)};
                }

                //!
                //! \brief Array subscript operator (SoAi data layout).
                //!
                //! The return value of `At(..)` is a tuple of references that is used for the proxy type construction.
                //!
                //! \tparam Enable used for multi-versioning depending on the data layout
                //! \param index the intra-stab index
                //! \return a const proxy type
                //!
                template <DataLayout Enable = C_Layout>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline auto operator[] (const SizeType index) const
                    -> std::enable_if_t<Enable == DataLayout::SoAi, ConstProxyType>
                {
                    return {ptr.At(stab_index, index)};
                }

                //!
                //! \brief Array subscript operator (SoA data layout).
                //!
                //! The return value of `At(..)` is a tuple of references that is used for the proxy type construction.
                //!
                //! \tparam Enable used for multi-versioning depending on the data layout
                //! \param index the intra-stab index
                //! \return a proxy type
                //!
                template <DataLayout Enable = C_Layout>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline auto operator[] (const SizeType index)
                    -> std::enable_if_t<Enable == DataLayout::SoA, ProxyType>
                {
                    return {ptr.At(stab_index * n[0] + index)};
                }

                //!
                //! \brief Array subscript operator (SoA data layout).
                //!
                //! The return value of `At(..)` is a tuple of references that is used for the proxy type construction.
                //!
                //! \tparam Enable used for multi-versioning depending on the data layout
                //! \param index the intra-stab index
                //! \return a const proxy type
                //!
                template <DataLayout Enable = C_Layout>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline auto operator[] (const SizeType index) const
                    -> std::enable_if_t<Enable == DataLayout::SoA, ConstProxyType>
                {
                    return {ptr.At(stab_index * n[0] + index)};
                }

            private:
                PointerType& ptr;
                const SizeArray& n;
                const SizeType stab_index;
            };
        }
        
        // Forward declaration.
        template <typename T, SizeType C_Dimension, ::XXX_NAMESPACE::memory::DataLayout C_Layout = ::XXX_NAMESPACE::memory::DataLayout::AoS>
        class Field;

        namespace internal
        {
            //!
            //! \brief A container type.
            //!
            //! This type encapsulates a shared pointer to some memory and allows access to the memory through `Pointer`s and `MultiPointer`s
            //! (base pointers) and `Accessor`s.
            //!
            //! \tparam ValueT element type
            //! \tparam C_Dimension the dimension of the field
            //! \tparam C_Layout any of AoS, SoAi, SoA
            //! \tparam C_Target the target platform
            //!
            template <typename ValueT, SizeType C_Dimension, ::XXX_NAMESPACE::memory::DataLayout C_Layout, target C_Target>
            class Container
            {
                using DataLayout = ::XXX_NAMESPACE::memory::DataLayout;
                template <typename T>
                using Traits = ::XXX_NAMESPACE::internal::Traits<T, C_Layout>;
                using SizeArray = ::XXX_NAMESPACE::dataTypes::SizeArray<C_Dimension>;
                using ConstValueType = typename Traits<ValueT>::ConstType;
                template <typename T>
                using BasePointerType = typename Traits<T>::BasePointerType;
                using AllocatorT = typename BasePointerType<ValueT>::Allocator;
                using AllocationShape = typename AllocatorT::AllocationShape;

                // Friend declarations.
                friend class ::XXX_NAMESPACE::dataTypes::Field<ValueT, C_Dimension, C_Layout>;

            public:
                // Template parameters.
                using ValueType = ValueT;
                static constexpr SizeType Dimension = C_Dimension;
                static constexpr DataLayout Layout = C_Layout;
                static constexpr target Target = C_Target;

            private:
                struct Deleter
                {
                    auto operator()(BasePointerType<ValueT>* pointer) const
                        -> void
                    {
                        AllocatorT::template Deallocate<C_Target>(*pointer);
                    }
                };

                Container()
                    :
                    n{},
                    allocation_shape{}
                {}

                Container(const SizeArray& n)
                    :
                    n(n),
                    allocation_shape(AllocatorT::template GetAllocationShape<C_Layout>(n)),
                    data(new BasePointerType<ValueT>(AllocatorT::template Allocate<Target>(allocation_shape), allocation_shape.n_0), Deleter()),
                    ptr(*data),
                    const_ptr(*data)
                {}

                static constexpr bool UseProxyType = (C_Layout != DataLayout::AoS && ::XXX_NAMESPACE::internal::ProvidesProxyType<ValueT>::value);
                using return_type = std::conditional_t<(C_Dimension == 1), std::conditional_t<UseProxyType, typename Traits<ValueT>::ProxyType, ValueT&>, internal::Accessor<ValueT, C_Dimension - 1, C_Dimension, C_Layout>>;
                using const_return_type = std::conditional_t<(C_Dimension == 1), std::conditional_t<UseProxyType, const typename Traits<const ValueT>::ProxyType, const ValueT&>, internal::Accessor<ConstValueType, C_Dimension - 1, C_Dimension, C_Layout>>;

            public:

                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline auto operator[](const SizeType index)
                    -> return_type
                {
                    return internal::Accessor<ValueT, C_Dimension, C_Dimension, C_Layout>(ptr, n)[index];
                }

                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline auto operator[](const SizeType index) const
                    -> const_return_type
                {
                    return internal::Accessor<ConstValueType, C_Dimension, C_Dimension, C_Layout>(const_ptr, n)[index];
                }

                template <typename FuncT>
                auto Set(FuncT func)
                {
                    if (data.get())
                    {   
                        for (size_t stab_index = 0; stab_index < allocation_shape.num_stabs; ++stab_index)
                        {
                            for (size_t i = 0; i < n[0]; ++i)
                            {
                                std::get<0>(data.get()->At(stab_index, i)) = func();
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
                    return allocation_shape.GetByteSize();
                }

                auto GetBasePointer()
                {
                    return data.get()->GetBasePointer();
                }
                
                SizeArray n;
                AllocationShape allocation_shape;
                std::shared_ptr<BasePointerType<ValueT>> data;
                BasePointerType<ValueT> ptr;
                BasePointerType<ConstValueType> const_ptr;
            };
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //!
        //! \brief Multi-dimensional field data type
        //!
        //! This field data type uses a plain C-pointer to dynamically adapt to the needed memory requirement.
        //! In case of C_Dimension > 1, its is determined as the product of all dimensions with the
        //! innermost dimension padded according to ValueT, the data layout and the (default) data alignment.
        //! All memory is contiguous and can be moved as a whole (important e.g. for data transfers).
        //! The field, however, allows At to the data using array subscript operator chaining together with proxy objects.
        //! \n\n
        //! In case of Data_layout = AoS, data is stored with all elements placed in main memory one after the other.
        //! \n
        //! In case of Data_layout = SoA (applied only if ValueT is a record type) the individual members
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
        //! You can At the individual components of field<vec<double, 3>, 2, SoA> b({3,2})
        //! as usual, e.g., b[1][0].x = ...
        //! \n
        //! GNU and Clang/LLVM seem to optimize the proxies away.
        //!
        //! \tparam ValueT data type
        //! \tparam C_Dimension dimension
        //! \tparam C_Layout any of SoA (struct of arrays) and AoS (array of structs)
        //!
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        template <typename T, SizeType C_Dimension, ::XXX_NAMESPACE::memory::DataLayout C_Layout>
        class Field
        {
            static_assert(!std::is_const<T>::value, "error: field with const elements is not allowed");

        public:

            using element_type = T;
            static constexpr SizeType dimension = C_Dimension;
            static constexpr ::XXX_NAMESPACE::memory::DataLayout layout = C_Layout;

        private:

            using const_element_type = typename ::XXX_NAMESPACE::internal::Traits<element_type, C_Layout>::ConstType;
            template <::XXX_NAMESPACE::target C_Target>
            using ContainerType = internal::Container<T, C_Dimension, C_Layout, C_Target>;
            
        public:

            Field() = default;
                
            Field(const ::XXX_NAMESPACE::dataTypes::SizeArray<C_Dimension>& n, const bool initialize_to_zero = false)
                :
                n(n)
            {
                Resize(n, initialize_to_zero);
            }

            auto Resize(const ::XXX_NAMESPACE::dataTypes::SizeArray<C_Dimension>& new_n, const bool initialize_to_zero = false)
                -> void
            {
                if (n != new_n)
                {
                    n = new_n;
                    data = ContainerType<::XXX_NAMESPACE::target::Host>(n);
                    
                    if (initialize_to_zero)
                    {
                        data.Set([] () { return 0; });
                    }

                #if defined(__CUDACC__)
                    device_data = ContainerType<::XXX_NAMESPACE::target::GPU_CUDA>(n);

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

            static constexpr bool UseProxyType = (C_Layout != ::XXX_NAMESPACE::memory::DataLayout::AoS && ::XXX_NAMESPACE::internal::ProvidesProxyType<T>::value);
            using value_type = std::conditional_t<(C_Dimension == 1), std::conditional_t<UseProxyType, typename ::XXX_NAMESPACE::internal::Traits<T, C_Layout>::ProxyType, T&>, internal::Accessor<element_type, C_Dimension - 1, C_Dimension, C_Layout>>;
            using const_value_type = std::conditional_t<(C_Dimension == 1), std::conditional_t<UseProxyType, const typename ::XXX_NAMESPACE::internal::Traits<const T, C_Layout>::ProxyType, const T&>, internal::Accessor<const_element_type, C_Dimension - 1, C_Dimension, C_Layout>>;

            inline auto operator[](const SizeType index)
                -> value_type
            {
                return data[index];
            }

            inline auto operator[](const SizeType index) const
                -> const_value_type
            {
                return data[index];
            }

            HOST_VERSION
            CUDA_DEVICE_VERSION    
            inline auto size() const
                -> const ::XXX_NAMESPACE::dataTypes::SizeArray<C_Dimension>&
            {
                return n;   
            }

        #if defined(__CUDACC__)
            auto GetDeviceAccess(const bool sync_with_host = false)
                -> ContainerType<::XXX_NAMESPACE::target::GPU_CUDA>&
            {
                if (device_data.IsEmpty())
                {
                    device_data = ContainerType<::XXX_NAMESPACE::target::GPU_CUDA>(n);
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

            ::XXX_NAMESPACE::dataTypes::SizeArray<C_Dimension> n;
            ContainerType<::XXX_NAMESPACE::target::Host> data;
        #if defined(__CUDACC__)
            ContainerType<::XXX_NAMESPACE::target::GPU_CUDA> device_data;
        #endif
        };
    }
}

#endif
