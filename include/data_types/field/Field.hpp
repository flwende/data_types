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
            //! For Dimension > 1, the number of stabs is determined as the reduction over all indices, but without accounting for
            //! the innermost dimension: [k][j][i] -> '(k * n[1] + j) * n[0] + i' -> 'stab_index = k * n[1] + j'
            //! The 'memory' type holds a base-pointer-like memory reference to [0][0]..[0] and can deduce the final
            //! memory reference from 'stab_index', 'n[0]' and 'i'.
            //! The result (recursion anchor, Dimension=1) of the array subscript operator chaining is either a reference of type 
            //! 'ValueT' in case of the AoS data layout or if there is no proxy type available with the SoA data layout, or
            //! a proxy type that is initialized through the final memory reference type in case of SoA.
            //!
            //! \tparam ValueT element type
            //! \tparam Level recursion level
            //! \tparam Dimension the dimension of the field
            //! \tparam Layout any of AoS, SoAi, SoA
            //!
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            template <typename ValueT, SizeT Level, SizeT Dimension, ::XXX_NAMESPACE::memory::DataLayout Layout>
            class Accessor
            {
                using BasePointer = typename ::XXX_NAMESPACE::internal::Traits<ValueT, Layout>::BasePointer;
                using Pointer = std::conditional_t<std::is_const<ValueT>::value, const BasePointer, BasePointer>;
                using DataLayout = ::XXX_NAMESPACE::memory::DataLayout;
                using SizeArray = ::XXX_NAMESPACE::dataTypes::SizeArray<Dimension>;

            public:            
                //!
                //! \brief Constructor.
                //!
                //! \param ptr base-pointer-like memory reference
                //! \param n extent of the Dimension-dimensional field
                //! \param stab_index the offset in units of 'innermost dimension n[0]'
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                Accessor(Pointer& ptr, const SizeArray& n, const SizeT stab_index = 0) 
                    : 
                    ptr(ptr), 
                    n(n), 
                    stab_index(stab_index) 
                {}
            
                //!
                //! \brief Array subscript operator.
                //!
                //! This function returns a lower-dimensional accessor type with the `stab_index` shifted 
                //! by the number of stabs in the (Dimension-1)-dimension sub-volume according to the `index` value.
                //!
                //! \param index element index
                //! \return a lower-dimensional accessor type with a shifted stab_index
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline auto operator[] (const SizeT index)
                    -> Accessor<ValueT, Level - 1, Dimension, Layout>
                {
                    return {ptr, n, stab_index + index * n.ReduceMul(1, Level - 1)};
                }
            
                //!
                //! \brief Array subscript operator.
                //!
                //! This function returns a lower-dimensional accessor type with the `stab_index` shifted 
                //! by the number of stabs in the (Dimension-1)-dimension sub-volume according to the `index` value.
                //!
                //! \param index element index
                //! \return a lower-dimensional accessor type with a shifted stab_index
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline auto operator[] (const SizeT index) const
                    -> Accessor<ValueT, Level - 1, Dimension, Layout>
                {
                    return {ptr, n, stab_index + index * n.ReduceMul(1, Level - 1)};
                }

            private:
                Pointer& ptr;
                const SizeArray& n;
                const SizeT stab_index;
            };
            
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            //!
            //! \brief Accessor type for array subscript operator chaining [][]..[] (recursion anchor).
            //!
            //! This is the recursion anchor (`Level=1`). Depending on the element type and the data layout, either a reference
            //! to an element or a proxy type is returned by the array subscript operator.
            //!
            //! \tparam ValueT element type
            //! \tparam Dimension the dimension of the field
            //! \tparam Layout any of AoS, SoAi, SoA
            //!
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            template <typename ValueT, SizeT Dimension, ::XXX_NAMESPACE::memory::DataLayout Layout>
            class Accessor<ValueT, 1, Dimension, Layout>
            {
                using BasePointer = typename ::XXX_NAMESPACE::internal::Traits<ValueT, Layout>::BasePointer;
                using Pointer = std::conditional_t<std::is_const<ValueT>::value, const BasePointer, BasePointer>;
                using Proxy = typename ::XXX_NAMESPACE::internal::Traits<ValueT, Layout>::Proxy;
                using ConstProxy = const typename ::XXX_NAMESPACE::internal::Traits<const ValueT, Layout>::Proxy;
                using SizeArray = ::XXX_NAMESPACE::dataTypes::SizeArray<Dimension>;
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
                Accessor(Pointer& ptr, const SizeArray& n, const SizeT stab_index = 0) 
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
                template <DataLayout Enable = Layout>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline auto operator[] (const SizeT index)
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
                template <DataLayout Enable = Layout>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline auto operator[] (const SizeT index) const
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
                template <DataLayout Enable = Layout>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline auto operator[] (const SizeT index)
                    -> std::enable_if_t<Enable == DataLayout::SoAi, Proxy>
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
                template <DataLayout Enable = Layout>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline auto operator[] (const SizeT index) const
                    -> std::enable_if_t<Enable == DataLayout::SoAi, ConstProxy>
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
                template <DataLayout Enable = Layout>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline auto operator[] (const SizeT index)
                    -> std::enable_if_t<Enable == DataLayout::SoA, Proxy>
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
                template <DataLayout Enable = Layout>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline auto operator[] (const SizeT index) const
                    -> std::enable_if_t<Enable == DataLayout::SoA, ConstProxy>
                {
                    return {ptr.At(stab_index * n[0] + index)};
                }

            private:
                Pointer& ptr;
                const SizeArray& n;
                const SizeT stab_index;
            };
        }
        
        // Forward declaration.
        template <typename T, SizeT Dimension, ::XXX_NAMESPACE::memory::DataLayout Layout = ::XXX_NAMESPACE::memory::DataLayout::AoS>
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
            //! \tparam Dimension the dimension of the field
            //! \tparam Layout any of AoS, SoAi, SoA
            //! \tparam Target the target platform
            //!
            template <typename ValueT, SizeT Dimension, ::XXX_NAMESPACE::memory::DataLayout Layout, target Target>
            class Container
            {
                using DataLayout = ::XXX_NAMESPACE::memory::DataLayout;
                template <typename T>
                using Traits = ::XXX_NAMESPACE::internal::Traits<T, Layout>;
                using ConstValueT = typename Traits<ValueT>::ConstT;
                using SizeArray = ::XXX_NAMESPACE::dataTypes::SizeArray<Dimension>;
                template <typename T>
                using BasePointer = typename Traits<T>::BasePointer;
                using Allocator = typename BasePointer<ValueT>::Allocator;
                using AllocationShape = typename Allocator::AllocationShape;

                // Friend declarations.
                friend class ::XXX_NAMESPACE::dataTypes::Field<ValueT, Dimension, Layout>;

            public:
                // Template parameters.
                using TParam_ValueT = ValueT;
                static constexpr SizeT TParam_Dimension = Dimension;
                static constexpr DataLayout TParam_Layout = Layout;
                static constexpr target TParam_Target = Target;

            private:
                struct Deleter
                {
                    auto operator()(BasePointer<ValueT>* pointer) const
                        -> void
                    {
                        Allocator::template Deallocate<Target>(*pointer);
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
                    allocation_shape(Allocator::template GetAllocationShape<Layout>(n)),
                    data(new BasePointer<ValueT>(Allocator::template Allocate<Target>(allocation_shape), allocation_shape.n_0), Deleter()),
                    ptr(*data),
                    const_ptr(*data)
                {}

                static constexpr bool UseProxy = (Layout != DataLayout::AoS && ::XXX_NAMESPACE::internal::ProvidesProxy<ValueT>::value);
                using return_type = std::conditional_t<(Dimension == 1), std::conditional_t<UseProxy, typename Traits<ValueT>::Proxy, ValueT&>, internal::Accessor<ValueT, Dimension - 1, Dimension, Layout>>;
                using const_return_type = std::conditional_t<(Dimension == 1), std::conditional_t<UseProxy, const typename Traits<const ValueT>::Proxy, const ValueT&>, internal::Accessor<ConstValueT, Dimension - 1, Dimension, Layout>>;

            public:

                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline auto operator[](const SizeT index)
                    -> return_type
                {
                    return internal::Accessor<ValueT, Dimension, Dimension, Layout>(ptr, n)[index];
                }

                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline auto operator[](const SizeT index) const
                    -> const_return_type
                {
                    return internal::Accessor<ConstValueT, Dimension, Dimension, Layout>(const_ptr, n)[index];
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
                std::shared_ptr<BasePointer<ValueT>> data;
                BasePointer<ValueT> ptr;
                BasePointer<ConstValueT> const_ptr;
            };
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //!
        //! \brief Multi-dimensional field data type
        //!
        //! This field data type uses a plain C-pointer to dynamically adapt to the needed memory requirement.
        //! In case of Dimension > 1, its is determined as the product of all dimensions with the
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
        //! \tparam Dimension dimension
        //! \tparam Layout any of SoA (struct of arrays) and AoS (array of structs)
        //!
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        template <typename T, SizeT Dimension, ::XXX_NAMESPACE::memory::DataLayout Layout>
        class Field
        {
            static_assert(!std::is_const<T>::value, "error: field with const elements is not allowed");

        public:

            using element_type = T;
            static constexpr SizeT dimension = Dimension;
            static constexpr ::XXX_NAMESPACE::memory::DataLayout layout = Layout;

        private:

            using const_element_type = typename ::XXX_NAMESPACE::internal::Traits<element_type, Layout>::ConstT;
            template <::XXX_NAMESPACE::target Target>
            using Container = internal::Container<T, Dimension, Layout, Target>;
            
        public:

            Field() = default;
                
            Field(const ::XXX_NAMESPACE::dataTypes::SizeArray<Dimension>& n, const bool initialize_to_zero = false)
                :
                n(n)
            {
                Resize(n, initialize_to_zero);
            }

            auto Resize(const ::XXX_NAMESPACE::dataTypes::SizeArray<Dimension>& new_n, const bool initialize_to_zero = false)
                -> void
            {
                if (n != new_n)
                {
                    n = new_n;
                    data = Container<::XXX_NAMESPACE::target::Host>(n);
                    
                    if (initialize_to_zero)
                    {
                        data.Set([] () { return 0; });
                    }

                #if defined(__CUDACC__)
                    device_data = Container<::XXX_NAMESPACE::target::GPU_CUDA>(n);

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

            static constexpr bool UseProxy = (Layout != ::XXX_NAMESPACE::memory::DataLayout::AoS && ::XXX_NAMESPACE::internal::ProvidesProxy<T>::value);
            using value_type = std::conditional_t<(Dimension == 1), std::conditional_t<UseProxy, typename ::XXX_NAMESPACE::internal::Traits<T, Layout>::Proxy, T&>, internal::Accessor<element_type, Dimension - 1, Dimension, Layout>>;
            using const_value_type = std::conditional_t<(Dimension == 1), std::conditional_t<UseProxy, const typename ::XXX_NAMESPACE::internal::Traits<const T, Layout>::Proxy, const T&>, internal::Accessor<const_element_type, Dimension - 1, Dimension, Layout>>;

            inline auto operator[](const SizeT index)
                -> value_type
            {
                return data[index];
            }

            inline auto operator[](const SizeT index) const
                -> const_value_type
            {
                return data[index];
            }

            HOST_VERSION
            CUDA_DEVICE_VERSION    
            inline auto size() const
                -> const ::XXX_NAMESPACE::dataTypes::SizeArray<Dimension>&
            {
                return n;   
            }

        #if defined(__CUDACC__)
            auto GetDeviceAccess(const bool sync_with_host = false)
                -> Container<::XXX_NAMESPACE::target::GPU_CUDA>&
            {
                if (device_data.IsEmpty())
                {
                    device_data = Container<::XXX_NAMESPACE::target::GPU_CUDA>(n);
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

            ::XXX_NAMESPACE::dataTypes::SizeArray<Dimension> n;
            Container<::XXX_NAMESPACE::target::Host> data;
        #if defined(__CUDACC__)
            Container<::XXX_NAMESPACE::target::GPU_CUDA> device_data;
        #endif
        };
    }
}

#endif
