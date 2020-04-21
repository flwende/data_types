// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(DATA_TYPES_FIELD_FIELD_HPP)
#define DATA_TYPES_FIELD_FIELD_HPP

#include <memory>
#include <type_traits>
#include <vector>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#include <common/DataLayout.hpp>
#include <common/Memory.hpp>
#include <common/Traits.hpp>
#include <common/SmartPointer.hpp>
#include <tuple/Get.hpp>
#include <platform/Target.hpp>
#include <DataTypes.hpp>

namespace XXX_NAMESPACE
{
    namespace dataTypes
    {
        using namespace ::XXX_NAMESPACE::memory;

        using ::XXX_NAMESPACE::internal::Traits;
        using ::XXX_NAMESPACE::internal::ProvidesProxy;
        using ::XXX_NAMESPACE::memory::DataLayout;
        using ::XXX_NAMESPACE::memory::SmartPointer;
        using ::XXX_NAMESPACE::platform::Identifier;
        using ::XXX_NAMESPACE::variadic::IsInvocable;

        // Forward declaration.
        template <typename, SizeT, DataLayout>
        class Field;

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
            template <typename ValueT, SizeT Level, SizeT Dimension, DataLayout Layout>
            class Accessor
            {
                using ConstValueT = typename Traits<ValueT, Layout>::ConstT;
                using BasePointer = typename Traits<std::decay_t<ValueT>, Layout>::BasePointer;
                using Pointer = std::conditional_t<std::is_const<ValueT>::value, const BasePointer, BasePointer>;

              public:
                //!
                //! \brief Constructor.
                //!
                //! \param pointer base-pointer-like memory reference
                //! \param n extent of the Dimension-dimensional field
                //! \param stab_index the offset in units of 'innermost dimension n[0]'
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                Accessor(Pointer& pointer, const SizeArray<Dimension>& n, const SizeT stab_index = 0) : pointer(pointer), n(n), stab_index(stab_index)
                {
                    assert(pointer.IsValid());
                }

                //!
                //! \brief Array subscript operator.
                //!
                //! This function returns a lower-dimensional accessor type with the `stab_index` shifted
                //! by the number of stabs in the (Dimension-1)-dimension sub-volume according to the `index` value.
                //!
                //! \param index element index
                //! \return a lower-dimensional (const) `Accessor` type with a shifted stab_index
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline auto operator[](const SizeT index) -> Accessor<ValueT, Level - 1, Dimension, Layout>
                { 
                    assert(index < n[Level - 1]);

                    return {pointer, n, stab_index + index * n.ReduceMul(1, Level - 1)};
                }

                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline auto operator[](const SizeT index) const -> Accessor<ConstValueT, Level - 1, Dimension, Layout>
                {
                    assert(index < n[Level - 1]);

                    return {pointer, n, stab_index + index * n.ReduceMul(1, Level - 1)};
                }

              protected:
                Pointer& pointer;
                const SizeArray<Dimension>& n;
                const SizeT stab_index;
            };

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
            template <typename ValueT, SizeT Dimension, DataLayout Layout>
            class Accessor<ValueT, 1, Dimension, Layout>
            {
                using ConstValueT = typename Traits<ValueT, Layout>::ConstT;
                using BasePointer = typename Traits<std::decay_t<ValueT>, Layout>::BasePointer;
                using Pointer = std::conditional_t<std::is_const<ValueT>::value, const BasePointer, BasePointer>;
                using Proxy = typename Traits<ValueT, Layout>::Proxy;
                using ConstProxy = typename Traits<const ValueT, Layout>::Proxy;

                struct IndexPair
                {
                    const SizeT stab_index;
                    const SizeT intra_stab_index;
                };

                //!
                //! \brief Calculate the stab and intra-stab index for a given index.
                //! 
                //! The calculation is relative to this accessor's internal state.
                //! Further, it is multi-versioned according to the specified data layout.
                //!
                //! \param index the index
                //! \return stab and intra-stab index for `index`
                //!
                template <DataLayout Enable = Layout>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline auto GetStabAndIntraStabIndex(const SizeT index) const -> std::enable_if_t<Enable == DataLayout::SoA, IndexPair>
                {
                    return {0, stab_index * n[0] + index};
                }

                template <DataLayout Enable = Layout>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline auto GetStabAndIntraStabIndex(const SizeT index) const -> std::enable_if_t<Enable == DataLayout::AoSoA, IndexPair>
                {
                    constexpr SizeT N0 = BasePointer::InnerArraySize;

                    return {stab_index * ((n[0] + N0 - 1) / N0) + (index / N0), index % N0};
                }

                template <DataLayout Enable = Layout>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline auto GetStabAndIntraStabIndex(const SizeT index) const -> std::enable_if_t<!(Enable == DataLayout::SoA || Enable == DataLayout::AoSoA), IndexPair>
                {
                    return {stab_index, index};
                }

              public:
                //!
                //! \brief Constructor.
                //!
                //! \param pointer base pointer
                //! \param n extent of the field
                //! \param stab_index the stab_index
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                Accessor(Pointer& pointer, const SizeArray<Dimension>& n, const SizeT stab_index = 0) : pointer(pointer), n(n), stab_index(stab_index) 
                {
                    assert(pointer.IsValid());
                }

                //!
                //! \brief Array subscript operator (AoS data layout).
                //!
                //! The return value of `At(..)` is a tuple with a single reference to some variable of type `ValueT`.
                //! Get the reference through `Gget<0>`.
                //!
                //! \tparam Enable used for multi-versioning depending on the data layout
                //! \param index the intra-stab index
                //! \return a (const) reference to a variable of type `ValueT`
                //!
                template <DataLayout Enable = Layout>
                HOST_VERSION CUDA_DEVICE_VERSION inline auto operator[](const SizeT index) -> std::enable_if_t<Enable == DataLayout::AoS, ValueT&>
                {
                    assert(index < n[0]);

                    return Get<0>(pointer.At(stab_index, index));
                }

                template <DataLayout Enable = Layout>
                HOST_VERSION CUDA_DEVICE_VERSION inline auto operator[](const SizeT index) const -> std::enable_if_t<Enable == DataLayout::AoS, ConstValueT&>
                {
                    assert(index < n[0]);

                    return Get<0>(pointer.At(stab_index, index));
                }

                //!
                //! \brief Array subscript operator (SoA data layout).
                //!
                //! The return value of `At(..)` is a tuple of references that is used for the proxy type construction.
                //!
                //! \tparam Enable used for multi-versioning depending on the data layout
                //! \param index the intra-stab index
                //! \return a (const) proxy type
                //!
                template <DataLayout Enable = Layout>
                HOST_VERSION CUDA_DEVICE_VERSION inline auto operator[](const SizeT index) -> std::enable_if_t<Enable != DataLayout::AoS, Proxy>
                {
                    assert(index < n[0]);

                    const auto& indices = GetStabAndIntraStabIndex(index);

                    return {pointer.At(indices.stab_index, indices.intra_stab_index)};
                }

                template <DataLayout Enable = Layout>
                HOST_VERSION CUDA_DEVICE_VERSION inline auto operator[](const SizeT index) const -> std::enable_if_t<Enable != DataLayout::AoS, ConstProxy>
                {
                    assert(index < n[0]);

                    const auto& indices = GetStabAndIntraStabIndex(index);

                    return {pointer.At(indices.stab_index, indices.intra_stab_index)};
                }

                //!
                //! \brief Request an `Accessor` with dimension 0 that points to a specific position.
                //! 
                //! \param index the position
                //! \return a (const) `Accessor` with dimension 0 that points to position `index`
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline auto At(const SizeT index) -> Accessor<ValueT, 0, 0, Layout>
                { 
                    assert(index < n[0]);

                    const auto& indices = GetStabAndIntraStabIndex(index);
                    
                    return {pointer, n[0], indices.stab_index, indices.intra_stab_index};
                }

                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline auto At(const SizeT index) const -> Accessor<ConstValueT, 0, 0, Layout>
                { 
                    assert(index < n[0]);

                    const auto& indices = GetStabAndIntraStabIndex(index);
                    
                    return {pointer, n[0], indices.stab_index, indices.intra_stab_index};
                }

              protected:
                Pointer& pointer;
                const SizeArray<Dimension>& n;
                const SizeT stab_index;
            };

            //!
            //! \brief Accessor type for array subscript operator chaining [][]..[] (special version).
            //!
            //! This is the recursion anchor (`Level=1`). Depending on the element type and the data layout, either a reference
            //! to an element or a proxy type is returned by the array subscript operator.
            //!
            //! \tparam ValueT element type
            //! \tparam Dimension the dimension of the field
            //! \tparam Layout any of AoS, SoAi, SoA
            //!
            template <typename ValueT, DataLayout Layout>
            class Accessor<ValueT, 0, 0, Layout>
            {
                using ConstValueT = const ValueT;
                using BasePointer = typename Traits<std::decay_t<ValueT>, Layout>::BasePointer;
                using Pointer = std::conditional_t<std::is_const<ValueT>::value, const BasePointer, BasePointer>;
                using Proxy = typename Traits<ValueT, Layout>::Proxy;
                using ConstProxy = typename Traits<const ValueT, Layout>::Proxy;

              public:
                using ValueType = ValueT;

                //!
                //! \brief Constructor.
                //!
                //! \param pointer base pointer
                //! \param n extent of the field
                //! \param stab_index the stab_index
                //! \param intra_stab_index the index offset within a stab
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                Accessor(Pointer& pointer, const SizeT n_0, const SizeT stab_index = 0, const SizeT intra_stab_index = 0) : pointer(pointer), n_0(n_0), stab_index(stab_index), intra_stab_index(intra_stab_index)
                {
                    assert(pointer.IsValid());
                }
                
                //!
                //! \brief Array subscript operator (AoS data layout).
                //!
                //! The return value of `At(..)` is a tuple with a single reference to some variable of type `ValueT`.
                //! Get the reference through `Gget<0>`.
                //!
                //! \tparam Enable used for multi-versioning depending on the data layout
                //! \param index the intra-stab index
                //! \return a (const) reference to a variable of type `ValueT`
                //!
                template <DataLayout Enable = Layout>
                HOST_VERSION CUDA_DEVICE_VERSION inline auto operator[](const SizeT index) -> std::enable_if_t<Enable == DataLayout::AoS, ValueT&>
                {
                    assert((intra_stab_index + index) < n_0);

                    return Get<0>(pointer.At(stab_index, intra_stab_index + index));
                }

                template <DataLayout Enable = Layout>
                HOST_VERSION CUDA_DEVICE_VERSION inline auto operator[](const SizeT index) const -> std::enable_if_t<Enable == DataLayout::AoS, const ValueT&>
                {
                    assert((intra_stab_index + index) < n_0);

                    return Get<0>(pointer.At(stab_index, intra_stab_index + index));
                }
                
                //!
                //! \brief Array subscript operator (all other layouts).
                //!
                //! The return value of `At(..)` is a tuple of references that is used for the proxy type construction.
                //!
                //! \tparam Enable used for multi-versioning depending on the data layout
                //! \param index the intra-stab index
                //! \return a (const) proxy type
                //!
                template <DataLayout Enable = Layout>
                HOST_VERSION CUDA_DEVICE_VERSION inline auto operator[](const SizeT index) -> std::enable_if_t<Enable != DataLayout::AoS, Proxy>
                {
                    assert((intra_stab_index + index) < n_0);

                    return {pointer.At(stab_index, intra_stab_index + index)};
                }

                template <DataLayout Enable = Layout>
                HOST_VERSION CUDA_DEVICE_VERSION inline auto operator[](const SizeT index) const -> std::enable_if_t<Enable != DataLayout::AoS, ConstProxy>
                {
                    assert((intra_stab_index + index) < n_0);

                    return {pointer.At(stab_index, intra_stab_index + index)};
                }

                HOST_VERSION CUDA_DEVICE_VERSION inline auto& operator++()
                {
                    ++intra_stab_index;
                    return *this;
                }

                //!
                //! \brief Request an `Accessor` with dimension 0 that points to a specific position.
                //! 
                //! \param index the position
                //! \return a (const) `Accessor` with dimension 0 that points to position `index`
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline auto At(const SizeT index) -> Accessor<ValueT, 0, 0, Layout>
                { 
                    assert((intra_stab_index + index) < n_0);

                    return {pointer, n_0, stab_index, intra_stab_index + index};
                }

                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline auto At(const SizeT index) const -> Accessor<ConstValueT, 0, 0, Layout>
                { 
                    assert((intra_stab_index + index) < n_0);

                    return {pointer, n_0, stab_index, intra_stab_index + index};
                }

                //! @{
                //! Some iterator functionality.
                //!
                HOST_VERSION CUDA_DEVICE_VERSION inline auto operator==(const Accessor& other) const
                {
                    return (stab_index == other.stab_index && intra_stab_index == other.intra_stab_index);
                }

                HOST_VERSION CUDA_DEVICE_VERSION inline auto operator!=(const Accessor& other) const
                {
                    return !(*this == other);
                }

                template <DataLayout Enable = Layout>
                HOST_VERSION CUDA_DEVICE_VERSION inline auto operator*() -> std::enable_if_t<Enable != DataLayout::AoS, Proxy>
                {
                    return (*this)[0];
                }

                template <DataLayout Enable = Layout>
                HOST_VERSION CUDA_DEVICE_VERSION inline auto operator*() const -> std::enable_if_t<Enable != DataLayout::AoS, ConstProxy>
                {
                    return (*this)[0];
                }
                //! @}

              protected:
                Pointer& pointer;
                const SizeT n_0;
                const SizeT stab_index;
                SizeT intra_stab_index;
            };

            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            template <typename ValueT, SizeT Dimension, DataLayout Layout, Identifier Target>
            class Container
            {
                template <typename T>
                using Traits = Traits<T, Layout>;
                using ConstValueT = typename Traits<ValueT>::ConstT;
                template <typename T>
                using BasePointer = typename Traits<T>::BasePointer;
                using Allocator = typename BasePointer<ValueT>::Allocator;
                using AllocationShape = typename Allocator::AllocationShape;
                template <typename T, SizeT L, SizeT D = Dimension>
                using Accessor = internal::Accessor<T, L, D, Layout>;
                static constexpr bool UseProxy = (Layout != DataLayout::AoS && ProvidesProxy<ValueT>::value);
                using Proxy = typename Traits<ValueT>::Proxy;
                using ConstProxy = typename Traits<ConstValueT>::Proxy;
                using ReturnT = std::conditional_t<Dimension == 1, std::conditional_t<UseProxy, Proxy, ValueT&>, Accessor<ValueT, Dimension - 1>>;
                using ConstReturnT = std::conditional_t<Dimension == 1, std::conditional_t<UseProxy, ConstProxy, const ValueT&>, Accessor<ConstValueT, Dimension - 1>>;

                template <typename T>
                using SmartPointerHost = SmartPointer<T, Deleter<Identifier::Host>>;
#if defined(__CUDACC__)
                template <typename T>
                using SmartPointerGPU = SmartPointer<T, Deleter<Identifier::GPU_CUDA>>;
#endif

                // Friend declarations.
                friend class ::XXX_NAMESPACE::dataTypes::Field<ValueT, Dimension, Layout>;

              public:
                // Template parameters.
                using TParam_ValueT = ValueT;
                static constexpr SizeT TParam_Dimension = Dimension;
                static constexpr DataLayout TParam_Layout = Layout;
                static constexpr Identifier TParam_Target = Target;

              private:
                //!
                //! \brief A deleter type for shared pointer deallocation.
                //!
                struct Deleter
                {
                    //!
                    //! \brief Callable for shared pointer deallocation.
                    //!
                    //! \param pointer a pointer to either `Pointer` or `MultiPointer`
                    //!
                    auto operator()(BasePointer<ValueT>* pointer) const -> void
                    { 
                        assert(pointer != nullptr);

                        Allocator::template Deallocate<Target>(*pointer);
                    }
                };

                //!
                //! \brief Standard constructor (private).
                //!
                //! Create an empty `Container`.
                //!
                Container() : n{}, allocation_shape{}, base_pointer{}, pointer{} {}

                //!
                //! \brief Constructor (private).
                //!
                //! Create a `Container` from a `SizeArray`.
                //!
                //! \param n a `SizeArray` (e.g. extent of a `Field`)
                //!
                Container(const SizeArray<Dimension>& n)
                    : n(n), allocation_shape(Allocator::template GetAllocationShape<Layout>(n)),
#if defined(__CUDACC__)
                      base_pointer(new BasePointer<ValueT>(Allocator::template Allocate<Target>(allocation_shape), allocation_shape.n_0), Deleter()), 
#else
                      base_pointer(new BasePointer<ValueT>(Allocator::template Allocate<Target>(allocation_shape), allocation_shape.n_0)), 
#endif
                      pointer(*base_pointer)
                {
                }

              public:
                //!
                //! \brief Get the extent of the innermost array (meaningful for AoSoA data layout only).
                //!
                //! \return the extent of the innermost array
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                static constexpr inline auto GetInnerArraySize() { return BasePointer<ValueT>::InnerArraySize; }

                //!
                //! \brief Array subscript operator.
                //!
                //! The type of the return value is inherited from the `Accessor`'s subscript operator (it is determined above).
                //! Depending on the dimensionality of the `Container`, this function returns either a reference to a variable of
                //! type `ValueT` or an `Accessor` of lower dimension.
                //!
                //! \param index an index value for the data access
                //! \return a (const) reference to a variable of type `ValueT` or an `Accessor` of lower dimension
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline auto operator[](const SizeT index) -> ReturnT { return Accessor<ValueT, Dimension>(pointer, n)[index]; }

                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline auto operator[](const SizeT index) const -> ConstReturnT { return Accessor<ConstValueT, Dimension>(pointer, n)[index]; }

                //!
                //! \brief Request an `Accessor` with dimension 0 that points to a specific position.
                //! 
                //! This function exists only for `Dimension` 1.
                //!
                //! \param index the position
                //! \return a (const) `Accessor` with dimension 0 that points to position `index`
                //!
                template <SizeT D = Dimension>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline auto At(const SizeT index) -> std::enable_if_t<D == 1, Accessor<ValueT, 0, 0>> { return Accessor<ValueT, 1>(pointer, n).At(index); }

                template <SizeT D = Dimension>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline auto At(const SizeT index) const -> std::enable_if_t<D == 1, Accessor<ConstValueT, 0, 0>> { return Accessor<ConstValueT, 1>(pointer, n).At(index); }

                //!
                //! \brief Set the content of the container.
                //!
                //! This function uses a callable `func` for value assignment to the container elements.
                //! This function uses a two dimensional `Accessor` to iterator over the stabs and within each stab.
                //! 
                //! \tparam FuncT the type of the callable
                //! \param func a callable
                //!
                template <typename FuncT>
                auto Set(FuncT func) -> void
                {
                    static_assert(IsInvocable<FuncT, SizeT>::value, "error: callable is not invocable. void (*) (SizeT) expected.");

                    if (Dimension == 1)
                    {
                        internal::Accessor<ValueT, 1, Dimension, Layout> accessor(pointer, n);

                        for (SizeT i = 0; i < n[0]; ++i)
                        {
                            accessor[i] = func(i);
                        }
                    } 
                    else
                    {
                        for (SizeT k = 0; k < n.ReduceMul(2); ++k)
                        {
                            internal::Accessor<ValueT, 2, Dimension, Layout> accessor(pointer, n, k * n[1]);

                            for (SizeT stab_index = 0; stab_index < n[1]; ++stab_index)
                            {
                                for (SizeT i = 0; i < n[0]; ++i)
                                {
                                    accessor[stab_index][i] = func((k * n[1] + stab_index) * n[0] + i);
                                }
                            }
                        }
                    }
                }

                //!
                //! \brief Copy all data into a (vector) container.
                //!
                //! All data is serialized: the output data layout is AoS.
                //! This function uses a two dimensional (const) `Accessor` to iterator over the stabs and within each stab.
                //! 
                //! \return a (vector) container holding all the data
                //!
                auto Get() const
                {
                    std::vector<ValueT> data;
                    data.reserve(n.ReduceMul());
                    
                    if (Dimension == 1)
                    {
                        internal::Accessor<ConstValueT, 1, Dimension, Layout> accessor(pointer, n);

                        for (SizeT i = 0; i < n[0]; ++i)
                        {
                            data.push_back(accessor[i]);
                        }
                    } 
                    else
                    {
                        for (SizeT k = 0; k < n.ReduceMul(2); ++k)
                        {
                            internal::Accessor<ConstValueT, 2, Dimension, Layout> accessor(pointer, n, k * n[1]);

                            for (SizeT stab_index = 0; stab_index < n[1]; ++stab_index)
                            {
                                for (SizeT i = 0; i < n[0]; ++i)
                                {
                                    data.push_back(accessor[stab_index][i]);
                                }
                            }
                        }
                    }

                    return data;
                }

                //!
                //! \brief Get the size of the container.
                //!
                //! \return the size of the container
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline const auto& Size() const { return n; }

                //!
                //! \brief Get the size of the container for a specific dimension.
                //!
                //! \param dimension the requested dimension
                //! \return the size of the container for the requested dimension
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline auto Size(const SizeT dimension) const
                { 
                    assert(dimension < Dimension);

                    return n[dimension];
                }

                HOST_VERSION
                CUDA_DEVICE_VERSION
                inline auto Pitch() const
                { 
                    return allocation_shape.n_0;
                }

                //!
                //! \brief Test for this container being empty.
                //!
                //! \return `true` if the container is empty, otherwise `false`
                //!
                inline auto IsEmpty() const { return (base_pointer.get() == nullptr || n.ReduceMul() == 0); }

              protected:
                //!
                //! \brief Get the total size of the container in bytes.
                //!
                //! \return the total size of the container in bytes
                //!
                auto GetByteSize() const { return allocation_shape.GetByteSize(); }

                //!
                //! \brief Get the base pointer of this container.
                //!
                //! \return the base pointer of this container
                //!
                auto GetBasePointer() const
                { 
                    return pointer.GetBasePointer();
                }

                SizeArray<Dimension> n;
                AllocationShape allocation_shape;
#if defined(__CUDACC__)
                std::shared_ptr<BasePointer<ValueT>> base_pointer;
#else
                std::unique_ptr<BasePointer<ValueT>, Deleter> base_pointer;
#endif
                BasePointer<ValueT> pointer;
                //BasePointer<ConstValueT> const_pointer;
            };
        } // namespace internal

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
        template <typename ValueT, SizeT Dimension, DataLayout Layout = DataLayout::AoS>
        class Field
        {
            static_assert(Dimension > 0, "error: a field without zero-dimension is not valid.");
            static_assert(!std::is_const<ValueT>::value, "error: field with const elements is not allowed.");

            template <typename T>
            using Traits = Traits<T, Layout>;
            using ConstValueT = typename Traits<ValueT>::ConstT;
            template <Identifier Target>
            using Container = internal::Container<ValueT, Dimension, Layout, Target>;
            template <typename T, SizeT L, SizeT D = Dimension>
            using Accessor = internal::Accessor<T, L, D, Layout>;
            static constexpr bool UseProxy = (Layout != DataLayout::AoS && ProvidesProxy<ValueT>::value);
            using ReturnT = std::conditional_t<Dimension == 1, std::conditional_t<UseProxy, typename Traits<ValueT>::Proxy, ValueT&>, Accessor<ValueT, Dimension - 1>>;
            using ConstReturnT = std::conditional_t<Dimension == 1, std::conditional_t<UseProxy, typename Traits<ConstValueT>::Proxy, const ValueT&>, Accessor<ConstValueT, Dimension - 1>>;

          public:
            // Template parameters.
            using TParam_ValueT = ValueT;
            static constexpr SizeT TParam_Dimension = Dimension;
            static constexpr DataLayout TParam_Layout = Layout;

          public:
            //!
            //! \brief Standard constructor.
            //!
            //! Create an empty `Field`.
            //!
            
#if defined(__CUDACC__)
            Field() : n{}, data{}, d_data{} {}
#else
            Field() : n{}, data{} {}
#endif

            //!
            //! \brief Constructor.
            //!
            //! Create a `Field` from a `SizeArray` and optionally zero all its elements.
            //!
            //! \param n the extent of the field
            //! \param initialize_to_zero (optional) if `true`, zero all its elements
            //!
            Field(const SizeArray<Dimension>& n, const bool initialize_to_zero = false) : Field() { Resize(n, initialize_to_zero); }

            //!
            //! \brief Resize the container.
            //!
            //! Resizing this field means to replace its `data` member by a new one with the requested size.
            //! The container type internally uses a shared pointer to reference its memory.
            //! Memory deallocation of this field thus is managed by the container and its shared pointer.
            //!
            //! \param n_new the new extent of the field
            //! \param initialize_to_zero (optional) if `true`, zero all its elements
            //!
            auto Resize(const SizeArray<Dimension>& n_new, const bool initialize_to_zero = false) -> void
            {
                if (n != n_new)
                {
                    n = n_new;

                    data = Container<Identifier::Host>(n);

                    if (initialize_to_zero)
                    {
                        data.Set([](SizeT) { return 0; });
                    }

#if defined(__CUDACC__)
                    // Resize only of there is already a non-empty device container.
                    if (!d_data.IsEmpty())
                    {
                        DeviceResize(initialize_to_zero);
                    }
#endif
                }
            }

            const auto* GetBasePointer() const
            {
                return data.GetBasePointer();
            }

            //!
            //! \brief Set the content of the field.
            //!
            //! This function uses a callable `func` for value assignment to the field elements.
            //! It redirects this task to its `data` member.
            //!
            //! \tparam FuncT the type of the callable
            //! \param func a callable
            //!
            template <typename FuncT>
            auto Set(FuncT func, [[maybe_unused]] const bool sync_with_device = true)
            {
                data.Set(func);
                
#if defined(__CUDACC__)
                // Copy data to device only if there is already a device container.
                if (sync_with_device)
                {
                    if (d_data.IsEmpty())
                    {
                        DeviceResize();
                    }

                    CopyHostToDevice();
                }
#endif                
            }

            template <typename T = ValueT>
            void Get(std::vector<T>& buffer, const bool sync_with_device = false) const
            {
                static_assert(std::is_convertible<ValueT, T>::value, "error: types are not convertible.");

#if defined(__CUDACC__)
                if (sync_with_device)
                {
                    CopyDeviceToHost();
                }
#endif

                buffer = data.Get();
            }

            auto Get(const bool sync_with_device = false) const
            {
#if defined(__CUDACC__)
                if (sync_with_device)
                {
                    CopyDeviceToHost();
                }
#endif

                return data.Get();
            }

            //!
            //! \brief Array subscript operator.
            //!
            //! The type of the return value is inherited from the `Accessor`'s subscript operator (it is determined above).
            //! Depending on the dimensionality of the `Container`, this function returns either a reference to a variable of
            //! type `ValueT` or an `Accessor` of lower dimension.
            //!
            //! \param index an index value for the data access
            //! \return a (const) reference to a variable of type `ValueT` or an `Accessor` of lower dimension
            //!
            inline auto operator[](const SizeT index) -> ReturnT { return data[index]; }

            inline auto operator[](const SizeT index) const -> ConstReturnT { return data[index]; }

            //!
            //! \brief Request an `Accessor` with dimension 0 that points to a specific position.
            //! 
            //! This function exists only for `Dimension` 1.
            //!
            //! \param index the position
            //! \return a (const) `Accessor` with dimension 0 that points to position `index`
            //!
            template <SizeT D = Dimension>
            inline auto At(const SizeT index) -> std::enable_if_t<D == 1, Accessor<ValueT, 0, 0>> { return data.At(index); }

            template <SizeT D = Dimension>
            inline auto At(const SizeT index) const -> std::enable_if_t<D == 1, Accessor<ConstValueT, 0, 0>> { return data.At(index); }

            //!
            //! \brief Get the size of the container.
            //!
            //! \return the size of the container
            //!
            inline const auto& Size() const { return n; }

            //!
            //! \brief Get the size of the container for a specific dimension.
            //!
            //! \param dimension the requested dimension
            //! \return the size of the container for the requested dimension
            //!
            inline auto Size(const SizeT dimension) const
            { 
                assert(dimension < Dimension);

                return n[dimension];
            }

            inline auto Pitch() const
            { 
                return data.Pitch();
            }

            static constexpr inline auto GetInnerArraySize() { return Container<Identifier::Host>::GetInnerArraySize(); }

#if defined(__CUDACC__)
            //!
            //! \brief Get access to the device data.
            //!
            //! If the device container has not been initialized properly yet, create it.
            //!
            //! \param sync_with_host (optional) if `true`, copy all data from the host to the device
            //! \return a reference to the device container
            //!
            auto DeviceData(const bool sync_with_host = false) -> Container<Identifier::GPU_CUDA>&
            {
                if (d_data.IsEmpty())
                {
                    DeviceResize(sync_with_host);
                }

                return d_data;
            }

            //!
            //! \brief Copy device data to the host.
            //!
            auto CopyDeviceToHost() const -> void
            {
                assert(!data.IsEmpty());

                if (!data.IsEmpty())
                {
                    assert(data.GetBasePointer() != nullptr);
                    assert(d_data.GetBasePointer() != nullptr);

                    cudaMemcpy((void*)data.GetBasePointer(), (const void*)d_data.GetBasePointer(), data.GetByteSize(), cudaMemcpyDeviceToHost);
                }
            }

            //!
            //! \brief Copy host data to the device.
            //!
            auto CopyHostToDevice() -> void
            {
                // Create device container if not already there.
                if (d_data.IsEmpty())
                {
                    DeviceResize();
                }

                assert(data.GetBasePointer() != nullptr);
                assert(d_data.GetBasePointer() != nullptr);

                cudaMemcpy((void*)d_data.GetBasePointer(), (const void*)data.GetBasePointer(), data.GetByteSize(), cudaMemcpyHostToDevice);
            }

          protected:
            //!
            //! \brief Resize the device container.
            //!
            //! \param sync_with_host (optional) if `true`, copy all data from the host to the device
            //!
            auto DeviceResize(const bool sync_with_host = false) -> void
            {
                // Resize only of there is already a non-empty device container.
                d_data = Container<Identifier::GPU_CUDA>(n);

                if (sync_with_host)
                {
                    CopyHostToDevice();
                }
            }
#endif
          protected:
            SizeArray<Dimension> n;
            Container<Identifier::Host> data;
#if defined(__CUDACC__)
            Container<Identifier::GPU_CUDA> d_data;
#endif
        };
    } // namespace dataTypes
} // namespace XXX_NAMESPACE

#endif
