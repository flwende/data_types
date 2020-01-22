// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(DATA_TYPES_VEC_SIMDVEC_HPP)
#define DATA_TYPES_VEC_SIMDVEC_HPP

#include <array>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#include <auxiliary/Function.hpp>
#include <common/DataLayout.hpp>
#include <common/Memory.hpp>
#include <common/Traits.hpp>
#include <DataTypes.hpp>

namespace XXX_NAMESPACE
{
    namespace dataTypes
    {
        using ::XXX_NAMESPACE::memory::DataLayout;
        using ::XXX_NAMESPACE::internal::Traits;
        using ::XXX_NAMESPACE::auxiliary::AssignAll;

        namespace internal
        {
            template <typename, SizeT, SizeT, DataLayout>
            class Accessor;
        }

        template <typename ValueT, DataLayout Layout>
        class SimdVecRef
        {
            using ConstValueT = const ValueT;
            using ReturnValueT = std::conditional_t<Layout == DataLayout::AoS, ValueT&, typename Traits<ValueT, Layout>::Proxy>;
            using ConstReturnValueT = std::conditional_t<Layout == DataLayout::AoS, ConstValueT&, typename Traits<ConstValueT, Layout>::Proxy>;

        public:
            HOST_VERSION
            CUDA_DEVICE_VERSION
            SimdVecRef(internal::Accessor<ValueT, 0, 0, Layout>&& accessor, const SizeT size) : accessor(std::move(accessor)), size(size) {}
            
            template <typename Accessor, typename Filter>
            HOST_VERSION
            CUDA_DEVICE_VERSION
            auto& Load(const Accessor& external_data, Filter filter, const SizeT n)
            {
                assert(n <= size);

                #pragma omp simd
                for (SizeT i = 0; i < n; ++i)
                {
                    filter(external_data[i], accessor[i]);
                }

                return *this;
            }

            template <typename Accessor, typename Filter>
            HOST_VERSION
            CUDA_DEVICE_VERSION
            const auto& Store(Accessor&& external_data, Filter filter, const SizeT n) const
            {
                assert(n <= size);

                #pragma omp simd
                for (SizeT i = 0; i < n; ++i)
                {
                    filter(accessor[i], external_data[i]);
                } 

                return *this;
            }

            template <typename Accessor>
            HOST_VERSION
            CUDA_DEVICE_VERSION
            auto& Load(const Accessor& external_data, const SizeT n)
            {
                return Load(external_data, AssignAll, n);
            }

            template <typename Accessor>
            HOST_VERSION
            CUDA_DEVICE_VERSION
            auto& Store(Accessor&& external_data, const SizeT n) const
            {
                return Store(external_data, AssignAll, n);
            }

            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline auto operator[](const SizeT index) -> ReturnValueT
            {
                return accessor[index];
            }
            
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline auto operator[](const SizeT index) const -> ConstReturnValueT
            {
                return accessor[index];
            }
            
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline auto begin()
            {
                return accessor;
            }

            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline auto end()
            {
                return accessor;
            }

            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline auto begin() const
            {
                return accessor.At(size);
            }

            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline auto end() const
            {
                return accessor.At(size);
            }
        
        protected:
            internal::Accessor<ValueT, 0, 0, Layout> accessor;
            const SizeT size;
        };

        template <typename ValueT, SizeT Size, DataLayout Layout>
        class SimdVec
        {
            using ConstValueT = const ValueT;
            using Pointer = typename Traits<ValueT, Layout>::BasePointer;
            using PointerValueT = typename Traits<ValueT, Layout>::BasePointerValueT;
            using ReturnValueT = std::conditional_t<Layout == DataLayout::AoS, ValueT&, typename Traits<ValueT, Layout>::Proxy>;
            using ConstReturnValueT = std::conditional_t<Layout == DataLayout::AoS, ConstValueT&, typename Traits<ConstValueT, Layout>::Proxy>;

            static constexpr SizeT PaddingFactor = Traits<ValueT, Layout>::PaddingFactor;
            static constexpr SizeT NumElements = ((Size + (PaddingFactor - 1)) / PaddingFactor) * PaddingFactor;

        public:
            HOST_VERSION
            CUDA_DEVICE_VERSION
            SimdVec() : pointer(reinterpret_cast<PointerValueT*>(&data[0]), NumElements) {}
        
            template <typename Accessor, typename Filter>
            HOST_VERSION
            CUDA_DEVICE_VERSION
            auto& Load(const Accessor& external_data, const Filter& filter, const SizeT n)
            {
                #pragma omp simd
                for (SizeT i = 0; i < n; ++i)
                {
                    filter(external_data[i], (*this)[i]);
                }

                return *this;
            }

            template <typename Accessor, typename Filter>
            HOST_VERSION
            CUDA_DEVICE_VERSION
            auto& Load(const Accessor& external_data, const Filter& filter)
            {
                return Load(external_data, filter, Size);
            }

            template <typename Accessor>
            HOST_VERSION
            CUDA_DEVICE_VERSION
            auto& Load(const Accessor& external_data, SizeT n)
            {
                return Load(external_data, AssignAll, n);
            }

            template <typename Accessor, typename Filter>
            HOST_VERSION
            CUDA_DEVICE_VERSION
            const auto& Store(Accessor&& external_data, const Filter& filter, const SizeT n) const
            {
                #pragma omp simd
                for (SizeT i = 0; i < n; ++i)
                {
                    filter((*this)[i], external_data[i]);
                }

                return *this;
            }

            template <typename Accessor, typename Filter>
            HOST_VERSION
            CUDA_DEVICE_VERSION
            const auto& Store(Accessor&& external_data, const Filter& filter) const
            {
                return Store(std::forward<Accessor>(external_data), filter, Size);
            }

            template <typename Accessor>
            HOST_VERSION
            CUDA_DEVICE_VERSION
            const auto& Store(Accessor&& external_data, SizeT n) const
            {
                return Store(std::forward<Accessor>(external_data), AssignAll, n);
            }

            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline auto operator[](const SizeT index) -> ReturnValueT
            {
                return {pointer.At(0, index)};
            }
            
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline auto operator[](const SizeT index) const -> ConstReturnValueT
            {
                return {pointer.At(0, index)};
            }
            
        protected:
            ValueT data[NumElements];
            Pointer pointer;
        };
    }
}

#endif