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

        template <typename ValueT, SizeT D, DataLayout Layout>
        class SimdVecRef
        {
            using ConstValueT = const ValueT;
            using ReturnValueT = std::conditional_t<Layout == DataLayout::AoS, ValueT&, typename Traits<ValueT, Layout>::Proxy>;
            using ConstReturnValueT = std::conditional_t<Layout == DataLayout::AoS, ConstValueT&, typename Traits<ConstValueT, Layout>::Proxy>;

        public:
            SimdVecRef(internal::Accessor<ValueT, 0, D, Layout>&& accessor, const SizeT size) : accessor(std::move(accessor)), size(size) {}
            
            template <typename Accessor, typename Filter>
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
            auto& Load(const Accessor& external_data, const SizeT n)
            {
                return Load(external_data, AssignAll, n);
            }

            template <typename Accessor>
            auto& Store(Accessor&& external_data, const SizeT n) const
            {
                return Store(external_data, AssignAll, n);
            }

            inline auto operator[](const SizeT index) -> ReturnValueT
            {
                return accessor[index];
            }
            
            inline auto operator[](const SizeT index) const -> ConstReturnValueT
            {
                return accessor[index];
            }
            
            inline auto begin() -> internal::Accessor<ValueT, 0, D, Layout>
            {
                return accessor;
            }

            inline auto end() -> internal::Accessor<ValueT, 0, D, Layout>
            {
                return accessor;
            }

            inline auto begin() const -> internal::Accessor<ConstValueT, 0, D, Layout>
            {
                return accessor.At(size);
            }

            inline auto end() const -> internal::Accessor<ConstValueT, 0, D, Layout>
            {
                return accessor.At(size);
            }
            
        protected:
            internal::Accessor<ValueT, 0, D, Layout> accessor;
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
            SimdVec() : pointer(reinterpret_cast<PointerValueT*>(data.data()), NumElements), ref(internal::Accessor<ValueT, 0, 1, Layout>(pointer, Size), Size) {}
        
            template <typename Accessor, typename Filter>
            auto& Load(const Accessor& external_data, Filter filter, const SizeT n)
            {
                ref.Load(external_data, filter, n);

                return *this;
            }

            template <typename Accessor, typename Filter>
            auto& Load(const Accessor& external_data, Filter filter)
            {
                return Load(external_data, filter, Size);
            }

            template <typename Accessor>
            auto& Load(const Accessor& external_data, SizeT n)
            {
                return Load(external_data, AssignAll, n);
            }

            template <typename Accessor, typename Filter>
            const auto& Store(Accessor&& external_data, Filter filter, const SizeT n) const
            {
                ref.Store(std::forward<Accessor>(external_data), filter, n);

                return *this;
            }

            template <typename Accessor, typename Filter>
            const auto& Store(Accessor&& external_data, Filter filter) const
            {
                return Store(std::forward<Accessor>(external_data), filter, Size);
            }

            template <typename Accessor>
            const auto& Store(Accessor&& external_data, SizeT n) const
            {
                return Store(std::forward<Accessor>(external_data), AssignAll, n);
            }

            inline auto operator[](const SizeT index) -> ReturnValueT
            {
                return ref[index];
            }
            
            inline auto operator[](const SizeT index) const -> ConstReturnValueT
            {
                return ref[index];
            }
            
            inline auto begin() -> internal::Accessor<ValueT, 0, 1, Layout>
            {
                return ref.begin();
            }

            inline auto end() -> internal::Accessor<ValueT, 0, 1, Layout>
            {
                return ref.end();
            }

            inline auto begin() const -> internal::Accessor<ConstValueT, 0, 1, Layout>
            {
                return ref.begin();
            }

            inline auto end() const -> internal::Accessor<ConstValueT, 0, 1, Layout>
            {
                return ref.end();
            }
            
        protected:
            std::array<ValueT, NumElements> data;
            Pointer pointer;
            SimdVecRef<ValueT, 1, Layout> ref;
        };
    }
}

#endif
