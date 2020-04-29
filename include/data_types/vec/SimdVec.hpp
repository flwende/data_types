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
        using ::XXX_NAMESPACE::internal::ProvidesProxy;
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
            static constexpr bool UseProxy = (Layout != DataLayout::AoS && ProvidesProxy<ValueT>::value);
            using ReturnValueT = std::conditional_t<UseProxy, typename Traits<ValueT, Layout>::Proxy, ValueT&>;
            using ConstReturnValueT = std::conditional_t<UseProxy, typename Traits<ConstValueT, Layout>::Proxy, ConstValueT&>;

          public:
            HOST_VERSION
            CUDA_DEVICE_VERSION
            SimdVecRef(internal::Accessor<ValueT, 0, 0, Layout>&& accessor, const SizeT size) : accessor(std::move(accessor)), size(size) {}
            
            template <typename Accessor>
            HOST_VERSION
            CUDA_DEVICE_VERSION
            auto& Load(const Accessor& external_data, const SizeT n)
            {
                assert(n <= size);

                for (SizeT i = 0; i < n; ++i)
                {
                    accessor[i] = external_data[i];
                }

                return *this;
            }

            template <typename Accessor>
            HOST_VERSION
            CUDA_DEVICE_VERSION
            auto& Load(const Accessor& external_data)
            {
                return Load(external_data, size);
            }

            template <typename Accessor>
            HOST_VERSION
            CUDA_DEVICE_VERSION
            const auto& Store(Accessor&& external_data, const SizeT n) const
            {
                assert(n <= size);

                for (SizeT i = 0; i < n; ++i)
                {
                    external_data[i] = accessor[i];
                } 

                return *this;
            }

            template <typename Accessor>
            HOST_VERSION
            CUDA_DEVICE_VERSION
            auto& Store(Accessor&& external_data) const
            {
                return Store(external_data, size);
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
            
          protected:
            internal::Accessor<ValueT, 0, 0, Layout> accessor;
            const SizeT size;
        };

        template <typename ValueT, SizeT Size, DataLayout Layout = DataLayout::AoS>
        class SimdVec
        {
            template <typename T>
            using Traits = Traits<T, Layout>;
            using ConstValueT = typename Traits<ValueT>::ConstT;
            using Pointer = typename Traits<ValueT>::BasePointer;
            using PointerValueT = typename Traits<ValueT>::BasePointerValueT;
            static constexpr bool UseProxy = ProvidesProxy<ValueT>::value;
            using Proxy = typename Traits<ValueT>::Proxy;
            using ConstProxy = typename Traits<ConstValueT>::Proxy;

            static constexpr SizeT PaddingFactor = Traits<ValueT>::PaddingFactor;
            static constexpr SizeT NumElements = ((Size + (PaddingFactor - 1)) / PaddingFactor) * PaddingFactor;

          public:
            HOST_VERSION
            CUDA_DEVICE_VERSION
            SimdVec() : data{}, pointer(reinterpret_cast<PointerValueT*>(&data[0]), NumElements) 
            {
                for (SizeT i = 0; i < Size; ++i)
                {
                    data[i] = PointerValueT{};
                }
            }
        
            HOST_VERSION
            CUDA_DEVICE_VERSION
            SimdVec(const SimdVec& other) : data{}, pointer(reinterpret_cast<PointerValueT*>(&data[0]), NumElements)
            {
                for (SizeT i = 0; i < Size; ++i)
                {
                    data[i] = other.data[i];
                }
            }

            HOST_VERSION
            CUDA_DEVICE_VERSION
            SimdVec(SimdVec&& other) : data{}, pointer(reinterpret_cast<PointerValueT*>(&data[0]), NumElements)
            {
                for (SizeT i = 0; i < Size; ++i)
                {
                    data[i] = other.data[i];
                }
            }

            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline auto& operator=(const SimdVec& other)
            {
                if (this != &other)
                {
                    for (SizeT i = 0; i < Size; ++i)
                    {
                        data[i] = other.data[i];
                    }
                }

                return *this;
            }

            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline auto& operator=(SimdVec&& other)
            {
                if (this != &other)
                {
                    for (SizeT i = 0; i < Size; ++i)
                    {
                        data[i] = other.data[i];
                    }
                }

                return *this;
            }

            template <typename Accessor>
            HOST_VERSION
            CUDA_DEVICE_VERSION
            auto& Load(const Accessor& external_data, const SizeT n = Size)
            {
                assert(n <= Size);

                for (SizeT i = 0; i < n; ++i)
                {
                    (*this)[i] = external_data[i];
                }

                return *this;
            }

            template <typename Accessor>
            HOST_VERSION
            CUDA_DEVICE_VERSION
            const auto& Store(Accessor&& external_data, const SizeT n = Size) const
            {
                assert(n <= Size);

                for (SizeT i = 0; i < n; ++i)
                {
                    external_data[i] = (*this)[i];
                }

                return *this;
            }

            template <bool Enable = UseProxy>
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline auto operator[](const SizeT index) -> std::enable_if_t<!Enable, ValueT&>
            {
                return Get<0>(pointer.At(0, index));
            }
            
            template <bool Enable = UseProxy>
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline auto operator[](const SizeT index) const -> std::enable_if_t<!Enable, ConstValueT&>
            {
                return Get<0>(pointer.At(0, index));
            }

            template <bool Enable = UseProxy>
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline auto operator[](const SizeT index) -> std::enable_if_t<Enable, Proxy>
            {
                return {pointer.At(0, index)};
            }
            
            template <bool Enable = UseProxy>
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline auto operator[](const SizeT index) const -> std::enable_if_t<Enable, ConstProxy>
            {
                return {pointer.At(0, index)};
            }

            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline auto ReduceAdd() const
            {
                ValueT aggregate{};

                for (SizeT i = 0; i < Size; ++i)
                {
                    aggregate += data[i];
                }
                
                return aggregate;
            }
            
          protected:
            ValueT data[NumElements];
            Pointer pointer;
        };

        // Data layout AoS: `pointer` member is not needed!
        template <typename ValueT, SizeT Size>
        class SimdVec<ValueT, Size, DataLayout::AoS>
        {
          public:
            HOST_VERSION
            CUDA_DEVICE_VERSION
            SimdVec() : data{} {}

            template <typename Accessor>
            HOST_VERSION
            CUDA_DEVICE_VERSION
            auto& Load(const Accessor& external_data, const SizeT n = Size)
            {
                assert(n <= Size);

                for (SizeT i = 0; i < n; ++i)
                {
                    data[i] = external_data[i];
                }

                return *this;
            }

            template <typename Accessor>
            HOST_VERSION
            CUDA_DEVICE_VERSION
            const auto& Store(Accessor&& external_data, const SizeT n = Size) const
            {
                assert(n <= Size);

                for (SizeT i = 0; i < n; ++i)
                {
                    external_data[i] = data[i];
                }

                return *this;
            }

            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline auto& operator[](const SizeT index)
            {
                return data[index];
            }
            
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline const auto& operator[](const SizeT index) const
            {
                return data[index];
            }

            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline auto ReduceAdd() const
            {
                ValueT aggregate{};

                for (SizeT i = 0; i < Size; ++i)
                {
                    aggregate += data[i];
                }
                
                return aggregate;
            }
            
          protected:
            ValueT data[Size];
        };

        namespace
        {
            template <typename T>
            struct VectorSize
            {
                static constexpr SizeT value = 1;
            };

            template <typename ValueT, SizeT Size, DataLayout Layout>
            struct VectorSize<SimdVec<ValueT, Size, Layout>>
            {
                static constexpr SizeT value = Size;
            };
        }

        template <typename T>
        inline constexpr SizeT GetVectorSize()
        {
            return VectorSize<T>::value;
        }
    }
}

#endif