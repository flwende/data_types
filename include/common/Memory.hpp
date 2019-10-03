// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(COMMON_MEMORY_HPP)
#define COMMON_MEMORY_HPP

#include <cassert>
#include <tuple>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#include <auxiliary/CPPStandard.hpp>
#include <auxiliary/Template.hpp>
#include <common/DataLayout.hpp>
#include <common/Math.hpp>
#include <data_types/DataTypes.hpp>
#include <platform/Target.hpp>
#include <platform/simd/Simd.hpp>

namespace XXX_NAMESPACE
{
    namespace dataTypes
    {
        namespace internal
        {
            // Forward declarations.
            template <typename, SizeT, ::XXX_NAMESPACE::memory::DataLayout, ::XXX_NAMESPACE::target>
            class Container;
        }
    }

    namespace memory
    {
        namespace
        {
            //!
            //! \brief Allocator base class.
            //!
            //! Provides AllocationShape data structure and Allocate and Deallocate member functions.
            //!
            //! \tparam T the type of the memory allocation
            //!
            template <typename T>
            class AllocatorBase
            {
              protected:
                static constexpr SizeT DefaultAlignment = SIMD_NAMESPACE::simd::alignment;

                //!
                //! \brief Allocation shape type.
                //!
                template <SizeT UnitSize>
                struct AllocationShape
                {
                    //!
                    //! \brief Standard constructor.
                    //!
                    //! Create an invalid shape.
                    //!
                    AllocationShape() : n_0{}, num_stabs{}, alignment{} {}

                    //!
                    //! \brief Constructor.
                    //!
                    //! \param n_0 the innermost extent of the allocation
                    //! \param num_stabs the number of stabs
                    //! \param alignment the alignment (bytes) used for padding of `n_0`
                    //!
                    AllocationShape(const SizeT n_0, const SizeT num_stabs, const SizeT alignment) : n_0(n_0), num_stabs(num_stabs), alignment(alignment) {}

                    //!
                    //! \brief Get the number of bytes of this allocation shape.
                    //!
                    //! \param allocation_shape the allocation shape
                    //! \return the number of bytes of the allocation shape
                    //!
                    auto GetByteSize() const -> SizeT { return n_0 * num_stabs * UnitSize; }

                    SizeT n_0;
                    SizeT num_stabs;
                    SizeT alignment;
                };

              public:
                //!
                //! \brief Memory allocation (Host).
                //!
                //! NOTE: aligned_alloc results in a segfault here -> use _mm_malloc.
                //!
                //! \tparam Target the target platform memory should be allocated for/on
                //! \tparam AllocationShapeT the type of the allocation shape
                //! \param allocation_shape the allocation shape used for the memory allocation
                //! \return a pointer to memory according to the allocation shape
                //!
                template <::XXX_NAMESPACE::target Target, typename AllocationShapeT>
                static auto Allocate(const AllocationShapeT& allocation_shape) -> typename std::enable_if<Target == ::XXX_NAMESPACE::target::Host, T*>::type
                {
                    return reinterpret_cast<T*>(_mm_malloc(allocation_shape.GetByteSize(), allocation_shape.alignment));
                }

                //!
                //! \brief Memory deallocation (Host).
                //!
                //! \tparam Target the target platform memory should be allocated for/on
                //! \tparam PointerT the type of the pointer
                //! \param pointer a pointer variable
                //!
                template <::XXX_NAMESPACE::target Target, typename PointerT>
                static auto Deallocate(PointerT& pointer) -> typename std::enable_if<Target == ::XXX_NAMESPACE::target::Host, void>::type
                {
                    if (pointer.GetBasePointer())
                    {
                        _mm_free(pointer.GetBasePointer());
                    }
                }

#if defined(__CUDACC__)
                //!
                //! \brief Memory allocation (GPU).
                //!
                //! \tparam Target the target platform memory should be allocated for/on
                //! \tparam AllocationShapeT the type of the allocation shape
                //! \param allocation_shape the allocation shape used for the memory allocation
                //! \return a pointer to memory according to the allocation shape
                //!
                template <::XXX_NAMESPACE::target Target, typename AllocationShapeT>
                static auto Allocate(const AllocationShapeT& allocation_shape) -> typename std::enable_if<Target == ::XXX_NAMESPACE::target::GPU_CUDA, T*>::type
                {
                    T* d_ptr = nullptr;

                    cudaMalloc((void**)&d_ptr, allocation_shape.GetByteSize());

                    return d_ptr;
                }

                //!
                //! \brief Memory deallocation (GPU).
                //!
                //! \tparam Target the target platform memory should be allocated for/on
                //! \tparam PointerT the type of the pointer
                //! \param pointer a pointer variable
                //!
                template <::XXX_NAMESPACE::target Target, typename PointerT>
                static auto Deallocate(PointerT& pointer) -> typename std::enable_if<Target == ::XXX_NAMESPACE::target::GPU_CUDA, void>::type
                {
                    if (pointer.GetBasePointer())
                    {
                        cudaFree(pointer.GetBasePointer());
                    }
                }
#endif
            };
        } // namespace

        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        //!
        //! \brief A pointer wrapper for homogeneous structured types (HST) including fundamental types.
        //!
        //! It provides functionality for memory (de)allocation and accessing it in the multi-
        //! dimensional case with different data layouts: AoS, SoA, SoAi.
        //!
        //! Idea: Multidimensional fields are contiguous sequences of stabs (innermost dimension n_0).
        //! Stabs are separated by 'NumParameters x n_0' elements of type T, with NumParameters being the size of the parameter pack (for HSTs).
        //! All elements of the field can be Ated through jumping to the stab using a stab index and the
        //! base pointer to the 1st member of the 0th element of the field, and within the stab by adding
        //! a multiple of n_0 according to the member that should be Ated.
        //! The resulting pointer then is shifted by an intra-stab index to access the actual data member.
        //!
        //! THE POINTER MANAGED BY THIS CLASS IS EXTERNAL!
        //!
        //! \tparam T parameter pack (one parameter for each data member; all of the same size)
        //!
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        template <typename... T>
        class Pointer
        {
            // Number of parameters (members of the HST).
            static constexpr SizeT NumParameters = ::XXX_NAMESPACE::variadic::Pack<T...>::Size;
            static_assert(NumParameters > 0, "error: empty parameter pack");

            // All members have the same type: get this type.
            using ValueT = typename ::XXX_NAMESPACE::variadic::Pack<T...>::template Type<0>;

            // Check if all types are the same (same size is sufficient).
            static constexpr bool IsHomogeneous = ::XXX_NAMESPACE::variadic::Pack<T...>::SameSize();
            static_assert(IsHomogeneous, "error: use the inhomogeneous MultiPointer instead");

            // Friend declarations.
            template <typename...>
            friend class Pointer;
            template <typename, SizeT, DataLayout, ::XXX_NAMESPACE::target>
            friend class ::XXX_NAMESPACE::dataTypes::internal::Container;
            friend class AllocatorBase<ValueT>;

            //!
            //! \brief Create a tuple of member references from the base pointer.
            //!
            //! This function sets up a tuple of references pointing to the members of an HST for
            //! field index 'stab_index * n_0 + index'.
            //!
            //! \tparam I parameter pack used for indexed array access
            //! \param stab_index the stab index
            //! \param index the intra-stab index
            //! \param unnamed used for template paramter deduction
            //! \return a tuple of references (one reference for each member of the HST)
            //!
            template <SizeT... I>
            HOST_VERSION CUDA_DEVICE_VERSION inline auto GetValues(const SizeT stab_index, const SizeT index, std::integer_sequence<SizeT, I...>) -> std::tuple<T&...>
            {
                return {reinterpret_cast<T&>(ptr[(stab_index * NumParameters + I) * n_0 + index])...};
            }

            //!
            //! \brief Create a tuple of member references from the base pointer.
            //!
            //! This function sets up a tuple of references pointing to the members of an HST for
            //! field index 'stab_index * n_0 + index'.
            //!
            //! \tparam I parameter pack used for indexed array access
            //! \param stab_index the stab index
            //! \param index the intra-stab index
            //! \param unnamed used for template paramter deduction
            //! \return a tuple of references (one reference for each member of the HST)
            //!
            template <SizeT... I>
            HOST_VERSION CUDA_DEVICE_VERSION inline auto GetValues(const SizeT stab_index, const SizeT index, std::integer_sequence<SizeT, I...>) const -> std::tuple<const T&...>
            {
                return {reinterpret_cast<const T&>(ptr[(stab_index * NumParameters + I) * n_0 + index])...};
            }

            //!
            //! \brief Get the base pointer.
            //!
            //! \return the base pointer
            //!
            inline auto GetBasePointer() -> ValueT* { return ptr; }

            //!
            //! \brief Get the base pointer.
            //!
            //! \return the base pointer
            //!
            inline auto GetBasePointer() const -> const ValueT* { return ptr; }

          public:
            //!
            //! \brief Standard constructor.
            //!
            //! Create an invalid Pointer.
            //!
            Pointer() : n_0(0), ptr(nullptr) {}

            //!
            //! \brief Constructor.
            //!
            //! Set up a Pointer from an external pointer.
            //!
            //! \param ptr an external pointer that is used as the base pointer internally
            //! \param n_0 the innermost dimension of the field
            //!
            Pointer(ValueT* ptr, const SizeT n_0) : n_0(n_0), ptr(ptr) { assert(ptr != nullptr); }

            //!
            //! \brief Copy/conversion constructor.
            //!
            //! Create a copy of another Pointer if their types are convertible.
            //!
            //! \tparam OtherT parameter pack (member types) of the other Pointer
            //! \param other another Pointer
            //!
            template <typename... OtherT>
            Pointer(const Pointer<OtherT...>& other) : n_0(other.n_0), ptr(reinterpret_cast<ValueT*>(other.ptr))
            {
                static_assert(::XXX_NAMESPACE::variadic::Pack<ValueT, OtherT...>::IsConvertible(), "error: types are not convertible");
            }

            //!
            //! \brief Exchange this Pointer's members with another Pointer.
            //!
            //! \param other another Pointer
            //! \return this Pointer
            //!
            inline auto Swap(Pointer& other) -> Pointer&
            {
                SizeT this_n_0 = n_0;
                n_0 = other.n_0;
                other.n_0 = this_n_0;

                ValueT* this_ptr = ptr;
                ptr = other.ptr;
                other.ptr = this_ptr;

                return *this;
            }

            //!
            //! \brief Access the field through the base pointer.
            //!
            //! This implementation uses a single intra-stab index within stab 0.
            //!
            //! \param index the intra-stab index
            //! \return a tuple of references (one reference for each member of the HST)
            //!
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline auto At(const SizeT index) { return GetValues(0, index, std::make_integer_sequence<SizeT, NumParameters>{}); }

            //!
            //! \brief Access the field through the base pointer.
            //!
            //! This implementation uses a single intra-stab index within stab 0.
            //!
            //! \param index the intra-stab index
            //! \return a tuple of references (one reference for each member of the HST)
            //!
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline auto At(const SizeT index) const { return GetValues(0, index, std::make_integer_sequence<SizeT, NumParameters>{}); }

            //!
            //! \brief Access the field through the base pointer.
            //!
            //! This implementation uses a stab index and an intra-stab index.
            //!
            //! \param stab_index the stab index
            //! \param index the intra-stab index
            //! \return a tuple of references (one reference for each member of the HST)
            //!
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline auto At(const SizeT stab_index, const SizeT index) { return GetValues(stab_index, index, std::make_integer_sequence<SizeT, NumParameters>{}); }

            //!
            //! \brief Access the field through the base pointer.
            //!
            //! This implementation uses a stab index and an intra-stab index.
            //!
            //! \param stab_index the stab index
            //! \param index the intra-stab index
            //! \return a tuple of references (one reference for each member of the HST)
            //!
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline auto At(const SizeT stab_index, const SizeT index) const { return GetValues(stab_index, index, std::make_integer_sequence<SizeT, NumParameters>{}); }

            //!
            //! \brief An allocator class.
            //!
            //! This class implements data allocation and deallocation functionality Padding and data alignment
            //! for the different target platforms.
            //!
            class Allocator : public AllocatorBase<ValueT>
            {
                using Base = AllocatorBase<ValueT>;

                static constexpr SizeT DefaultAlignment = Base::DefaultAlignment;

                //!
                //! \brief Pad the value of n to a given alignment.
                //!
                //! \param n the value the padding should be applied to
                //! \param alignment the alignment
                //! \return `n` padded to `alignment`
                //!
                static auto Padding(const SizeT n, const SizeT alignment = DefaultAlignment)
                {
                    assert(::XXX_NAMESPACE::math::IsPowerOf<2>(alignment));

                    const SizeT n_unit = ::XXX_NAMESPACE::math::LeastCommonMultiple(alignment, static_cast<SizeT>(sizeof(ValueT))) / static_cast<SizeT>(sizeof(ValueT));

                    return ((n + n_unit - 1) / n_unit) * n_unit;
                }

              public:
                using AllocationShape = typename Base::template AllocationShape<sizeof(ValueT)>;

                //!
                //! \brief Get the allocation shape for given SizeArray (not SoA data layout).
                //!
                //! Allocation shape:
                //!     1st component: innermost extent of the SizeArray padded according to the alignment.
                //!     2nd component: product of all other extents and the number of members of the HST (the number of stabs).
                //!
                //! \tparam Layout the data layout
                //! \tparam N the dimension of the SizeArray
                //! \param n a SizeArray
                //! \param alignment the alignment (bytes) to be used for the padding of the innermost extent of `n`
                //! \return an allocation shape
                //!
                template <DataLayout Layout, SizeT N>
                static auto GetAllocationShape(const ::XXX_NAMESPACE::dataTypes::SizeArray<N>& n, const SizeT alignment = DefaultAlignment)
                    -> std::enable_if_t<Layout != DataLayout::SoA, AllocationShape>
                {
                    return {Padding(n[0], alignment), NumParameters * n.ReduceMul(1), alignment};
                }

                //!
                //! \brief Get the allocation shape for given SizeArray (SoA data layout).
                //!
                //! Allocation shape:
                //!     1st component: total number of elements in SizeArray padded according to the alignment.
                //!     2nd component: number of members of the HST.
                //!
                //! \tparam Layout the data layout
                //! \tparam N the dimension of the SizeArray
                //! \param n a SizeArray
                //! \param alignment the alignment to be used for the padding of the innermost extent of `n`
                //! \return an allocation shape
                //!
                template <DataLayout Layout, SizeT N>
                static auto GetAllocationShape(const ::XXX_NAMESPACE::dataTypes::SizeArray<N>& n, const SizeT alignment = DefaultAlignment)
                    -> std::enable_if_t<Layout == DataLayout::SoA, AllocationShape>
                {
                    return {Padding(n.ReduceMul(), alignment), NumParameters, alignment};
                }
            };

          private:
            // Extent of the innermost dimension (w.r.t. a multidimensional field declaration).
            SizeT n_0;
            // Base pointer.
            ValueT* ptr;
        };

        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        //
        // NumParameters-dimensional homogeneous structured type (HST).
        //
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        namespace
        {
            // Defines 'template <T, NumParameters> struct TypeGen {..};'.
            MACRO_TYPE_GEN(::XXX_NAMESPACE::memory::Pointer);
        } // namespace

        //!
        //! \brief A homogeneous structured type witn `N` members.
        //!
        //! \tparam T the type of the members
        //! \tparam N the number of members
        //!
        template <typename T, SizeT N>
        using PointerN = typename TypeGen<T, N>::Type;

        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        //!
        //! \brief A pointer wrapper for inhomogeneous structured types (IST) and SoA[i] data layout
        //!
        //! It provides functionality for memory (de)allocation and accessing it in the multi-
        //! dimensional case with different data layouts: SoA, SoAi.
        //!
        //! Idea: Similar to the Pointer type, but a bit more complicated to implement as
        //! multiple base pointers need to be managed internally, one for each data member of the inhomogeneous
        //! structured type.
        //!
        //! THE POINTER MANAGED BY THIS CLASS IS EXTERNAL!
        //!
        //! \tparam T parameter pack (one parameter for each data member)
        //!
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        template <typename... T>
        class MultiPointer
        {
            // Number of parameters (members of the HST).
            static constexpr SizeT NumParameters = ::XXX_NAMESPACE::variadic::Pack<T...>::Size;
            static_assert(NumParameters > 0, "error: empty parameter pack");

            // All members have different type: use std::uint8_t as the base pointer type.
            using ValueT = std::uint8_t;

            // All members have the same type: we don't want that here.
            static constexpr bool IsHomogeneous = ::XXX_NAMESPACE::variadic::Pack<T...>::SameSize();
            static_assert(!IsHomogeneous, "error: use the Pointer instead");

            // Find out the byte-size of the largest parameter type.
            static constexpr SizeT SizeOfLargestParameter = ::XXX_NAMESPACE::variadic::Pack<T...>::SizeOfLargestParameter();

            // Size of the inhomogeneous structured type.
            static constexpr SizeT RecordSize = ::XXX_NAMESPACE::variadic::Pack<T...>::SizeOfPack();

            // The member type sizes in relative to the size of the largest paramter type.
            static constexpr ::XXX_NAMESPACE::dataTypes::SizeArray<NumParameters> SizeScalingFactor{(SizeOfLargestParameter / sizeof(T))...};

            // Friend declarations.
            template <typename... X>
            friend class MultiPointer;
            template <typename, SizeT, DataLayout, ::XXX_NAMESPACE::target>
            friend class ::XXX_NAMESPACE::dataTypes::internal::Container;
            friend class AllocatorBase<ValueT>;

            //!
            //! \brief Create a tuple of (base) pointers from a pointer.
            //!
            //! (Base) pointers are separated from the pointer by the value of `n_0` and the byte-size of type of the memory they are pointing to.
            //! The latter is stored in `Offset[]` as exclusive prefix sums over the member types of the IST.
            //!
            //! \tparam I parameter pack used for indexed array access
            //! \param ptr the base pointer
            //! \param n_0 distance between successive pointers
            //! \param unnamed used for template paramter deduction
            //! \return a tuple of (base) pointers (one pointer for each member of the IST)
            //!
            template <SizeT... I>
            inline auto make_pointer_tuple(ValueT* ptr, const SizeT n_0, std::integer_sequence<SizeT, I...>) -> std::tuple<T*...>
            {
                // (Exclusive) prefix sums over the byte-sizes of the member types of the IST.
                constexpr ::XXX_NAMESPACE::dataTypes::SizeArray<NumParameters> Offset = ::XXX_NAMESPACE::math::PrefixSum(::XXX_NAMESPACE::dataTypes::SizeArray<NumParameters>{sizeof(T)...});

                return {reinterpret_cast<T*>(&ptr[Offset[I] * n_0])...};
            }

            //!
            //! \brief Create a tuple of member references from the (base) pointers.
            //!
            //! This function sets up a tuple of references pointing to the members of an IST for field index 'stab_index * n + index'.
            //! The value of `n` is calculated from the stab size of all members `n_0 * RecordSize` in terms of the member-type sizes.
            //! The actual calculation is: n = n_0 * RecordSize / sizeof(T)...
            //!                              = ((n_0 * RecordSize) / SizeOfLargestParameter) * (SizeOfLargestParameter / sizeof(T))...
            //!                              = n_0x * SizeScalingFactor[I]...
            //!
            //!
            //! \tparam I parameter pack used for indexed array access
            //! \param stab_index the stab index
            //! \param index the intra-stab index
            //! \param unnamed used for template paramter deduction
            //! \return a tuple of references (one reference for each member of the HST)
            //!
            template <SizeT... I>
            HOST_VERSION CUDA_DEVICE_VERSION inline auto GetValues(const SizeT stab_index, const SizeT index, std::integer_sequence<SizeT, I...>) -> std::tuple<T&...>
            {
                return {std::get<I>(ptr)[stab_index * (n_0x * SizeScalingFactor[I]) + index]...};
            }

            //!
            //! \brief Create a tuple of member references from the (base) pointers.
            //!
            //! This function sets up a tuple of references pointing to the members of an IST for field index 'stab_index * n + index'.
            //! The value of `n` is calculated from the stab size of all members `n_0 * RecordSize` in terms of the member-type sizes.
            //! The actual calculation is: n = n_0 * RecordSize / sizeof(T)...
            //!                              = ((n_0 * RecordSize) / SizeOfLargestParameter) * (SizeOfLargestParameter / sizeof(T))...
            //!                              = n_0x * SizeScalingFactor[I]...
            //!
            //!
            //! \tparam I parameter pack used for indexed array access
            //! \param stab_index the stab index
            //! \param index the intra-stab index
            //! \param unnamed used for template paramter deduction
            //! \return a tuple of references (one reference for each member of the HST)
            //!
            template <SizeT... I>
            HOST_VERSION CUDA_DEVICE_VERSION inline auto GetValues(const SizeT stab_index, const SizeT index, std::integer_sequence<SizeT, I...>) const -> std::tuple<const T&...>
            {
                return {std::get<I>(ptr)[stab_index * (n_0x * SizeScalingFactor[I]) + index]...};
            }

            //!
            //! \brief Get the base pointer.
            //!
            //! \return the base pointer
            //!
            inline auto GetBasePointer() -> ValueT* { return reinterpret_cast<ValueT*>(std::get<0>(ptr)); }

            //!
            //! \brief Get the base pointer.
            //!
            //! \return the base pointer
            //!
            inline auto GetBasePointer() const -> const ValueT* { return reinterpret_cast<const ValueT*>(std::get<0>(ptr)); }

          public:
            //!
            //! \brief Standard constructor.
            //!
            //! Create an invalid MultiPointer.
            //!
            MultiPointer() : n_0x(0), ptr{} {}

            //!
            //! \brief Constructor.
            //!
            //! Set up a MultiPointer from an external pointer.
            //!
            //! \param ptr an external pointer that is used as the base pointer internally (it is the 0th element of the pointer tuple)
            //! \param n_0 the innermost dimension of the field
            //!
            MultiPointer(ValueT* ptr, const SizeT n_0) : n_0x((n_0 * RecordSize) / SizeOfLargestParameter), ptr(make_pointer_tuple(ptr, n_0, std::make_integer_sequence<SizeT, NumParameters>{}))
            {
                assert(ptr != nullptr);
                assert(((n_0 * RecordSize) % SizeOfLargestParameter) == 0);
            }

            //!
            //! \brief Copy/conversion constructor.
            //!
            //! Create a copy of another MultiPointer if their types are convertible.
            //!
            //! \tparam OtherT parameter pack (member types) of the other MultiPointer
            //! \param other another MultiPointer
            //!
            template <typename... OtherT>
            MultiPointer(const MultiPointer<OtherT...>& other) : n_0x(other.n_0x), ptr(other.ptr)
            {
                static_assert(::XXX_NAMESPACE::variadic::Pack<ValueT, OtherT...>::IsConvertible(), "error: types are not convertible");
            }

            //!
            //! \brief Exchange this MultiPointer's members with another MultiPointer.
            //!
            //! \param other another MultiPointer
            //! \return this MultiPointer
            //!
            inline auto Swap(MultiPointer& other) -> MultiPointer&
            {
                SizeT this_num_units = n_0x;
                n_0x = other.n_0x;
                other.n_0x = this_num_units;

                std::tuple<T*...> this_ptr = ptr;
                ptr = other.ptr;
                other.ptr = this_ptr;

                return *this;
            }

            //!
            //! \brief Access the field through the base pointer.
            //!
            //! This implementation uses a single intra-stab index within stab 0.
            //!
            //! \param index the intra-stab index
            //! \return a tuple of references (one reference for each member of the HST)
            //!
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline auto At(const SizeT index) { return GetValues(0, index, std::make_integer_sequence<SizeT, NumParameters>{}); }

            //!
            //! \brief Access the field through the base pointer.
            //!
            //! This implementation uses a single intra-stab index within stab 0.
            //!
            //! \param index the intra-stab index
            //! \return a tuple of references (one reference for each member of the HST)
            //!
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline auto At(const SizeT index) const { return GetValues(0, index, std::make_integer_sequence<SizeT, NumParameters>{}); }

            //!
            //! \brief Access the field through the base pointer.
            //!
            //! This implementation uses a stab index and an intra-stab index.
            //!
            //! \param stab_index the stab index
            //! \param index the intra-stab index
            //! \return a tuple of references (one reference for each member of the HST)
            //!
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline auto At(const SizeT stab_index, const SizeT index) { return GetValues(stab_index, index, std::make_integer_sequence<SizeT, NumParameters>{}); }

            //!
            //! \brief Access the field through the base pointer.
            //!
            //! This implementation uses a stab index and an intra-stab index.
            //!
            //! \param stab_index the stab index
            //! \param index the intra-stab index
            //! \return a tuple of references (one reference for each member of the HST)
            //!
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline auto At(const SizeT stab_index, const SizeT index) const { return GetValues(stab_index, index, std::make_integer_sequence<SizeT, NumParameters>{}); }

            //!
            //! \brief An allocator class.
            //!
            //! This class implements data allocation and deallocation functionality Padding and data alignment
            //! for the different target platforms.
            //!
            class Allocator : public AllocatorBase<ValueT>
            {
                using Base = AllocatorBase<ValueT>;

                static constexpr SizeT DefaultAlignment = Base::DefaultAlignment;

                //!
                //! \brief Pad the value of n to a given alignment.
                //!
                //! \param n the value the padding should be applied to
                //! \param alignment the alignment
                //! \return `n` padded to `alignment`
                //!
                static auto Padding(const SizeT n, const SizeT alignment = DefaultAlignment)
                {
                    assert(::XXX_NAMESPACE::math::IsPowerOf<2>(alignment));

                    constexpr SizeT One = static_cast<SizeT>(1);
                    // Determine the total byte-size of all data members that have a size different from (smaller than) the largest parameter type.
                    constexpr SizeT SizeRest = ::XXX_NAMESPACE::variadic::Pack<T...>::SizeOfPackExcludingLargestParameter();
                    // Determine the number of ISTs that is needed so that their overall size is an integral multiple of each data member type.
                    constexpr SizeT RecordPaddingFactor = std::max(One, ::XXX_NAMESPACE::math::LeastCommonMultiple(SizeOfLargestParameter, SizeRest) / std::max(One, SizeRest));
                    const SizeT parameter_padding_factor = ::XXX_NAMESPACE::math::LeastCommonMultiple(alignment, SizeOfLargestParameter) / SizeOfLargestParameter;
                    const SizeT n_unit = ::XXX_NAMESPACE::math::LeastCommonMultiple(RecordPaddingFactor, parameter_padding_factor);

                    return ((n + n_unit - 1) / n_unit) * n_unit;
                }

              public:
                using AllocationShape = typename Base::template AllocationShape<RecordSize>;
                
                //!
                //! \brief Get the allocation shape for given SizeArray (not SoA data layout).
                //!
                //! Allocation shape:
                //!     1st component: innermost extent of the SizeArray padded according to the alignment.
                //!     2nd component: product of all other extents (the number of stabs).
                //!
                //! \tparam Layout the data layout
                //! \tparam N the dimension of the SizeArray
                //! \param n a SizeArray
                //! \param alignment the alignment (bytes) to be used for the padding of the innermost extent of `n`
                //! \return an allocation shape
                //!
                template <DataLayout Layout, SizeT N>
                static auto GetAllocationShape(const ::XXX_NAMESPACE::dataTypes::SizeArray<N>& n, const SizeT alignment = DefaultAlignment)
                    -> std::enable_if_t<Layout != DataLayout::SoA, AllocationShape>
                {
                    return {Padding(n[0], alignment), n.ReduceMul(1), alignment};
                }

                //!
                //! \brief Get the allocation shape for given SizeArray (SoA data layout).
                //!
                //! Allocation shape:
                //!     1st component: total number of elements in SizeArray padded according to the alignment.
                //!     2nd component: 1
                //!
                //! \tparam Layout the data layout
                //! \tparam N the dimension of the SizeArray
                //! \param n a SizeArray
                //! \param alignment the alignment to be used for the padding of the innermost extent of `n`
                //! \return an allocation shape
                //!
                template <DataLayout Layout, SizeT N>
                static auto GetAllocationShape(const ::XXX_NAMESPACE::dataTypes::SizeArray<N>& n, const SizeT alignment = DefaultAlignment)
                    -> std::enable_if_t<Layout == DataLayout::SoA, AllocationShape>
                {
                    return {Padding(n.ReduceMul(), alignment), 1, alignment};
                }
            };

          private:
            // Extent of the innermost dimension (w.r.t. a multidimensional field declaration).
            SizeT n_0x;
            // Base pointers (of different type).
            std::tuple<T*...> ptr;
        };
    } // namespace memory
} // namespace XXX_NAMESPACE

#endif