// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(DATA_TYPES_ARRAY_ARRAY_HPP)
#define DATA_TYPES_ARRAY_ARRAY_HPP

#include <array>
#include <cassert>
#include <iostream>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#include <auxiliary/CPPStandard.hpp>
#include <auxiliary/Pack.hpp>
#include <platform/Target.hpp>
#include <DataTypes.hpp>

namespace XXX_NAMESPACE
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Forward declarations.
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    namespace variadic
    {
        template <typename...>
        struct Pack;
    }

    namespace dataTypes
    {
        using ::XXX_NAMESPACE::variadic::Pack;

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //!
        //! \brief A fixed sized array.
        //!
        //! \tparam ValueT data type
        //! \tparam N number of array elements
        //!
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        template <typename ValueT, SizeT N>
        class Array
        {
            template <typename, SizeT>
            friend class Array;

          public:
            // Template parameters.
            using ValueType = ValueT;
            static constexpr SizeT Size = N;

            //!
            //! \brief Standard constructor.
            //!
            HOST_VERSION
            CUDA_DEVICE_VERSION
            constexpr Array() = default;

            //!
            //! \brief Constructor.
            //!
            //! Assignt the same value to each element.
            //!
            //! \param value some value
            //!
            HOST_VERSION
            CUDA_DEVICE_VERSION
            constexpr Array(const ValueT& value)
            {
                for (SizeT i = 0; i < N; ++i)
                {
                    data[i] = value;
                }
            }

            //!
            //! \brief Constructor taking N (or less) arguments for the array initialization.
            //!
            //! If the argument count is less than N, all elements above N are zeroed.
            //!
            //! \tparam T parameter pack
            //! \param args variadic argument list
            //!
            template <typename... T>
            HOST_VERSION CUDA_DEVICE_VERSION constexpr Array(const T... args) : data{static_cast<ValueT>(std::move(args))...}
            {
                static_assert(sizeof...(T) <= Size, "error: parameter list does not match the size of this array.");
                static_assert(Pack<T...>::template IsConvertibleTo<ValueT>(), "error: types are not convertible.");
            }

            //!
            //! \brief Copy constructor.
            //!
            //! \tparam OtherN the size of another Array
            //! \param other another Array
            //!
            template <SizeT OtherN>
            HOST_VERSION CUDA_DEVICE_VERSION constexpr Array(const Array<ValueT, OtherN>& other)
            {
                for (SizeT i = 0; i < std::min(OtherN, N); ++i)
                {
                    data[i] = other.data[i];
                }

                for (SizeT i = std::min(OtherN, N); i < N; ++i)
                {
                    data[i] = static_cast<ValueT>(0);
                }
            }

            //!
            //! \brief Constructor.
            //!
            //! Construct from and std::array type.
            //!
            //! \param std_array an std::array argument
            //!
            HOST_VERSION
            CUDA_DEVICE_VERSION
            constexpr Array(const std::array<ValueT, N>& std_array)
            {
                for (SizeT i = 0; i < N; ++i)
                {
                    data[i] = std_array[i];
                }
            }

            //!
            //! \brief Exchange the content with another Array.
            //!
            //! \param other another Array
            //! \return a reference to this object with the new content
            //!
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline auto Swap(Array& other) -> Array&
            {
                for (SizeT i = 0; i < N; ++i)
                {
                    const ValueT tmp = data[i];
                    data[i] = other.data[i];
                    other.data[i] = tmp;
                }

                return *this;
            }

            //!
            //! \brief Array subscript operator.
            //!
            //! \param index element index
            //! \return a (const) reference to the element
            //!
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline constexpr auto operator[](const SizeT index) -> ValueT&
            {
                assert(index < N);

                return data[index];
            }

            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline constexpr auto operator[](const SizeT index) const -> const ValueT&
            {
                assert(index < N);

                return data[index];
            }

            //!
            //! \brief Array access.
            //!
            //! \tparam Index element index
            //! \return a (const) reference to the element
            //!
            template <SizeT Index>
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline constexpr auto At() -> ValueT&
            {
                static_assert(Index < N, "error: out of bounds data access");

                return data[Index];
            }

            template <SizeT Index>
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline constexpr auto At() const -> const ValueT&
            {
                static_assert(Index < N, "error: out of bounds data access");

                return data[Index];
            }

            //!
            //! \brief Replace the element at position 'index' by a new one.
            //!
            //! The replacement happens in place.
            //!
            //! \param value replacement
            //! \param index element index
            //! \return a reference to this Array
            //!
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline constexpr auto Replace(const ValueT& value, const SizeT index) -> Array&
            {
                assert(index < N);

                data[index] = value;

                return *this;
            }

            //!
            //! \brief Replace the element at position 'index' by a new one.
            //!
            //! \param value replacement
            //! \param index element index
            //! \return a new Array object
            //!
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline constexpr auto Replace(const ValueT& value, const SizeT index) const
            {
                Array<ValueT, N> result{*this};

                return result.Replace(value, index);
            }

            //!
            //! \brief Insert a new element at position 'index'.
            //!
            //! The last element of the array will be removed (chopped)!
            //!
            //! \param value element to be inserted
            //! \param index element index
            //! \return a reference to this Array
            //!
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline constexpr auto InsertAndChop(const ValueT& value, const SizeT index = 0) -> Array&
            {
                static_assert(N > 0, "error: this is an empty array");
                assert(index <= N);

                // Keep everything behind position 'index'.
                for (SizeT i = (N - 1); i > index; --i)
                {
                    data[i] = data[i - 1];
                }

                // Insert element at position 'index'
                data[index] = value;

                return *this;
            }

            //!
            //! \brief Insert a new element at position 'index'.
            //!
            //! The last element of the array will be removed (chopped)!
            //!
            //! \param value element to be inserted
            //! \param index element index
            //! \return a new Array object
            //!
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline constexpr auto InsertAndChop(const ValueT& value, const SizeT index = 0) const
            {
                Array<ValueT, N> result{*this};

                return result.InsertAndChop(value, index);
            }

            //!
            //! \brief Insert element at position 'index'
            //!
            //! \param value element to be inserted
            //! \param index element index
            //! \return a new Array object
            //!
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline constexpr auto Insert(const ValueT& value, const SizeT index = 0) const
            {
                assert(index <= N);

                Array<ValueT, N + 1> result{};

                for (SizeT i = 0; i < std::min(index, N); ++i)
                {
                    result.data[i] = data[i];
                }

                if (index <= N)
                {
                    // Insert element at position 'index'.
                    result.data[index] = value;

                    // Keep everything behind position 'index'.
                    for (SizeT i = index; i < N; ++i)
                    {
                        result.data[i + 1] = data[i];
                    }
                }

                return result;
            }

            //!
            //! \brief Test for two Arrays having the same content.
            //!
            //! \param other another Array
            //! \return `true` if the content is the same, otherwise `false`
            //!
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline constexpr auto operator==(const Array& other) const
            {
                bool is_equal = true;

                for (SizeT i = 0; i < N; ++i)
                {
                    is_equal &= (data[i] == other.data[i]);
                }

                return is_equal;
            }

            //!
            //! \brief Test for two Arrays having not the same content.
            //!
            //! \param other another Array
            //! \return `false` if the content is the same, otherwise `true`
            //!
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline constexpr auto operator!=(const Array& other) const { return !(*this == other); }

            //!
            //! \brief Reduction across all elements of the Array.
            //!
            //! \tparam FuncT the type of the callable
            //! \param func a callable implementing the reduction
            //! \param initial_value the initial value of the aggregate
            //! \param begin the lower bound index for the reduction
            //! \param end the upper bound index for the reduciton
            //! \return the value of the aggregate
            //!
            template <typename FuncT>
            HOST_VERSION CUDA_DEVICE_VERSION inline constexpr auto Reduce(const FuncT func, const ValueT& initial_value, const SizeT begin = 0, const SizeT end = N) const
            {
                assert(end <= N);

                ValueT aggregate = initial_value;

                for (SizeT i = begin; i < end; ++i)
                {
                    aggregate = func(aggregate, data[i]);
                }

                return aggregate;
            }

            //!
            //! \brief //! \brief Multiply reduction across all elements of the Array.
            //!
            //! \param begin the lower bound index for the reduction
            //! \param end the upper bound index for the reduciton
            //! \return the value of the aggregate
            //!
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline constexpr ValueT ReduceMul(const SizeT begin = 0, const SizeT end = N) const
            {
                assert(end <= N);
             
#if (__cplusplus > 201402L)
                return Reduce([](const ValueT product, const ValueT element) { return (product * element); }, 1, begin, end);
#else
                ValueT aggregate = static_cast<ValueT>(1);

                for (SizeT i = begin; i < end; ++i)
                {
                    aggregate *= data[i];
                }

                return aggregate;
#endif
            }

          private:
            ValueT data[N];
        };

        template <SizeT N>
        using SizeArray = Array<SizeT, N>;

        template <typename T, typename ValueT, SizeT N>
        bool operator==(const T& value, const Array<ValueT, N>& array)
        {
            return array == Array<ValueT, N>(value);
        }

        template <typename T, typename ValueT, SizeT N>
        bool operator==(const Array<ValueT, N>& array, const T& value)
        {
            return array == Array<ValueT, N>(value);
        }

        template <typename ValueT, SizeT N>
        std::ostream& operator<<(std::ostream& os, const Array<ValueT, N>& array)
        {
            for (SizeT i = 0; i < N; ++i)
            {
                os << array[i] << ((i + 1) < N ? " " : "");
            }

            return os;
        }
    } // namespace dataTypes
} // namespace XXX_NAMESPACE

#endif