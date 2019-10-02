// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(DATA_TYPES_ARRAY_ARRAY_HPP)
#define DATA_TYPES_ARRAY_ARRAY_HPP

#include <array>
#include <cassert>
//#include <cstdint>
#include <iostream>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#include <auxiliary/CPPStandard.hpp>
#include <auxiliary/Template.hpp>
#include <data_types/DataTypes.hpp>
#include <platform/Target.hpp>

namespace XXX_NAMESPACE
{
    namespace dataTypes
    {
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //!
        //! \brief A fixed sized array.
        //!
        //! \tparam T data type
        //! \tparam C_N array size
        //!
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        template <typename T, SizeType C_N>
        class Array
        {
            template <typename, SizeType>
            friend class Array;
        
            T data[C_N];

        public:

            // Template paramteres.
            using ValueType = T;
            static constexpr SizeType N = C_N;

            //!
            //! \brief Standard constructor.
            //!
            HOST_VERSION
            CUDA_DEVICE_VERSION
            constexpr Array() 
                : 
                data{} 
            {}

            //!
            //! \brief Constructor.
            //!
            //! Assignt the same value to each element.
            //!
            //! \param x some value
            //!
            HOST_VERSION
            CUDA_DEVICE_VERSION
            constexpr Array(const T& x) 
            {
                for (SizeType i = 0; i < C_N; ++i)
                {
                    data[i] = x;
                }
            }

            //!
            //! \brief Constructor taking C_N (or less) arguments for the array initialization.
            //!
            //! If the argument count is less than C_N, all elements above C_N are zeroed.
            //!
            //! \tparam Args parameter pack
            //! \param args variadic argument list
            //!
            template <typename ...Args>
            HOST_VERSION
            CUDA_DEVICE_VERSION
            constexpr Array(const Args... args) 
                : 
                data{static_cast<T>(std::move(args))...}
            {}

            //!
            //! \brief Copy constructor.
            //!
            //! \tparam C_X the size of another Array
            //! \param array another Array
            //!
            template <SizeType C_X>
            HOST_VERSION
            CUDA_DEVICE_VERSION
            constexpr Array(const Array<T, C_X>& array)
            {
                for (SizeType i = 0; i < std::min(C_X, C_N); ++i)
                {
                    data[i] = array.data[i];
                }

                for (SizeType i = std::min(C_X, C_N); i < C_N; ++i)
                {
                    data[i] = static_cast<T>(0);
                }
            }

            //!
            //! \brief Constructor.
            //!
            //! Construct from and std::array type.
            //!
            //! \param array an std::array argument
            //!
            HOST_VERSION
            CUDA_DEVICE_VERSION
            constexpr Array(const std::array<T, C_N>& array) 
            {
                for (SizeType i = 0; i < C_N; ++i)
                {
                    data[i] = array[i];
                }
            }

            //!
            //! \brief Exchange the content with another Array.
            //!
            //! \param array another Array
            //! \return a reference to this object with the new content
            //!
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline auto Swap(Array& array) -> Array&
            {
                for (SizeType i = 0; i < C_N; ++i)
                {
                    const T tmp = data[i];
                    data[i] = array.data[i];
                    array.data[i] = tmp;
                }

                return *this;
            }

            //!
            //! \brief Array subscript operator.
            //!
            //! \param index element index
            //! \return reference to the element
            //!
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline constexpr auto operator[](const SizeType index) -> T&
            {
                assert(index < C_N);
                
                return data[index];
            }

            //!
            //! \brief Array subscript operator.
            //!
            //! \param index element index
            //! \return const reference to the element
            //!
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline constexpr auto operator[](const SizeType index) const -> const T&
            {
                assert(index < C_N);

                return data[index];
            }

            //!
            //! \brief Replace the element at position 'index' by a new one.
            //!
            //! The replacement happens in place.
            //!
            //! \param x replacement
            //! \param index element index
            //! \return a reference to this Array
            //!
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline constexpr auto Replace(const T x, const SizeType index) -> Array&
            {
                assert(index < C_N);
                
                data[index] = x;
                
                return *this;
            }

            //!
            //! \brief Replace the element at position 'index' by a new one.
            //!
            //! \param x replacement
            //! \param index element index
            //! \return a new Array object
            //!
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline constexpr auto Replace(const T x, const SizeType index) const
            {
                Array<T, C_N> result{*this};

                return result.Replace(x, index);
            }

            //!
            //! \brief Insert a new element at position 'index'.
            //!
            //! The last element of the array will be removed (chopped)!
            //!
            //! \param x element to be inserted
            //! \param index element index
            //! \return a reference to this Array
            //!
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline constexpr auto InsertAndChop(const T x, const SizeType index = 0) -> Array&
            {
                static_assert(C_N > 0, "error: this is an empty array");
                assert(index <= C_N);

                // Keep everything behind position 'index'.
                for (SizeType i = (C_N - 1); i > index; --i)
                {
                    data[i] = data[i - 1];
                }

                // Insert element at position 'index'
                data[index] = x;

                return *this;
            }

            //!
            //! \brief Insert a new element at position 'index'.
            //!
            //! The last element of the array will be removed (chopped)!
            //!
            //! \param x element to be inserted
            //! \param index element index
            //! \return a new Array object
            //!
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline constexpr auto InsertAndChop(const T x, const SizeType index = 0) const
            {
                Array<T, C_N> result{*this};

                return result.InsertAndChop(x, index);
            }

            //!
            //! \brief Insert element at position 'index'
            //!
            //! \param x element to be inserted
            //! \param index element index
            //! \return a new Array object
            //!
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline constexpr auto Insert(const T x, const SizeType index = 0) const
            {
                assert(index <= C_N);

                Array<T, C_N + 1> result{};

                for (SizeType i = 0; i < std::min(index, C_N); ++i)
                {
                    result.data[i] = data[i];
                }

                if (index <= C_N)
                {
                    // Insert element at position 'index'.
                    result.data[index] = x;

                    // Keep everything behind position 'index'.
                    for (SizeType i = index; i < C_N; ++i)
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
                bool is_same = true;

                for (SizeType i = 0; i < C_N; ++i)
                {
                    is_same &= (data[i] == other.data[i]);
                }

                return is_same;
            }

            //!
            //! \brief Test for two Arrays having not the same content.
            //!
            //! \param other another Array
            //! \return `false` if the content is the same, otherwise `true`
            //!
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline constexpr auto operator!=(const Array& other) const
            {
                return !(*this == other);
            }

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
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline constexpr auto Reduce(const FuncT func, const T initial_value, const SizeType begin = 0, const SizeType end = C_N) const
            {
                T aggregate = initial_value;

                for (SizeType i = begin; i < end; ++i)
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
            inline constexpr T ReduceMul(const SizeType begin = 0, const SizeType end = C_N) const
            {
            #if (__cplusplus > 201402L)
                return Reduce([](const T product, const T element) { return (product * element); }, 1, begin, end);
            #else
                T aggregate = static_cast<T>(1);

                for (SizeType i = begin; i < end; ++i)
                {
                    aggregate *= data[i];
                }
                
                return aggregate;
            #endif
            }
        };

        template <typename T, SizeType C_N>
        std::ostream& operator<<(std::ostream& os, const Array<T, C_N>& array)
        {
            for (SizeType i = 0; i < C_N; ++i)
            {
                os << array[i] << ((i + 1) < C_N ? " " : "");
            }

            return os;
        }
    }
}

#endif