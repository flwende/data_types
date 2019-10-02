// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(DATA_TYPES_ARRAY_ARRAY_HPP)
#define DATA_TYPES_ARRAY_ARRAY_HPP

#include <array>
//#include <cstdint>
#include <iostream>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#include <data_types/DataTypes.hpp>
#include <platform/Target.hpp>

namespace XXX_NAMESPACE
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //! \brief A fixed sized array
    //!
    //! \tparam T data type
    //! \tparam C_N array size
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <typename T, SizeType C_N>
    class sarray
    {
        template <typename, SizeType>
        friend class sarray;
     
        T data[C_N];

    public:

        using element_type = T;
        static constexpr SizeType size = C_N;

        //! \brief Standard constructor
        constexpr sarray() 
            : 
            data{} {}

        constexpr sarray(const T& x) 
        {
            for (SizeType i = 0; i < C_N; ++i)
                data[i] = x;
        }

        //! \brief Constructor taking C_N (or less) arguments to initialize the array
        //!
        //! \tparam Args variadic template type parameter list
        //! \param args parameters
        template <typename ... Args>
        constexpr sarray(const Args ... args) 
            : 
            data{ static_cast<T>(std::move(args)) ... } {}

        //! \brief Copy constructor
        //!
        //! \tparam X array size
        //! \param x an sarray object of type T with size X
        template <SizeType X>
        constexpr sarray(const sarray<T, X>& x)
        {
            constexpr SizeType i_max = (X < C_N ? X : C_N);

            for (SizeType i = 0; i < i_max; ++i)
            {
                data[i] = x.data[i];
            }

            for (SizeType i = i_max; i < C_N; ++i)
            {
                data[i] = static_cast<T>(0);
            }
        }

        //! \brief Construct from std::array
        //!
        //! \param x an std::array object
        constexpr sarray(const std::array<T, C_N>& x) 
        {
            for (SizeType i = 0; i < C_N; ++i)
            {
                data[i] = x[i];
            }
        }

        //! \brief Swap content
        //!
        //! \param x an std::array object
        //! \return a reference to this object with the new content
        inline constexpr sarray& swap(sarray& x)
        {
            for (SizeType i = 0; i < C_N; ++i)
            {
                const T tmp = data[i];
                data[i] = x.data[i];
                x.data[i] = tmp;
            }

            return *this;
        }

        //! \brief Array subscript operator
        //!
        //! \param idx element index
        //! \return reference to the element
        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline constexpr T& operator[](const SizeType idx)
        {
            return data[idx];
        }

        //! \brief Array subscript operator
        //!
        //! \param idx element index
        //! \return const reference to the element
        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline constexpr const T& operator[](const SizeType idx) const
        {
            return data[idx];
        }

        //! \brief Replace element at position 'idx'
        //!
        //! \param x replacement
        //! \param idx element index
        //! \return an sarray object
        inline constexpr sarray<T, C_N> replace(const T x, const SizeType idx) const 
        {
            sarray<T, C_N> result = *this;

            if (idx < C_N) 
            {
                result.data[idx] = x;
            }

            return result;
        }

        //! \brief Insert element at position 'idx'
        //!
        //! Note: the last element of the array will be removed (chopped)!
        //!
        //! \param x element to be inserted
        //! \param idx element index
        //! \return an sarray object
        inline constexpr sarray<T, C_N> insert_and_chop(const T x, const SizeType idx = 0) const
        {
            if (idx >= C_N) return *this;

            sarray<T, C_N> result;

            // take everything before position 'idx'
            for (SizeType i = 0; i < idx; ++i)
            {
                result.data[i] = data[i];
            }

            // insert element at position 'idx'
            result.data[idx] = x;

            // take everything behind position 'idx' 
            for (SizeType i = idx; i < (C_N - 1); ++i)
            {
                result.data[i + 1] = data[i];
            }

            return result;
        }

        //! \brief Insert element at position 'idx'
        //!
        //! \param x element to be inserted
        //! \param idx element index
        //! \return an sarray object
        inline constexpr sarray<T, C_N + 1> insert(const T x, const SizeType idx = 0) const
        {
            sarray<T, C_N + 1> result(*this);

            if (idx < (C_N + 1))
            {
                // insert element at position 'idx'
                result.data[idx] = x;

                // take everything behind position 'idx' 
                for (SizeType i = idx; i < C_N; ++i)
                {
                    result.data[i + 1] = data[i];
                }
            }

            return result;
        }

        //! \brief Test whether two sarrays are the same
        //!
        //! \param rhs
        //! \return result of the element-wise comparison of the sarray contents
        inline constexpr bool operator==(const sarray& rhs) const
        {
            bool is_same = true;

            for (SizeType i = 0; i < C_N; ++i)
            {
                is_same &= (data[i] == rhs.data[i]);
            }

            return is_same;
        }

        //! \brief Test whether two sarrays are the same
        //!
        //! \param rhs
        //! \return result of the element-wise comparison of the sarray contents
        inline constexpr bool operator!=(const sarray& rhs) const
        {
            return !(*this == rhs);
        }

        //! \brief Reduce across all entries
        //!
        //! \param func a lambda implementing the reduction
        //! \param r_0 start value for the reduction
        //! \return reduction
        template <typename F>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline constexpr T reduce(F func, const T r_0, const SizeType begin = 0, const SizeType end = C_N) const
        {
            T r = r_0;

            for (SizeType i = begin; i < end; ++i)
            {
                r = func(r, data[i]);
            }
            
            return r;
        }

        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline constexpr T reduce_mul(const SizeType begin = 0, const SizeType end = C_N) const
        {
            return reduce([&](const T product, const T element) { return (product * element); }, 1, begin, end);
        }
    };

    template <typename T, typename SizeType, SizeType C_N>
    std::ostream& operator<<(std::ostream& os, const sarray<T, C_N>& x)
    {
        for (SizeType i = 0; i < C_N; ++i)
        {
            os << x[i] << ((i + 1) < C_N ? " " : "");
        }

        return os;
    }
}

#endif