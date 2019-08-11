// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(SARRAY_SARRAY_HPP)
#define SARRAY_SARRAY_HPP

#include <array>
#include <cstdint>
#include <iostream>

#include <platform/target.hpp>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

namespace XXX_NAMESPACE
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //! \brief A fixed sized array
    //!
    //! \tparam T data type
    //! \tparam D array size
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <typename T, std::size_t D>
    class sarray
    {
        template <typename TT, std::size_t DD>
        friend class sarray;
     
        T data[D];

    public:

        using element_type = T;
        static constexpr std::size_t size = D;

        //! \brief Standard constructor
        constexpr sarray() 
            : 
            data{} {}

        constexpr sarray(const T& x) 
        {
            for (std::size_t i = 0; i < D; ++i)
                data[i] = x;
        }

        //! \brief Constructor taking D (or less) arguments to initialize the array
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
        template <std::size_t X>
        constexpr sarray(const sarray<T, X>& x)
        {
            constexpr std::size_t i_max = (X < D ? X : D);

            for (std::size_t i = 0; i < i_max; ++i)
            {
                data[i] = x.data[i];
            }

            for (std::size_t i = i_max; i < D; ++i)
            {
                data[i] = static_cast<T>(0);
            }
        }

        //! \brief Construct from std::array
        //!
        //! \param x an std::array object
        constexpr sarray(const std::array<T, D>& x) 
        {
            for (std::size_t i = 0; i < D; ++i)
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
            for (std::size_t i = 0; i < D; ++i)
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
        inline constexpr T& operator[](const std::size_t idx)
        {
            return data[idx];
        }

        //! \brief Array subscript operator
        //!
        //! \param idx element index
        //! \return const reference to the element
        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline constexpr const T& operator[](const std::size_t idx) const
        {
            return data[idx];
        }

        //! \brief Replace element at position 'idx'
        //!
        //! \param x replacement
        //! \param idx element index
        //! \return an sarray object
        inline constexpr sarray<T, D> replace(const T x, const std::size_t idx) const 
        {
            sarray<T, D> result = *this;

            if (idx < D) 
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
        inline constexpr sarray<T, D> insert_and_chop(const T x, const std::size_t idx = 0) const
        {
            if (idx >= D) return *this;

            sarray<T, D> result;

            // take everything before position 'idx'
            for (std::size_t i = 0; i < idx; ++i)
            {
                result.data[i] = data[i];
            }

            // insert element at position 'idx'
            result.data[idx] = x;

            // take everything behind position 'idx' 
            for (std::size_t i = idx; i < (D - 1); ++i)
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
        inline constexpr sarray<T, D + 1> insert(const T x, const std::size_t idx = 0) const
        {
            sarray<T, D + 1> result(*this);

            if (idx < (D + 1))
            {
                // insert element at position 'idx'
                result.data[idx] = x;

                // take everything behind position 'idx' 
                for (std::size_t i = idx; i < D; ++i)
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

            for (std::size_t i = 0; i < D; ++i)
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
        inline constexpr T reduce(F func, const T r_0, const std::size_t begin = 0, const std::size_t end = D) const
        {
            T r = r_0;

            for (std::size_t i = begin; i < end; ++i)
            {
                r = func(r, data[i]);
            }
            
            return r;
        }

        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline constexpr T reduce_mul(const std::size_t begin = 0, const std::size_t end = D) const
        {
            return reduce([&](const T product, const T element) { return (product * element); }, 1, begin, end);
        }
    };

    template <typename T, std::size_t D>
    std::ostream& operator<<(std::ostream& os, const sarray<T, D>& x)
    {
        for (std::size_t i = 0; i < D; ++i)
        {
            os << x[i] << ((i + 1) < D ? " " : "");
        }

        return os;
    }
}

#endif