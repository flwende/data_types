// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(DATA_TYPES_INTEGER_SEQUENCE_INTEGER_SEQUENCE_HPP)
#define DATA_TYPES_INTEGER_SEQUENCE_INTEGER_SEQUENCE_HPP

#include <cstdint>
#include <type_traits>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#include <platform/Target.hpp>
#include <DataTypes.hpp>

namespace XXX_NAMESPACE
{
    namespace dataTypes
    {
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        //!
        //! \brief An integer sequence type.
        //!
        //! The integer sequence type is defined recursively.
        //!
        //! \tparam IntegerT the type of the integer
        //! \tparam I a variadic list of integers
        //!
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        template <typename IntegerT, IntegerT... I>
        struct IntegerSequence
        {
        };

        namespace
        {
            //!
            //! \brief Definition of the integer sequence type (recursive).
            //!
            //! The recusion level (the integer) is decreased and prepended to the existing integer sequence.
            //!
            //! \tparam IntegerT the type of the integer
            //! \tparam N the recursion level
            //! \tparam I a variadic list of integers
            //!
            template <typename IntegerT, std::size_t N, IntegerT... I>
            struct IntegerSequenceImplementation
            {
                using Type = typename IntegerSequenceImplementation<IntegerT, N - 1, N - 1, I...>::Type;
            };

            //!
            //! \brief Definition of the integer sequence type (recursion anchor).
            //!
            //! The existing integer sequence is used to define the `IntegerSequence` type.
            //!
            //! \tparam IntegerT the type of the integer
            //! \tparam I a variadic list of integers
            //!
            template <typename IntegerT, IntegerT... I>
            struct IntegerSequenceImplementation<IntegerT, 0, I...>
            {
                using Type = IntegerSequence<IntegerT, I...>;
            };
        } // namespace

        //!
        //! \brief Create an integer sequence.
        //!
        //! This function creates the sequence `0..(N - 1)`.
        //! If `N` equals zero, the list is empty.
        //!
        //! \tparam IntegerT the type of the integer
        //! \tparam N the length of the sequence
        //! \return an integer sequence of length `N`
        //!
        template <typename IntegerT, std::size_t N>
        HOST_VERSION CUDA_DEVICE_VERSION inline constexpr auto MakeIntegerSequence()
        {
            using ReturnT = std::conditional_t<N == 0, IntegerSequence<IntegerT>, typename IntegerSequenceImplementation<IntegerT, N>::Type>;

            return ReturnT{};
        }

        //!
        //! \brief The type of an integer sequence.
        //!
        //! \tparam IntegerT the type of the integer
        //! \tparam N the length of the integer sequence
        //!
        template <typename IntegerT, std::size_t N>
        using IntegerSequenceT = decltype(MakeIntegerSequence<IntegerT, N>());

        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        //!
        //! \brief An index sequence type.
        //!
        //! Special case of an integer sequence with `IntegerT` set to `SizeT`.
        //!
        //! \tparam I a variadic list of integers
        //!
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        template <SizeT... I>
        using IndexSequence = IntegerSequence<SizeT, I...>;

        //!
        //! \brief Create an index sequence.
        //!
        //! This function creates the sequence `0..(N - 1)`.
        //! If `N` equals zero, the list is empty.
        //!
        //! \tparam N the length of the sequence
        //! \return an index sequence of length `N`
        //!
        template <SizeT N>
        HOST_VERSION CUDA_DEVICE_VERSION inline constexpr auto MakeIndexSequence()
        {
            return MakeIntegerSequence<SizeT, N>();
        }

        //!
        //! \brief The type of an index sequence.
        //!
        //! \tparam N the length of the index sequence
        //!
        template <SizeT N>
        using IndexSequenceT = decltype(MakeIndexSequence<N>());
    } // namespace dataTypes
} // namespace XXX_NAMESPACE

#endif