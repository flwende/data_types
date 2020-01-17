// Copyright (c) 2020 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(AUXILIARY_ACCUMULATE_HPP)
#define AUXILIARY_ACCUMULATE_HPP

#include <cmath>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#include <auxiliary/CPPStandard.hpp>

namespace XXX_NAMESPACE
{
    namespace variadic
    {
        namespace
        {
            //! @{
            //! Accumulate the variadic argument list.
            //!
            template <typename AggregateT, typename... ParameterList>
            struct AccumulateImplementation;

            template <typename AggregateT, typename Head, typename... Tail>
            struct AccumulateImplementation<AggregateT, Head, Tail...>
            {
                static inline constexpr auto Add(AggregateT aggregate, Head head, Tail... tail) { return AccumulateImplementation<AggregateT, Tail...>::Add(aggregate + head, tail...); }

                static inline constexpr auto Max(AggregateT aggregate, Head head, Tail... tail) { return AccumulateImplementation<AggregateT, Tail...>::Max(std::max(aggregate, head), tail...); }
            };

            template <typename AggregateT>
            struct AccumulateImplementation<AggregateT>
            {
                static inline constexpr auto Add(AggregateT aggregate) { return aggregate; }

                static inline constexpr auto Max(AggregateT aggregate) { return aggregate; }
            };
            //! @}
        } // namespace

        //! @{
        //! Accumulate the variadic argument list.
        //!
        //! \tparam AggregateT type of the aggregate
        //!
        template <typename AggregateT, typename... T>
        constexpr inline auto AccumulateAdd(AggregateT aggregate, T... values) -> AggregateT
        {
            return AccumulateImplementation<AggregateT, T...>::Add(aggregate, values...);
        }

        template <typename AggregateT, typename... T>
        constexpr inline auto AccumulateMax(AggregateT aggregate, T... values) -> AggregateT
        {
            return AccumulateImplementation<AggregateT, T...>::Max(aggregate, values...);
        }
        //! @}
    } // namespace variadic
} // namespace XXX_NAMESPACE

#endif