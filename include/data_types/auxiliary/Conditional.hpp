// Copyright (c) 2020 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(AUXILIARY_CONDITIONAL_HPP)
#define AUXILIARY_CONDITIONAL_HPP

#include <type_traits>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#include <auxiliary/CPPStandard.hpp>
#include <auxiliary/Function.hpp>
#include <platform/Target.hpp>

namespace XXX_NAMESPACE
{
    namespace compileTime
    {
        using ::XXX_NAMESPACE::variadic::IsInvocable;

        //! @{
        //! A compile time if-else construct.
        //! If any of the arguments is invocable, its return values is returned conditionally.
        //!
        template <bool Predicate, typename T_1, typename T_2>
        HOST_VERSION CUDA_DEVICE_VERSION constexpr auto IfElse(T_1 x, T_2 y) -> std::enable_if_t<Predicate && !IsInvocable<T_1>::value, T_1>
        {
            return x;
        }

        template <bool Predicate, typename T_1, typename T_2>
        HOST_VERSION CUDA_DEVICE_VERSION constexpr auto IfElse(T_1 x, T_2 y) -> std::enable_if_t<Predicate && IsInvocable<T_1>::value, decltype(x())>
        {
            return x();
        }

        template <bool Predicate, typename T_1, typename T_2>
        HOST_VERSION CUDA_DEVICE_VERSION constexpr auto IfElse(T_1 x, T_2 y) -> std::enable_if_t<!Predicate && !IsInvocable<T_2>::value, T_2>
        {
            return y;
        }

        template <bool Predicate, typename T_1, typename T_2>
        HOST_VERSION CUDA_DEVICE_VERSION constexpr auto IfElse(T_1 x, T_2 y) -> std::enable_if_t<!Predicate && IsInvocable<T_2>::value, decltype(y())>
        {
            return y();
        }
        //! @}
    } // namespace compileTime
} // namespace XXX_NAMESPACE

#endif