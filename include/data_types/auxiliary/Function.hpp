// Copyright (c) 2020 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(AUXILIARY_FUNCTION_HPP)
#define AUXILIARY_FUNCTION_HPP

#include <functional>
#include <type_traits>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#include <auxiliary/CPPStandard.hpp>

namespace XXX_NAMESPACE
{
    namespace variadic
    {
        //!
        //! Test for invocability of a callable.
        //!
        template <typename FuncT, typename ...T>
        struct IsInvocable
        {
            static constexpr bool value = std::is_constructible<std::function<void(T...)>, std::reference_wrapper<std::remove_reference_t<FuncT>>>::value;
        };
    } // namespace variadic
} // namespace XXX_NAMESPACE

#endif