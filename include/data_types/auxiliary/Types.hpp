// Copyright (c) 2020 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(AUXILIARY_TYPES_HPP)
#define AUXILIARY_TYPES_HPP

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#include <auxiliary/CPPStandard.hpp>

namespace XXX_NAMESPACE
{
    namespace variadic
    {
        //! @{
        //! Generate type `T<..>` with reversed template parameter list.
        //!
        template <typename T, typename... ParameterList>
        struct AssembleTypeReverse;

        template <template <typename...> typename T, typename... InverseParameterList, typename Head, typename... Tail>
        struct AssembleTypeReverse<T<InverseParameterList...>, Head, Tail...>
        {
            using Type = typename AssembleTypeReverse<T<Head, InverseParameterList...>, Tail...>::Type;
        };

        template <template <typename...> typename T, typename... InverseParameterList>
        struct AssembleTypeReverse<T<InverseParameterList...>>
        {
            using Type = T<InverseParameterList...>;
        };
        //! @}
    } // namespace variadic

    namespace auxiliary
    {
        template <typename T>
        struct IsConstReference
        {
            static constexpr bool value = std::is_reference<T>::value && std::is_const<std::remove_reference_t<T>>::value;
        };

        template <typename T>
        struct ConstReferenceToReference
        {
            using Type = std::conditional_t<IsConstReference<T>::value, std::add_lvalue_reference_t<std::decay_t<T>>, T>;
        };
    }
} // namespace XXX_NAMESPACE

#endif