// Copyright (c) 2020 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(AUXILIARY_PACK_HPP)
#define AUXILIARY_PACK_HPP

#include <type_traits>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#include <auxiliary/Accumulate.hpp>
#include <auxiliary/CPPStandard.hpp>
#include <auxiliary/Types.hpp>
#include <DataTypes.hpp>

namespace XXX_NAMESPACE
{
    namespace variadic
    {
        using SizeT = ::XXX_NAMESPACE::dataTypes::SizeT;

        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        //
        // Some helpers.
        //
        ////////////////////////////////////////////////////////////////////////////////////////////////////////

        namespace internal
        {
            using ::XXX_NAMESPACE::auxiliary::ConstReferenceToReference;

            //! @{
            //! Get the value of the N-th parameter and its type in the variadic list.
            //!
            template <SizeT N, typename ...ParameterList>
            struct Parameter;

            template <SizeT N, typename Head, typename... Tail>
            struct Parameter<N, Head, Tail...>
            {
                using Type = typename Parameter<N - 1, Tail...>::Type;

                static inline constexpr auto Value(Head, Tail... tail) -> typename Parameter<N - 1, Tail...>::Type
                { 
                    return Parameter<N - 1, Tail...>::Value(tail...);
                }
            };

            //! Recursion anchor: N = 0.
            template <typename Head, typename... Tail>
            struct Parameter<0, Head, Tail...>
            {
                using Type = Head;

                static inline constexpr auto Value(Head head, Tail...) -> Head
                { 
                    return head; 
                }
            };

            template <>
            struct Parameter<0>
            {
                using Type = void;

                static inline constexpr auto Value() { return 0; }
            };
            //! @}

            template <typename T_1, typename T_2>
            struct IsConvertibleTo
            {
                static constexpr bool value = !(std::is_const<T_1>::value && !std::is_const<T_2>::value) && 
                    std::is_convertible<typename ConstReferenceToReference<T_1>::Type, typename ConstReferenceToReference<T_2>::Type>::value;
            };
        } // namespace

        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        //
        // Some operations on variadic parameter packs.
        //
        ////////////////////////////////////////////////////////////////////////////////////////////////////////

        template <typename... ParameterList>
        struct Pack
        {
            using Head = typename internal::Parameter<0, ParameterList...>::Type;

            template <SizeT N>
            using Type = typename internal::Parameter<N, ParameterList...>::Type;

            static constexpr SizeT Size = sizeof...(ParameterList);

            template <SizeT N>
            static inline constexpr auto Value(ParameterList... values) -> typename internal::Parameter<N, ParameterList...>::Type
            {
                return internal::Parameter<N, ParameterList...>::Value(values...);
            }

            template <template <typename...> typename TypeName>
            using TypeWithReverseParameters = typename AssembleTypeReverse<TypeName<>, ParameterList...>::Type;

            //!
            //! \return `true` if all parameters have the same type, otherwise `false`
            //!
            static inline constexpr auto IsSame()
            {
                return AccumulateAdd<SizeT>(0, (std::is_same<Head, ParameterList>::value ? 1 : 0)...) == Size;
            }

            //!
            //! \return `true` if all parameters have the same size, otherwise `false`
            //!
            static inline constexpr auto SameSize()
            {
                return AccumulateAdd<SizeT>(0, (sizeof(Head) == sizeof(ParameterList) ? 1 : 0)...) == Size;
            }

            //!
            //! \return `true` if any parameter is constant qualified, otherwise `false`
            //!
            static inline constexpr auto IsConst()
            {
                return AccumulateAdd<SizeT>(0, (std::is_const<ParameterList>::value ? 1 : 0)...) > 0;
            }

            //!
            //! \return `true` if all parameters are unsigned, otherwise `false`
            //!
            static inline constexpr auto IsUnsigned()
            {
                return AccumulateAdd<SizeT>(0, (std::is_unsigned<ParameterList>::value ? 1 : 0)...) == Size;
            }

            //!
            //! \return `true` if all parameters are fundamental, otherwise `false`
            //!
            static inline constexpr auto IsFundamental()
            {
                return AccumulateAdd<SizeT>(0, (std::is_fundamental<ParameterList>::value ? 1 : 0)...) == Size;
            }

            //!
            //! \return `true` if any parameter is a `void` type, otherwise `false`
            //!
            static inline constexpr auto IsVoid()
            {
                return AccumulateAdd<SizeT>(0, (std::is_void<ParameterList>::value ? 1 : 0)...) > 0;
            }

            //!
            //! \return `true` if any parameter is volatile qualified, otherwise `false`
            //!
            static inline constexpr auto IsVolatile()
            {
                return AccumulateAdd<SizeT>(0, (std::is_volatile<ParameterList>::value ? 1 : 0)...) > 0;
            }

            //!
            //! \return `true` if all parameters are reference values, otherwise `false`
            //!
            static inline constexpr auto IsReference()
            {
                return AccumulateAdd<SizeT>(0, (std::is_reference<ParameterList>::value ? 1 : 0)...) == Size;
            }

            //! @{
            //! Test for all parameters of this `Pack` being convertible to/from `X...`.
            //!
            //! \return `true` if all parameters are convertible amongst eachother, otherwise `false`
            //!
            template <typename ...X>
            static inline constexpr auto IsConvertibleTo() -> std::enable_if_t<(Size > 1 && sizeof...(X) == 1), bool>
            {
                return AccumulateAdd<SizeT>(0, (internal::IsConvertibleTo<ParameterList, typename internal::Parameter<0, X...>::Type>::value ? 1 : 0)...) == std::max(Size, sizeof...(X));
            }

            template <typename ...X>
            static inline constexpr auto IsConvertibleTo() -> std::enable_if_t<(Size == 1 && sizeof...(X) > 1), bool>
            {
                return AccumulateAdd<SizeT>(0, (internal::IsConvertibleTo<typename internal::Parameter<0, ParameterList...>::Type, X>::value ? 1 : 0)...) == std::max(Size, sizeof...(X));
            }

            template <typename ...X>
            static inline constexpr auto IsConvertibleTo() -> std::enable_if_t<(Size == 1 && Size == sizeof...(X)), bool>
            {
                return internal::IsConvertibleTo<typename internal::Parameter<0, ParameterList...>::Type, typename internal::Parameter<0, X...>::Type>::value;
            }

            template <typename ...X>
            static inline constexpr auto IsConvertibleTo() -> std::enable_if_t<(Size > 1 && Size == sizeof...(X)), bool>
            {
                return AccumulateAdd<SizeT>(0, (internal::IsConvertibleTo<ParameterList, X>::value ? 1 : 0)...) == Size;
            }

            template <typename ...X>
            static inline constexpr auto IsConvertibleFrom()
            {
                return Pack<X...>::template IsConvertibleTo<ParameterList...>();
            }
            //! @}

            //!
            //! \return the byte size of the parameter with the largest type
            //!
            static inline constexpr auto SizeOfLargestParameter()
            {
                return AccumulateMax<SizeT>(0, sizeof(ParameterList)...);
            }

            //!
            //! \return the total byte size of the parameter pack
            //!
            static inline constexpr auto SizeOfPack()
            {
                return AccumulateAdd<SizeT>(0, sizeof(ParameterList)...);
            }

            //!
            //! \return the total byte size of all parameters excluding those with the largest type
            //!
            static inline constexpr auto SizeOfPackExcludingLargestParameter()
            {
                return AccumulateAdd<SizeT>(0, (sizeof(ParameterList) == SizeOfLargestParameter() ? 0 : sizeof(ParameterList))...);
            }
        };
    } // namespace variadic
} // namespace XXX_NAMESPACE

#endif