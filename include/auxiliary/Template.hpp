// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(AUXILIARY_TEMPLATE_HPP)
#define AUXILIARY_TEMPLATE_HPP

#include <functional>
#include <type_traits>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#include <auxiliary/CPPStandard.hpp>
#include <data_types/DataTypes.hpp>
#include <platform/Target.hpp>

namespace XXX_NAMESPACE
{
    namespace variadic
    {
        using SizeT = ::XXX_NAMESPACE::dataTypes::SizeT;

        namespace
        {
            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            //!
            //! \brief Get the number of parameters in the variadic list.
            //!
            //! \tparam T variadic parameter list
            //! \return number of parameters
            //!
            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            template <typename... T>
            constexpr auto GetNumParameters()
            {
                return sizeof...(T);
            }

            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            //!
            //! \brief Get the value of the N-th parameter and its type in the variadic list.
            //!
            //! \tparam N parameter index
            //! \tparam T variadic parameter list
            //!
            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            template <SizeT N, typename ...T>
            struct Parameter;

            //!
            //! \brief Special case for empty parameter list.
            //!
            template <>
            struct Parameter<0>
            {
                using Type = void;

                static constexpr auto Value() { return 0; }
            };

            //!
            //! \brief Get the value of the N-th parameter and its type in the variadic list.
            //!
            //! \tparam N parameter index
            //! \tparam Head type of the front most parameter (head)
            //! \tparam Tail variadic parameter list (tail)
            //!
            template <SizeT N, typename Head, typename... Tail>
            struct Parameter<N, Head, Tail...>
            {
                //! Type of the N-th parameter
                using Type = typename Parameter<N - 1, Tail...>::Type;

                //!
                //! \brief Get the value of the N-th parameter through recursion.
                //!
                //! \param head frontmost argument
                //! \param tail remaining arguments
                //! \return recursive list definition
                //!
                static constexpr auto Value(Head head, Tail... tail) -> typename Parameter<N - 1, Tail...>::Type
                { 
                    return Parameter<N - 1, Tail...>::Value(tail...);
                }
            };

            //!
            //! \brief N = 0 specialization (recursion ancher definition).
            //!
            //! \tparam Head type of the N-th parameter
            //! \tparam Tail variadic parameter list (tail)
            //!
            template <typename Head, typename... Tail>
            struct Parameter<0, Head, Tail...>
            {
                //! \brief Type of the N-th parameter.
                using Type = Head;

                //!
                //! \brief Value of the N-th parameter.
                //!
                //! \param head N-the argument
                //! \param tail remaining arguments
                //! \return the N-th argument
                //!
                static constexpr auto Value(Head head, Tail... tail) -> Head
                { 
                    return head; 
                }
            };

            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            //!
            //! \brief Accumulate the variadic parameter/argument list.
            //!
            //! \tparam AggregateT type of the aggregate
            //! \tparam T variadic parameter list
            //!
            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            template <typename AggregateT, typename... T>
            struct Accumulate;

            //!
            //! \brief Special case for empty parameter list.
            //!
            template <typename AggregateT>
            struct Accumulate<AggregateT>
            {
                static constexpr auto Add(AggregateT aggregate) { return aggregate; }

                static constexpr auto Max(AggregateT aggregate) { return aggregate; }
            };

            //!
            //! \brief Accumulate the variadic parameter/argument list (recursive).
            //!
            //! \tparam AggregateT type of the aggregate
            //! \tparam Head type of the front most parameter (head)
            //! \tparam Tail variadic parameter list (tail)
            //!
            template <typename AggregateT, typename Head, typename... Tail>
            struct Accumulate<AggregateT, Head, Tail...>
            {
                //!
                //! \brief Get the sum over all arguments.
                //!
                //! \param aggregate value of the aggregate
                //! \param head frontmost argument
                //! \param tail remaining arguments
                //! \return recursive list definition
                //!
                static constexpr auto Add(AggregateT aggregate, const Head head, const Tail... tail) { return Accumulate<AggregateT, Tail...>::Add(aggregate + head, tail...); }

                //!
                //! \brief Get the maximum value among all arguments.
                //!
                //! \param aggregate value of the aggregate
                //! \param head frontmost argument
                //! \param tail remaining arguments
                //! \return recursive list definition
                //!
                static constexpr auto Max(AggregateT aggregate, const Head head, const Tail... tail) { return Accumulate<AggregateT, Tail...>::Max(std::max(aggregate, head), tail...); }
            };

            //!
            //! \brief Accumulate the variadic parameter/argument list (recursion anchor).
            //!
            //! \tparam AggregateT type of the aggregate
            //! \tparam Head type of the front most parameter (head)
            //!
            template <typename AggregateT, typename Head>
            struct Accumulate<AggregateT, Head>
            {
                //!
                //! \brief Get the sum over all arguments.
                //!
                //! \param aggregate value of the aggregate
                //! \param head frontmost argument
                //! \return the sum over all arguments
                //!
                static constexpr auto Add(AggregateT aggregate, const Head head) { return aggregate + head; }

                //!
                //! \brief Get the maximum value among all arguments.
                //!
                //! \param aggregate value of the aggregate
                //! \param head frontmost argument
                //! \return the maximum value among all arguments
                //!
                static constexpr auto Max(AggregateT aggregate, const Head head) { return std::max(aggregate, head); }
            };
        } // namespace

        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        //!
        //! \brief A template parameter pack info data structure.
        //!
        //! \tparam T variadic parameter list
        //!
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        template <typename... T>
        struct Pack
        {
            //! \brief Type of the N-th parameter.
            template <SizeT N>
            using Type = typename Parameter<N, T...>::Type;

            //! \brief Number of parameters.
            static constexpr SizeT Size = GetNumParameters<T...>();

            //!
            //! \brief Get the value of the N-th parameter.
            //!
            //! \tparam N parameter index
            //! \param values variadic argument list
            //! \return the N-th argument
            //!
            template <SizeT N>
            static constexpr auto Value(T... values) -> typename Parameter<N, T...>::Type
            {
                return Parameter<N, T...>::Value(values...);
            }

            //!
            //! \brief Test for all parameters having the same type.
            //!
            //! \return `true` if all parameters have the same type, otherwise `false`
            //!
            static constexpr auto IsSame()
            {
                using Head = typename Parameter<0, T...>::Type;

                return Accumulate<SizeT, std::conditional_t<sizeof(T) == 0, T, SizeT>...>::Add(0, (std::is_same<Head, T>::value ? 1 : 0)...) == Size;
            }

            //!
            //! \brief Test for all parameters having the same size.
            //!
            //! \return `true` if all parameters have the same size, otherwise `false`
            //!
            static constexpr auto SameSize()
            {
                using Head = typename Parameter<0, T...>::Type;

                return Accumulate<SizeT, std::conditional_t<sizeof(T) == 0, T, SizeT>...>::Add(0, (sizeof(Head) == sizeof(T) ? 1 : 0)...) == Size;
            }

            //!
            //! \brief Test for all parameters being convertible amongst eachother.
            //!
            //! \tparam X a type-parameter list to be considered for conversion
            //! \return `true` if all parameters are convertible amongst eachother, otherwise `false`
            //!
            template <typename ...X>
            static constexpr auto IsConvertibleTo() -> std::enable_if_t<(sizeof...(X) == 1), bool>
            {
                using Head = typename Parameter<0, X...>::Type;

                return Accumulate<SizeT, std::conditional_t<sizeof(T) == 0, T, SizeT>...>::Add(0, (std::is_convertible<T, Head>::value ? 1 : 0)...) == Size;
            }

            template <typename ...X>
            static constexpr auto IsConvertibleTo() -> std::enable_if_t<(sizeof...(X) > 1), bool>
            {
                static_assert(sizeof...(X) == Size, "error: parameter lists have different size.");

                return Accumulate<SizeT, std::conditional_t<sizeof(T) == 0, T, SizeT>...>::Add(0, (std::is_convertible<T, X>::value ? 1 : 0)...) == Size;
            }

            template <typename ...X>
            static constexpr auto IsConvertibleFrom()
            {
                return IsConvertibleTo<X...>();
            }

            //!
            //! \brief Test for any parameter being constant qualified.
            //!
            //! \return `true` if any parameter is constant qualified, otherwise `false`
            //!
            static constexpr auto IsConst()
            {
                return Accumulate<SizeT, std::conditional_t<sizeof(T) == 0, T, SizeT>...>::Add(0, (std::is_const<T>::value ? 1 : 0)...) > 0;
            }

            //!
            //! \brief Test for all parameters being unsigned.
            //!
            //! \return `true` if all parameters are unsigned, otherwise `false`
            //!
            static constexpr auto IsUnsigned()
            {
                return Accumulate<SizeT, std::conditional_t<sizeof(T) == 0, T, SizeT>...>::Add(0, (std::is_unsigned<T>::value ? 1 : 0)...) == Size;
            }

            //!
            //! \brief Test for all parameters being fundamental.
            //!
            //! \return `true` if all parameters are fundamental, otherwise `false`
            //!
            static constexpr auto IsFundamental()
            {
                return Accumulate<SizeT, std::conditional_t<sizeof(T) == 0, T, SizeT>...>::Add(0, (std::is_fundamental<T>::value ? 1 : 0)...) == Size;
            }

            //!
            //! \brief Test for all parameters being void types.
            //!
            //! \return `true` if all parameters are void types, otherwise `false`
            //!
            static constexpr auto IsVoid()
            {
                return Accumulate<SizeT, std::conditional_t<sizeof(T) == 0, T, SizeT>...>::Add(0, (std::is_void<T>::value ? 1 : 0)...) > 0;
            }

            //!
            //! \brief Test for all parameters being volatile qualified.
            //!
            //! \return `true` if all parameters are volatile qualified, otherwise `false`
            //!
            static constexpr auto IsVolatile()
            {
                return Accumulate<SizeT, std::conditional_t<sizeof(T) == 0, T, SizeT>...>::Add(0, (std::is_volatile<T>::value ? 1 : 0)...) > 0;
            }

            //!
            //! \brief Test for all parameters being volatile qualified.
            //!
            //! \return `true` if all parameters are volatile qualified, otherwise `false`
            //!
            static constexpr auto IsReference()
            {
                return Accumulate<SizeT, std::conditional_t<sizeof(T) == 0, T, SizeT>...>::Add(0, (std::is_reference<T>::value ? 1 : 0)...) == Size;
            }

            //!
            //! \brief Get the size of the parameter with the largest type.
            //!
            //! \return the size of the parameter with the largest type
            //!
            static constexpr auto SizeOfLargestParameter()
            {
                return Accumulate<SizeT, std::conditional_t<sizeof(T) == 0, T, SizeT>...>::Max(0, sizeof(T)...);
            }

            //!
            //! \brief Get the total size of the parameter pack.
            //!
            //! \return the total size of the parameter pack
            //!
            static constexpr auto SizeOfPack()
            {
                return Accumulate<SizeT, std::conditional_t<sizeof(T) == 0, T, SizeT>...>::Add(0, sizeof(T)...);
            }

            //!
            //! \brief Get the total size of all parameters excluding those with the largest type.
            //!
            //! \return the total size of all parameters excluding those with the largest type
            //!
            static constexpr auto SizeOfPackExcludingLargestParameter()
            {
                constexpr SizeT size_of_largest_parameter = SizeOfLargestParameter();

                return Accumulate<SizeT, std::conditional_t<sizeof(T) == 0, T, SizeT>...>::Add(0, (sizeof(T) == size_of_largest_parameter ? 0 : sizeof(T))...);
            }
        };

        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        //!
        //! \brief Test for invocability of a callable.
        //!
        //! \tparam FuncT the type of a callable
        //! \tparam T variadic parameter list
        //!
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        template <typename FuncT, typename ...T>
        struct IsInvocable
        {
            static constexpr bool value = std::is_constructible<std::function<void(T...)>, std::reference_wrapper<std::remove_reference_t<FuncT>>>::value;
        };
    } // namespace variadic

    namespace compileTime
    {
        using SizeT = ::XXX_NAMESPACE::dataTypes::SizeT;

        /////////////////////////////////////////////////////////////////
        //
        // A compile-time loop.
        //
        /////////////////////////////////////////////////////////////////
        namespace
        {
            //!
            //! \brief Definition of the loop (recursive).
            //!
            //! \tparam I value of the iteration variable
            //! \tparam C_Begin start value of the iteration variable
            //!
            template <SizeT I, SizeT C_Begin>
            struct LoopImplementation
            {
                //!
                //! \brief Loop body invocation.
                //!
                //! The recursion must happen before the loop body invocation.
                //!
                //! \tparam FuncT the type of a callable (loop body)
                //! \param loop_body the loop body
                //!
                template <typename FuncT>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                static constexpr auto Execute(FuncT loop_body) -> void
                {
                    LoopImplementation<I - 1, C_Begin>::Execute(loop_body);

                    loop_body(std::integral_constant<SizeT, C_Begin + I>{});
                }
            };

            //!
            //! \brief Definition of the loop (recursion anchor).
            //!
            //! This is the first iteration of the loop.
            //!
            //! \tparam C_Begin start value of the iteration variable
            //!
            template <SizeT C_Begin>
            struct LoopImplementation<0, C_Begin>
            {
                //!
                //! \brief Loop body invocation.
                //!
                //! The recursion stops here.
                //!
                //! \tparam FuncT the type of a callable (loop body)
                //! \param loop_body the loop body
                //!
                template <typename FuncT>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                static constexpr auto Execute(FuncT loop_body) -> void
                {
                    static_assert(::XXX_NAMESPACE::variadic::IsInvocable<FuncT, std::integral_constant<SizeT, C_Begin>>::value, "error: callable is not invocable. void (*) (std::integral_constant<SizeT,..>) expected.");

                    loop_body(std::integral_constant<SizeT, C_Begin>{});
                }
            };
        } // namespace

        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        //!
        //! \brief A compile-time loop.
        //!
        //! ```
        //! Loop<C_Begin, C_End>::Execute([..] (const auto I) { loop_body(I); })
        //! ```
        //! corresponds to
        //! ```
        //!    for (SizeT I = C_Begin; I < C_End; ++I)
        //!        loop_body(I);
        //! ```
        //!
        //! \tparam C_Begin lower loop bound
        //! \tparam C_End upper loop bound
        //!
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        template <SizeT C_Begin, SizeT C_End = 0>
        struct Loop
        {
            static constexpr SizeT Begin = (C_End == 0 ? 0 : C_Begin);
            static constexpr SizeT End = (C_End == 0 ? C_Begin : C_End);

            static_assert(Begin < End, "error: End <= Begin");

            //!
            //! \brief Loop body invocation.
            //!
            //! \tparam FuncT the type of a callable (loop body)
            //! \param loop_body the loop body
            //!
            template <typename F>
            HOST_VERSION
            CUDA_DEVICE_VERSION
            static constexpr auto Execute(F loop_body) -> void
            {
                LoopImplementation<(End - Begin - 1), Begin>::Execute(loop_body);
            }
        };

        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        //!
        //! \brief A compile-time if-else.
        //!
        //! Mutli-versioning according to the value of the `Predicate` and
        //! invocability of the argument(s).
        //!
        //! \tparam Predicate a predicate that is either `true` or `false`
        //! \tparam T_1 type of `if` branch callable or argument
        //! \tparam T_2 type of `else` branch callable or argument
        //! \param x `if` branch callable or argument
        //! \param y `else` branch callable or argument
        //!
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        template <bool Predicate, typename T_1, typename T_2>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        constexpr auto IfElse(T_1 x, T_2 y)
            -> std::enable_if_t<Predicate && !::XXX_NAMESPACE::variadic::IsInvocable<T_1>::value, T_1>
        {
            return x;
        }

        template <bool Predicate, typename T_1, typename T_2>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        constexpr auto IfElse(T_1 x, T_2 y)
            -> std::enable_if_t<Predicate && ::XXX_NAMESPACE::variadic::IsInvocable<T_1>::value, decltype(x())>
        {
            return x();
        }

        template <bool Predicate, typename T_1, typename T_2>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        constexpr auto IfElse(T_1 x, T_2 y)
            -> std::enable_if_t<!Predicate && !::XXX_NAMESPACE::variadic::IsInvocable<T_2>::value, T_2>
        {
            return y;
        }

        template <bool Predicate, typename T_1, typename T_2>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        constexpr auto IfElse(T_1 x, T_2 y)
            -> std::enable_if_t<!Predicate && ::XXX_NAMESPACE::variadic::IsInvocable<T_2>::value, decltype(y())>
        {
            return y();
        }
    } // namespace compileTime
} // namespace XXX_NAMESPACE

#endif