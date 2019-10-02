// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(AUXILIARY_TEMPLATE_HPP)
#define AUXILIARY_TEMPLATE_HPP

#include <type_traits>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#include <auxiliary/CPPStandard.hpp>
#include <auxiliary/Macro.hpp>
#include <data_types/DataTypes.hpp>

namespace XXX_NAMESPACE
{
    namespace variadic
    {
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
            //! \brief Get the value of the C_N-th parameter and its type in the variadic list.
            //!
            //! \tparam C_N parameter index
            //! \tparam Head type of the front most parameter (head)
            //! \tparam Tail variadic parameter list (tail)
            //!
            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            template <SizeType C_N, typename Head, typename... Tail>
            struct Parameter
            {
                //! Type of the C_N-th parameter
                using Type = typename Parameter<C_N - 1, Tail...>::Type;

                //!
                //! \brief Get the value of the C_N-th parameter through recursion.
                //!
                //! \param head frontmost argument
                //! \param tail remaining arguments
                //! \return recursive list definition
                //!
                static constexpr auto Value(Head head, Tail... tail) { return Parameter<C_N - 1, Tail...>::Value(tail...); }
            };

            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            //!
            //! \brief C_N = 0 specialization (recursion ancher definition).
            //!
            //! \tparam Head type of the C_N-th parameter
            //! \tparam Tail variadic parameter list (tail)
            //!
            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            template <typename Head, typename... Tail>
            struct Parameter<0, Head, Tail...>
            {
                //! \brief Type of the C_N-th parameter.
                using Type = Head;

                //!
                //! \brief Value of the C_N-th parameter.
                //!
                //! \param head C_N-the argument
                //! \param tail remaining arguments
                //! \return the C_N-th argument
                //!
                static constexpr auto Value(Head head, Tail... tail) { return head; }
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
            //! \brief Accumulate the variadic parameter/argument list.
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

        //!
        //! \brief A template parameter pack info data structure.
        //!
        //! \tparam T variadic parameter list
        //!
        template <typename... T>
        struct Pack
        {
            //! \brief Type of the C_N-th parameter.
            template <SizeType C_N>
            using Type = typename Parameter<C_N, T...>::Type;

            //! \brief Number of parameters.
            static constexpr SizeType Size = GetNumParameters<T...>();

            //!
            //! \brief Get the value of the C_N-th parameter.
            //!
            //! \tparam C_N parameter index
            //! \param values variadic argument list
            //! \return the C_N-th argument
            //!
            template <SizeType C_N>
            static constexpr auto Value(T... values)
            {
                static_assert(Size > 0, "error: empty parameter pack");

                return Parameter<C_N, T...>::Value(values...);
            }

            //!
            //! \brief Test for all parameters having the same type.
            //!
            //! \return `true` if all parameters have the same type, otherwise `false`
            //!
            static constexpr auto IsSame()
            {
                static_assert(Size > 0, "error: empty parameter pack");

                using Head = typename Parameter<0, T...>::Type;

                return Accumulate<SizeType, std::conditional_t<sizeof(T) == 0, T, SizeType>...>::Add(0, (std::is_same<Head, T>::value ? 1 : 0)...) == Size;
            }

            //!
            //! \brief Test for all parameters having the same size.
            //!
            //! \return `true` if all parameters have the same size, otherwise `false`
            //!
            static constexpr auto SameSize()
            {
                static_assert(Size > 0, "error: empty parameter pack");

                using Head = typename Parameter<0, T...>::Type;

                return Accumulate<SizeType, std::conditional_t<sizeof(T) == 0, T, SizeType>...>::Add(0, (sizeof(Head) == sizeof(T) ? 1 : 0)...) == Size;
            }

            //!
            //! \brief Test for all parameters are convertible amongst eachother.
            //!
            //! \return `true` if all parameters are convertible amongst eachother, otherwise `false`
            //!
            static constexpr auto IsConvertible()
            {
                static_assert(Size > 0, "error: empty parameter pack");

                using Head = typename Parameter<0, T...>::Type;

                return Accumulate<SizeType, std::conditional_t<sizeof(T) == 0, T, SizeType>...>::Add(0, (std::is_convertible<Head, T>::value ? 1 : 0)...) == Size;
            }

            //!
            //! \brief Test for all parameters are constant qualified.
            //!
            //! \return `true` if all parameters are constant qualified, otherwise `false`
            //!
            static constexpr auto IsConst()
            {
                static_assert(Size > 0, "error: empty parameter pack");

                return Accumulate<SizeType, std::conditional_t<sizeof(T) == 0, T, SizeType>...>::Add(0, (std::is_const<T>::value ? 1 : 0)...) == Size;
            }

            //!
            //! \brief Get the size of the parameter with the largest type.
            //!
            //! \return the size of the parameter with the largest type
            //!
            static constexpr auto SizeOfLargestParameter()
            {
                static_assert(Size > 0, "error: empty parameter pack");

                return Accumulate<SizeType, std::conditional_t<sizeof(T) == 0, T, SizeType>...>::Max(0, sizeof(T)...);
            }

            //!
            //! \brief Get the total size of the parameter pack.
            //!
            //! \return the total size of the parameter pack
            //!
            static constexpr auto SizeOfPack()
            {
                static_assert(Size > 0, "error: empty parameter pack");

                return Accumulate<SizeType, std::conditional_t<sizeof(T) == 0, T, SizeType>...>::Add(0, sizeof(T)...);
            }

            //!
            //! \brief Get the total size of all parameters excluding those with the largest type.
            //!
            //! \return the total size of all parameters excluding those with the largest type
            //!
            static constexpr auto SizeOfPackExcludingLargestParameter()
            {
                static_assert(Size > 0, "error: empty parameter pack");

                constexpr SizeType size_of_largest_parameter = SizeOfLargestParameter();

                return Accumulate<SizeType, std::conditional_t<sizeof(T) == 0, T, SizeType>...>::Add(0, (sizeof(T) == size_of_largest_parameter ? 0 : sizeof(T))...);
            }
        };

        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        //!
        //! \brief Generate a type with up to 'FW_SEQ_MAX_N' template parameters
        //!
        //! Remove from a parameter pack (Head..., Tail...) 'M = |Head...| = (FW_SEQ_MAX_N - C_N) >= 0'
        //! parameters recursively, and take the remaining parameter list (Tail...) of length 'C_N' to
        //! define the type TYPE_NAME<Tail...>.
        //!
        //! \param TYPE_NAME name of the type to be generated
        //!
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
#define MACRO_TYPE_GEN(TYPE_NAME)                                                                                                                                                                                          \
    template <SizeType C_N, typename Head, typename... Tail>                                                                                                                                                               \
    struct TypeGenImplementation                                                                                                                                                                                           \
    {                                                                                                                                                                                                                      \
        using Type = typename TypeGenImplementation<C_N - 1, Tail...>::Type;                                                                                                                                               \
    };                                                                                                                                                                                                                     \
                                                                                                                                                                                                                           \
    template <typename Head, typename... Tail>                                                                                                                                                                             \
    struct TypeGenImplementation<0, Head, Tail...>                                                                                                                                                                         \
    {                                                                                                                                                                                                                      \
        using Type = TYPE_NAME<Head, Tail...>;                                                                                                                                                                             \
    };                                                                                                                                                                                                                     \
                                                                                                                                                                                                                           \
    template <typename T, SizeType C_N>                                                                                                                                                                                    \
    struct TypeGen                                                                                                                                                                                                         \
    {                                                                                                                                                                                                                      \
        static_assert(C_N > 0, "error: no template parameters specified");                                                                                                                                                 \
        static_assert(C_N <= FW_SEQ_MAX_N, "error: not implemented");                                                                                                                                                      \
                                                                                                                                                                                                                           \
        using Type = typename TypeGenImplementation<FW_SEQ_MAX_N - C_N, FW_SEQ_N(T)>::Type;                                                                                                                                \
    };
    } // namespace variadic

    namespace compileTime
    {
        /////////////////////////////////////////////////////////////////
        //
        // A compile-time loop.
        //
        /////////////////////////////////////////////////////////////////
        namespace
        {
            //!
            //! \brief Recursive definition of the loop.
            //!
            //! \tparam I value of the iteration variable
            //! \tparam C_Begin start value of the iteration variable
            //!
            template <SizeType I, SizeType C_Begin>
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
                static constexpr auto Execute(FuncT loop_body) -> void
                {
                    LoopImplementation<I - 1, C_Begin>::Execute(loop_body);

                    loop_body(std::integral_constant<SizeType, C_Begin + I>{});
                }
            };

            //!
            //! \brief Recursive definition of the loop (recursion anchor).
            //!
            //! This is the first iteration of the loop.
            //!
            //! \tparam C_Begin start value of the iteration variable
            //!
            template <SizeType C_Begin>
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
                static constexpr auto Execute(FuncT loop_body) -> void
                {
                    loop_body(std::integral_constant<SizeType, C_Begin>{});
                }
            };
        } // namespace

        /////////////////////////////////////////////////////////////////
        //!
        //! \brief A compile-time loop.
        //!
        //! ```
        //! Loop<C_Begin, C_End>::Execute([..] (const auto I) { loop_body(I); })
        //! ```
        //! corresponds to
        //! ```
        //!    for (SizeType I = C_Begin; I < C_End; ++I)
        //!        loop_body(I);
        //! ```
        //!
        //! \tparam C_Begin lower loop bound
        //! \tparam C_End upper loop bound
        //!
        template <SizeType C_Begin, SizeType C_End = 0>
        struct Loop
        {
            static constexpr SizeType Begin = (C_End == 0 ? 0 : C_Begin);
            static constexpr SizeType End = (C_End == 0 ? C_Begin : C_End);

            static_assert(Begin < End, "error: End <= Begin");

            //!
            //! \brief Loop body invocation.
            //!
            //! \tparam FuncT the type of a callable (loop body)
            //! \param loop_body the loop body
            //!
            template <typename F>
            static constexpr auto Execute(F loop_body) -> void
            {
                LoopImplementation<(End - Begin - 1), Begin>::Execute(loop_body);
            }
        };

        /////////////////////////////////////////////////////////////////
        //
        // A compile-time if-else.
        //
        /////////////////////////////////////////////////////////////////
        template <bool C_Predicate, typename T_1, typename T_2>
        auto IfElse(const T_1& x, const T_2& y)
            -> std::enable_if_t<C_Predicate, T_1&>
        {
            return x;
        }

        template <bool C_Predicate, typename T_1, typename T_2>
        auto IfElse(const T_1& x, const T_2& y)
            -> std::enable_if_t<C_Predicate, T_2&>
        {
            return y;
        }

    } // namespace compileTime
} // namespace XXX_NAMESPACE

#endif