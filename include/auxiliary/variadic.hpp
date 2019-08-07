// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(AUXILIARY_VARIADIC_HPP)
#define AUXILIARY_VARIADIC_HPP

#include <cstdint>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#if !defined(AUXILIARY_NAMESPACE_NAMESPACE)
#define AUXILIARY_NAMESPACE XXX_NAMESPACE
#endif

#include "macro.hpp"

namespace AUXILIARY_NAMESPACE
{
    namespace variadic
    {
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        //! \brief Get the number of arguments in the variadic list
        //!
        //! \tparam T variadic argument list
        //! \return number of arguments
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        template <typename ...T>
        constexpr std::size_t get_num_arguments()
        {
            return sizeof ...(T);
        }

        namespace
        {
            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            //! \brief Get the value of the N-th argument and its type in the variadic list
            //!
            //! \tparam N element to access
            //! \tparam T_head data type (head)
            //! \tparam T_tail variadic argument list (tail)
            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            template <std::size_t N, typename T_head, typename ...T_tail>
            struct argument
            {
                //! Data type of the N-th argument
                using type = typename argument<N - 1, T_tail...>::type;

                //! \brief Get the value of the N-th argument through recursion
                //!
                //! \param head frontmost argument
                //! \param tail remaining arguments
                //! \return argument list without the frontmost argument
                static constexpr T_head value(T_head head, T_tail... tail)
                {
                    return argument<N - 1, T_tail...>::value(tail ...);
                }
            };

            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            //! \brief N = 0 Specialization (recursion ancher definition)
            //!
            //! \tparam T_head data type of the N-th element
            //! \tparam T_tail variadic argument list (tail)
            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            template <typename T_head, typename ...T_tail>
            struct argument<0, T_head, T_tail...>
            {
                //! Data type of the N-the element
                using type = T_head;

                //! \brief Value of the N-th argument
                //!
                //! \param head N-the argument
                //! \param tail remaining arguments
                //! \return N-the argument
                static constexpr T_head value(T_head head, T_tail... tail)
                {
                    return head;
                }
            };

            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            //! \brief Accumulate values of the variadic list
            //!
            //! \tparam T variadic argument list
            //! \return accumulate
            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            template <typename A, typename ...T>
            struct accumulate;

            template <typename A, typename Head, typename ...Tail>
            struct accumulate<A, Head, Tail...>
            {
                static constexpr A add(A aggregate, const Head head, const Tail... tail)
                {
                    return accumulate<A, Tail...>::add(aggregate + head, tail...);
                }

                static constexpr A max(A aggregate, const Head head, const Tail... tail)
                {
                    return accumulate<A, Tail...>::max(std::max(aggregate, head), tail...);
                }
            };

            template <typename A, typename Head>
            struct accumulate<A, Head>
            {
                static constexpr A add(A aggregate, const Head head)
                {
                    return aggregate + head;
                }

                static constexpr A max(A aggregate, const Head head)
                {
                    return std::max(aggregate, head);
                }
            };
        }

        template <typename ...T>
        struct pack
        {
            template <std::size_t N>
            using type = typename argument<N, T...>::type;

            template <std::size_t N>
            static constexpr auto value(T... values)
            {
                static_assert(sizeof...(T) > 0, "error: empty parameter pack");

                return argument<N, T...>::value(values...);
            }

            static constexpr bool is_same()
            {
                static_assert(sizeof...(T) > 0, "error: empty parameter pack");

                using T_Head = typename argument<0, T...>::type;

                return accumulate<std::size_t, typename std::conditional<sizeof(T) == 0, T, std::size_t>::type...>::add(0, (std::is_same<T_Head, T>::value ? 1 : 0)...) == sizeof...(T);
            }

            static constexpr bool has_same_size()
            {
                static_assert(sizeof...(T) > 0, "error: empty parameter pack");

                using T_Head = typename argument<0, T...>::type;

                return accumulate<std::size_t, typename std::conditional<sizeof(T) == 0, T, std::size_t>::type...>::add(0, (sizeof(T_Head) == sizeof(T) ? 1 : 0)...) == sizeof...(T);
            }

            static constexpr bool is_convertible()
            {
                static_assert(sizeof...(T) > 0, "error: empty parameter pack");

                using T_Head = typename argument<0, T...>::type;

                return accumulate<std::size_t, typename std::conditional<sizeof(T) == 0, T, std::size_t>::type...>::add(0, (std::is_convertible<T_Head, T>::value ? 1 : 0)...) == sizeof...(T);
            }

            static constexpr bool is_const()
            {
                static_assert(sizeof...(T) > 0, "error: empty parameter pack");

                return accumulate<std::size_t, typename std::conditional<sizeof(T) == 0, T, std::size_t>::type...>::add(0, (std::is_const<T>::value ? 1 : 0)...) == sizeof...(T);
            }

            static constexpr std::size_t size_of_largest_type()
            {
                static_assert(sizeof...(T) > 0, "error: empty parameter pack");

                return accumulate<std::size_t, typename std::conditional<sizeof(T) == 0, T, std::size_t>::type...>::max(0, sizeof(T)...);
            }

            static constexpr std::size_t size_of_pack()
            {
                static_assert(sizeof...(T) > 0, "error: empty parameter pack");

                return accumulate<std::size_t, typename std::conditional<sizeof(T) == 0, T, std::size_t>::type...>::add(0, sizeof(T)...);
            }

            static constexpr std::size_t size_of_pack_excluding_largest_type()
            {
                static_assert(sizeof...(T) > 0, "error: empty parameter pack");

                constexpr std::size_t size_largest_type = size_of_largest_type();

                return accumulate<std::size_t, typename std::conditional<sizeof(T) == 0, T, std::size_t>::type...>::add(0, (sizeof(T) == size_largest_type ? 0 : sizeof(T))...);
            }
        };

        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        //! \brief Fold function
        //!
        //! \tparam F function type (must be constexpr: C++17)
        //! \tparam A aggregate type  
        //! \tparam T_head data type (head)
        //! \tparam T_tail variadic argument list (tail)
        //! \param op lambda to be used for folding
        //! \param aggregate 
        //! \param head frontmost argument
        //! \param tail remaining arguments
        //! \return aggregate
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        template <typename F, typename A>
        constexpr A fold(F op, A aggregate)
        {
            return aggregate;
        }

        template <typename F, typename A, typename T_head, typename ...T_tail>
        constexpr A fold(F op, A aggregate, T_head head, T_tail... tail)
        {
            return fold(op, op(aggregate, head), tail...);
        }        

        /////////////////////////////////////////////////////////////////
        // compile time loop
        //
        // NOTE: loop<begin, end>::execute([..](auto& I) { constexpr std::size_t i = I.value; body(i[,..]); }) 
        //       corresponds to
        //       
        //       for (std::size_t i = begin; i <= end; ++i)
        //         body(i[,..]);
        /////////////////////////////////////////////////////////////////
        namespace
        {
            template <typename T, T X>
            struct template_parameter_t
            {
                static constexpr T value = X;
            };

            template <std::size_t I, std::size_t Shift>
            struct loop_implementation
            {
                template <typename F>
                static void execute(F body)
                {
                    loop_implementation<I - 1, Shift>::execute(body);

                    template_parameter_t<std::size_t, Shift + I> i; 
                    body(i);
                }
            };

            template <std::size_t Shift>
            struct loop_implementation<0, Shift>
            {
                template <typename F>
                static void execute(F body)
                {
                    template_parameter_t<std::size_t, Shift> i; 
                    body(i);
                }
            };
        }

        template <std::size_t A, std::size_t B = 0>
        struct loop
        {
            static constexpr std::size_t begin = (B == 0 ? 0 : A);
            static constexpr std::size_t end = (B == 0 ? A : B);

            static_assert(end > begin, "error: end <= begin");

            template <typename F>
            static void execute(F body)
            {
                loop_implementation<(end - begin - 1), begin>::execute(body);
            }
        };

        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        //! \brief Generate a variadic type with up to 'FW_SEQ_MAX_N' template parameters
        //!
        //! Remove from a parameter argument (T_front..., T_tail...) 'M = |T_front...| = (FW_SEQ_MAX_N - N) >= 0'
        //! parameters in a recursive way, and take the remaining parameter argument (T_tail...) of length 'N' to 
        //! define the type TYPE_NAME<T_tail...>.
        //!
        //! \param TYPE_NAME name of the variadic type to be generated
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
    #define MACRO_VARIADIC_TYPE_GEN(TYPE_NAME)                                          \
        template <std::size_t N, typename T_head, typename ...T_tail>                   \
        struct type_gen_impl                                                            \
        {                                                                               \
            using type = typename type_gen_impl<N - 1, T_tail...>::type;                \
        };                                                                              \
                                                                                        \
        template <typename T_head, typename ...T_tail>                                  \
        struct type_gen_impl<0, T_head, T_tail...>                                      \
        {                                                                               \
            using type = TYPE_NAME<T_head, T_tail...>;                                  \
        };                                                                              \
                                                                                        \
        template <typename T, std::size_t N>                                            \
        struct type_gen                                                                 \
        {                                                                               \
            static_assert(N > 0, "error: no template arguments specified");             \
            static_assert(N <= FW_SEQ_MAX_N, "error: not implemented");                 \
                                                                                        \
            using type = typename type_gen_impl<FW_SEQ_MAX_N - N, FW_SEQ_N(T)>::type;   \
        };
    }
}

#endif