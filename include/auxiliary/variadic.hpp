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
        template <typename ... T>
        constexpr std::size_t get_num_arguments()
        {
            return sizeof ... (T);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        //! \brief Get the value of the N-th argument and its type in the variadic list
        //!
        //! \tparam N element to access
        //! \tparam T_head data type (head)
        //! \tparam T_tail variadic argument list (tail)
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        template <std::size_t N, typename T_head, typename ... T_tail>
        struct argument
        {
            //! Data type of the N-th argument
            using type = typename argument<N - 1, T_tail ...>::type;

            //! \brief Get the value of the N-th argument through recursion
            //!
            //! \param head frontmost argument
            //! \param tail remaining arguments
            //! \return argument list without the frontmost argument
            static constexpr T_head value(T_head head, T_tail ... tail)
            {
                return argument<N - 1, T_tail ...>::value(tail ...);
            }
        };

        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        //! \brief N = 0 Specialization (recursion anchor definition)
        //!
        //! \tparam T_head data type of the N-th element
        //! \tparam T_tail variadic argument list (tail)
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        template <typename T_head, typename ... T_tail>
        struct argument<0, T_head, T_tail ...>
        {
            //! Data type of the N-the element
            using type = T_head;

            //! \brief Value of the N-th argument
            //!
            //! \param head N-the argument
            //! \param tail remaining arguments
            //! \return N-the argument
            static constexpr T_head value(T_head head, T_tail ... tail)
            {
                return head;
            }
        };

        /////////////////////////////////////////////////////////////////
        // compile time loop
        //
        // NOTE: loop<begin, end>::execute([..](auto& I) { constexpr std::size_t i = I.value; body(i[,..]); }) 
        //       corresponds to
        //       
        //       for (std::size_t i = begin; i < end; ++i)
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

        template <typename F, typename A, typename T_head, typename ... T_tail>
        constexpr A fold(F op, A aggregate, T_head head, T_tail ... tail)
        {
            return fold(op, op(aggregate, head), tail ...);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        //! \brief Generate a variadic type with up to 'FW_SEQ_MAX_N' template parameters
        //!
        //! Remove from a parameter pack (T_front ..., T_tail ...) 'M = |T_front ...| = (FW_SEQ_MAX_N - N) >= 0'
        //! parameters in a recursive way, and take the remaining parameter pack (T_tail ...) of length 'N' to 
        //! define the type TYPE_NAME<T_tail...>.
        //!
        //! \param TYPE_NAME name of the variadic type to be generated
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
    #define MACRO_VARIADIC_TYPE_GEN(TYPE_NAME)                                          \
        template <std::size_t N, typename T_head, typename ... T_tail>                  \
        struct type_gen_impl                                                            \
        {                                                                               \
            using type = typename type_gen_impl<N - 1, T_tail ...>::type;               \
        };                                                                              \
                                                                                        \
        template <typename T_head, typename ... T_tail>                                 \
        struct type_gen_impl<0, T_head, T_tail ...>                                     \
        {                                                                               \
            using type = TYPE_NAME<T_head, T_tail ...>;                                 \
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