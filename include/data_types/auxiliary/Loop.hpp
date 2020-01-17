// Copyright (c) 2020 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(AUXILIARY_LOOP_HPP)
#define AUXILIARY_LOOP_HPP

#include <type_traits>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#include <auxiliary/CPPStandard.hpp>
#include <auxiliary/Function.hpp>
#include <platform/Target.hpp>
#include <DataTypes.hpp>

namespace XXX_NAMESPACE
{
    namespace compileTime
    {
        using SizeT = ::XXX_NAMESPACE::dataTypes::SizeT;
        using ::XXX_NAMESPACE::variadic::IsInvocable;

        namespace
        {
            //! Recursive definition of a (forward) loop.
            //!
            //! \tparam I value of the iteration variable
            //! \tparam LoopBegin lower loop bound
            //!
            template <SizeT I, SizeT LoopBegin>
            struct LoopImplementation
            {
                //!
                //! The recursion must happen before the loop body invocation.
                //!
                template <typename LoopBody>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                static inline constexpr auto Execute(LoopBody loop_body) -> void
                {
                    LoopImplementation<I - 1, LoopBegin>::Execute(loop_body);

                    static_assert(IsInvocable<LoopBody, std::integral_constant<SizeT, LoopBegin + I>>::value, "error: callable is not invocable. void (*) (std::integral_constant<SizeT,..>) expected.");

                    loop_body(std::integral_constant<SizeT, LoopBegin + I>{});
                }
            };

            //!
            //! Recursion anchor: first iteration of the loop.
            //!
            //! \tparam LoopBegin lower loop bound
            //!
            template <SizeT LoopBegin>
            struct LoopImplementation<0, LoopBegin>
            {
                //!
                //! The recursion stops here: loop body invocation.
                //!
                template <typename LoopBody>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                static inline constexpr auto Execute(LoopBody loop_body) -> void
                {
                    static_assert(IsInvocable<LoopBody, std::integral_constant<SizeT, LoopBegin>>::value, "error: callable is not invocable. void (*) (std::integral_constant<SizeT,..>) expected.");

                    loop_body(std::integral_constant<SizeT, LoopBegin>{});
                }
            };
        } // namespace

        //!
        //! \brief A compile-time loop.
        //!
        //! ```
        //! Loop<LoopBegin, LoopEnd>::Execute([..] (const auto I) { loop_body(I); })
        //! ```
        //! corresponds to
        //! ```
        //!    for (SizeT I = LoopBegin; I < LoopEnd; ++I)
        //!        loop_body(I);
        //! ```
        //!
        //! \tparam LoopBegin lower loop bound
        //! \tparam LoopEnd upper loop bound
        //!
        template <SizeT LoopBegin, SizeT LoopEnd = 0>
        struct Loop
        {
            static constexpr SizeT Begin = (LoopEnd == 0 ? 0 : LoopBegin);
            static constexpr SizeT End = (LoopEnd == 0 ? LoopBegin : LoopEnd);

            static_assert(Begin < End, "error: End <= Begin");

            //!
            //! \tparam LoopBody the type of a callable (loop body)
            //! \param loop_body the loop body
            //!
            template <typename LoopBody>
            HOST_VERSION
            CUDA_DEVICE_VERSION
            static inline constexpr auto Execute(LoopBody loop_body) -> void
            {
                LoopImplementation<(End - Begin - 1), Begin>::Execute(loop_body);
            }
        };
    } // namespace compileTime
} // namespace XXX_NAMESPACE

#endif