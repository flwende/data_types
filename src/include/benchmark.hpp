// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(BENCHMARK_HPP)
#define BENCHMARK_HPP

#include <auxiliary/Template.hpp>
#include <kernel.hpp>

template <SizeT Dimension>
int benchmark(int argc, char** argv, const SizeArray<Dimension>& size);

template <typename ...T>
int benchmark(int argc, char** argv, const T... n)
{
    static_assert(::fw::variadic::Pack<T...>::IsUnsigned(), "error: values must be unsigned integers.");

    return benchmark(argc, argv, SizeArray<sizeof...(T)>{n...});
}

#endif