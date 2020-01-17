// Copyright (c) 2020 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(__cpp_constexpr)
static_assert(false, "ERROR: (constexpr) C++11 or newer is needed. Sorry.");
#endif

#if !defined(__cpp_generic_lambdas)
static_assert(false, "ERROR: (generic lambdas) C++14 or newer is needed. Sorry.");
#endif