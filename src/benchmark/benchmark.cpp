// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <array>
#include <sched.h>

#include <benchmark.hpp>

constexpr SizeT NX_DEFAULT = 128;
constexpr SizeT NY_DEFAULT = 128;
constexpr SizeT NZ_DEFAULT = 128;

void ApplyPinning();

int main(int argc, char** argv)
{
    // Command line arguments
    SizeT dimension = 0;
    const SizeT nx = (argc > 1 ? atoi(argv[++dimension]) : NX_DEFAULT);
    const SizeT ny = (argc > 2 ? atoi(argv[++dimension]) : NY_DEFAULT);
    const SizeT nz = (argc > 3 ? atoi(argv[++dimension]) : NZ_DEFAULT);

    ApplyPinning();

    if (dimension == 1)
    {
        return benchmark(argc, argv, nx);
    }
    else if (dimension == 2)
    {
        return benchmark(argc, argv, nx, ny);
    }
    else if (dimension == 3)
    {
        return benchmark(argc, argv, nx, ny, nz);
    }

    return 1;
}

void ApplyPinning()
{
    cpu_set_t mask;
    SizeT core_id = 0;

    if (const char* variable = std::getenv("PIN_TO_CORE"))
    {
        core_id = std::atoi(variable);
    }

    std::cout << "# INFO: use cpu core " << core_id << std::endl;

    CPU_ZERO(&mask);
    CPU_SET(core_id, &mask);

    sched_setaffinity(0, sizeof(mask), &mask);
}