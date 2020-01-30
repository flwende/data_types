// Copyright (c) 2020 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#include <iostream>
#include <memory>
#include <gtest/gtest.h>

#include <field/Field.hpp>
#include <tuple/Tuple.hpp>

using namespace ::fw::dataTypes;
using namespace ::fw::memory;

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}