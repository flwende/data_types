// Copyright (c) 2020 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#include <iostream>
#include <memory>
#include <gtest/gtest.h>

#include <tuple/Record.hpp>

using namespace ::fw::dataTypes;

TEST(Record, empty_compile_test)
{
    [[maybe_unused]] internal::Record<> record;
}

TEST(Record, with_default_initialization)
{
    {
        internal::Record<int> record;

        EXPECT_EQ(0, Get<0>(record));
    }
    {
        internal::Record<int, float, short> record;
        
        EXPECT_EQ(0, Get<0>(record));
        EXPECT_EQ(0, Get<1>(record));
        EXPECT_EQ(0, Get<2>(record));
    }
}

TEST(Record, single_argument_initialization)
{
    {
        internal::Record<int> record(2);

        EXPECT_EQ(2, Get<0>(record));
    }
    {
        internal::Record<int, float, long> record(2);
        
        EXPECT_EQ(2, Get<0>(record));
        EXPECT_EQ(2, Get<1>(record));
        EXPECT_EQ(2, Get<2>(record));
    }
}

TEST(Record, multi_argument_initialization)
{
    internal::Record<int, float, long> record(2, 3.1f, 4);
    
    EXPECT_EQ(2, Get<0>(record));
    EXPECT_EQ(3.1f, Get<1>(record));
    EXPECT_EQ(4, Get<2>(record));
}

TEST(Record, copy_initialization)
{
    internal::Record<int, float, long> record_1(2, 3.1f, 4);    
    internal::Record<int, float, long> record_2(record_1);

    EXPECT_EQ(2, Get<0>(record_2));
    EXPECT_EQ(3.1f, Get<1>(record_2));
    EXPECT_EQ(4, Get<2>(record_2));
}

TEST(Record, copy_initialization_with_same_type)
{
    internal::Record<double, int, float> record_1(2.3, 3, -4.1f);
    internal::Record<double, int, float> record_2(record_1);
    
    EXPECT_EQ(2.3, Get<0>(record_2));
    EXPECT_EQ(3, Get<1>(record_2));
    EXPECT_EQ(-4.1f, Get<2>(record_2));
}

TEST(Record, move_initialization_with_same_type)
{
    internal::Record<double, int, float> record_1(2.3, 3, -4.1f);
    internal::Record<double, int, float> record_2(std::move(record_1));
    
    EXPECT_EQ(2.3, Get<0>(record_2));
    EXPECT_EQ(3, Get<1>(record_2));
    EXPECT_EQ(-4.1f, Get<2>(record_2));
}

TEST(Record, copy_initialization_with_other_type)
{
    internal::Record<double, int, float> record_1(2.3, 3, -4.1f);
    internal::Record<double, long, double> record_2(record_1);
    
    EXPECT_EQ(2.3, Get<0>(record_2));
    EXPECT_EQ(3, Get<1>(record_2));
    EXPECT_EQ(-4.1f, Get<2>(record_2));
}

TEST(Record, move_initialization_with_other_type)
{
    internal::Record<double, int, float> record_1(2.3, 3, -4.1f);
    internal::Record<double, long, double> record_2(std::move(record_1));
    
    EXPECT_EQ(2.3, Get<0>(record_2));
    EXPECT_EQ(3, Get<1>(record_2));
    EXPECT_EQ(-4.1f, Get<2>(record_2));
}

TEST(Record, copy_assignment_with_other_type)
{
    internal::Record<double, int, float> record_1(2.3, 3, -4.1f);
    internal::Record<double, long, double> record_2;
    
    record_2 = record_1;
    
    EXPECT_EQ(2.3, Get<0>(record_2));
    EXPECT_EQ(3, Get<1>(record_2));
    EXPECT_EQ(-4.1f, Get<2>(record_2));
}

TEST(Record, move_assignment_with_other_type)
{
    internal::Record<double, int, float> record_1(2.3, 3, -4.1f);
    internal::Record<double, long, double> record_2;
    
    record_2 = std::move(record_1);
    
    EXPECT_EQ(2.3, Get<0>(record_2));
    EXPECT_EQ(3, Get<1>(record_2));
    EXPECT_EQ(-4.1f, Get<2>(record_2));
}

TEST(Record, convert_references)
{
    int i = 1;
    float f = 4.0f;

    internal::Record<int&, float&> record_1(i, f);
    internal::Record<const int&, const float&> record_2(internal::Record<int&, float&>{i, f});

    EXPECT_EQ(1, Get<0>(record_1));
    EXPECT_EQ(4.0f, Get<1>(record_1));

    EXPECT_EQ(1, Get<0>(record_2));
    EXPECT_EQ(4.0f, Get<1>(record_2));
}

TEST(Record, constexpr_empty_compile_test)
{
    [[maybe_unused]] constexpr internal::Record<> record;
}

TEST(Record, constexpr_with_default_initialization)
{
    {
        constexpr internal::Record<int> record;

        EXPECT_EQ(0, Get<0>(record));
    }
    {
        constexpr internal::Record<int, float, short> record;
        
        EXPECT_EQ(0, Get<0>(record));
        EXPECT_EQ(0, Get<1>(record));
        EXPECT_EQ(0, Get<2>(record));
    }
}

TEST(Record, constexpr_single_argument_initialization)
{
    {
        constexpr internal::Record<int> record(2);

        EXPECT_EQ(2, Get<0>(record));
    }
    {
        constexpr internal::Record<int, float, long> record(2);
        
        EXPECT_EQ(2, Get<0>(record));
        EXPECT_EQ(2, Get<1>(record));
        EXPECT_EQ(2, Get<2>(record));
    }
}

TEST(Record, constexpr_multi_argument_initialization)
{
    constexpr internal::Record<int, float, long> record(2, 3.1f, 4);
    
    EXPECT_EQ(2, Get<0>(record));
    EXPECT_EQ(3.1f, Get<1>(record));
    EXPECT_EQ(4, Get<2>(record));
}

TEST(Record, constexpr_copy_initialization)
{
    constexpr internal::Record<int, float, long> record_1(2, 3.1f, 4);    
    constexpr internal::Record<int, float, long> record_2(record_1);

    EXPECT_EQ(2, Get<0>(record_2));
    EXPECT_EQ(3.1f, Get<1>(record_2));
    EXPECT_EQ(4, Get<2>(record_2));
}

TEST(Record, constexpr_copy_initialization_with_same_type)
{
    constexpr internal::Record<double, int, float> record_1(2.3, 3, -4.1f);
    constexpr internal::Record<double, int, float> record_2(record_1);
    
    EXPECT_EQ(2.3, Get<0>(record_2));
    EXPECT_EQ(3, Get<1>(record_2));
    EXPECT_EQ(-4.1f, Get<2>(record_2));
}

TEST(Record, constexpr_copy_initialization_with_other_type)
{
    constexpr internal::Record<double, int, float> record_1(2.3, 3, -4.1f);
    constexpr internal::Record<double, long, double> record_2(record_1);
    
    EXPECT_EQ(2.3, Get<0>(record_2));
    EXPECT_EQ(3, Get<1>(record_2));
    EXPECT_EQ(-4.1f, Get<2>(record_2));
}

TEST(Record, assign_different_types)
{
    internal::Record<double, int, float> record_1(2.3, 3, -4.1f);
    
    record_1 = 1;
    EXPECT_EQ(1, Get<0>(record_1));
    EXPECT_EQ(1, Get<1>(record_1));
    EXPECT_EQ(1, Get<2>(record_1));

    record_1 = {1, 2, 3};
    EXPECT_EQ(1, Get<0>(record_1));
    EXPECT_EQ(2, Get<1>(record_1));
    EXPECT_EQ(3, Get<2>(record_1));

    record_1 = internal::Record<int, int>{5, 6};
    EXPECT_EQ(5, Get<0>(record_1));
    EXPECT_EQ(6, Get<1>(record_1));
    EXPECT_EQ(0, Get<2>(record_1));

    record_1 = internal::Record<int, int, float, double>{5, 6, 12, 5};
    EXPECT_EQ(5, Get<0>(record_1));
    EXPECT_EQ(6, Get<1>(record_1));
    EXPECT_EQ(12, Get<2>(record_1));

    internal::Record<float> record_2(record_1);
    EXPECT_EQ(5, Get<0>(record_2));

    record_2 = 12;
    record_1 = std::move(record_2);
    EXPECT_EQ(12, Get<0>(record_1));
    EXPECT_EQ(0, Get<1>(record_1));
    EXPECT_EQ(0, Get<2>(record_1));
}

TEST(Record, constexpr_assign_different_types)
{
    {
        constexpr internal::Record<double, int, float> record_1(1);
        EXPECT_EQ(1, Get<0>(record_1));
        EXPECT_EQ(1, Get<1>(record_1));
        EXPECT_EQ(1, Get<2>(record_1));
    }
    {
        constexpr internal::Record<double, int, float> record_1 = {1, 2, 3};
        EXPECT_EQ(1, Get<0>(record_1));
        EXPECT_EQ(2, Get<1>(record_1));
        EXPECT_EQ(3, Get<2>(record_1));
    }
    {
        constexpr internal::Record<double, int, float> record_1 = internal::Record<int, int>{5, 6};
        EXPECT_EQ(5, Get<0>(record_1));
        EXPECT_EQ(6, Get<1>(record_1));
        EXPECT_EQ(0, Get<2>(record_1));
    }
    {
        constexpr internal::Record<double, int, float> record_1 = internal::Record<int, int, float, double>{5, 6, 12, 5};
        EXPECT_EQ(5, Get<0>(record_1));
        EXPECT_EQ(6, Get<1>(record_1));
        EXPECT_EQ(12, Get<2>(record_1));
    
        constexpr internal::Record<float> record_2(record_1);
        EXPECT_EQ(5, Get<0>(record_2));
    }
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}