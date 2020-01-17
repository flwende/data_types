// Copyright (c) 2020 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#include <iostream>
#include <memory>
#include <gtest/gtest.h>

#include <tuple/Tuple.hpp>

using namespace ::fw::dataTypes;

TEST(Tuple, empty_compile_test)
{
    [[maybe_unused]] Tuple<> tuple;
}

TEST(Tuple, with_default_initialization)
{
    {
        Tuple<int> tuple;

        EXPECT_EQ(0, Get<0>(tuple));
    }
    {
        Tuple<int, float, short> tuple;
        
        EXPECT_EQ(0, Get<0>(tuple));
        EXPECT_EQ(0, Get<1>(tuple));
        EXPECT_EQ(0, Get<2>(tuple));
    }
}

TEST(Tuple, single_argument_initialization)
{
    {
        Tuple<int> tuple(2);

        EXPECT_EQ(2, Get<0>(tuple));
    }
    {
        Tuple<int, float, long> tuple(2);
        
        EXPECT_EQ(2, Get<0>(tuple));
        EXPECT_EQ(2, Get<1>(tuple));
        EXPECT_EQ(2, Get<2>(tuple));
    }
}

TEST(Tuple, multi_argument_initialization)
{
    Tuple<int, float, long> tuple(2, 3.1f, 4);
    
    EXPECT_EQ(2, Get<0>(tuple));
    EXPECT_EQ(3.1f, Get<1>(tuple));
    EXPECT_EQ(4, Get<2>(tuple));
}

TEST(Tuple, record_argument_initialization)
{
    {
        Tuple<int, float, long> tuple(internal::Record<int, float, long>{2, 3.1f, 4});
        
        EXPECT_EQ(2, Get<0>(tuple));
        EXPECT_EQ(3.1f, Get<1>(tuple));
        EXPECT_EQ(4, Get<2>(tuple));
    }
    {
        Tuple<int, float, long> tuple(internal::Record<int, float>{2, 3.1f});
        
        EXPECT_EQ(2, Get<0>(tuple));
        EXPECT_EQ(3.1f, Get<1>(tuple));
        EXPECT_EQ(0, Get<2>(tuple));
    }
}

TEST(Tuple, copy_initialization)
{
    Tuple<int, float, long> record_1(2, 3.1f, 4);    
    Tuple<int, float, long> record_2(record_1);

    EXPECT_EQ(2, Get<0>(record_2));
    EXPECT_EQ(3.1f, Get<1>(record_2));
    EXPECT_EQ(4, Get<2>(record_2));
}

TEST(Tuple, copy_initialization_with_same_type)
{
    Tuple<double, int, float> record_1(2.3, 3, -4.1f);
    Tuple<double, int, float> record_2(record_1);
    
    EXPECT_EQ(2.3, Get<0>(record_2));
    EXPECT_EQ(3, Get<1>(record_2));
    EXPECT_EQ(-4.1f, Get<2>(record_2));
}

TEST(Tuple, move_initialization_with_same_type)
{
    Tuple<double, int, float> record_1(2.3, 3, -4.1f);
    Tuple<double, int, float> record_2(std::move(record_1));
    
    EXPECT_EQ(2.3, Get<0>(record_2));
    EXPECT_EQ(3, Get<1>(record_2));
    EXPECT_EQ(-4.1f, Get<2>(record_2));
}

TEST(Tuple, copy_initialization_with_other_type)
{
    Tuple<double, int, float> record_1(2.3, 3, -4.1f);
    Tuple<double, long, double> record_2(record_1);
    
    EXPECT_EQ(2.3, Get<0>(record_2));
    EXPECT_EQ(3, Get<1>(record_2));
    EXPECT_EQ(-4.1f, Get<2>(record_2));
}

TEST(Tuple, move_initialization_with_other_type)
{
    Tuple<double, int, float> record_1(2.3, 3, -4.1f);
    Tuple<double, long, double> record_2(std::move(record_1));
    
    EXPECT_EQ(2.3, Get<0>(record_2));
    EXPECT_EQ(3, Get<1>(record_2));
    EXPECT_EQ(-4.1f, Get<2>(record_2));
}

TEST(Tuple, copy_assignment_with_other_type)
{
    Tuple<double, int, float> record_1(2.3, 3, -4.1f);
    Tuple<double, long, double> record_2;
    
    record_2 = record_1;
    
    EXPECT_EQ(2.3, Get<0>(record_2));
    EXPECT_EQ(3, Get<1>(record_2));
    EXPECT_EQ(-4.1f, Get<2>(record_2));
}

TEST(Tuple, move_assignment_with_other_type)
{
    Tuple<double, int, float> record_1(2.3, 3, -4.1f);
    Tuple<double, long, double> record_2;
    
    record_2 = std::move(record_1);
    
    EXPECT_EQ(2.3, Get<0>(record_2));
    EXPECT_EQ(3, Get<1>(record_2));
    EXPECT_EQ(-4.1f, Get<2>(record_2));
}

TEST(Tuple, constexpr_empty_compile_test)
{
    [[maybe_unused]] constexpr Tuple<> tuple;
}

TEST(Tuple, constexpr_with_default_initialization)
{
    {
        constexpr Tuple<int> tuple;

        EXPECT_EQ(0, Get<0>(tuple));
    }
    {
        constexpr Tuple<int, float, short> tuple;
        
        EXPECT_EQ(0, Get<0>(tuple));
        EXPECT_EQ(0, Get<1>(tuple));
        EXPECT_EQ(0, Get<2>(tuple));
    }
}

TEST(Tuple, constexpr_single_argument_initialization)
{
    {
        constexpr Tuple<int> tuple(2);

        EXPECT_EQ(2, Get<0>(tuple));
    }
    {
        constexpr Tuple<int, float, long> tuple(2);
        
        EXPECT_EQ(2, Get<0>(tuple));
        EXPECT_EQ(2, Get<1>(tuple));
        EXPECT_EQ(2, Get<2>(tuple));
    }
}

TEST(Tuple, constexpr_multi_argument_initialization)
{
    constexpr Tuple<int, float, long> tuple(2, 3.1f, 4);
    
    EXPECT_EQ(2, Get<0>(tuple));
    EXPECT_EQ(3.1f, Get<1>(tuple));
    EXPECT_EQ(4, Get<2>(tuple));
}

TEST(Tuple, constexpr_copy_initialization)
{
    constexpr Tuple<int, float, long> record_1(2, 3.1f, 4);    
    constexpr Tuple<int, float, long> record_2(record_1);

    EXPECT_EQ(2, Get<0>(record_2));
    EXPECT_EQ(3.1f, Get<1>(record_2));
    EXPECT_EQ(4, Get<2>(record_2));
}

TEST(Tuple, constexpr_copy_initialization_with_same_type)
{
    constexpr Tuple<double, int, float> record_1(2.3, 3, -4.1f);
    constexpr Tuple<double, int, float> record_2(record_1);
    
    EXPECT_EQ(2.3, Get<0>(record_2));
    EXPECT_EQ(3, Get<1>(record_2));
    EXPECT_EQ(-4.1f, Get<2>(record_2));
}

TEST(Tuple, constexpr_copy_initialization_with_other_type)
{
    constexpr Tuple<double, int, float> record_1(2.3, 3, -4.1f);
    constexpr Tuple<double, long, double> record_2(record_1);
    
    EXPECT_EQ(2.3, Get<0>(record_2));
    EXPECT_EQ(3, Get<1>(record_2));
    EXPECT_EQ(-4.1f, Get<2>(record_2));
}

TEST(Tuple, member_access)
{
    {
        Tuple<double> record_1(2.3);
    
        EXPECT_EQ(2.3, record_1.x);
    }
    {
        Tuple<double, int> record_1(2.3, 3);
    
        EXPECT_EQ(2.3, record_1.x);
        EXPECT_EQ(3, record_1.y);
    }
    {
        Tuple<double, int, float> record_1(2.3, 3, -4.1f);
    
        EXPECT_EQ(2.3, record_1.x);
        EXPECT_EQ(3, record_1.y);
        EXPECT_EQ(-4.1f, record_1.z);
    }
}

TEST(Tuple, negate_sign)
{
    Tuple<double, int, float> record_1(2.3, 3, -4.1f);
    Tuple<double, long, double> record_2(-record_1);
    
    EXPECT_EQ(-2.3, Get<0>(record_2));
    EXPECT_EQ(-3, Get<1>(record_2));
    EXPECT_EQ(4.1f, Get<2>(record_2));
}

TEST(Tuple, constexpr_negate_sign)
{
    constexpr Tuple<double, int, float> record_1(2.3, 3, -4.1f);
    constexpr Tuple<double, long, double> record_2(-record_1);
    
    EXPECT_EQ(-2.3, Get<0>(record_2));
    EXPECT_EQ(-3, Get<1>(record_2));
    EXPECT_EQ(4.1f, Get<2>(record_2));
}

TEST(Tuple, add)
{
    using namespace ::fw::math;
    
    Tuple<double, int, float> record_1(0.0, 3, -4.0f);
    Tuple<double, long, double> record_2(-3.2, 12, 101.0);
    auto record_3 = record_1 + record_2;
    
    EXPECT_EQ(-3.2, Get<0>(record_3));
    EXPECT_EQ(15, Get<1>(record_3));
    EXPECT_EQ(97.0, Get<2>(record_3));
}

TEST(Tuple, constexpr_add)
{
    using namespace ::fw::math;
    
    constexpr Tuple<double, int, float> record_1(0.0, 3, -4.0f);
    constexpr Tuple<double, long, double> record_2(-3.2, 12, 101.0);
    constexpr auto record_3 = record_1 + record_2;
    
    EXPECT_EQ(-3.2, Get<0>(record_3));
    EXPECT_EQ(15, Get<1>(record_3));
    EXPECT_EQ(97.0, Get<2>(record_3));
}

TEST(Tuple, add_scalar)
{
    using namespace ::fw::math;
    
    Tuple<double, int, float> record_1(0.0, 3, -4.0f);
    auto record_2 = 2 + record_1 + 4;
    
    EXPECT_EQ(6.0, Get<0>(record_2));
    EXPECT_EQ(9, Get<1>(record_2));
    EXPECT_EQ(2.0, Get<2>(record_2));
}

TEST(Tuple, constexpr_add_scalar)
{
    using namespace ::fw::math;
    
    constexpr Tuple<double, int, float> record_1(0.0, 3, -4.0f);
    constexpr auto record_2 = 2 + record_1 + 4;
    
    EXPECT_EQ(6.0, Get<0>(record_2));
    EXPECT_EQ(9, Get<1>(record_2));
    EXPECT_EQ(2.0, Get<2>(record_2));
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}