// Copyright (c) 2020 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#include <iostream>
#include <memory>
#include <gtest/gtest.h>

#include <tuple/TupleProxy.hpp>

using namespace ::fw::dataTypes;

TEST(TupleProxy, create_from_record)
{
    int i = 12;
    float f = -3.0f;
    double d = 1024.0;

    internal::Record<int&, float&, double&> references(i, f, d);
    internal::TupleProxy<int, float, double> proxy(references);
    
    EXPECT_EQ(12, Get<0>(proxy));
    EXPECT_EQ(-3.0f, Get<1>(proxy));
    EXPECT_EQ(1024.0, Get<2>(proxy));

    proxy.y = 5.0f;

    EXPECT_EQ(5.0f, f);
}

TEST(TupleProxy, create_from_temporary_record)
{
    int i = 12;
    float f = -3.0f;
    double d = 1024.0;

    internal::TupleProxy<int, float, double> proxy(internal::Record<int&, float&, double&>{i, f, d});
    
    EXPECT_EQ(12, Get<0>(proxy));
    EXPECT_EQ(-3.0f, Get<1>(proxy));
    EXPECT_EQ(1024.0, Get<2>(proxy));

    proxy.y = 5.0f;

    EXPECT_EQ(5.0f, f);
}

TEST(TupleProxy, assign_scalar)
{
    int i = 12;
    float f = -3.0f;
    double d = 1024.0;

    internal::Record<int&, float&, double&> references(i, f, d);
    internal::TupleProxy<int, float, double> proxy(references);

    EXPECT_EQ(12, Get<0>(proxy));
    EXPECT_EQ(-3.0f, Get<1>(proxy));
    EXPECT_EQ(1024.0, Get<2>(proxy));

    proxy = 2;

    EXPECT_EQ(2, Get<0>(proxy));
    EXPECT_EQ(2.0f, Get<1>(proxy));
    EXPECT_EQ(2.0, Get<2>(proxy));
}

TEST(TupleProxy, assign_tuple)
{
    int i = 12;
    float f = -3.0f;
    double d = 1024.0;

    internal::Record<int&, float&, double&> references(i, f, d);
    internal::TupleProxy<int, float, double> proxy(references);

    EXPECT_EQ(12, Get<0>(proxy));
    EXPECT_EQ(-3.0f, Get<1>(proxy));
    EXPECT_EQ(1024.0, Get<2>(proxy));

    proxy = Tuple<int, float, double>{1, 2.0f, 3.0};

    EXPECT_EQ(1, Get<0>(proxy));
    EXPECT_EQ(2.0f, Get<1>(proxy));
    EXPECT_EQ(3.0, Get<2>(proxy));

    Tuple<int, float, double> tuple(3, 4.0f, 2.0);
    proxy = tuple;

    EXPECT_EQ(3, Get<0>(proxy));
    EXPECT_EQ(4.0f, Get<1>(proxy));
    EXPECT_EQ(2.0, Get<2>(proxy));
}

TEST(TupleProxy, assign_other_tuple)
{
    int i = 12;
    float f = -3.0f;
    double d = 1024.0;

    internal::Record<int&, float&, double&> references(i, f, d);
    internal::TupleProxy<int, float, double> proxy(references);

    EXPECT_EQ(12, Get<0>(proxy));
    EXPECT_EQ(-3.0f, Get<1>(proxy));
    EXPECT_EQ(1024.0, Get<2>(proxy));

    proxy = Tuple<float, int, double>{1.0f, 12, 3.0};

    EXPECT_EQ(1, Get<0>(proxy));
    EXPECT_EQ(12.0f, Get<1>(proxy));
    EXPECT_EQ(3.0, Get<2>(proxy));
}

TEST(TupleProxy, assign_proxy)
{
    int i_1 = 12;
    float f_1 = -3.0f;
    double d_1 = 1024.0;

    int i_2 = 2;
    float f_2 = -14.0f;
    double d_2 = 16.0;

    internal::TupleProxy<int, float, double> proxy_1(internal::Record<int&, float&, double&>{i_1, f_1, d_1});
    internal::TupleProxy<int, float, double> proxy_2(internal::Record<int&, float&, double&>{i_2, f_2, d_2});

    EXPECT_EQ(12, Get<0>(proxy_1));
    EXPECT_EQ(-3.0f, Get<1>(proxy_1));
    EXPECT_EQ(1024.0, Get<2>(proxy_1));

    proxy_1 = proxy_2;

    EXPECT_EQ(2, Get<0>(proxy_1));
    EXPECT_EQ(-14.0f, Get<1>(proxy_1));
    EXPECT_EQ(16.0, Get<2>(proxy_1));
}

TEST(TupleProxy, assign_other_proxy)
{
    int i_1 = 12;
    float f_1 = -3.0f;
    double d_1 = 1024.0;

    int i_2 = 2;
    float f_2 = -14.0f;
    double d_2 = 16.0;

    internal::TupleProxy<int, float, double> proxy_1(internal::Record<int&, float&, double&>{i_1, f_1, d_1});
    internal::TupleProxy<const float, const int, const double> proxy_2(internal::Record<float&, int&, double&>{f_2, i_2, d_2});

    EXPECT_EQ(12, Get<0>(proxy_1));
    EXPECT_EQ(-3.0f, Get<1>(proxy_1));
    EXPECT_EQ(1024.0, Get<2>(proxy_1));

    proxy_1 = proxy_2;

    EXPECT_EQ(-14, Get<0>(proxy_1));
    EXPECT_EQ(2.0f, Get<1>(proxy_1));
    EXPECT_EQ(16.0, Get<2>(proxy_1));
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}