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

auto loop_1d = [] (auto& field, auto&& loop_body) 
    {
        for (SizeT x = 0; x < field.Size(0); ++x) 
        { 
            loop_body(field[x]);
        } 
    };

auto loop_2d = [] (auto& field, auto&& loop_body) 
    {
        for (SizeT y = 0; y < field.Size(1); ++y) 
        {
            for (SizeT x = 0; x < field.Size(0); ++x) 
            { 
                loop_body(field[y][x]);
            } 
        }
    };

auto loop_3d = [] (auto& field, auto&& loop_body) 
    {
        for (SizeT z = 0; z < field.Size(2); ++z) 
        {
            for (SizeT y = 0; y < field.Size(1); ++y) 
            {
                for (SizeT x = 0; x < field.Size(0); ++x) 
                { 
                    loop_body(field[z][y][x]);
                } 
            }
        }
    };    

TEST(Field, aos_1d_default)
{
    using ElementT = Tuple<int, float, short>;

    Field<ElementT, 1, DataLayout::AoS> field;
    EXPECT_EQ(0, field.Size());

    field.Resize({1011});
    EXPECT_EQ(1011, field.Size());
    EXPECT_EQ(true, [&field]() { 
        bool all_zero = true;
        loop_1d(field, [&all_zero](auto&& item) { if (item != 0) all_zero = false; });
        return all_zero;
    } ());
}

TEST(Field, aos_2d_default)
{
    using ElementT = Tuple<int, float, short>;

    Field<ElementT, 2, DataLayout::AoS> field;
    EXPECT_EQ(SizeArray<2>(0), field.Size());

    field.Resize({133, 45});
    EXPECT_EQ(SizeArray<2>(133, 45), field.Size());
    EXPECT_EQ(true, [&field]() { 
        bool all_zero = true;
        loop_2d(field, [&all_zero](auto&& item) { if (item != 0) all_zero = false; });
        return all_zero;
    } ());
}

TEST(Field, aos_3d_default)
{
    using ElementT = Tuple<int, float, short>;

    Field<ElementT, 3, DataLayout::AoS> field;
    EXPECT_EQ(SizeArray<3>(0), field.Size());

    field.Resize({133, 21, 34});
    EXPECT_EQ(SizeArray<3>(133, 21, 34), field.Size());
    EXPECT_EQ(true, [&field]() { 
        bool all_zero = true;
        loop_3d(field, [&all_zero](auto&& item) { if (item != 0) all_zero = false; });
        return all_zero;
    } ());
}

TEST(Field, aos_1d_assign_1dindex_as_value)
{
    using ElementT = Tuple<int, float, short>;

    Field<ElementT, 1, DataLayout::AoS> field;
    EXPECT_EQ(0, field.Size());

    field.Resize({1011});
    EXPECT_EQ(1011, field.Size());
    SizeT value = 0;
    field.Set([&value](auto) { SizeT a = value++, b = value++, c = value++; return ElementT(a, b, c); });
    EXPECT_EQ(true, [&field]() { 
        bool all_values_correct = true;
        SizeT index = 0;
        loop_1d(field, [&all_values_correct, &index](auto&& item) { if (item != ElementT(3 * index + 0, 3 * index + 1, 3 * index + 2)) all_values_correct = false; ++index; });
        return all_values_correct;
    } ());

    // Check data in memory
    EXPECT_EQ(true, [&field]() {
        const ElementT *ptr = field.GetBasePointer();
        for (SizeT i = 0; i < field.Size().ReduceMul(); ++i)
        {
            if (ptr[i] != ElementT(3 * i + 0, 3 * i + 1, 3 * i + 2))
            {
                return false;
            }
        }
        return true;
    } ());
}

TEST(Field, aos_2d_assign_1dindex_as_value)
{
    using ElementT = Tuple<int, float, short>;

    Field<ElementT, 2, DataLayout::AoS> field;
    EXPECT_EQ(0, field.Size());

    field.Resize({254, 33});
    EXPECT_EQ(SizeArray<2>(254, 33), field.Size());
    SizeT value = 0;
    field.Set([&value](auto) { SizeT a = value++, b = value++, c = value++; return ElementT(a, b, c); });
    EXPECT_EQ(true, [&field]() { 
        bool all_values_correct = true;
        SizeT index = 0;
        loop_2d(field, [&all_values_correct, &index](auto&& item) { if (item != ElementT(3 * index + 0, 3 * index + 1, 3 * index + 2)) all_values_correct = false; ++index; });
        return all_values_correct;
    } ());

    // Check data in memory
    EXPECT_EQ(true, [&field]() {
        const ElementT *ptr = field.GetBasePointer();
        for (SizeT stab_index = 0; stab_index < field.Size().ReduceMul(1); ++stab_index)
        {
            for (SizeT x = 0; x < field.Size(0); ++x)
            {
                const SizeT index = stab_index * field.Size(0) + x;
                const SizeT i = stab_index * field.Pitch() + x;
                if (ptr[i] != ElementT(3 * index + 0, 3 * index + 1, 3 * index + 2))
                {
                    return false;
                }
            }
        }
        return true;
    } ());
}

TEST(Field, aos_3d_assign_1dindex_as_value)
{
    using ElementT = Tuple<int, float, short>;

    Field<ElementT, 3, DataLayout::AoS> field;
    EXPECT_EQ(0, field.Size());

    field.Resize({254, 33, 52});
    EXPECT_EQ(SizeArray<3>(254, 33, 52), field.Size());
    SizeT value = 0;
    field.Set([&value](auto) { SizeT a = value++, b = value++, c = value++; return ElementT(a, b, c); });
    EXPECT_EQ(true, [&field]() { 
        bool all_values_correct = true;
        SizeT index = 0;
        loop_3d(field, [&all_values_correct, &index](auto&& item) { if (item != ElementT(3 * index + 0, 3 * index + 1, 3 * index + 2)) all_values_correct = false; ++index; });
        return all_values_correct;
    } ());

    // Check data in memory
    EXPECT_EQ(true, [&field]() {
        const ElementT *ptr = field.GetBasePointer();
        for (SizeT stab_index = 0; stab_index < field.Size().ReduceMul(1); ++stab_index)
        {
            for (SizeT x = 0; x < field.Size(0); ++x)
            {
                const SizeT index = stab_index * field.Size(0) + x;
                const SizeT i = stab_index * field.Pitch() + x;
                if (ptr[i] != ElementT(3 * index + 0, 3 * index + 1, 3 * index + 2))
                {
                    return false;
                }
            }
        }
        return true;
    } ());
}

TEST(Field, soa_soai_1d_assign_1dindex_as_value)
{
    using ElementT = Tuple<int, float, short>;

    Field<ElementT, 1, DataLayout::SoA> field;
    EXPECT_EQ(0, field.Size());

    field.Resize({1011});
    EXPECT_EQ(1011, field.Size());
    SizeT value = 0;
    field.Set([&value](auto) { SizeT a = value++, b = value++, c = value++; return ElementT(a, b, c); });
    EXPECT_EQ(true, [&field]() { 
        bool all_values_correct = true;
        SizeT index = 0;
        loop_1d(field, [&all_values_correct, &index](auto&& item) { if (item != ElementT(3 * index + 0, 3 * index + 1, 3 * index + 2)) all_values_correct = false; ++index; });
        return all_values_correct;
    } ());

    // Check data in memory
    EXPECT_EQ(true, [&field]() {
        const char* stab_ptr = reinterpret_cast<const char*>(field.GetBasePointer());
        const int *i_ptr = reinterpret_cast<const int*>(stab_ptr);
        const float *f_ptr = reinterpret_cast<const float*>(stab_ptr + field.Pitch() * sizeof(int));
        const short *s_ptr = reinterpret_cast<const short*>(stab_ptr + field.Pitch() * (sizeof(int) + sizeof(float)));
        for (SizeT i = 0; i < field.Size().ReduceMul(); ++i)
        {
            if (i_ptr[i] != static_cast<int>(3 * i) || f_ptr[i] != static_cast<float>(3 * i + 1) || s_ptr[i] != static_cast<short>(3 * i + 2))
            {
                return false;
            }
        }
        return true;
    } ());
}

TEST(Field, soa_2d_assign_1dindex_as_value)
{
    using ElementT = Tuple<int, float, short>;

    Field<ElementT, 2, DataLayout::SoA> field;
    EXPECT_EQ(0, field.Size());

    field.Resize({254, 33});
    EXPECT_EQ(SizeArray<2>(254, 33), field.Size());
    SizeT value = 0;
    field.Set([&value](auto) { SizeT a = value++, b = value++, c = value++; return ElementT(a, b, c); });
    EXPECT_EQ(true, [&field]() { 
        bool all_values_correct = true;
        SizeT index = 0;
        loop_2d(field, [&all_values_correct, &index](auto&& item) { if (item != ElementT(3 * index + 0, 3 * index + 1, 3 * index + 2)) all_values_correct = false; ++index; });
        return all_values_correct;
    } ());

    // Check data in memory
    EXPECT_EQ(true, [&field]() {
        const char* stab_ptr = reinterpret_cast<const char*>(field.GetBasePointer());
        const int *i_ptr = reinterpret_cast<const int*>(stab_ptr);
        const float *f_ptr = reinterpret_cast<const float*>(stab_ptr + field.Pitch() * sizeof(int));
        const short *s_ptr = reinterpret_cast<const short*>(stab_ptr + field.Pitch() * (sizeof(int) + sizeof(float)));
        for (SizeT i = 0; i < field.Size().ReduceMul(); ++i)
        {
            if (i_ptr[i] != static_cast<int>(3 * i) || f_ptr[i] != static_cast<float>(3 * i + 1) || s_ptr[i] != static_cast<short>(3 * i + 2))
            {
                return false;
            }
        }
        return true;
    } ());
}

TEST(Field, soai_2d_assign_1dindex_as_value)
{
    using ElementT = Tuple<int, float, short>;

    Field<ElementT, 2, DataLayout::SoAi> field;
    EXPECT_EQ(0, field.Size());

    field.Resize({254, 33});
    EXPECT_EQ(SizeArray<2>(254, 33), field.Size());
    SizeT value = 0;
    field.Set([&value](auto) { SizeT a = value++, b = value++, c = value++; return ElementT(a, b, c); });
    EXPECT_EQ(true, [&field]() { 
        bool all_values_correct = true;
        SizeT index = 0;
        loop_2d(field, [&all_values_correct, &index](auto&& item) { if (item != ElementT(3 * index + 0, 3 * index + 1, 3 * index + 2)) all_values_correct = false; ++index; });
        return all_values_correct;
    } ());

    // Check data in memory
    EXPECT_EQ(true, [&field]() {
        for (SizeT stab_index = 0; stab_index < field.Size().ReduceMul(1); ++stab_index)
        {
            const char* stab_ptr = reinterpret_cast<const char*>(field.GetBasePointer()) + stab_index * field.Pitch() * (sizeof(int) + sizeof(float) + sizeof(short));
            const int *i_ptr = reinterpret_cast<const int*>(stab_ptr);
            const float *f_ptr = reinterpret_cast<const float*>(stab_ptr + field.Pitch() * sizeof(int));
            const short *s_ptr = reinterpret_cast<const short*>(stab_ptr + field.Pitch() * (sizeof(int) + sizeof(float)));
            for (SizeT i = 0; i < field.Size(0); ++i)
            {
                const SizeT index = stab_index * field.Size(0) + i;
                if (i_ptr[i] != static_cast<int>(3 * index) || f_ptr[i] != static_cast<float>(3 * index + 1) || s_ptr[i] != static_cast<short>(3 * index + 2))
                {
                    return false;
                }
            }
        }
        return true;
    } ());
}

TEST(Field, soa_3d_assign_1dindex_as_value)
{
    using ElementT = Tuple<int, float, short>;

    Field<ElementT, 3, DataLayout::SoA> field;
    EXPECT_EQ(0, field.Size());

    field.Resize({254, 33, 52});
    EXPECT_EQ(SizeArray<3>(254, 33, 52), field.Size());
    SizeT value = 0;
    field.Set([&value](auto) { SizeT a = value++, b = value++, c = value++; return ElementT(a, b, c); });
    EXPECT_EQ(true, [&field]() { 
        bool all_values_correct = true;
        SizeT index = 0;
        loop_3d(field, [&all_values_correct, &index](auto&& item) { if (item != ElementT(3 * index + 0, 3 * index + 1, 3 * index + 2)) all_values_correct = false; ++index; });
        return all_values_correct;
    } ());

    // Check data in memory
    EXPECT_EQ(true, [&field]() {
        const char* stab_ptr = reinterpret_cast<const char*>(field.GetBasePointer());
        const int *i_ptr = reinterpret_cast<const int*>(stab_ptr);
        const float *f_ptr = reinterpret_cast<const float*>(stab_ptr + field.Pitch() * sizeof(int));
        const short *s_ptr = reinterpret_cast<const short*>(stab_ptr + field.Pitch() * (sizeof(int) + sizeof(float)));
        for (SizeT i = 0; i < field.Size().ReduceMul(); ++i)
        {
            if (i_ptr[i] != static_cast<int>(3 * i) || f_ptr[i] != static_cast<float>(3 * i + 1) || s_ptr[i] != static_cast<short>(3 * i + 2))
            {
                return false;
            }
        }
        return true;
    } ());
}

TEST(Field, soai_3d_assign_1dindex_as_value)
{
    using ElementT = Tuple<int, float, short>;

    Field<ElementT, 3, DataLayout::SoAi> field;
    EXPECT_EQ(0, field.Size());

    field.Resize({254, 33, 52});
    EXPECT_EQ(SizeArray<3>(254, 33, 52), field.Size());
    SizeT value = 0;
    field.Set([&value](auto) { SizeT a = value++, b = value++, c = value++; return ElementT(a, b, c); });
    EXPECT_EQ(true, [&field]() { 
        bool all_values_correct = true;
        SizeT index = 0;
        loop_3d(field, [&all_values_correct, &index](auto&& item) { if (item != ElementT(3 * index + 0, 3 * index + 1, 3 * index + 2)) all_values_correct = false; ++index; });
        return all_values_correct;
    } ());

    // Check data in memory
    EXPECT_EQ(true, [&field]() {
        for (SizeT stab_index = 0; stab_index < field.Size().ReduceMul(1); ++stab_index)
        {
            const char* stab_ptr = reinterpret_cast<const char*>(field.GetBasePointer()) + stab_index * field.Pitch() * (sizeof(int) + sizeof(float) + sizeof(short));
            const int *i_ptr = reinterpret_cast<const int*>(stab_ptr);
            const float *f_ptr = reinterpret_cast<const float*>(stab_ptr + field.Pitch() * sizeof(int));
            const short *s_ptr = reinterpret_cast<const short*>(stab_ptr + field.Pitch() * (sizeof(int) + sizeof(float)));
            for (SizeT i = 0; i < field.Size(0); ++i)
            {
                const SizeT index = stab_index * field.Size(0) + i;
                if (i_ptr[i] != static_cast<int>(3 * index) || f_ptr[i] != static_cast<float>(3 * index + 1) || s_ptr[i] != static_cast<short>(3 * index + 2))
                {
                    return false;
                }
            }
        }
        return true;
    } ());
}

TEST(Field, aosoa_1d_assign_1dindex_as_value)
{
    using ElementT = Tuple<int, float, short>;
    constexpr SizeT BlockingFactor = ::fw::internal::Traits<ElementT, DataLayout::AoSoA>::BlockingFactor;

    Field<ElementT, 1, DataLayout::AoSoA> field;
    EXPECT_EQ(0, field.Size());

    field.Resize({1011});
    EXPECT_EQ(1011, field.Size());
    SizeT value = 0;
    field.Set([&value](auto) { SizeT a = value++, b = value++, c = value++; return ElementT(a, b, c); });
    EXPECT_EQ(true, [&field]() { 
        bool all_values_correct = true;
        SizeT index = 0;
        loop_1d(field, [&all_values_correct, &index](auto&& item) { if (item != ElementT(3 * index + 0, 3 * index + 1, 3 * index + 2)) all_values_correct = false; ++index; });
        return all_values_correct;
    } ());

    // Check data in memory
    EXPECT_EQ(true, [&field]() {
        const SizeT num_stabs = (field.Size(0) + (BlockingFactor - 1)) / BlockingFactor;
        for (SizeT stab_index = 0; stab_index < num_stabs; ++stab_index)
        {
            const char* stab_ptr = reinterpret_cast<const char*>(field.GetBasePointer()) + stab_index * field.Pitch() * (sizeof(int) + sizeof(float) + sizeof(short));
            const int *i_ptr = reinterpret_cast<const int*>(stab_ptr);
            const float *f_ptr = reinterpret_cast<const float*>(stab_ptr + field.Pitch() * sizeof(int));
            const short *s_ptr = reinterpret_cast<const short*>(stab_ptr + field.Pitch() * (sizeof(int) + sizeof(float)));
            for (SizeT i = 0; i < BlockingFactor; ++i)
            {
                const SizeT index = stab_index * BlockingFactor + i;
                if (index >= field.Size().ReduceMul()) continue;    
                if (i_ptr[i] != static_cast<int>(3 * index) || f_ptr[i] != static_cast<float>(3 * index + 1) || s_ptr[i] != static_cast<short>(3 * index + 2))
                {
                    return false;
                }
            }
        }
        return true;
    } ());
}

TEST(Field, aosoa_2d_assign_1dindex_as_value)
{
    using ElementT = Tuple<int, float, short>;
    constexpr SizeT BlockingFactor = ::fw::internal::Traits<ElementT, DataLayout::AoSoA>::BlockingFactor;
    constexpr SizeT PaddingFactor = ::fw::internal::Traits<ElementT, DataLayout::AoSoA>::PaddingFactor;
    constexpr SizeT Pitch = ((BlockingFactor + (PaddingFactor - 1)) / PaddingFactor) * PaddingFactor;

    Field<ElementT, 2, DataLayout::AoSoA> field;
    EXPECT_EQ(0, field.Size());

    field.Resize({312, 43});
    EXPECT_EQ(SizeArray<2>(312, 43), field.Size());
    SizeT value = 0;
    field.Set([&value](auto) { SizeT a = value++, b = value++, c = value++; return ElementT(a, b, c); });
    EXPECT_EQ(true, [&field]() { 
        bool all_values_correct = true;
        SizeT index = 0;
        loop_2d(field, [&all_values_correct, &index](auto&& item) { if (item != ElementT(3 * index + 0, 3 * index + 1, 3 * index + 2)) all_values_correct = false; ++index; });
        return all_values_correct;
    } ());

    // Check data in memory
    EXPECT_EQ(true, [&field]() {
        const SizeT num_stabs = (field.Size().ReduceMul() + (BlockingFactor - 1)) / BlockingFactor;
        const SizeT num_inner_stabs = (field.Size(0) + (BlockingFactor - 1)) / BlockingFactor;
        for (SizeT n_1 = 0, stab_index = 0; n_1 < num_stabs; n_1 += num_inner_stabs, ++stab_index)
        {
            for (SizeT n_0 = 0; n_0 < num_inner_stabs; ++n_0)
            {
                const char* stab_ptr = reinterpret_cast<const char*>(field.GetBasePointer()) + (n_1 + n_0) * Pitch * (sizeof(int) + sizeof(float) + sizeof(short));
                const int *i_ptr = reinterpret_cast<const int*>(stab_ptr);
                const float *f_ptr = reinterpret_cast<const float*>(stab_ptr + Pitch * sizeof(int));
                const short *s_ptr = reinterpret_cast<const short*>(stab_ptr + Pitch * (sizeof(int) + sizeof(float)));
                for (SizeT i = 0; i < BlockingFactor; ++i)
                {
                    const SizeT intra_stab_index = n_0 * BlockingFactor + i;
                    if (intra_stab_index >= field.Size(0)) continue;
                    const SizeT index = stab_index * field.Size(0) + intra_stab_index;
                    if (i_ptr[i] != static_cast<int>(3 * index) || f_ptr[i] != static_cast<float>(3 * index + 1) || s_ptr[i] != static_cast<short>(3 * index + 2))
                    {
                        return false;
                    }
                }
            }
        }
        return true;
    } ());
}

TEST(Field, aosoa_3d_assign_1dindex_as_value)
{
    using ElementT = Tuple<int, float, short>;
    constexpr SizeT BlockingFactor = ::fw::internal::Traits<ElementT, DataLayout::AoSoA>::BlockingFactor;
    constexpr SizeT PaddingFactor = ::fw::internal::Traits<ElementT, DataLayout::AoSoA>::PaddingFactor;
    constexpr SizeT Pitch = ((BlockingFactor + (PaddingFactor - 1)) / PaddingFactor) * PaddingFactor;

    Field<ElementT, 3, DataLayout::AoSoA> field;
    EXPECT_EQ(0, field.Size());

    field.Resize({254, 33, 52});
    EXPECT_EQ(SizeArray<3>(254, 33, 52), field.Size());
    SizeT value = 0;
    field.Set([&value](auto) { SizeT a = value++, b = value++, c = value++; return ElementT(a, b, c); });
    EXPECT_EQ(true, [&field]() { 
        bool all_values_correct = true;
        SizeT index = 0;
        loop_3d(field, [&all_values_correct, &index](auto&& item) { if (item != ElementT(3 * index + 0, 3 * index + 1, 3 * index + 2)) all_values_correct = false; ++index; });
        return all_values_correct;
    } ());

    // Check data in memory
    EXPECT_EQ(true, [&field]() {
        const SizeT num_stabs = (field.Size().ReduceMul() + (BlockingFactor - 1)) / BlockingFactor;
        const SizeT num_inner_stabs = (field.Size(0) + (BlockingFactor - 1)) / BlockingFactor;
        for (SizeT n_1 = 0, stab_index = 0; n_1 < num_stabs; n_1 += num_inner_stabs, ++stab_index)
        {
            for (SizeT n_0 = 0; n_0 < num_inner_stabs; ++n_0)
            {
                const char* stab_ptr = reinterpret_cast<const char*>(field.GetBasePointer()) + (n_1 + n_0) * Pitch * (sizeof(int) + sizeof(float) + sizeof(short));
                const int *i_ptr = reinterpret_cast<const int*>(stab_ptr);
                const float *f_ptr = reinterpret_cast<const float*>(stab_ptr + Pitch * sizeof(int));
                const short *s_ptr = reinterpret_cast<const short*>(stab_ptr + Pitch * (sizeof(int) + sizeof(float)));
                for (SizeT i = 0; i < BlockingFactor; ++i)
                {
                    const SizeT intra_stab_index = n_0 * BlockingFactor + i;
                    if (intra_stab_index >= field.Size(0)) continue;
                    const SizeT index = stab_index * field.Size(0) + intra_stab_index;
                    if (i_ptr[i] != static_cast<int>(3 * index) || f_ptr[i] != static_cast<float>(3 * index + 1) || s_ptr[i] != static_cast<short>(3 * index + 2))
                    {
                        return false;
                    }
                }
            }
        }
        return true;
    } ());
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}