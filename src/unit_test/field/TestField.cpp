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

using TypeX = int;
using TypeY = float;
using TypeZ = short;

/*
using TypeX = short;
using TypeY = short;
using TypeZ = short;
*/
/*
using TypeX = char;
using TypeY = char;
using TypeZ = char;
*/

using ElementT = Tuple<TypeX, TypeY, TypeZ>;

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

TEST(Field, soa_soai_1d_assign_1dindex_as_value_1)
{
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
        const TypeX *i_ptr = reinterpret_cast<const TypeX*>(stab_ptr);
        const TypeY *f_ptr = reinterpret_cast<const TypeY*>(stab_ptr + field.Pitch() * sizeof(TypeX));
        const TypeZ *s_ptr = reinterpret_cast<const TypeZ*>(stab_ptr + field.Pitch() * (sizeof(TypeX) + sizeof(TypeY)));
        for (SizeT i = 0; i < field.Size().ReduceMul(); ++i)
        {
            if (i_ptr[i] != static_cast<TypeX>(3 * i) || f_ptr[i] != static_cast<TypeY>(3 * i + 1) || s_ptr[i] != static_cast<TypeZ>(3 * i + 2))
            {
                return false;
            }
        }
        return true;
    } ());
}

TEST(Field, soa_soai_1d_assign_1dindex_as_value_2)
{
    using TypeX = float;
    using ElementT = Tuple<TypeX>;

    Field<ElementT, 1, DataLayout::SoA> field;
    EXPECT_EQ(0, field.Size());

    field.Resize({1011});
    EXPECT_EQ(1011, field.Size());
    SizeT value = 0;
    field.Set([&value](auto) { SizeT a = value++; return ElementT(a); });
    EXPECT_EQ(true, [&field]() { 
        bool all_values_correct = true;
        SizeT index = 0;
        loop_1d(field, [&all_values_correct, &index](auto&& item) { if (item != ElementT(index)) all_values_correct = false; ++index; });
        return all_values_correct;
    } ());

    // Check data in memory
    EXPECT_EQ(true, [&field]() {
        const char* stab_ptr = reinterpret_cast<const char*>(field.GetBasePointer());
        const TypeX *f_ptr = reinterpret_cast<const TypeX*>(stab_ptr);
        for (SizeT i = 0; i < field.Size().ReduceMul(); ++i)
        {
            if (f_ptr[i] != static_cast<TypeX>(i))
            {
                return false;
            }
        }
        return true;
    } ());
}

TEST(Field, soa_soai_1d_assign_1dindex_as_value_3)
{
    using TypeX = float;
    using TypeY = short;
    using ElementT = Tuple<TypeX, TypeY>;

    Field<ElementT, 1, DataLayout::SoA> field;
    EXPECT_EQ(0, field.Size());

    field.Resize({1011});
    EXPECT_EQ(1011, field.Size());
    SizeT value = 0;
    field.Set([&value](auto) { SizeT a = value++, b = value++; return ElementT(a, b); });
    EXPECT_EQ(true, [&field]() { 
        bool all_values_correct = true;
        SizeT index = 0;
        loop_1d(field, [&all_values_correct, &index](auto&& item) { if (item != ElementT(2 * index + 0, 2 * index + 1)) all_values_correct = false; ++index; });
        return all_values_correct;
    } ());

    // Check data in memory
    EXPECT_EQ(true, [&field]() {
        const char* stab_ptr = reinterpret_cast<const char*>(field.GetBasePointer());
        const TypeX *f_ptr = reinterpret_cast<const TypeX*>(stab_ptr);
        const TypeY *i_ptr = reinterpret_cast<const TypeY*>(stab_ptr + field.Pitch() * sizeof(TypeX));
        for (SizeT i = 0; i < field.Size().ReduceMul(); ++i)
        {
            if (f_ptr[i] != static_cast<TypeX>(2 * i) || i_ptr[i] != static_cast<TypeX>(2 * i + 1))
            {
                return false;
            }
        }
        return true;
    } ());
}

TEST(Field, soa_2d_assign_1dindex_as_value)
{
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
        const TypeX *i_ptr = reinterpret_cast<const TypeX*>(stab_ptr);
        const TypeY *f_ptr = reinterpret_cast<const TypeY*>(stab_ptr + field.Pitch() * sizeof(TypeX));
        const TypeZ *s_ptr = reinterpret_cast<const TypeZ*>(stab_ptr + field.Pitch() * (sizeof(TypeX) + sizeof(TypeY)));
        for (SizeT i = 0; i < field.Size().ReduceMul(); ++i)
        {
            if (i_ptr[i] != static_cast<TypeX>(3 * i) || f_ptr[i] != static_cast<TypeY>(3 * i + 1) || s_ptr[i] != static_cast<TypeZ>(3 * i + 2))
            {
                return false;
            }
        }
        return true;
    } ());
}

TEST(Field, soai_2d_assign_1dindex_as_value)
{
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
            const char* stab_ptr = reinterpret_cast<const char*>(field.GetBasePointer()) + stab_index * field.Pitch() * (sizeof(TypeX) + sizeof(TypeY) + sizeof(TypeZ));
            const TypeX *i_ptr = reinterpret_cast<const TypeX*>(stab_ptr);
            const TypeY *f_ptr = reinterpret_cast<const TypeY*>(stab_ptr + field.Pitch() * sizeof(TypeX));
            const TypeZ *s_ptr = reinterpret_cast<const TypeZ*>(stab_ptr + field.Pitch() * (sizeof(TypeX) + sizeof(TypeY)));
            for (SizeT i = 0; i < field.Size(0); ++i)
            {
                const SizeT index = stab_index * field.Size(0) + i;
                if (i_ptr[i] != static_cast<TypeX>(3 * index) || f_ptr[i] != static_cast<TypeY>(3 * index + 1) || s_ptr[i] != static_cast<TypeZ>(3 * index + 2))
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
        const TypeX *i_ptr = reinterpret_cast<const TypeX*>(stab_ptr);
        const TypeY *f_ptr = reinterpret_cast<const TypeY*>(stab_ptr + field.Pitch() * sizeof(TypeX));
        const TypeZ *s_ptr = reinterpret_cast<const TypeZ*>(stab_ptr + field.Pitch() * (sizeof(TypeX) + sizeof(TypeY)));
        for (SizeT i = 0; i < field.Size().ReduceMul(); ++i)
        {
            if (i_ptr[i] != static_cast<TypeX>(3 * i) || f_ptr[i] != static_cast<TypeY>(3 * i + 1) || s_ptr[i] != static_cast<TypeZ>(3 * i + 2))
            {
                return false;
            }
        }
        return true;
    } ());
}

TEST(Field, soai_3d_assign_1dindex_as_value)
{
    Field<ElementT, 3, DataLayout::SoAi> field;
    EXPECT_EQ(0, field.Size());

    field.Resize({254, 33, 52});
    EXPECT_EQ(SizeArray<3>(254, 33, 52), field.Size());
    //field.Resize({255, 256, 256});
    //EXPECT_EQ(SizeArray<3>(255, 256, 256), field.Size());
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
            const char* stab_ptr = reinterpret_cast<const char*>(field.GetBasePointer()) + stab_index * field.Pitch() * (sizeof(TypeX) + sizeof(TypeY) + sizeof(TypeZ));
            const TypeX *i_ptr = reinterpret_cast<const TypeX*>(stab_ptr);
            const TypeY *f_ptr = reinterpret_cast<const TypeY*>(stab_ptr + field.Pitch() * sizeof(TypeX));
            const TypeZ *s_ptr = reinterpret_cast<const TypeZ*>(stab_ptr + field.Pitch() * (sizeof(TypeX) + sizeof(TypeY)));
            for (SizeT i = 0; i < field.Size(0); ++i)
            {
                const SizeT index = stab_index * field.Size(0) + i;
                if (i_ptr[i] != static_cast<TypeX>(3 * index) || f_ptr[i] != static_cast<TypeY>(3 * index + 1) || s_ptr[i] != static_cast<TypeZ>(3 * index + 2))
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
            const char* stab_ptr = reinterpret_cast<const char*>(field.GetBasePointer()) + stab_index * field.Pitch() * (sizeof(TypeX) + sizeof(TypeY) + sizeof(TypeZ));
            const TypeX *i_ptr = reinterpret_cast<const TypeX*>(stab_ptr);
            const TypeY *f_ptr = reinterpret_cast<const TypeY*>(stab_ptr + field.Pitch() * sizeof(TypeX));
            const TypeZ *s_ptr = reinterpret_cast<const TypeZ*>(stab_ptr + field.Pitch() * (sizeof(TypeX) + sizeof(TypeY)));
            for (SizeT i = 0; i < BlockingFactor; ++i)
            {
                const SizeT index = stab_index * BlockingFactor + i;
                if (index >= field.Size().ReduceMul()) continue;    
                if (i_ptr[i] != static_cast<TypeX>(3 * index) || f_ptr[i] != static_cast<TypeY>(3 * index + 1) || s_ptr[i] != static_cast<TypeZ>(3 * index + 2))
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
    constexpr SizeT BlockingFactor = ::fw::internal::Traits<ElementT, DataLayout::AoSoA>::BlockingFactor;
    
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
                const char* stab_ptr = reinterpret_cast<const char*>(field.GetBasePointer()) + (n_1 + n_0) * field.Pitch() * (sizeof(TypeX) + sizeof(TypeY) + sizeof(TypeZ));
                const TypeX *i_ptr = reinterpret_cast<const TypeX*>(stab_ptr);
                const TypeY *f_ptr = reinterpret_cast<const TypeY*>(stab_ptr + field.Pitch() * sizeof(TypeX));
                const TypeZ *s_ptr = reinterpret_cast<const TypeZ*>(stab_ptr + field.Pitch() * (sizeof(TypeX) + sizeof(TypeY)));
                for (SizeT i = 0; i < BlockingFactor; ++i)
                {
                    const SizeT intra_stab_index = n_0 * BlockingFactor + i;
                    if (intra_stab_index >= field.Size(0)) continue;
                    const SizeT index = stab_index * field.Size(0) + intra_stab_index;
                    if (i_ptr[i] != static_cast<TypeX>(3 * index) || f_ptr[i] != static_cast<TypeY>(3 * index + 1) || s_ptr[i] != static_cast<TypeZ>(3 * index + 2))
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
    constexpr SizeT BlockingFactor = ::fw::internal::Traits<ElementT, DataLayout::AoSoA>::BlockingFactor;
    
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
                const char* stab_ptr = reinterpret_cast<const char*>(field.GetBasePointer()) + (n_1 + n_0) * field.Pitch() * (sizeof(TypeX) + sizeof(TypeY) + sizeof(TypeZ));
                const TypeX *i_ptr = reinterpret_cast<const TypeX*>(stab_ptr);
                const TypeY *f_ptr = reinterpret_cast<const TypeY*>(stab_ptr + field.Pitch() * sizeof(TypeX));
                const TypeZ *s_ptr = reinterpret_cast<const TypeZ*>(stab_ptr + field.Pitch() * (sizeof(TypeX) + sizeof(TypeY)));
                for (SizeT i = 0; i < BlockingFactor; ++i)
                {
                    const SizeT intra_stab_index = n_0 * BlockingFactor + i;
                    if (intra_stab_index >= field.Size(0)) continue;
                    const SizeT index = stab_index * field.Size(0) + intra_stab_index;
                    if (i_ptr[i] != static_cast<TypeX>(3 * index) || f_ptr[i] != static_cast<TypeY>(3 * index + 1) || s_ptr[i] != static_cast<TypeZ>(3 * index + 2))
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