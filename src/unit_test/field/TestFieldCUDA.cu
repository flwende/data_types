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

using ElementT = Tuple<TypeX, TypeY, TypeZ>;

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

template <typename Container>
__global__
void Kernel_1(Container field)
{
  const SizeT x = blockIdx.x * blockDim.x + threadIdx.x;
  const SizeT y = blockIdx.y * blockDim.y + threadIdx.y;
  const SizeT z = blockIdx.z * blockDim.z + threadIdx.z;
  const SizeT index = (z * field.Size(1) + y) * field.Size(0) + x;

  if (x < field.Size(0) && y < field.Size(1) && z < field.Size(2))
    {
      field[z][y][x] = index;
    }
}

template <typename Container>
__global__
void Kernel_2(Container field)
{
  const SizeT x = blockIdx.x * blockDim.x + threadIdx.x;
  const SizeT y = blockIdx.y * blockDim.y + threadIdx.y;
  const SizeT z = blockIdx.z * blockDim.z + threadIdx.z;
  const SizeT index = (z * field.Size(1) + y) * field.Size(0) + x;

  if (x < field.Size(0) && y < field.Size(1) && z < field.Size(2))
    {
      field[z][y][x].x = 3 * index + 0;
      field[z][y][x].y = 3 * index + 1;
      field[z][y][x].z = 3 * index + 2;      
    }
}

TEST(Field, aos_3d_assign_1dindex_as_value)
{
  constexpr SizeT nx = 23;
  constexpr SizeT ny = 7;	
  constexpr SizeT nz = 13;		
  
  Field<ElementT, 3, DataLayout::AoS> field({nx, ny, nz}, true);
  
  EXPECT_EQ(true, [&field]() {
		    bool all_zero = true;
		    loop_3d(field, [&all_zero](auto&& item) { if (item != 0) all_zero = false; });
		    return all_zero;
		  } ());
  
  dim3 block{128, 1, 1};
  dim3 grid{(nx + (block.x - 1)) / block.x, (ny + (block.y - 1)) / block.y, (nz + (block.z - 1)) / block.z};
    
  Kernel_1<<<grid, block>>>(field.DeviceData());
  field.CopyDeviceToHost();

  EXPECT_EQ(true, [&field]() {
		    bool output_is_correct = true;
		    SizeT index = 0;
		    loop_3d(field, [&output_is_correct, &index](auto&& item) { if (item != index++) output_is_correct = false; });
		    return output_is_correct;
		  } ());
}

TEST(Field, aos_3d_assign_1dindex_as_value_componentwise)
{
  constexpr SizeT nx = 23;
  constexpr SizeT ny = 7;	
  constexpr SizeT nz = 13;		
  
  Field<ElementT, 3, DataLayout::AoS> field({nx, ny, nz}, true);
  
  EXPECT_EQ(true, [&field]() {
		    bool all_zero = true;
		    loop_3d(field, [&all_zero](auto&& item) { if (item != 0) all_zero = false; });
		    return all_zero;
		  } ());
  
  dim3 block{128, 1, 1};
  dim3 grid{(nx + (block.x - 1)) / block.x, (ny + (block.y - 1)) / block.y, (nz + (block.z - 1)) / block.z};
    
  Kernel_2<<<grid, block>>>(field.DeviceData());
  field.CopyDeviceToHost();

  EXPECT_EQ(true, [&field]() {
		    bool output_is_correct = true;
		    SizeT index = 0;
		    loop_3d(field, [&output_is_correct, &index](auto&& item) { if (item != ElementT{3 * index + 0, 3 * index + 1, 3 * index + 2}) output_is_correct = false; ++index; });
		    return output_is_correct;
		  } ());
}

TEST(Field, soa_3d_assign_1dindex_as_value)
{
  constexpr SizeT nx = 23;
  constexpr SizeT ny = 7;	
  constexpr SizeT nz = 13;		
  
  Field<ElementT, 3, DataLayout::SoA> field({nx, ny, nz}, true);
  
  EXPECT_EQ(true, [&field]() {
		    bool all_zero = true;
		    loop_3d(field, [&all_zero](auto&& item) { if (item != ElementT(0)) all_zero = false; });
		    return all_zero;
		  } ());
  
  dim3 block{128, 1, 1};
  dim3 grid{(nx + (block.x - 1)) / block.x, (ny + (block.y - 1)) / block.y, (nz + (block.z - 1)) / block.z};
    
  Kernel_1<<<grid, block>>>(field.DeviceData());
  field.CopyDeviceToHost();
  
  EXPECT_EQ(true, [&field]() {
		    bool output_is_correct = true;
		    SizeT index = 0;
		    loop_3d(field, [&output_is_correct, &index](auto&& item) { if (item.x != index || item.y != index || item.z != index) output_is_correct = false; ++index; });
		    return output_is_correct;
		  } ());
}

TEST(Field, soa_3d_assign_1dindex_as_value_componentwise)
{
  constexpr SizeT nx = 23;
  constexpr SizeT ny = 7;	
  constexpr SizeT nz = 13;		
  
  Field<ElementT, 3, DataLayout::SoA> field({nx, ny, nz}, true);
  
  EXPECT_EQ(true, [&field]() {
		    bool all_zero = true;
		    loop_3d(field, [&all_zero](auto&& item) { if (item != ElementT(0)) all_zero = false; });		    
		    return all_zero;
		  } ());
  
  dim3 block{128, 1, 1};
  dim3 grid{(nx + (block.x - 1)) / block.x, (ny + (block.y - 1)) / block.y, (nz + (block.z - 1)) / block.z};
    
  Kernel_2<<<grid, block>>>(field.DeviceData());
  field.CopyDeviceToHost();

  EXPECT_EQ(true, [&field]() {
		    bool output_is_correct = true;
		    SizeT index = 0;
		    loop_3d(field, [&output_is_correct, &index](auto&& item) { if (item.x != (3 * index) || item.y != (3 * index + 1) || item.z != (3 * index + 2)) output_is_correct = false; ++index; });
		    return output_is_correct;
		  } ());
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
