# Data Types
This repository contains a collection of data types.

# SArray
A static array definition.

# Vec
A Simple D-dimensional vector with components `x`, `y` and `z`.

# Buffer
A multi-dimensional container with contiguous (dynamically allocated) memory and support for AoS (Array of Structs) and SoA (Struct of Arrays) data layouts.
Internally, the innermost dimension is padded according to the desired (or the default) data alignment.
In case of the Buffer holds Vec data elements, elementwise access through proxy-objects is possible, that is, you can write

```
buffer<vec<float, 3>, 2,...> data({nx, ny});
for (std::size_t j = 0; j < ny; ++j)
{
  for (std::size_t i = 0; i < ny; ++i)
  {
    data[j][i].x = 0.5F;
    data[j][i].z = logf(data[j][i].y);
  }
}
```
The advantage: if you choose the SoA data layout, components `x`, `y` and `z` are contiguous in memory each, and SIMD vectorization should work out of the box.

### Performance
The table below lists the performance of `buffer<vec<float, 3>,...>` (execution time in seconds) in D = 1 and D = 3 dimensions for calculating the element-wise logarithm and exponential of a given field using the implementation in `src/test_buffer.cpp` and the provided Makefiles. In the 1-dimensional case, we also list execution times with Intel's SIMD data layout templates (https://software.intel.com/en-us/code-samples/intel-compiler/intel-compiler-features/intel-sdlt) using `soa1d_container`.
However, there is (currently) no SDLT container for D > 1.

```
                            D = 1 (nx = 65536)       D = 3 (nx/ny/nz = 128/128/128)
                            AoS   SoA   SDLT-SoA  AoS   SoA
g++-7.3, glibc-2.25         3.7   0.7   0.7       112   12.1
clang++-5.0, Intel SVML     4.0   0.5   -         123   9.8
```
The execution happened on a dual socket Intel Xeon E5-2630v3 compute node.

### TODOs
SYCL integration.
