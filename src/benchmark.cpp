#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <array>

template <typename T>
class DEBUG;

#include <kernels.hpp>

constexpr SizeType NX_DEFAULT = 128;
constexpr SizeType NY_DEFAULT = 128;
constexpr SizeType NZ_DEFAULT = 128;

constexpr double SPREAD = 1.0;
constexpr double OFFSET = 3.0;

// Implements the benchmark in different dimensions.
template <typename ...T>
int benchmark(const T... n);

int main(int argc, char** argv)
{
    // Command line arguments
    SizeType dimension = 0;
    const SizeType nx = (argc > 1 ? atoi(argv[++dimension]) : NX_DEFAULT);
    const SizeType ny = (argc > 2 ? atoi(argv[++dimension]) : NY_DEFAULT);
    const SizeType nz = (argc > 3 ? atoi(argv[++dimension]) : NZ_DEFAULT);

    if (dimension == 1)
    {
        return benchmark(nx);
    }
    else if (dimension == 2)
    {
        return benchmark(nx, ny);
    }
    else if (dimension == 3)
    {
        return benchmark(nx, ny, nz);
    }

    return 1;
}

template <typename FieldT, typename RandomT>
void assign_random_numbers(FieldT&& field, const SizeType size, RandomT generator)
{
    for (SizeType i = 0; i < size; ++i)
    {
        field[i] = generator();
    }
}

template <SizeType C_Dimension, typename FieldT, typename SizeT, typename RandomT>
void assign_random_numbers(FieldT&& field, const SizeT& size, RandomT generator)
{
    if constexpr (C_Dimension == 1)
    {
        assign_random_numbers(field, size[0], generator);
    }
    else
    {
        for (std::size_t i = 0; i < size[C_Dimension - 1]; ++i)
        {
            assign_random_numbers<C_Dimension - 1>(field[i], size, generator);
        }
    }
}

template <typename T, SizeType C_Dimension>
void setup_field(field_type<T, C_Dimension>& field, const SizeType seed = 1)
{
    const auto& size = field.size();
    
    srand48(1);
    assign_random_numbers<C_Dimension>(field, size, [] () { return static_cast<T>((2.0 * drand48() -1.0) * SPREAD + OFFSET); });

    /*
    for (SizeType k = 0; k < nz; ++k)
        {
            for (SizeType j = 0; j < ny; ++j)
            {
                for (SizeType i = 0; i < nx; ++i)
                {
                    const SizeType index = k * nx * ny + j * nx + i;
                    const type s_1 = static_cast<type>((2.0 * drand48() -1.0) * SPREAD + OFFSET);
                    const type s_2 = static_cast<type>((2.0 * drand48() -1.0) * SPREAD + OFFSET);
                    const type s_3 = static_cast<type>((2.0 * drand48() -1.0) * SPREAD + OFFSET);
                    in_orig_1[index] = {static_cast<type_x>(s_1 * value), static_cast<type_y>(s_2 * value), static_cast<type_z>(s_3 * value)};
                    #if defined(USE_1D_BUFFER)
                        in_1[index] = in_orig_1[index];
                        #if defined(INPLACE)
                        out_2[index] = in_1[index];
                        #endif
                    #else
                        in_1[k][j][i] = in_orig_1[index];
                        #if defined(INPLACE)
                        out_2[k][j][i] = in_1[k][j][i];
                        #endif
                    #endif

                    #if defined(VECTOR_PRODUCT)
                    {
                        const type s_1 = static_cast<type>((2.0 * drand48() -1.0) * SPREAD + OFFSET);
                        const type s_2 = static_cast<type>((2.0 * drand48() -1.0) * SPREAD + OFFSET);
                        const type s_3 = static_cast<type>((2.0 * drand48() -1.0) * SPREAD + OFFSET);
                        in_orig_2[index] = {static_cast<type_x>(s_1 * value), static_cast<type_y>(s_2 * value), static_cast<type_z>(s_3 * value)};
                        #if defined(USE_1D_BUFFER)
                            in_2[index] = in_orig_2[index];
                        #else
                            in_2[k][j][i] = in_orig_2[index];
                        #endif
                    }
                    #endif
                }
            }
        }
    */ 
}

// Implements the benchmark in different dimensions.
// The number of arguments defines the dimension.
template <typename ...T>
int benchmark(const T... n)
{
    constexpr SizeType Dimension{sizeof...(T)};
    const SizeType size[]{n...};
    field_type<real_type, Dimension> field{{n...}};

    //    setup_field(field);
    //field.set([] () { return static_cast<real_type>((2.0 * drand48() -1.0) * SPREAD + OFFSET); });
    field.set([] () { static SizeType index = 0; return index++; });

    if constexpr (Dimension == 3)
    {
        for (SizeType k = 0; k < size[2]; ++k)
            for (SizeType j = 0; j < size[1]; ++j)
            {
                for (SizeType i = 0; i < size[0]; ++i)
                    std::cout << field[k][j][i] << " ";
                std::cout << std::endl;
            }
    }

    return 0;
}
