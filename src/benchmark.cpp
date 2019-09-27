#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <array>

template <typename T>
class DEBUG;

#include <kernels.hpp>

constexpr size_type NX_DEFAULT = 128;
constexpr size_type NY_DEFAULT = 128;
constexpr size_type NZ_DEFAULT = 128;

constexpr double SPREAD = 1.0;
constexpr double OFFSET = 3.0;

// Implements the benchmark in different dimensions.
template <typename ...T>
int benchmark(const T... n);

int main(int argc, char** argv)
{
    // Command line arguments
    size_type dimension = 0;
    const size_type nx = (argc > 1 ? atoi(argv[++dimension]) : NX_DEFAULT);
    const size_type ny = (argc > 2 ? atoi(argv[++dimension]) : NY_DEFAULT);
    const size_type nz = (argc > 3 ? atoi(argv[++dimension]) : NZ_DEFAULT);

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
void assign_random_numbers(FieldT&& field, const size_type size, RandomT generator)
{
    for (size_type i = 0; i < size; ++i)
    {
        field[i] = generator();
    }
}

template <size_type C_Dimension, typename FieldT, typename SizeT, typename RandomT>
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

template <typename T, size_type C_Dimension>
void setup_field(field_type<T, C_Dimension>& field, const size_type seed = 1)
{
    const auto& size = field.size();
    
    srand48(1);
    assign_random_numbers<C_Dimension>(field, size, [] () { return static_cast<T>((2.0 * drand48() -1.0) * SPREAD + OFFSET); });

    /*
    for (size_type k = 0; k < nz; ++k)
        {
            for (size_type j = 0; j < ny; ++j)
            {
                for (size_type i = 0; i < nx; ++i)
                {
                    const size_type index = k * nx * ny + j * nx + i;
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
    constexpr size_type Dimension{sizeof...(T)};
    const size_type size[]{n...};
    field_type<real_type, Dimension> field{{n...}};

    //    setup_field(field);
    //field.set([] () { return static_cast<real_type>((2.0 * drand48() -1.0) * SPREAD + OFFSET); });
    field.set([] () { static size_type index = 0; return index++; });

    if constexpr (Dimension == 3)
    {
        for (size_type k = 0; k < size[2]; ++k)
            for (size_type j = 0; j < size[1]; ++j)
            {
                for (size_type i = 0; i < size[0]; ++i)
                    std::cout << field[k][j][i] << " ";
                std::cout << std::endl;
            }
    }

    return 0;
}
