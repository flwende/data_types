#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <buffer/buffer.hpp>
#include <vec/vec.hpp>
#include <tuple/tuple.hpp>

using namespace VEC_NAMESPACE;
using namespace TUPLE_NAMESPACE;

//constexpr data_layout layout = data_layout::SoA;
constexpr data_layout layout = data_layout::AoS;

using type = double;

//constexpr std::size_t scale = 3;
//using vtype = vec<type, scale>;
using vtype = fw::tuple<std::uint16_t, std::int64_t, std::uint32_t>;
constexpr std::size_t scale = 14;

using buffer_type = fw::buffer<vtype, 2, layout>;

int main(int argc, char** argv)
{
    const std::size_t n_0 = (argc > 1 ? atoi(argv[1]) : 16);
    const std::size_t n_1 = (argc > 2 ? atoi(argv[2]) : 4);

    buffer_type b(sarray<std::size_t, 2>(n_0, n_1));
    auto acc = b.read_write();
    auto acc_read = b.read();    

/*
    vec<double, scale> v(3.2, 1.3, -4.5);
    vec<float, scale> v2 = -1.2;

    internal::vec_proxy<double, scale> vp(v);
    internal::vec_proxy<float, scale> vp2(v2);

    vp = vp2;
    vp *= v2;
    std::cout << v << std::endl;

    acc[3][4] = sqrt(2 + log(vp * 2U));
*/

    srand48(1);
    for (std::size_t j = 0; j < n_1; ++j)
    {
        for (std::size_t i = 0; i < n_0; ++i)
        {
            acc[j][i].x = 1.0 + 0xFF * drand48();
            acc[j][i].y = 1.0 + 0xFF * drand48();
            acc[j][i].z = 1.0 + 0xFF * drand48();
        }
    }

    for (std::size_t j = 0; j < n_1; ++j)
    {
        for (std::size_t i = 0; i < n_0; ++i)
        {
            acc[j][i].x = std::log(acc[j][i].z);
            //acc[j][i] = cross(v, 2.0 * sqrt(0.5 + log(acc[j][i])));
        }
    }

    if (true)//layout == data_layout::AoS)
    {
        for (std::size_t j = 0; j < n_1; ++j)
        {
            for (std::size_t i = 0; i < n_0; ++i)
            {
                std::cout << acc_read[j][i] << ", ";
            }
            std::cout << std::endl << std::endl;
        }
    }
    else
    {
        //const std::uint8_t* ptr = reinterpret_cast<const std::uint8_t*>(&acc_read[0][0].x);
        const type* ptr = reinterpret_cast<const type*>(&acc_read[0][0].x);
        const std::size_t n_0_internal = b.n[0];

        for (std::size_t j = 0; j < (n_1 * scale); ++j)
        {
            if (j > 0 && (j % scale) == 0) std::cout << std::endl;

            for (std::size_t i = 0; i < n_0; ++i)
            {
                //std::cout << static_cast<std::uint64_t>(ptr[j * n_0_internal + i]) << ", ";
                std::cout << ptr[j * n_0_internal + i] << ", ";
            }
            std::cout << std::endl;
        }
    }

    /*
const std::size_t n_0_internal = b.n[0];

    srand48(1);
    for (std::size_t j = 0; j < n_1; ++j)
    {
        for (std::size_t i = 0; i < n_0; ++i)
        {
            acc[j][i] = 0.1 + drand48();
        }
    }

    for (std::size_t j = 0; j < n_1; ++j)
    {
        for (std::size_t i = 0; i < n_0; ++i)
        {
            acc[j][i].x = std::log(acc[j][i].y);
        }
    }

    const type* ptr = reinterpret_cast<const type*>(&acc[0][0].x);
    for (std::size_t j = 0; j < (n_1 * scale); ++j)
    {
        if (j > 0 && (j % scale) == 0) std::cout << std::endl;

        for (std::size_t i = 0; i < n_0; ++i)
        {
            std::cout << static_cast<type>(ptr[j * n_0_internal + i]) << ", ";
        }
        std::cout << std::endl;
    }
    */
        
    return 0;
}