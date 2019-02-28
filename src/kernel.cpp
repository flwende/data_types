// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#include <cstdint>
#include <omp.h>

#include "kernel.hpp"

using namespace fw;

#if defined(AOS_LAYOUT)
    #if defined(VECTOR_PRODUCT)
        template <>
        template <>
        double kernel<element_type>::cross<1>(const fw::buffer<element_type, 1, fw::data_layout::AoS>& x_1, const fw::buffer<element_type, 1, fw::data_layout::AoS>& x_2, fw::buffer<element_type, 1, fw::data_layout::AoS>& y)
        {
            double time = omp_get_wtime();

            #if defined(__INTEL_COMPILER)
            #pragma forceinline recursive
            #endif
            #pragma omp simd        
            for (std::size_t i = 0; i < x_1.n[0]; ++i)
            {
                y[i] = 0.5 + fw::math<element_type>::log(1.0 + fw::cross(x_1[i], fw::math<element_type>::exp(x_2[i])) * fw::cross(x_1[i], fw::math<element_type>::exp(x_2[i])));
            }

            return (omp_get_wtime() - time);
        }

        template <>
        template <>
        double kernel<element_type>::cross<2>(const fw::buffer<element_type, 2, fw::data_layout::AoS>& x_1, const fw::buffer<element_type, 2, fw::data_layout::AoS>& x_2, fw::buffer<element_type, 2, fw::data_layout::AoS>& y)
        {
            double time = omp_get_wtime();

            for (std::size_t j = 0; j < x_1.n[1]; ++j)
            {
                #if defined(__INTEL_COMPILER)
                #pragma forceinline recursive
                #endif
                #pragma omp simd
                for (std::size_t i = 0; i < x_1.n[0]; ++i)
                {
                    y[j][i] = 0.5 + fw::math<element_type>::log(1.0 + fw::cross(x_1[j][i], fw::math<element_type>::exp(x_2[j][i])) * fw::cross(x_1[j][i], fw::math<element_type>::exp(x_2[j][i])));
                }
            }

            return (omp_get_wtime() - time);
        }

        template <>
        template <>
        double kernel<element_type>::cross<3>(const fw::buffer<element_type, 3, fw::data_layout::AoS>& x_1, const fw::buffer<element_type, 3, fw::data_layout::AoS>& x_2, fw::buffer<element_type, 3, fw::data_layout::AoS>& y)
        {
            double time = omp_get_wtime();

            for (std::size_t k = 0; k < x_1.n[2]; ++k)
            {
                for (std::size_t j = 0; j < x_1.n[1]; ++j)
                {
                    #if defined(__INTEL_COMPILER)
                    #pragma forceinline recursive
                    #endif
                    #pragma omp simd
                    for (std::size_t i = 0; i < x_1.n[0]; ++i)
                    {
                        y[k][j][i] = 0.5 + fw::math<element_type>::log(1.0 + fw::cross(x_1[k][j][i], fw::math<element_type>::exp(x_2[k][j][i])) * fw::cross(x_1[k][j][i], fw::math<element_type>::exp(x_2[k][j][i])));
                    }
                }
            }

            return (omp_get_wtime() - time);
        }
    #else
        template <>
        template <>
        double kernel<element_type>::exp<1>(fw::buffer<element_type, 1, fw::data_layout::AoS>& x)
        {
            double time = omp_get_wtime();
            
            #if defined(__INTEL_COMPILER)
            #pragma forceinline recursive
            #endif
            #pragma omp simd
            for (std::size_t i = 0; i < x.n[0]; ++i)
            {
                #if defined(ELEMENT_ACCESS)
                x[i].y = std::exp(x[i].y);
                #else
                x[i] = fw::math<element_type>::exp(x[i]);
                #endif
            }

            return (omp_get_wtime() - time);
        }

        template <>
        template <>
        double kernel<element_type>::exp<2>(fw::buffer<element_type, 2, fw::data_layout::AoS>& x)
        {
            double time = omp_get_wtime();
            
            for (std::size_t j = 0; j < x.n[1]; ++j)
            {
                #if defined(__INTEL_COMPILER)
                #pragma forceinline recursive
                #endif
                #pragma omp simd
                for (std::size_t i = 0; i < x.n[0]; ++i)
                {
                    #if defined(ELEMENT_ACCESS)
                    x[j][i].y = std::exp(x[j][i].y);
                    #else
                    x[j][i] = fw::math<element_type>::exp(x[j][i]);
                    #endif
                }
            }

            return (omp_get_wtime() - time);
        }

        template <>
        template <>
        double kernel<element_type>::exp<3>(fw::buffer<element_type, 3, fw::data_layout::AoS>& x)
        {
            double time = omp_get_wtime();
            
            for (std::size_t k = 0; k < x.n[2]; ++k)
            {
                for (std::size_t j = 0; j < x.n[1]; ++j)
                {
                    #if defined(__INTEL_COMPILER)
                    #pragma forceinline recursive
                    #endif
                    #pragma omp simd
                    for (std::size_t i = 0; i < x.n[0]; ++i)
                    {
                        #if defined(ELEMENT_ACCESS)
                        x[k][j][i].y = std::exp(x[k][j][i].y);
                        #else
                        //x[k][j][i] = fw::math<element_type>::exp(x[k][j][i]);
                        x[k][j][i] = fw::math<element_type>::exp(x[k][j][i]);
                        #endif
                    }
                }
            }

            return (omp_get_wtime() - time);
        }

        template <>
        template <>
        double kernel<element_type>::log<1>(fw::buffer<element_type, 1, fw::data_layout::AoS>& x)
        {
            double time = omp_get_wtime();
            
            #if defined(__INTEL_COMPILER)
            #pragma forceinline recursive
            #endif
            #pragma omp simd
            for (std::size_t i = 0; i < x.n[0]; ++i)
            {
                #if defined(ELEMENT_ACCESS)
                x[i].y = std::log(x[i].y);
                #else
                x[i] = fw::math<element_type>::log(x[i]);
                #endif
            }

            return (omp_get_wtime() - time);
        }

        template <>
        template <>
        double kernel<element_type>::log<2>(fw::buffer<element_type, 2, fw::data_layout::AoS>& x)
        {
            double time = omp_get_wtime();
            
            for (std::size_t j = 0; j < x.n[1]; ++j)
            {
                #if defined(__INTEL_COMPILER)
                #pragma forceinline recursive
                #endif
                #pragma omp simd
                for (std::size_t i = 0; i < x.n[0]; ++i)
                {
                    #if defined(ELEMENT_ACCESS)
                    x[j][i].y = std::log(x[j][i].y);
                    #else
                    x[j][i] = fw::math<element_type>::log(x[j][i]);
                    #endif
                }
            }

            return (omp_get_wtime() - time);
        }

        template <>
        template <>
        double kernel<element_type>::log<3>(fw::buffer<element_type, 3, fw::data_layout::AoS>& x)
        {
            double time = omp_get_wtime();

            for (std::size_t k = 0; k < x.n[2]; ++k)
            {
                for (std::size_t j = 0; j < x.n[1]; ++j)
                {
                    #if defined(__INTEL_COMPILER)
                    #pragma forceinline recursive
                    #endif
                    #pragma omp simd
                    for (std::size_t i = 0; i < x.n[0]; ++i)
                    {
                        #if defined(ELEMENT_ACCESS)
                        x[k][j][i].y = std::log(x[k][j][i].y);
                        #else
                        //x[k][j][i] = fw::math<element_type>::log(x[k][j][i]);
                        x[k][j][i] = fw::math<element_type>::log(x[k][j][i]);
                        #endif
                    }
                }
            }
            
            return (omp_get_wtime() - time);
        }

        template <>
        template <>
        double kernel<element_type>::exp<1>(const fw::buffer<element_type, 1, fw::data_layout::AoS>& x, fw::buffer<element_type, 1, fw::data_layout::AoS>& y)
        {
            double time = omp_get_wtime();
            
            #if defined(__INTEL_COMPILER)
            #pragma forceinline recursive
            #endif
            #pragma omp simd
            for (std::size_t i = 0; i < x.n[0]; ++i)
            {
                #if defined(ELEMENT_ACCESS)
                y[i].y = std::exp(x[i].y);
                #else
                y[i] = fw::math<element_type>::exp(x[i]);
                #endif
            }

            return (omp_get_wtime() - time);
        }

        template <>
        template <>
        double kernel<element_type>::exp<2>(const fw::buffer<element_type, 2, fw::data_layout::AoS>& x, fw::buffer<element_type, 2, fw::data_layout::AoS>& y)
        {
            double time = omp_get_wtime();
            
            for (std::size_t j = 0; j < x.n[1]; ++j)
            {
                #if defined(__INTEL_COMPILER)
                #pragma forceinline recursive
                #endif
                #pragma omp simd
                for (std::size_t i = 0; i < x.n[0]; ++i)
                {
                    #if defined(ELEMENT_ACCESS)
                    y[j][i].y = std::exp(x[j][i].y);
                    #else
                    y[j][i] = fw::math<element_type>::exp(x[j][i]);
                    #endif
                }
            }

            return (omp_get_wtime() - time);
        }

        template <>
        template <>
        double kernel<element_type>::exp<3>(const fw::buffer<element_type, 3, fw::data_layout::AoS>& x, fw::buffer<element_type, 3, fw::data_layout::AoS>& y)
        {
            double time = omp_get_wtime();
            
            for (std::size_t k = 0; k < x.n[2]; ++k)
            {
                for (std::size_t j = 0; j < x.n[1]; ++j)
                {
                    #if defined(__INTEL_COMPILER)
                    #pragma forceinline recursive
                    #endif
                    #pragma omp simd
                    for (std::size_t i = 0; i < x.n[0]; ++i)
                    {
                        #if defined(ELEMENT_ACCESS)
                        y[k][j][i].y = std::exp(x[k][j][i].y);
                        #else
                        y[k][j][i] = fw::math<element_type>::exp(x[k][j][i]);
                        #endif
                    }
                }
            }

            return (omp_get_wtime() - time);
        }

        template <>
        template <>
        double kernel<element_type>::log<1>(const fw::buffer<element_type, 1, fw::data_layout::AoS>& x, fw::buffer<element_type, 1, fw::data_layout::AoS>& y)
        {
            double time = omp_get_wtime();
            
            #if defined(__INTEL_COMPILER)
            #pragma forceinline recursive
            #endif
            #pragma omp simd
            for (std::size_t i = 0; i < x.n[0]; ++i)
            {
                #if defined(ELEMENT_ACCESS)
                y[i].y = std::log(x[i].y);
                #else
                y[i] = fw::math<element_type>::log(x[i]);
                #endif
            }

            return (omp_get_wtime() - time);
        }

        template <>
        template <>
        double kernel<element_type>::log<2>(const fw::buffer<element_type, 2, fw::data_layout::AoS>& x, fw::buffer<element_type, 2, fw::data_layout::AoS>& y)
        {
            double time = omp_get_wtime();
            
            for (std::size_t j = 0; j < x.n[1]; ++j)
            {
                #if defined(__INTEL_COMPILER)
                #pragma forceinline recursive
                #endif
                #pragma omp simd
                for (std::size_t i = 0; i < x.n[0]; ++i)
                {
                    #if defined(ELEMENT_ACCESS)
                    y[j][i].y = std::log(x[j][i].y);
                    #else
                    y[j][i] = fw::math<element_type>::log(x[j][i]);
                    #endif
                }
            }

            return (omp_get_wtime() - time);
        }

        template <>
        template <>
        double kernel<element_type>::log<3>(const fw::buffer<element_type, 3, fw::data_layout::AoS>& x, fw::buffer<element_type, 3, fw::data_layout::AoS>& y)
        {
            double time = omp_get_wtime();

            for (std::size_t k = 0; k < x.n[2]; ++k)
            {
                for (std::size_t j = 0; j < x.n[1]; ++j)
                {
                    #if defined(__INTEL_COMPILER)
                    #pragma forceinline recursive
                    #endif
                    #pragma omp simd
                    for (std::size_t i = 0; i < x.n[0]; ++i)
                    {
                        #if defined(ELEMENT_ACCESS)
                        y[k][j][i].y = std::log(x[k][j][i].y);
                        #else
                        y[k][j][i] = fw::math<element_type>::log(x[k][j][i]);
                        #endif
                    }
                }
            }
            
            return (omp_get_wtime() - time);
        }
    #endif
#else
    #if defined(VECTOR_PRODUCT)
        template <>
        template <>
        double kernel<element_type>::cross<1>(const fw::buffer<element_type, 1, fw::data_layout::SoA>& x_1, const fw::buffer<element_type, 1, fw::data_layout::SoA>& x_2, fw::buffer<element_type, 1, fw::data_layout::SoA>& y)
        {
            double time = omp_get_wtime();

            #if defined(__INTEL_COMPILER)
            #pragma forceinline recursive
            #endif
            #pragma omp simd        
            for (std::size_t i = 0; i < x_1.n[0]; ++i)
            {
                y[i] = 0.5 + fw::math<element_type>::log(1.0 + fw::cross(x_1[i], fw::math<element_type>::exp(x_2[i])) * fw::cross(x_1[i], fw::math<element_type>::exp(x_2[i])));
            }

            return (omp_get_wtime() - time);
        }

        template <>
        template <>
        double kernel<element_type>::cross<2>(const fw::buffer<element_type, 2, fw::data_layout::SoA>& x_1, const fw::buffer<element_type, 2, fw::data_layout::SoA>& x_2, fw::buffer<element_type, 2, fw::data_layout::SoA>& y)
        {
            double time = omp_get_wtime();

            for (std::size_t j = 0; j < x_1.n[1]; ++j)
            {
                #if defined(__INTEL_COMPILER)
                #pragma forceinline recursive
                #endif
                #pragma omp simd
                for (std::size_t i = 0; i < x_1.n[0]; ++i)
                {
                    y[j][i] = 0.5 + fw::math<element_type>::log(1.0 + fw::cross(x_1[j][i], fw::math<element_type>::exp(x_2[j][i])) * fw::cross(x_1[j][i], fw::math<element_type>::exp(x_2[j][i])));
                }
            }

            return (omp_get_wtime() - time);
        }

        template <>
        template <>
        double kernel<element_type>::cross<3>(const fw::buffer<element_type, 3, fw::data_layout::SoA>& x_1, const fw::buffer<element_type, 3, fw::data_layout::SoA>& x_2, fw::buffer<element_type, 3, fw::data_layout::SoA>& y)
        {
            double time = omp_get_wtime();

            for (std::size_t k = 0; k < x_1.n[2]; ++k)
            {
                for (std::size_t j = 0; j < x_1.n[1]; ++j)
                {
                    #if defined(__INTEL_COMPILER)
                    #pragma forceinline recursive
                    #endif
                    #pragma omp simd
                    for (std::size_t i = 0; i < x_1.n[0]; ++i)
                    {
                        y[k][j][i] = 0.5 + fw::math<element_type>::log(1.0 + fw::cross(x_1[k][j][i], fw::math<element_type>::exp(x_2[k][j][i])) * fw::cross(x_1[k][j][i], fw::math<element_type>::exp(x_2[k][j][i])));
                    }
                }
            }

            return (omp_get_wtime() - time);
        }
    #else
        template <>
        template <>
        double kernel<element_type>::exp<1>(fw::buffer<element_type, 1, fw::data_layout::SoA>& x)
        {
            double time = omp_get_wtime();
            
            #if defined(__INTEL_COMPILER)
            #pragma forceinline recursive
            #endif
            #pragma omp simd
            for (std::size_t i = 0; i < x.n[0]; ++i)
            {
                #if defined(ELEMENT_ACCESS)
                x[i].y = std::exp(x[i].y);
                #else
                x[i] = fw::math<element_type>::exp(x[i]);
                #endif
            }

            return (omp_get_wtime() - time);
        }

        template <>
        template <>
        double kernel<element_type>::exp<2>(fw::buffer<element_type, 2, fw::data_layout::SoA>& x)
        {
            double time = omp_get_wtime();
            
            for (std::size_t j = 0; j < x.n[1]; ++j)
            {
                #if defined(__INTEL_COMPILER)
                #pragma forceinline recursive
                #endif
                #pragma omp simd
                for (std::size_t i = 0; i < x.n[0]; ++i)
                {
                    #if defined(ELEMENT_ACCESS)
                    x[j][i].y = std::exp(x[j][i].y);
                    #else
                    x[j][i] = fw::math<element_type>::exp(x[j][i]);
                    #endif
                }
            }

            return (omp_get_wtime() - time);
        }

        template <>
        template <>
        double kernel<element_type>::exp<3>(fw::buffer<element_type, 3, fw::data_layout::SoA>& x)
        {
            double time = omp_get_wtime();
            
            for (std::size_t k = 0; k < x.n[2]; ++k)
            {
                for (std::size_t j = 0; j < x.n[1]; ++j)
                {
                    #if defined(__INTEL_COMPILER)
                    #pragma forceinline recursive
                    #endif
                    #pragma omp simd
                    for (std::size_t i = 0; i < x.n[0]; ++i)
                    {
                        #if defined(ELEMENT_ACCESS)
                        x[k][j][i].y = std::exp(x[k][j][i].y);
                        #else
                        x[k][j][i] = fw::math<element_type>::exp(x[k][j][i]);
                        #endif
                    }
                }
            }

            return (omp_get_wtime() - time);
        }

        template <>
        template <>
        double kernel<element_type>::log<1>(fw::buffer<element_type, 1, fw::data_layout::SoA>& x)
        {
            double time = omp_get_wtime();
            
            #if defined(__INTEL_COMPILER)
            #pragma forceinline recursive
            #endif
            #pragma omp simd
            for (std::size_t i = 0; i < x.n[0]; ++i)
            {
                #if defined(ELEMENT_ACCESS)
                x[i].y = std::log(x[i].y);
                #else
                x[i] = fw::math<element_type>::log(x[i]);
                #endif
            }

            return (omp_get_wtime() - time);
        }

        template <>
        template <>
        double kernel<element_type>::log<2>(fw::buffer<element_type, 2, fw::data_layout::SoA>& x)
        {
            double time = omp_get_wtime();
            
            for (std::size_t j = 0; j < x.n[1]; ++j)
            {
                #if defined(__INTEL_COMPILER)
                #pragma forceinline recursive
                #endif
                #pragma omp simd
                for (std::size_t i = 0; i < x.n[0]; ++i)
                {
                    #if defined(ELEMENT_ACCESS)
                    x[j][i].y = std::log(x[j][i].y);
                    #else
                    x[j][i] = fw::math<element_type>::log(x[j][i]);
                    #endif
                }
            }

            return (omp_get_wtime() - time);
        }

        template <>
        template <>
        double kernel<element_type>::log<3>(fw::buffer<element_type, 3, fw::data_layout::SoA>& x)
        {
            double time = omp_get_wtime();

            for (std::size_t k = 0; k < x.n[2]; ++k)
            {
                for (std::size_t j = 0; j < x.n[1]; ++j)
                {
                    #if defined(__INTEL_COMPILER)
                    #pragma forceinline recursive
                    #endif
                    #pragma omp simd
                    for (std::size_t i = 0; i < x.n[0]; ++i)
                    {
                        #if defined(ELEMENT_ACCESS)
                        x[k][j][i].y = std::log(x[k][j][i].y);
                        #else
                        x[k][j][i] = fw::math<element_type>::log(x[k][j][i]);
                        #endif
                    }
                }
            }
            
            return (omp_get_wtime() - time);
        }

        template <>
        template <>
        double kernel<element_type>::exp<1>(const fw::buffer<element_type, 1, fw::data_layout::SoA>& x, fw::buffer<element_type, 1, fw::data_layout::SoA>& y)
        {
            double time = omp_get_wtime();
            
            #if defined(__INTEL_COMPILER)
            #pragma forceinline recursive
            #endif
            #pragma omp simd
            for (std::size_t i = 0; i < x.n[0]; ++i)
            {
                #if defined(ELEMENT_ACCESS)
                y[i].y = std::exp(x[i].y);
                #else
                y[i] = fw::math<element_type>::exp(x[i]);
                #endif
            }

            return (omp_get_wtime() - time);
        }

        template <>
        template <>
        double kernel<element_type>::exp<2>(const fw::buffer<element_type, 2, fw::data_layout::SoA>& x, fw::buffer<element_type, 2, fw::data_layout::SoA>& y)
        {
            double time = omp_get_wtime();
            
            for (std::size_t j = 0; j < x.n[1]; ++j)
            {
                #if defined(__INTEL_COMPILER)
                #pragma forceinline recursive
                #endif
                #pragma omp simd
                for (std::size_t i = 0; i < x.n[0]; ++i)
                {
                    #if defined(ELEMENT_ACCESS)
                    y[j][i].y = std::exp(x[j][i].y);
                    #else
                    y[j][i] = fw::math<element_type>::exp(x[j][i]);
                    #endif
                }
            }

            return (omp_get_wtime() - time);
        }

        template <>
        template <>
        double kernel<element_type>::exp<3>(const fw::buffer<element_type, 3, fw::data_layout::SoA>& x, fw::buffer<element_type, 3, fw::data_layout::SoA>& y)
        {
            double time = omp_get_wtime();
            
            for (std::size_t k = 0; k < x.n[2]; ++k)
            {
                for (std::size_t j = 0; j < x.n[1]; ++j)
                {
                    #if defined(__INTEL_COMPILER)
                    #pragma forceinline recursive
                    #endif
                    #pragma omp simd
                    for (std::size_t i = 0; i < x.n[0]; ++i)
                    {
                        #if defined(ELEMENT_ACCESS)
                        y[k][j][i].y = std::exp(x[k][j][i].y);
                        #else
                        y[k][j][i] = fw::math<element_type>::exp(x[k][j][i]);
                        #endif
                    }
                }
            }

            return (omp_get_wtime() - time);
        }

        template <>
        template <>
        double kernel<element_type>::log<1>(const fw::buffer<element_type, 1, fw::data_layout::SoA>& x, fw::buffer<element_type, 1, fw::data_layout::SoA>& y)
        {
            double time = omp_get_wtime();
            
            #if defined(__INTEL_COMPILER)
            #pragma forceinline recursive
            #endif
            #pragma omp simd
            for (std::size_t i = 0; i < x.n[0]; ++i)
            {
                #if defined(ELEMENT_ACCESS)
                y[i].y = std::log(x[i].y);
                #else
                y[i] = fw::math<element_type>::log(x[i]);
                #endif
            }

            return (omp_get_wtime() - time);
        }

        template <>
        template <>
        double kernel<element_type>::log<2>(const fw::buffer<element_type, 2, fw::data_layout::SoA>& x, fw::buffer<element_type, 2, fw::data_layout::SoA>& y)
        {
            double time = omp_get_wtime();
            
            for (std::size_t j = 0; j < x.n[1]; ++j)
            {
                #if defined(__INTEL_COMPILER)
                #pragma forceinline recursive
                #endif
                #pragma omp simd
                for (std::size_t i = 0; i < x.n[0]; ++i)
                {
                    #if defined(ELEMENT_ACCESS)
                    y[j][i].y = std::log(x[j][i].y);
                    #else
                    y[j][i] = fw::math<element_type>::log(x[j][i]);
                    #endif
                }
            }

            return (omp_get_wtime() - time);
        }

        template <>
        template <>
        double kernel<element_type>::log<3>(const fw::buffer<element_type, 3, fw::data_layout::SoA>& x, fw::buffer<element_type, 3, fw::data_layout::SoA>& y)
        {
            double time = omp_get_wtime();

            for (std::size_t k = 0; k < x.n[2]; ++k)
            {
                for (std::size_t j = 0; j < x.n[1]; ++j)
                {
                    #if defined(__INTEL_COMPILER)
                    #pragma forceinline recursive
                    #endif
                    #pragma omp simd
                    for (std::size_t i = 0; i < x.n[0]; ++i)
                    {
                        #if defined(ELEMENT_ACCESS)
                        y[k][j][i].y = std::log(x[k][j][i].y);
                        #else
                        y[k][j][i] = fw::math<element_type>::log(x[k][j][i]);
                        #endif
                    }
                }
            }
            
            return (omp_get_wtime() - time);
        }
    #endif
#endif