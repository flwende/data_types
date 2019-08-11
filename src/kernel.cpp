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
        double kernel<element_type>::cross<1, 1>(const fw::field<element_type, 1, fw::data_layout::AoS>& x_1, const fw::field<element_type, 1, fw::data_layout::AoS>& x_2, fw::field<element_type, 1, fw::data_layout::AoS>& y, const array_type<1>& n)
        {
            double time = omp_get_wtime();

            #if defined(__INTEL_COMPILER)
            #pragma forceinline recursive
            #endif
            #pragma omp simd        
            for (size_type i = 0; i < n[0]; ++i)
            {
                y[i] = 0.5 + fw::math<element_type>::log(1.0 + fw::cross(x_1[i], fw::math<element_type>::exp(x_2[i])));
            }

            return (omp_get_wtime() - time);
        }

        template <>
        template <>
        double kernel<element_type>::cross<2, 2>(const fw::field<element_type, 2, fw::data_layout::AoS>& x_1, const fw::field<element_type, 2, fw::data_layout::AoS>& x_2, fw::field<element_type, 2, fw::data_layout::AoS>& y, const array_type<2>& n)
        {
            double time = omp_get_wtime();

            for (size_type j = 0; j < n[1]; ++j)
            {
                #if defined(__INTEL_COMPILER)
                #pragma forceinline recursive
                #endif
                #pragma omp simd
                for (size_type i = 0; i < n[0]; ++i)
                {
                    y[j][i] = 0.5 + fw::math<element_type>::log(1.0 + fw::cross(x_1[j][i], fw::math<element_type>::exp(x_2[j][i])));
                }
            }

            return (omp_get_wtime() - time);
        }

        template <>
        template <>
        double kernel<element_type>::cross<3, 3>(const fw::field<element_type, 3, fw::data_layout::AoS>& x_1, const fw::field<element_type, 3, fw::data_layout::AoS>& x_2, fw::field<element_type, 3, fw::data_layout::AoS>& y, const array_type<3>& n)
        {
            double time = omp_get_wtime();

            for (size_type k = 0; k < n[2]; ++k)
            {
                for (size_type j = 0; j < n[1]; ++j)
                {
                    #if defined(__INTEL_COMPILER)
                    #pragma forceinline recursive
                    #endif
                    #pragma omp simd
                    for (size_type i = 0; i < n[0]; ++i)
                    {
                        y[k][j][i] = 0.5 + fw::math<element_type>::log(1.0 + fw::cross(x_1[k][j][i], fw::math<element_type>::exp(x_2[k][j][i])));
                    }
                }
            }

            return (omp_get_wtime() - time);
        }

        template <>
        template <>
        double kernel<element_type>::cross<3, 1>(const fw::field<element_type, 1, fw::data_layout::AoS>& x_1, const fw::field<element_type, 1, fw::data_layout::AoS>& x_2, fw::field<element_type, 1, fw::data_layout::AoS>& y, const array_type<3>& n)
        {
            double time = omp_get_wtime();

            for (size_type k = 0; k < n[2]; ++k)
            {
                for (size_type j = 0; j < n[1]; ++j)
                {
                    #if defined(__INTEL_COMPILER)
                    #pragma forceinline recursive
                    #endif
                    #pragma omp simd
                    for (size_type i = 0; i < n[0]; ++i)
                    {
                        const size_type index = (k * n[1] + j) * n[0] + i;
                        y[index] = 0.5 + fw::math<element_type>::log(1.0 + fw::cross(x_1[index], fw::math<element_type>::exp(x_2[index])));
                    }
                }
            }

            return (omp_get_wtime() - time);
        }
    #else
        template <>
        template <>
        double kernel<element_type>::exp<1, 1>(fw::field<element_type, 1, fw::data_layout::AoS>& x, const array_type<1>& n)
        {
            double time = omp_get_wtime();
            
            #if defined(__INTEL_COMPILER)
            #pragma forceinline recursive
            #endif
            #pragma omp simd
            for (size_type i = 0; i < n[0]; ++i)
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
        double kernel<element_type>::exp<2, 2>(fw::field<element_type, 2, fw::data_layout::AoS>& x, const array_type<2>& n)
        {
            double time = omp_get_wtime();
            
            for (size_type j = 0; j < n[1]; ++j)
            {
                #if defined(__INTEL_COMPILER)
                #pragma forceinline recursive
                #endif
                #pragma omp simd
                for (size_type i = 0; i < n[0]; ++i)
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
        double kernel<element_type>::exp<3, 3>(fw::field<element_type, 3, fw::data_layout::AoS>& x, const array_type<3>& n)
        {
            double time = omp_get_wtime();
            
            for (size_type k = 0; k < n[2]; ++k)
            {
                for (size_type j = 0; j < n[1]; ++j)
                {
                    #if defined(__INTEL_COMPILER)
                    #pragma forceinline recursive
                    #endif
                    #pragma omp simd
                    for (size_type i = 0; i < n[0]; ++i)
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
        double kernel<element_type>::exp<3, 1>(fw::field<element_type, 1, fw::data_layout::AoS>& x, const array_type<3>& n)
        {
            double time = omp_get_wtime();
            
            for (size_type k = 0; k < n[2]; ++k)
            {
                for (size_type j = 0; j < n[1]; ++j)
                {
                    #if defined(__INTEL_COMPILER)
                    #pragma forceinline recursive
                    #endif
                    #pragma omp simd
                    for (size_type i = 0; i < n[0]; ++i)
                    {
                        const size_type index = (k * n[1] + j) * n[0] + i;
                        #if defined(ELEMENT_ACCESS)
                        x[index].y = std::exp(x[index].y);
                        #else
                        x[index] = fw::math<element_type>::exp(x[index]);
                        #endif
                    }
                }
            }

            return (omp_get_wtime() - time);
        }

        template <>
        template <>
        double kernel<element_type>::log<1, 1>(fw::field<element_type, 1, fw::data_layout::AoS>& x, const array_type<1>& n)
        {
            double time = omp_get_wtime();
            
            #if defined(__INTEL_COMPILER)
            #pragma forceinline recursive
            #endif
            #pragma omp simd
            for (size_type i = 0; i < n[0]; ++i)
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
        double kernel<element_type>::log<2, 2>(fw::field<element_type, 2, fw::data_layout::AoS>& x, const array_type<2>& n)
        {
            double time = omp_get_wtime();
            
            for (size_type j = 0; j < n[1]; ++j)
            {
                #if defined(__INTEL_COMPILER)
                #pragma forceinline recursive
                #endif
                #pragma omp simd
                for (size_type i = 0; i < n[0]; ++i)
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
        double kernel<element_type>::log<3, 3>(fw::field<element_type, 3, fw::data_layout::AoS>& x, const array_type<3>& n)
        {
            double time = omp_get_wtime();

            for (size_type k = 0; k < n[2]; ++k)
            {
                for (size_type j = 0; j < n[1]; ++j)
                {
                    #if defined(__INTEL_COMPILER)
                    #pragma forceinline recursive
                    #endif
                    #pragma omp simd
                    for (size_type i = 0; i < n[0]; ++i)
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
        double kernel<element_type>::log<3, 1>(fw::field<element_type, 1, fw::data_layout::AoS>& x, const array_type<3>& n)
        {
            double time = omp_get_wtime();

            for (size_type k = 0; k < n[2]; ++k)
            {
                for (size_type j = 0; j < n[1]; ++j)
                {
                    #if defined(__INTEL_COMPILER)
                    #pragma forceinline recursive
                    #endif
                    #pragma omp simd
                    for (size_type i = 0; i < n[0]; ++i)
                    {
                        const size_type index = (k * n[1] + j) * n[0] + i;
                        #if defined(ELEMENT_ACCESS)
                        x[index].y = std::log(x[index].y);
                        #else
                        x[index] = fw::math<element_type>::log(x[index]);
                        #endif
                    }
                }
            }
            
            return (omp_get_wtime() - time);
        }

        template <>
        template <>
        double kernel<element_type>::exp<1, 1>(const fw::field<element_type, 1, fw::data_layout::AoS>& x, fw::field<element_type, 1, fw::data_layout::AoS>& y, const array_type<1>& n)
        {
            double time = omp_get_wtime();
            
            #if defined(__INTEL_COMPILER)
            #pragma forceinline recursive
            #endif
            #pragma omp simd
            for (size_type i = 0; i < n[0]; ++i)
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
        double kernel<element_type>::exp<2, 2>(const fw::field<element_type, 2, fw::data_layout::AoS>& x, fw::field<element_type, 2, fw::data_layout::AoS>& y, const array_type<2>& n)
        {
            double time = omp_get_wtime();
            
            for (size_type j = 0; j < n[1]; ++j)
            {
                #if defined(__INTEL_COMPILER)
                #pragma forceinline recursive
                #endif
                #pragma omp simd
                for (size_type i = 0; i < n[0]; ++i)
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
        double kernel<element_type>::exp<3, 3>(const fw::field<element_type, 3, fw::data_layout::AoS>& x, fw::field<element_type, 3, fw::data_layout::AoS>& y, const array_type<3>& n)
        {
            double time = omp_get_wtime();
            
            for (size_type k = 0; k < n[2]; ++k)
            {
                for (size_type j = 0; j < n[1]; ++j)
                {
                    #if defined(__INTEL_COMPILER)
                    #pragma forceinline recursive
                    #endif
                    #pragma omp simd
                    for (size_type i = 0; i < n[0]; ++i)
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
        double kernel<element_type>::exp<3, 1>(const fw::field<element_type, 1, fw::data_layout::AoS>& x, fw::field<element_type, 1, fw::data_layout::AoS>& y, const array_type<3>& n)
        {
            double time = omp_get_wtime();
            
            for (size_type k = 0; k < n[2]; ++k)
            {
                for (size_type j = 0; j < n[1]; ++j)
                {
                    #if defined(__INTEL_COMPILER)
                    #pragma forceinline recursive
                    #endif
                    #pragma omp simd
                    for (size_type i = 0; i < n[0]; ++i)
                    {
                        const size_type index = (k * n[1] + j) * n[0] + i;
                        #if defined(ELEMENT_ACCESS)
                        y[index].y = std::exp(x[index].y);
                        #else
                        y[index] = fw::math<element_type>::exp(x[index]);
                        #endif
                    }
                }
            }

            return (omp_get_wtime() - time);
        }

        template <>
        template <>
        double kernel<element_type>::log<1, 1>(const fw::field<element_type, 1, fw::data_layout::AoS>& x, fw::field<element_type, 1, fw::data_layout::AoS>& y, const array_type<1>& n)
        {
            double time = omp_get_wtime();
            
            #if defined(__INTEL_COMPILER)
            #pragma forceinline recursive
            #endif
            #pragma omp simd
            for (size_type i = 0; i < n[0]; ++i)
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
        double kernel<element_type>::log<2, 2>(const fw::field<element_type, 2, fw::data_layout::AoS>& x, fw::field<element_type, 2, fw::data_layout::AoS>& y, const array_type<2>& n)
        {
            double time = omp_get_wtime();
            
            for (size_type j = 0; j < n[1]; ++j)
            {
                #if defined(__INTEL_COMPILER)
                #pragma forceinline recursive
                #endif
                #pragma omp simd
                for (size_type i = 0; i < n[0]; ++i)
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
        double kernel<element_type>::log<3, 3>(const fw::field<element_type, 3, fw::data_layout::AoS>& x, fw::field<element_type, 3, fw::data_layout::AoS>& y, const array_type<3>& n)
        {
            double time = omp_get_wtime();

            for (size_type k = 0; k < n[2]; ++k)
            {
                for (size_type j = 0; j < n[1]; ++j)
                {
                    #if defined(__INTEL_COMPILER)
                    #pragma forceinline recursive
                    #endif
                    #pragma omp simd
                    for (size_type i = 0; i < n[0]; ++i)
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

        template <>
        template <>
        double kernel<element_type>::log<3, 1>(const fw::field<element_type, 1, fw::data_layout::AoS>& x, fw::field<element_type, 1, fw::data_layout::AoS>& y, const array_type<3>& n)
        {
            double time = omp_get_wtime();

            for (size_type k = 0; k < n[2]; ++k)
            {
                for (size_type j = 0; j < n[1]; ++j)
                {
                    #if defined(__INTEL_COMPILER)
                    #pragma forceinline recursive
                    #endif
                    #pragma omp simd
                    for (size_type i = 0; i < n[0]; ++i)
                    {
                        const size_type index = (k * n[1] + j) * n[0] + i;
                        #if defined(ELEMENT_ACCESS)
                        y[index].y = std::log(x[index].y);
                        #else
                        y[index] = fw::math<element_type>::log(x[index]);
                        #endif
                    }
                }
            }
            
            return (omp_get_wtime() - time);
        }
    #endif
#elif defined(SOA_LAYOUT)
    #if defined(VECTOR_PRODUCT)
        template <>
        template <>
        double kernel<element_type>::cross<1, 1>(const fw::field<element_type, 1, fw::data_layout::SoA>& x_1, const fw::field<element_type, 1, fw::data_layout::SoA>& x_2, fw::field<element_type, 1, fw::data_layout::SoA>& y, const array_type<1>& n)
        {
            double time = omp_get_wtime();

            #if defined(__INTEL_COMPILER)
            #pragma forceinline recursive
            #endif
            #pragma omp simd        
            for (size_type i = 0; i < n[0]; ++i)
            {
                y[i] = 0.5 + fw::math<element_type>::log(1.0 + fw::cross(x_1[i], fw::math<element_type>::exp(x_2[i])));
            }

            return (omp_get_wtime() - time);
        }

        template <>
        template <>
        double kernel<element_type>::cross<2, 2>(const fw::field<element_type, 2, fw::data_layout::SoA>& x_1, const fw::field<element_type, 2, fw::data_layout::SoA>& x_2, fw::field<element_type, 2, fw::data_layout::SoA>& y, const array_type<2>& n)
        {
            double time = omp_get_wtime();

            for (size_type j = 0; j < n[1]; ++j)
            {
                #if defined(__INTEL_COMPILER)
                #pragma forceinline recursive
                #endif
                #pragma omp simd
                for (size_type i = 0; i < n[0]; ++i)
                {
                    y[j][i] = 0.5 + fw::math<element_type>::log(1.0 + fw::cross(x_1[j][i], fw::math<element_type>::exp(x_2[j][i])));
                }
            }

            return (omp_get_wtime() - time);
        }

        template <>
        template <>
        double kernel<element_type>::cross<3, 3>(const fw::field<element_type, 3, fw::data_layout::SoA>& x_1, const fw::field<element_type, 3, fw::data_layout::SoA>& x_2, fw::field<element_type, 3, fw::data_layout::SoA>& y, const array_type<3>& n)
        {
            double time = omp_get_wtime();

            for (size_type k = 0; k < n[2]; ++k)
            {
                for (size_type j = 0; j < n[1]; ++j)
                {
                    #if defined(__INTEL_COMPILER)
                    #pragma forceinline recursive
                    #endif
                    #pragma omp simd
                    for (size_type i = 0; i < n[0]; ++i)
                    {
                        y[k][j][i] = 0.5 + fw::math<element_type>::log(1.0 + fw::cross(x_1[k][j][i], fw::math<element_type>::exp(x_2[k][j][i])));
                    }
                }
            }

            return (omp_get_wtime() - time);
        }

        template <>
        template <>
        double kernel<element_type>::cross<3, 1>(const fw::field<element_type, 1, fw::data_layout::SoA>& x_1, const fw::field<element_type, 1, fw::data_layout::SoA>& x_2, fw::field<element_type, 1, fw::data_layout::SoA>& y, const array_type<3>& n)
        {
            double time = omp_get_wtime();

            for (size_type k = 0; k < n[2]; ++k)
            {
                for (size_type j = 0; j < n[1]; ++j)
                {
                    #if defined(__INTEL_COMPILER)
                    #pragma forceinline recursive
                    #endif
                    #pragma omp simd
                    for (size_type i = 0; i < n[0]; ++i)
                    {
                        const size_type index = (k * n[1] + j) * n[0] + i;
                        y[index] = 0.5 + fw::math<element_type>::log(1.0 + fw::cross(x_1[index], fw::math<element_type>::exp(x_2[index])));
                    }
                }
            }

            return (omp_get_wtime() - time);
        }
    #else
        template <>
        template <>
        double kernel<element_type>::exp<1, 1>(fw::field<element_type, 1, fw::data_layout::SoA>& x, const array_type<1>& n)
        {
            double time = omp_get_wtime();
            
            #if defined(__INTEL_COMPILER)
            #pragma forceinline recursive
            #endif
            #pragma omp simd
            for (size_type i = 0; i < n[0]; ++i)
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
        double kernel<element_type>::exp<2, 2>(fw::field<element_type, 2, fw::data_layout::SoA>& x, const array_type<2>& n)
        {
            double time = omp_get_wtime();
            
            for (size_type j = 0; j < n[1]; ++j)
            {
                #if defined(__INTEL_COMPILER)
                #pragma forceinline recursive
                #endif
                #pragma omp simd
                for (size_type i = 0; i < n[0]; ++i)
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
        double kernel<element_type>::exp<3, 3>(fw::field<element_type, 3, fw::data_layout::SoA>& x, const array_type<3>& n)
        {
            double time = omp_get_wtime();
            
            for (size_type k = 0; k < n[2]; ++k)
            {
                for (size_type j = 0; j < n[1]; ++j)
                {
                    #if defined(__INTEL_COMPILER)
                    #pragma forceinline recursive
                    #endif
                    #pragma omp simd
                    for (size_type i = 0; i < n[0]; ++i)
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
        double kernel<element_type>::exp<3, 1>(fw::field<element_type, 1, fw::data_layout::SoA>& x, const array_type<3>& n)
        {
            double time = omp_get_wtime();
            
            for (size_type k = 0; k < n[2]; ++k)
            {
                for (size_type j = 0; j < n[1]; ++j)
                {
                    #if defined(__INTEL_COMPILER)
                    #pragma forceinline recursive
                    #endif
                    #pragma omp simd
                    for (size_type i = 0; i < n[0]; ++i)
                    {
                        const size_type index = (k * n[1] + j) * n[0] + i;
                        #if defined(ELEMENT_ACCESS)
                        x[index].y = std::exp(x[index].y);
                        #else
                        x[index] = fw::math<element_type>::exp(x[index]);
                        #endif
                    }
                }
            }

            return (omp_get_wtime() - time);
        }

        template <>
        template <>
        double kernel<element_type>::log<1, 1>(fw::field<element_type, 1, fw::data_layout::SoA>& x, const array_type<1>& n)
        {
            double time = omp_get_wtime();
            
            #if defined(__INTEL_COMPILER)
            #pragma forceinline recursive
            #endif
            #pragma omp simd
            for (size_type i = 0; i < n[0]; ++i)
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
        double kernel<element_type>::log<2, 2>(fw::field<element_type, 2, fw::data_layout::SoA>& x, const array_type<2>& n)
        {
            double time = omp_get_wtime();
            
            for (size_type j = 0; j < n[1]; ++j)
            {
                #if defined(__INTEL_COMPILER)
                #pragma forceinline recursive
                #endif
                #pragma omp simd
                for (size_type i = 0; i < n[0]; ++i)
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
        double kernel<element_type>::log<3, 3>(fw::field<element_type, 3, fw::data_layout::SoA>& x, const array_type<3>& n)
        {
            double time = omp_get_wtime();

            for (size_type k = 0; k < n[2]; ++k)
            {
                for (size_type j = 0; j < n[1]; ++j)
                {
                    #if defined(__INTEL_COMPILER)
                    #pragma forceinline recursive
                    #endif
                    #pragma omp simd
                    for (size_type i = 0; i < n[0]; ++i)
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
        double kernel<element_type>::log<3, 1>(fw::field<element_type, 1, fw::data_layout::SoA>& x, const array_type<3>& n)
        {
            double time = omp_get_wtime();

            for (size_type k = 0; k < n[2]; ++k)
            {
                for (size_type j = 0; j < n[1]; ++j)
                {
                    #if defined(__INTEL_COMPILER)
                    #pragma forceinline recursive
                    #endif
                    #pragma omp simd
                    for (size_type i = 0; i < n[0]; ++i)
                    {
                        const size_type index = (k * n[1] + j) * n[0] + i;
                        #if defined(ELEMENT_ACCESS)
                        x[index].y = std::log(x[index].y);
                        #else
                        x[index] = fw::math<element_type>::log(x[index]);
                        #endif
                    }
                }
            }
            
            return (omp_get_wtime() - time);
        }

        template <>
        template <>
        double kernel<element_type>::exp<1, 1>(const fw::field<element_type, 1, fw::data_layout::SoA>& x, fw::field<element_type, 1, fw::data_layout::SoA>& y, const array_type<1>& n)
        {
            double time = omp_get_wtime();
            
            #if defined(__INTEL_COMPILER)
            #pragma forceinline recursive
            #endif
            #pragma omp simd
            for (size_type i = 0; i < n[0]; ++i)
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
        double kernel<element_type>::exp<2, 2>(const fw::field<element_type, 2, fw::data_layout::SoA>& x, fw::field<element_type, 2, fw::data_layout::SoA>& y, const array_type<2>& n)
        {
            double time = omp_get_wtime();
            
            for (size_type j = 0; j < n[1]; ++j)
            {
                #if defined(__INTEL_COMPILER)
                #pragma forceinline recursive
                #endif
                #pragma omp simd
                for (size_type i = 0; i < n[0]; ++i)
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
        double kernel<element_type>::exp<3, 3>(const fw::field<element_type, 3, fw::data_layout::SoA>& x, fw::field<element_type, 3, fw::data_layout::SoA>& y, const array_type<3>& n)
        {
            double time = omp_get_wtime();
            
            for (size_type k = 0; k < n[2]; ++k)
            {
                for (size_type j = 0; j < n[1]; ++j)
                {
                    #if defined(__INTEL_COMPILER)
                    #pragma forceinline recursive
                    #endif
                    #pragma omp simd
                    for (size_type i = 0; i < n[0]; ++i)
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
        double kernel<element_type>::exp<3, 1>(const fw::field<element_type, 1, fw::data_layout::SoA>& x, fw::field<element_type, 1, fw::data_layout::SoA>& y, const array_type<3>& n)
        {
            double time = omp_get_wtime();
            
            for (size_type k = 0; k < n[2]; ++k)
            {
                for (size_type j = 0; j < n[1]; ++j)
                {
                    #if defined(__INTEL_COMPILER)
                    #pragma forceinline recursive
                    #endif
                    #pragma omp simd
                    for (size_type i = 0; i < n[0]; ++i)
                    {
                        const size_type index = (k * n[1] + j) * n[0] + i;
                        #if defined(ELEMENT_ACCESS)
                        y[index].y = std::exp(x[index].y);
                        #else
                        y[index] = fw::math<element_type>::exp(x[index]);
                        #endif
                    }
                }
            }

            return (omp_get_wtime() - time);
        }

        template <>
        template <>
        double kernel<element_type>::log<1, 1>(const fw::field<element_type, 1, fw::data_layout::SoA>& x, fw::field<element_type, 1, fw::data_layout::SoA>& y, const array_type<1>& n)
        {
            double time = omp_get_wtime();
            
            #if defined(__INTEL_COMPILER)
            #pragma forceinline recursive
            #endif
            #pragma omp simd
            for (size_type i = 0; i < n[0]; ++i)
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
        double kernel<element_type>::log<2, 2>(const fw::field<element_type, 2, fw::data_layout::SoA>& x, fw::field<element_type, 2, fw::data_layout::SoA>& y, const array_type<2>& n)
        {
            double time = omp_get_wtime();
            
            for (size_type j = 0; j < n[1]; ++j)
            {
                #if defined(__INTEL_COMPILER)
                #pragma forceinline recursive
                #endif
                #pragma omp simd
                for (size_type i = 0; i < n[0]; ++i)
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
        double kernel<element_type>::log<3, 3>(const fw::field<element_type, 3, fw::data_layout::SoA>& x, fw::field<element_type, 3, fw::data_layout::SoA>& y, const array_type<3>& n)
        {
            double time = omp_get_wtime();

            for (size_type k = 0; k < n[2]; ++k)
            {
                for (size_type j = 0; j < n[1]; ++j)
                {
                    #if defined(__INTEL_COMPILER)
                    #pragma forceinline recursive
                    #endif
                    #pragma omp simd
                    for (size_type i = 0; i < n[0]; ++i)
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

        template <>
        template <>
        double kernel<element_type>::log<3, 1>(const fw::field<element_type, 1, fw::data_layout::SoA>& x, fw::field<element_type, 1, fw::data_layout::SoA>& y, const array_type<3>& n)
        {
            double time = omp_get_wtime();

            for (size_type k = 0; k < n[2]; ++k)
            {
                for (size_type j = 0; j < n[1]; ++j)
                {
                    #if defined(__INTEL_COMPILER)
                    #pragma forceinline recursive
                    #endif
                    #pragma omp simd
                    for (size_type i = 0; i < n[0]; ++i)
                    {
                        const size_type index = (k * n[1] + j) * n[0] + i;
                        #if defined(ELEMENT_ACCESS)
                        y[index].y = std::log(x[index].y);
                        #else
                        y[index] = fw::math<element_type>::log(x[index]);
                        #endif
                    }
                }
            }
            
            return (omp_get_wtime() - time);
        }
    #endif
#elif defined(SOAI_LAYOUT)
    #if defined(VECTOR_PRODUCT)
        template <>
        template <>
        double kernel<element_type>::cross<1, 1>(const fw::field<element_type, 1, fw::data_layout::SoAi>& x_1, const fw::field<element_type, 1, fw::data_layout::SoAi>& x_2, fw::field<element_type, 1, fw::data_layout::SoAi>& y, const array_type<1>& n)
        {
            double time = omp_get_wtime();

            #if defined(__INTEL_COMPILER)
            #pragma forceinline recursive
            #endif
            #pragma omp simd        
            for (size_type i = 0; i < n[0]; ++i)
            {
                y[i] = 0.5 + fw::math<element_type>::log(1.0 + fw::cross(x_1[i], fw::math<element_type>::exp(x_2[i])));
            }

            return (omp_get_wtime() - time);
        }

        template <>
        template <>
        double kernel<element_type>::cross<2, 2>(const fw::field<element_type, 2, fw::data_layout::SoAi>& x_1, const fw::field<element_type, 2, fw::data_layout::SoAi>& x_2, fw::field<element_type, 2, fw::data_layout::SoAi>& y, const array_type<2>& n)
        {
            double time = omp_get_wtime();

            for (size_type j = 0; j < n[1]; ++j)
            {
                #if defined(__INTEL_COMPILER)
                #pragma forceinline recursive
                #endif
                #pragma omp simd
                for (size_type i = 0; i < n[0]; ++i)
                {
                    y[j][i] = 0.5 + fw::math<element_type>::log(1.0 + fw::cross(x_1[j][i], fw::math<element_type>::exp(x_2[j][i])));
                }
            }

            return (omp_get_wtime() - time);
        }

        template <>
        template <>
        double kernel<element_type>::cross<3, 3>(const fw::field<element_type, 3, fw::data_layout::SoAi>& x_1, const fw::field<element_type, 3, fw::data_layout::SoAi>& x_2, fw::field<element_type, 3, fw::data_layout::SoAi>& y, const array_type<3>& n)
        {
            double time = omp_get_wtime();

            for (size_type k = 0; k < n[2]; ++k)
            {
                for (size_type j = 0; j < n[1]; ++j)
                {
                    #if defined(__INTEL_COMPILER)
                    #pragma forceinline recursive
                    #endif
                    #pragma omp simd
                    for (size_type i = 0; i < n[0]; ++i)
                    {
                        y[k][j][i] = 0.5 + fw::math<element_type>::log(1.0 + fw::cross(x_1[k][j][i], fw::math<element_type>::exp(x_2[k][j][i])));
                    }
                }
            }

            return (omp_get_wtime() - time);
        }

        template <>
        template <>
        double kernel<element_type>::cross<3, 1>(const fw::field<element_type, 1, fw::data_layout::SoAi>& x_1, const fw::field<element_type, 1, fw::data_layout::SoAi>& x_2, fw::field<element_type, 1, fw::data_layout::SoAi>& y, const array_type<3>& n)
        {
            double time = omp_get_wtime();

            for (size_type k = 0; k < n[2]; ++k)
            {
                for (size_type j = 0; j < n[1]; ++j)
                {
                    #if defined(__INTEL_COMPILER)
                    #pragma forceinline recursive
                    #endif
                    #pragma omp simd
                    for (size_type i = 0; i < n[0]; ++i)
                    {
                        const size_type index = (k * n[1] + j) * n[0] + i;
                        y[index] = 0.5 + fw::math<element_type>::log(1.0 + fw::cross(x_1[index], fw::math<element_type>::exp(x_2[index])));
                    }
                }
            }

            return (omp_get_wtime() - time);
        }
    #else
        template <>
        template <>
        double kernel<element_type>::exp<1, 1>(fw::field<element_type, 1, fw::data_layout::SoAi>& x, const array_type<1>& n)
        {
            double time = omp_get_wtime();
            
            #if defined(__INTEL_COMPILER)
            #pragma forceinline recursive
            #endif
            #pragma omp simd
            for (size_type i = 0; i < n[0]; ++i)
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
        double kernel<element_type>::exp<2, 2>(fw::field<element_type, 2, fw::data_layout::SoAi>& x, const array_type<2>& n)
        {
            double time = omp_get_wtime();
            
            for (size_type j = 0; j < n[1]; ++j)
            {
                #if defined(__INTEL_COMPILER)
                #pragma forceinline recursive
                #endif
                #pragma omp simd
                for (size_type i = 0; i < n[0]; ++i)
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
        double kernel<element_type>::exp<3, 3>(fw::field<element_type, 3, fw::data_layout::SoAi>& x, const array_type<3>& n)
        {
            double time = omp_get_wtime();
            
            for (size_type k = 0; k < n[2]; ++k)
            {
                for (size_type j = 0; j < n[1]; ++j)
                {
                    #if defined(__INTEL_COMPILER)
                    #pragma forceinline recursive
                    #endif
                    #pragma omp simd
                    for (size_type i = 0; i < n[0]; ++i)
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
        double kernel<element_type>::exp<3, 1>(fw::field<element_type, 1, fw::data_layout::SoAi>& x, const array_type<3>& n)
        {
            double time = omp_get_wtime();
            
            for (size_type k = 0; k < n[2]; ++k)
            {
                for (size_type j = 0; j < n[1]; ++j)
                {
                    #if defined(__INTEL_COMPILER)
                    #pragma forceinline recursive
                    #endif
                    #pragma omp simd
                    for (size_type i = 0; i < n[0]; ++i)
                    {
                        const size_type index = (k * n[1] + j) * n[0] + i;
                        #if defined(ELEMENT_ACCESS)
                        x[index].y = std::exp(x[index].y);
                        #else
                        x[index] = fw::math<element_type>::exp(x[index]);
                        #endif
                    }
                }
            }

            return (omp_get_wtime() - time);
        }

        template <>
        template <>
        double kernel<element_type>::log<1, 1>(fw::field<element_type, 1, fw::data_layout::SoAi>& x, const array_type<1>& n)
        {
            double time = omp_get_wtime();
            
            #if defined(__INTEL_COMPILER)
            #pragma forceinline recursive
            #endif
            #pragma omp simd
            for (size_type i = 0; i < n[0]; ++i)
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
        double kernel<element_type>::log<2, 2>(fw::field<element_type, 2, fw::data_layout::SoAi>& x, const array_type<2>& n)
        {
            double time = omp_get_wtime();
            
            for (size_type j = 0; j < n[1]; ++j)
            {
                #if defined(__INTEL_COMPILER)
                #pragma forceinline recursive
                #endif
                #pragma omp simd
                for (size_type i = 0; i < n[0]; ++i)
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
        double kernel<element_type>::log<3, 3>(fw::field<element_type, 3, fw::data_layout::SoAi>& x, const array_type<3>& n)
        {
            double time = omp_get_wtime();

            for (size_type k = 0; k < n[2]; ++k)
            {
                for (size_type j = 0; j < n[1]; ++j)
                {
                    #if defined(__INTEL_COMPILER)
                    #pragma forceinline recursive
                    #endif
                    #pragma omp simd
                    for (size_type i = 0; i < n[0]; ++i)
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
        double kernel<element_type>::log<3, 1>(fw::field<element_type, 1, fw::data_layout::SoAi>& x, const array_type<3>& n)
        {
            double time = omp_get_wtime();

            for (size_type k = 0; k < n[2]; ++k)
            {
                for (size_type j = 0; j < n[1]; ++j)
                {
                    #if defined(__INTEL_COMPILER)
                    #pragma forceinline recursive
                    #endif
                    #pragma omp simd
                    for (size_type i = 0; i < n[0]; ++i)
                    {
                        const size_type index = (k * n[1] + j) * n[0] + i;
                        #if defined(ELEMENT_ACCESS)
                        x[index].y = std::log(x[index].y);
                        #else
                        x[index] = fw::math<element_type>::log(x[index]);
                        #endif
                    }
                }
            }
            
            return (omp_get_wtime() - time);
        }

        template <>
        template <>
        double kernel<element_type>::exp<1, 1>(const fw::field<element_type, 1, fw::data_layout::SoAi>& x, fw::field<element_type, 1, fw::data_layout::SoAi>& y, const array_type<1>& n)
        {
            double time = omp_get_wtime();
            
            #if defined(__INTEL_COMPILER)
            #pragma forceinline recursive
            #endif
            #pragma omp simd
            for (size_type i = 0; i < n[0]; ++i)
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
        double kernel<element_type>::exp<2, 2>(const fw::field<element_type, 2, fw::data_layout::SoAi>& x, fw::field<element_type, 2, fw::data_layout::SoAi>& y, const array_type<2>& n)
        {
            double time = omp_get_wtime();
            
            for (size_type j = 0; j < n[1]; ++j)
            {
                #if defined(__INTEL_COMPILER)
                #pragma forceinline recursive
                #endif
                #pragma omp simd
                for (size_type i = 0; i < n[0]; ++i)
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
        double kernel<element_type>::exp<3, 3>(const fw::field<element_type, 3, fw::data_layout::SoAi>& x, fw::field<element_type, 3, fw::data_layout::SoAi>& y, const array_type<3>& n)
        {
            double time = omp_get_wtime();
            
            for (size_type k = 0; k < n[2]; ++k)
            {
                for (size_type j = 0; j < n[1]; ++j)
                {
                    #if defined(__INTEL_COMPILER)
                    #pragma forceinline recursive
                    #endif
                    #pragma omp simd
                    for (size_type i = 0; i < n[0]; ++i)
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
        double kernel<element_type>::exp<3, 1>(const fw::field<element_type, 1, fw::data_layout::SoAi>& x, fw::field<element_type, 1, fw::data_layout::SoAi>& y, const array_type<3>& n)
        {
            double time = omp_get_wtime();
            
            for (size_type k = 0; k < n[2]; ++k)
            {
                for (size_type j = 0; j < n[1]; ++j)
                {
                    #if defined(__INTEL_COMPILER)
                    #pragma forceinline recursive
                    #endif
                    #pragma omp simd
                    for (size_type i = 0; i < n[0]; ++i)
                    {
                        const size_type index = (k * n[1] + j) * n[0] + i;
                        #if defined(ELEMENT_ACCESS)
                        y[index].y = std::exp(x[index].y);
                        #else
                        y[index] = fw::math<element_type>::exp(x[index]);
                        #endif
                    }
                }
            }

            return (omp_get_wtime() - time);
        }

        template <>
        template <>
        double kernel<element_type>::log<1, 1>(const fw::field<element_type, 1, fw::data_layout::SoAi>& x, fw::field<element_type, 1, fw::data_layout::SoAi>& y, const array_type<1>& n)
        {
            double time = omp_get_wtime();
            
            #if defined(__INTEL_COMPILER)
            #pragma forceinline recursive
            #endif
            #pragma omp simd
            for (size_type i = 0; i < n[0]; ++i)
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
        double kernel<element_type>::log<2, 2>(const fw::field<element_type, 2, fw::data_layout::SoAi>& x, fw::field<element_type, 2, fw::data_layout::SoAi>& y, const array_type<2>& n)
        {
            double time = omp_get_wtime();
            
            for (size_type j = 0; j < n[1]; ++j)
            {
                #if defined(__INTEL_COMPILER)
                #pragma forceinline recursive
                #endif
                #pragma omp simd
                for (size_type i = 0; i < n[0]; ++i)
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
        double kernel<element_type>::log<3, 3>(const fw::field<element_type, 3, fw::data_layout::SoAi>& x, fw::field<element_type, 3, fw::data_layout::SoAi>& y, const array_type<3>& n)
        {
            double time = omp_get_wtime();

            for (size_type k = 0; k < n[2]; ++k)
            {
                for (size_type j = 0; j < n[1]; ++j)
                {
                    #if defined(__INTEL_COMPILER)
                    #pragma forceinline recursive
                    #endif
                    #pragma omp simd
                    for (size_type i = 0; i < n[0]; ++i)
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

        template <>
        template <>
        double kernel<element_type>::log<3, 1>(const fw::field<element_type, 1, fw::data_layout::SoAi>& x, fw::field<element_type, 1, fw::data_layout::SoAi>& y, const array_type<3>& n)
        {
            double time = omp_get_wtime();

            for (size_type k = 0; k < n[2]; ++k)
            {
                for (size_type j = 0; j < n[1]; ++j)
                {
                    #if defined(__INTEL_COMPILER)
                    #pragma forceinline recursive
                    #endif
                    #pragma omp simd
                    for (size_type i = 0; i < n[0]; ++i)
                    {
                        const size_type index = (k * n[1] + j) * n[0] + i;
                        #if defined(ELEMENT_ACCESS)
                        y[index].y = std::log(x[index].y);
                        #else
                        y[index] = fw::math<element_type>::log(x[index]);
                        #endif
                    }
                }
            }
            
            return (omp_get_wtime() - time);
        }
    #endif    
#endif