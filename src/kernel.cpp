// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#include <cstdint>
#include <omp.h>

#include "kernel.hpp"

using namespace fw;

#if defined(AOS_LAYOUT)
    template <>
    template <>
    double kernel<element_type>::exp<1>(fw::buffer<element_type, 1, fw::data_layout::AoS>& x)
    {
        double time = omp_get_wtime();
        
        #if defined(__INTEL_COMPILER)
        #pragma forceinline recursive
        #pragma omp simd
        #endif
        for (std::size_t i = 0; i < x.n[0]; ++i)
        {
            #if defined(ELEMENT_ACCESS)
            x[i].y = std::exp(x[i].y);
            #else
            x[i] = fw::exp(x[i]);
            #endif
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
                #pragma omp simd
                #endif
                for (std::size_t i = 0; i < x.n[0]; ++i)
                {
                    #if defined(ELEMENT_ACCESS)
                    x[k][j][i].y = std::exp(x[k][j][i].y);
                    #else
                    x[k][j][i] = fw::exp(x[k][j][i]);
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
        #pragma omp simd
        #endif
        for (std::size_t i = 0; i < x.n[0]; ++i)
        {
            #if defined(ELEMENT_ACCESS)
            x[i].y = std::log(x[i].y);
            #else
            x[i] = fw::log(x[i]);
            #endif
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
                #pragma omp simd
                #endif
                for (std::size_t i = 0; i < x.n[0]; ++i)
                {
                    #if defined(ELEMENT_ACCESS)
                    x[k][j][i].y = std::log(x[k][j][i].y);
                    #else
                    x[k][j][i] = fw::log(x[k][j][i]);
                    #endif
                }
            }
        }
        
        return (omp_get_wtime() - time);
    }
#elif defined(SOA_LAYOUT)
    template <>
    template <>
    double kernel<element_type>::exp<1>(fw::buffer<element_type, 1, fw::data_layout::SoA>& x)
    {
        double time = omp_get_wtime();

        #if defined(__INTEL_COMPILER)
        #pragma forceinline recursive
        #pragma omp simd
        #endif
        for (std::size_t i = 0; i < x.n[0]; ++i)
        {
            #if defined(ELEMENT_ACCESS)
            x[i].y = std::exp(x[i].y);
            #else
            x[i] = fw::exp(x[i]);
            #endif
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
                #pragma omp simd
                #endif
                for (std::size_t i = 0; i < x.n[0]; ++i)
                {
                    #if defined(ELEMENT_ACCESS)
                    x[k][j][i].y = std::exp(x[k][j][i].y);
                    #else
                    x[k][j][i] = fw::exp(x[k][j][i]);
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
        #pragma omp simd
        #endif
        for (std::size_t i = 0; i < x.n[0]; ++i)
        {
            #if defined(ELEMENT_ACCESS)
            x[i].y = std::log(x[i].y);
            #else
            x[i] = fw::log(x[i]);
            #endif
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
                #pragma omp simd
                #endif
                for (std::size_t i = 0; i < x.n[0]; ++i)
                {
                    #if defined(ELEMENT_ACCESS)
                    x[k][j][i].y = std::log(x[k][j][i].y);
                    #else
                    x[k][j][i] = fw::log(x[k][j][i]);
                    #endif
                }
            }
        }
        
        return (omp_get_wtime() - time);
    }
#endif