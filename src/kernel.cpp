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
        
        auto a_x = x.read_write();

        #if defined(__INTEL_COMPILER)
        #pragma forceinline recursive
        #pragma omp simd
        #endif
        for (std::size_t i = 0; i < x.n[0]; ++i)
        {
            #if defined(ELEMENT_ACCESS)
            a_x[i].x = std::exp(a_x[i].x);
            #else
            a_x[i] = fw::exp(a_x[i]);
            #endif
        }

        return (omp_get_wtime() - time);
    }

    template <>
    template <>
    double kernel<element_type>::exp<3>(fw::buffer<element_type, 3, fw::data_layout::AoS>& x)
    {
        double time = omp_get_wtime();

        auto a_x = x.read_write();	
        
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
                    a_x[k][j][i].y = std::exp(a_x[k][j][i].y);
                    #else
                    a_x[k][j][i] = fw::exp(a_x[k][j][i]);
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
        
        auto a_x = x.read_write();
        
        #if defined(__INTEL_COMPILER)
        #pragma forceinline recursive
        #pragma omp simd
        #endif
        for (std::size_t i = 0; i < x.n[0]; ++i)
        {
            #if defined(ELEMENT_ACCESS)
            a_x[i].x = std::log(a_x[i].x);
            #else
            a_x[i] = fw::log(a_x[i]);
            #endif
        }

        return (omp_get_wtime() - time);
    }

    template <>
    template <>
    double kernel<element_type>::log<3>(fw::buffer<element_type, 3, fw::data_layout::AoS>& x)
    {
        double time = omp_get_wtime();

        auto a_x = x.read_write();	
        
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
                    a_x[k][j][i].y = std::log(a_x[k][j][i].y);
                    #else
                    a_x[k][j][i] = fw::log(a_x[k][j][i]);
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
        
        auto a_x = x.read_write();

        #if defined(__INTEL_COMPILER)
        #pragma forceinline recursive
        #pragma omp simd
        #endif
        for (std::size_t i = 0; i < x.n[0]; ++i)
        {
            #if defined(ELEMENT_ACCESS)
            a_x[i].x = std::exp(a_x[i].x);
            #else
            a_x[i] = fw::exp(a_x[i]);
            #endif
        }

        return (omp_get_wtime() - time);
    }

    template <>
    template <>
    double kernel<element_type>::exp<3>(fw::buffer<element_type, 3, fw::data_layout::SoA>& x)
    {
        double time = omp_get_wtime();

        auto a_x = x.read_write();	

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
                    a_x[k][j][i].y = std::exp(a_x[k][j][i].y);
                    #else
                    a_x[k][j][i] = fw::exp(a_x[k][j][i]);
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
        
        auto a_x = x.read_write();

        #if defined(__INTEL_COMPILER)
        #pragma forceinline recursive
        #pragma omp simd
        #endif
        for (std::size_t i = 0; i < x.n[0]; ++i)
        {
            #if defined(ELEMENT_ACCESS)
            a_x[i].x = std::log(a_x[i].x);
            #else
            a_x[i] = fw::log(a_x[i]);
            #endif
        }

        return (omp_get_wtime() - time);
    }


    template <>
    template <>
    double kernel<element_type>::log<3>(fw::buffer<element_type, 3, fw::data_layout::SoA>& x)
    {
        double time = omp_get_wtime();

        auto a_x = x.read_write();	
        
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
                    a_x[k][j][i].y = std::log(a_x[k][j][i].y);
                    #else
                    a_x[k][j][i] = fw::log(a_x[k][j][i]);
                    #endif
                }
            }
        }
        
        return (omp_get_wtime() - time);
    }
#endif