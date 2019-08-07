// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(BUFFER_BUFFER_HPP)
#define BUFFER_BUFFER_HPP

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#include "../common/data_layout.hpp"
#include "../common/memory.hpp"
#include "../common/traits.hpp"
#include "../sarray/sarray.hpp"

namespace XXX_NAMESPACE
{
    namespace internal
    {
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //! \brief A simple random access iterator
        //!
        //! \tparam P pointer type
        //! \tparam R return type for data access
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        template <typename P, typename R>
        class iterator
        {
            template <typename T, std::size_t N, std::size_t D, data_layout L>
            friend class accessor;

            template <typename T, std::size_t D, data_layout L>
            friend class buffer;

            P ptr;
            std::size_t pos;

            iterator(P ptr, const std::size_t pos)
                :
                ptr(ptr),
                pos(pos) {}

        public:

            inline iterator& operator++()
            {
                ++pos;
                return *this;
            }

            inline iterator operator++(int)
            {
                iterator it(ptr, pos);
                ++pos;
                return it;
            }

            inline iterator& operator+=(int inc)
            {
                pos += inc;
                return *this;
            }

            inline bool operator==(const iterator& it) const
            {
                return (pos == it.pos);
            }

            inline bool operator!=(const iterator& it) const
            {
                return (pos != it.pos);
            }

            inline R operator*()
            {
                return *(ptr.at(pos));
            }

            inline const R operator*() const
            {
                return *(ptr.at(pos));
            }

            inline R operator[](const std::size_t idx)
            {
                return *(ptr.at(pos + idx));
            }

            inline const R operator[](const std::size_t idx) const
            {
                return *(ptr.at(pos + idx));
            }
        };

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //! \brief Accessor type implementing array subscript operator chaining [][]..[]
        //!
        //! This data type basically collects all array indices and determines the final memory reference recursively.
        //! 
        //! Idea: all memory is allocated as a contiguous set of stabs (innermost dimension n[0] with padding).
        //! For D > 1, the number of stabs is determined as the reduction over all indices, but without accounting for
        //! the innermost dimension: [k][j][i] -> '(k * n[1] + j) * n[0] + i' -> 'stab_idx = k * n[1] + j'
        //! The 'memory' type holds a base-pointer-like memory reference to [0][0]..[0] and can deduce the final
        //! memory reference from 'stab_idx', 'n[0]' and 'i'.
        //! The result (recursion anchor, D=1) of the array subscript operator chaining is either a reference of type 
        //! 'T' in case of the AoS data layout or if there is no proxy type available with the SoA data layout, or
        //! a proxy type that is initialized through the final memory reference type in case of SoA.
        //!
        //! \tparam T data type
        //! \tparam N recursion level
        //! \tparam D dimension
        //! \tparam Data_layout any of SoA (struct of arrays) and AoS (array of structs)
        //! \tparam Enabled needed for partial specialization for data types providing a proxy type
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        template <typename T, std::size_t N, std::size_t D, data_layout L>
        class accessor
        {
            using base_pointer = typename internal::traits<T, L>::base_pointer;
            base_pointer& data;
            const sarray<std::size_t, D>& n;
            const std::size_t stab_idx;

        public:

            //! \brief Standard constructor
            //!
            //! \param data pointer-like memory reference
            //! \param n extent of the D-dimensional array
            //! \param stab_idx the offset in units of 'innermost dimension n[0]'
            HOST_VERSION
            CUDA_DEVICE_VERSION
            accessor(base_pointer& data, const sarray<std::size_t, D>& n, const std::size_t stab_idx = 0) 
                : 
                data(data), 
                n(n), 
                stab_idx(stab_idx) {}

            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline accessor<T, N - 1, D, L> operator[] (const std::size_t idx)
            {
                std::size_t delta = idx;

                for (std::size_t i = 1; i < (N - 1); ++i)
                {
                    delta *= n[i];
                }               

                return accessor<T, N - 1, D, L>(data, n, stab_idx + delta);
            }

            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline accessor<T, N - 1, D, L> operator[] (const std::size_t idx) const
            {
                std::size_t delta = idx;

                for (std::size_t i = 1; i < (N - 1); ++i)
                {
                    delta *= n[i];
                }               

                return accessor<T, N - 1, D, L>(data, n, stab_idx + delta);
            }

            inline accessor<T, N - 1, D, L> at(std::size_t idx)
            {
                if (idx >= n[N - 1])
                {
                    throw std::out_of_range("accessor<T, D>::at() : index out of bounds");
                }

                return operator[](idx);
            }

            inline accessor<T, N - 1, D, L> at(std::size_t idx) const
            {
                if (idx >= n[N - 1])
                {
                    throw std::out_of_range("accessor<T, D>::at() : index out of bounds");
                }

                return operator[](idx);
            }

            // iterator
            template <typename R>
            class stab_iterator
            {
                base_pointer& ptr;
                const sarray<std::size_t, D>& n;
                std::size_t pos;

            public:

                stab_iterator(base_pointer& ptr, const sarray<std::size_t, D>& n, const std::size_t pos = 0)
                    :
                    ptr(ptr),
                    n(n),
                    pos(pos) {}

                inline stab_iterator& operator++()
                {
                    ++pos;
                    return *this;
                }

                inline stab_iterator operator++(int)
                {
                    stab_iterator it(ptr, n, pos);
                    ++pos;
                    return it;
                }

                inline bool operator==(const stab_iterator& it) const
                {
                    return (pos == it.pos);
                }

                inline bool operator!=(const stab_iterator& it) const
                {
                    return (pos != it.pos);
                }

                inline R operator*()
                {
                    return R(ptr, n, pos);
                }

                inline const R operator*() const
                {
                    return R(ptr.at(pos, 0), n);
                }
            };

            using iterator = stab_iterator<accessor<T, 1, D, L>>;
            using const_iterator = stab_iterator<accessor<const T, 1, D, L>>;

            iterator begin() const
            {
                return iterator(data, n, stab_idx);
            }

            iterator end() const
            {
                return iterator(data, n, stab_idx + sarray<std::size_t, N>(n).reduce_mul(1));
            }

            const_iterator cbegin() const
            {
                return const_iterator(data, n, stab_idx);
            }

            const_iterator cend() const
            {
                return const_iterator(data, n, stab_idx + sarray<std::size_t, N>(n).reduce_mul(1));
            }
        };
        
        template <typename T, std::size_t D, data_layout L>
        class accessor<T, 1, D, L>
        {
            using base_pointer = typename internal::traits<T, L>::base_pointer;
            using const_base_pointer = typename internal::traits<const T, L>::base_pointer;
            using value_type = typename std::conditional<L == data_layout::SoA, typename internal::traits<T, data_layout::SoA>::proxy_type, T&>::type;
            using const_value_type = typename std::conditional<L == data_layout::SoA, const typename internal::traits<const T, data_layout::SoA>::proxy_type, const T&>::type;
            using iterator = typename internal::iterator<base_pointer, value_type>;
            using const_iterator = typename internal::iterator<const_base_pointer, const_value_type>;

            friend iterator;

            base_pointer& data;
            const sarray<std::size_t, D>& n;
            const std::size_t stab_idx;

        public:

            //! \brief Standard constructor
            //!
            //! \param ptr base pointer
            //! \param n extent of the D-dimensional array
            HOST_VERSION
            CUDA_DEVICE_VERSION
            accessor(base_pointer& data, const sarray<std::size_t, D>& n, const std::size_t stab_idx = 0) 
                : 
                data(data), 
                n(n), 
                stab_idx(stab_idx) {}

            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline value_type operator[] (const std::size_t idx)
            {
                #if defined(SOA_INNERMOST)
                return *(data.at(stab_idx, idx));
                #else
                return *(data.at(0, stab_idx * n[0] + idx));
                #endif
            }

            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline const value_type operator[] (const std::size_t idx) const
            {
                #if defined(SOA_INNERMOST)
                return *(data.at(stab_idx, idx));
                #else
                return *(data.at(0, stab_idx * n[0] + idx));
                #endif
            }

            inline value_type at(std::size_t idx)
            {
                if (idx >= n[0])
                {
                    throw std::out_of_range("accessor<T, 1>::at() : index out of bounds");
                }

                #if defined(SOA_INNERMOST)
                return *(data.at(stab_idx, idx));
                #else
                return *(data.at(0, stab_idx * n[0] + idx));
                #endif
            }

            inline const value_type at(std::size_t idx) const
            {
                if (idx >= n[0])
                {
                    throw std::out_of_range("accessor<T, 1>::at() : index out of bounds");
                }

                #if defined(SOA_INNERMOST)
                return *(data.at(stab_idx, idx));
                #else
                return *(data.at(0, stab_idx * n[0] + idx));
                #endif
            }
            
            // iterator            
            iterator begin() const
            {
                #if defined(SOA_INNERMOST)
                return iterator(data.at(stab_idx, 0), 0);
                #else
                return iterator(data.at(0, stab_idx * n[0]), 0);
                #endif
            }

            iterator end() const
            {
                #if defined(SOA_INNERMOST)
                return iterator(data.at(stab_idx, 0), n[0]);
                #else
                return iterator(data.at(0, stab_idx * n[0]), n[0]);
                #endif
            }

            const_iterator cbegin() const
            {
                #if defined(SOA_INNERMOST)
                return const_iterator(data.at(stab_idx, 0), 0);
                #else
                return const_iterator(data.at(0, stab_idx * n[0]), 0);
                #endif
            }

            const_iterator cend() const
            {
                #if defined(SOA_INNERMOST)
                return const_iterator(data.at(stab_idx, 0), n[0]);
                #else
                return const_iterator(data.at(0, stab_idx * n[0]), n[0]);
                #endif
            }
        };
    }
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //! \brief Multi-dimensional buffer data type
    //!
    //! This buffer data type uses a plain C-pointer to dynamically adapt to the needed memory requirement.
    //! In case of D > 1, its is determined as the product of all dimensions with the
    //! innermost dimension padded according to T, the data layout and the (default) data alignment.
    //! All memory is contiguous and can be moved as a whole (important e.g. for data transfers).
    //! The buffer, however, allows access to the data using array subscript operator chaining together with proxy objects.
    //! \n\n
    //! In case of Data_layout = AoS, data is stored with all elements placed in main memory one after the other.
    //! \n
    //! In case of Data_layout = SoA (applied only if T is a record type) the individual members
    //! are placed one after the other along the innermost dimension, e.g. for
    //! buffer<vec<double, 3>, 2, SoA>({3, 2}) the memory layout would be the following one:
    //! <pre>
    //!     [0][0].x
    //!     [0][1].x
    //!     [0][2].x
    //!     ######## (padding)
    //!     [0][0].y
    //!     [0][1].y
    //!     [0][2].y
    //!     ######## (padding)
    //!     [1][0].x
    //!     [1][1].x
    //!     [1][2].x
    //!     ######## (padding)
    //!     [1][0].y
    //!     [1][1].y
    //!     [1][2].y
    //! </pre>
    //! You can access the individual components of buffer<vec<double, 3>, 2, SoA> b({3,2})
    //! as usual, e.g., b[1][0].x = ...
    //! \n
    //! GNU and Clang/LLVM seem to optimize the proxies away.
    //!
    //! \tparam T data type
    //! \tparam D dimension
    //! \tparam Data_layout any of SoA (struct of arrays) and AoS (array of structs)
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <typename T, std::size_t D, data_layout L = data_layout::AoS>
    class buffer
    {
        static_assert(!std::is_const<T>::value, "error: buffer with const elements is not allowed");

        using element_type = T;
        using const_element_type = typename internal::traits<element_type, L>::const_type;

        template <typename X>
        using base_pointer = typename internal::traits<X, L>::base_pointer;
        
        void set_data(const element_type& value)
        {
            if (!data.get()) return;

            const std::size_t n_stabs = n.reduce_mul(1);
            
            for (std::size_t i_s = 0; i_s < n_stabs; ++i_s)
            {
                // get base_pointer to this stab, and use a 1d-accessor to access the elements in it
                base_pointer<element_type> data_stab = data->at(i_s, 0);
                internal::accessor<element_type, 1, D, L> stab(data_stab, n);
                
                #pragma omp simd
                for (std::size_t i = 0; i < n[0]; ++i)
                {
                    stab[i] = value;
                }
            }
        }

    public:

        using value_type = element_type;
        using allocator_type = typename base_pointer<element_type>::allocator;
        using size_type = std::size_t;

        sarray<std::size_t, D> n;

    private:

        sarray<std::size_t, D> n_internal;
        std::unique_ptr<base_pointer<element_type>> data;
        std::unique_ptr<base_pointer<const_element_type>> const_data;

    public:

        buffer()
            :
            n_internal(),
            n() {}
            
        buffer(const sarray<std::size_t, D>& n, const bool initialize_to_zero = false)
            :
            n(n),
            #if defined(SOA_INNERMOST)
            n_internal(n.replace(allocator_type::padding(n[0]), 0)),
            #else
            n_internal(sarray<std::size_t, D>{{1}}.replace(n.reduce_mul(), 0)), // {n_0 * .. * n_{D-1}, 1,.., 1}
            #endif
            data(std::make_unique<base_pointer<element_type>>(allocator_type::allocate(n_internal), n_internal[0])),
            const_data(std::make_unique<base_pointer<const_element_type>>(*data))
        {
            if (initialize_to_zero)
            {
                set_data({});
            }
        }
        
        ~buffer()
        {
            if (data.get())
            {
                allocator_type::deallocate(data->get_pointer());
                delete data.release();
            }

            delete const_data.release();
        }

        void resize(const sarray<std::size_t, D>& n, const bool initialize_to_zero = false)
        {
            this->n = n;
            #if defined(SOA_INNERMOST)
            this->n_internal = n.replace(allocator_type::padding(n[0]), 0);
            #else
            this->n_internal = sarray<std::size_t, D>{{1}}.replace(n.reduce_mul(), 0);
            #endif
    
            if (data.get())
            {
                allocator_type::deallocate(data->get_pointer());
                delete data.release();
            }

            delete const_data.release();

            data = std::make_unique<base_pointer<element_type>>(allocator_type::allocate(n_internal), n_internal[0]);
            const_data = std::make_unique<base_pointer<const_element_type>>(*data);

            if (initialize_to_zero)
            {
                set_data({});
            }
        }

        void swap(buffer& b)
        {
            if (n != b.n)
            {
                std::cerr << "error: buffer swapping not possible because of different extents" << std::endl;
                return;
            }

            n.swap(b.n);
            n_internal.swap(b.n_internal);
            data.swap(b.data);
            const_data.swap(b.const_data);
        }

        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto operator[](const std::size_t idx)
        {
            return internal::accessor<element_type, D, D, L>(*data, n)[idx];
        }

        HOST_VERSION
        CUDA_DEVICE_VERSION
        inline auto operator[](const std::size_t idx) const
        {
            return internal::accessor<const_element_type, D, D, L>(*const_data, n)[idx];
        }

        inline auto at(std::size_t idx)
        {
            return internal::accessor<element_type, D, D, L>(*data, n)[idx];
        }

        inline auto at(std::size_t idx) const
        {
            return internal::accessor<const_element_type, D, D, L>(*const_data, n)[idx];
        }

        allocator_type get_allocator() const
        {
            return allocator_type();
        }
    };
}

#endif