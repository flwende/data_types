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
        template <typename T, std::size_t N, std::size_t D, data_layout L, typename Enabled = void>
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
            accessor(base_pointer& data, const sarray<std::size_t, D>& n, const std::size_t stab_idx = 0) 
                : 
                data(data), 
                n(n), 
                stab_idx(stab_idx) {}

            inline accessor<T, N - 1, D, L> operator[] (const std::size_t idx)
            {
                std::size_t delta = idx;

                for (std::size_t i = 1; i < (N - 1); ++i)
                {
                    delta *= n[i];
                }               

                return accessor<T, N - 1, D, L>(data, n, stab_idx + delta);
            }

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
        };

        template <typename T, std::size_t D, data_layout L, typename Enabled>
        class accessor<T, 1, D, L, Enabled>
        {
            using base_pointer = typename internal::traits<T, L>::base_pointer;
            base_pointer& data;
            const sarray<std::size_t, D>& n;
            const std::size_t stab_idx;

        public:

            //! \brief Standard constructor
            //!
            //! \param ptr base pointer
            //! \param n extent of the D-dimensional array
            accessor(base_pointer& data, const sarray<std::size_t, D>& n, const std::size_t stab_idx = 0) 
                : 
                data(data), 
                n(n), 
                stab_idx(stab_idx) {}

            inline T& operator[] (const std::size_t idx)
            {
                return *(data.at(stab_idx, idx));
            }

            inline const T& operator[] (const std::size_t idx) const
            {
                return *(data.at(stab_idx, idx));
            }

            inline T& at(std::size_t idx)
            {
                if (idx >= n[0])
                {
                    throw std::out_of_range("accessor<T, 1>::at() : index out of bounds");
                }

                return *(data.at(stab_idx, idx));
            }

            inline const T& at(std::size_t idx) const
            {
                if (idx >= n[0])
                {
                    throw std::out_of_range("accessor<T, 1>::at() : index out of bounds");
                }

                return *(data.at(stab_idx, idx));
            }

            // a simple iterator class
            class iterator
            {
                using value_type = typename base_pointer::value_type;

                base_pointer base;

            public:

                iterator(base_pointer base)
                    :
                    base(base) {}

                inline iterator& operator++()
                {
                    ++base;
                    return *this;
                }

                inline iterator operator++(int)
                {
                    iterator it(base);
                    ++base;
                    return it;
                }

                inline iterator& operator+=(int inc)
                {
                    base += inc;
                    return *this;
                }

                inline bool operator==(iterator it) const
                {
                    return (base == it.base);
                }

                inline bool operator!=(iterator it) const
                {
                    return (base != it.base);
                }

                inline value_type& operator*()
                {
                    return *base;
                }

                inline const value_type& operator*() const
                {
                    return *base;
                }

                inline value_type& operator[](const std::size_t idx)
                {
                    return *(base.at(idx));
                }

                inline const value_type& operator[](const std::size_t idx) const
                {
                    return *(base.at(idx));
                }
            };

            iterator begin() const
            {
                return iterator(data.at(stab_idx, 0));
            }

            iterator end() const
            {
                return iterator(data.at(stab_idx, 0));
            }
        };

        template <typename T, std::size_t D>
        class accessor<T, 1, D, data_layout::SoA, typename std::enable_if<internal::provides_proxy_type<T>::value>::type>
        {
            using base_pointer = typename internal::traits<T, data_layout::SoA>::base_pointer;
            using proxy_type = typename internal::traits<T, data_layout::SoA>::proxy_type;
            base_pointer& data;
            const sarray<std::size_t, D>& n;
            const std::size_t stab_idx;

        public:

            //! \brief Standard constructor
            //!
            //! \param ptr base pointer
            //! \param n extent of the D-dimensional array
            accessor(base_pointer& data, const sarray<std::size_t, D>& n, const std::size_t stab_idx = 0) 
                : 
                data(data), 
                n(n), 
                stab_idx(stab_idx) {}

            inline proxy_type operator[] (const std::size_t idx)
            {
                return proxy_type(data.at(stab_idx, idx));
            }

            inline proxy_type operator[] (const std::size_t idx) const
            {
                return proxy_type(data.at(stab_idx, idx));
            }

            inline proxy_type at(std::size_t idx)
            {
                if (idx >= n[0])
                {
                    throw std::out_of_range("accessor<T, 1, SoA>::at() : index out of bounds");
                }

                return operator[](idx);
            }

            inline proxy_type at(std::size_t idx) const
            {
                if (idx >= n[0])
                {
                    throw std::out_of_range("accessor<T, 1, SoA>::at() : index out of bounds");
                }

                return operator[](idx);
            }
            
            // a simple iterator class
            class iterator
            {
                using value_type = typename base_pointer::value_type;

                base_pointer base;

            public:

                iterator(base_pointer base)
                    :
                    base(base) {}

                inline iterator& operator++()
                {
                    ++base;
                    return *this;
                }

                inline iterator operator++(int)
                {
                    iterator it(base);
                    ++base;
                    return it;
                }

                inline iterator& operator+=(int inc)
                {
                    base += inc;
                    return *this;
                }

                inline bool operator==(iterator it) const
                {
                    return (base == it.base);
                }

                inline bool operator!=(iterator it) const
                {
                    return (base != it.base);
                }

                inline proxy_type operator*()
                {
                    return proxy_type(base);
                }

                inline proxy_type operator*() const
                {
                    return proxy_type(base);
                }

                inline proxy_type operator[](const std::size_t idx)
                {
                    return proxy_type(base.at(0, idx));
                }

                inline const proxy_type operator[](const std::size_t idx) const
                {
                    return proxy_type(base.at(0, idx));
                }
            };

            iterator begin() const
            {
                return iterator(data.at(stab_idx, 0));
            }

            iterator end() const
            {
                return iterator(data.at(stab_idx, n[0]));
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

    public:

        using value_type = element_type;
        using allocator_type = typename base_pointer<element_type>::allocator;
        using size_type = std::size_t;

        sarray<std::size_t, D> n;
        sarray<std::size_t, D> n_internal;

    private:

        std::unique_ptr<base_pointer<element_type>> data;
        std::unique_ptr<base_pointer<const_element_type>> const_data;
        
        inline internal::accessor<element_type, D, D, L> read_write()
        {
            return internal::accessor<element_type, D, D, L>(*data, n);
        }

        inline internal::accessor<const_element_type, D, D, L> read() const 
        {
            return internal::accessor<const_element_type, D, D, L>(*const_data, n);
        }

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

        buffer()
            :
            n(),
            n_internal() {}
            
        buffer(const sarray<std::size_t, D>& n, const bool initialize_to_zero = false)
            :
            n(n),
            n_internal(n.replace(allocator_type::padding(n[0]), 0)),
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
            this->n_internal = n.replace(allocator_type::padding(n[0]), 0);
    
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

        using dm1_accessor_type = internal::accessor<element_type, D - 1, D, L>;
        using const_dm1_accessor_type = internal::accessor<const_element_type, D - 1, D, L>;
        using proxy_type = typename internal::traits<element_type, L>::proxy_type;
        using const_proxy_type = typename internal::traits<const_element_type, L>::proxy_type;
        static constexpr bool return_type_is_proxy = (L == data_layout::SoA && internal::provides_proxy_type<element_type>::value);

        using return_type = typename std::conditional<D == 1,
            typename std::conditional<return_type_is_proxy, proxy_type, element_type&>::type, 
            dm1_accessor_type>::type;

        using const_return_type = typename std::conditional<D == 1, 
            typename std::conditional<return_type_is_proxy, const_proxy_type, const_element_type&>::type, 
            const_dm1_accessor_type>::type;
        
        inline return_type operator[] (const std::size_t idx)
        {
            return read_write()[idx];
        }

        inline const_return_type operator[] (const std::size_t idx) const
        {
            return read()[idx];
        }

        inline return_type at(std::size_t idx)
        {
            return read_write().at(idx);
        }

        inline const_return_type at(std::size_t idx) const
        {
            return read().at(idx);
        }

        allocator_type get_allocator() const
        {
            return allocator_type();
        }
    };
}

#endif