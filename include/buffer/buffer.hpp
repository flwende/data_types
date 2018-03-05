// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(DATA_TYPES_BUFFER_HPP)
#define DATA_TYPES_BUFFER_HPP

#if defined(HAVE_SYCL)
#include <CL/sycl.hpp>
#endif

#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <type_traits>
#include <vector>
#include <boost/align/aligned_allocator.hpp>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#include "../misc/template_stuff.hpp"
#include "../static_array/sarray.hpp"

#include "buffer_constants.hpp"
#include "buffer_fdecl.hpp"
#include "buffer_vec.hpp"
#include "buffer_proxy_vec.hpp"

namespace XXX_NAMESPACE
{
	enum data_layout { AoS = 1, SoA = 2 };

	#if defined(HAVE_SYCL)
	enum target { host = 1, device = 2, host_device = 3 };
	#else
	enum target { host = 1 };
	#endif

	namespace detail
	{
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//! \brief Proxy data type implementing the index computation for subscript operator chaining
		//! when accessing the elements of buffer<T, D, Layout, Alignment>
		//!
		//! Proxies are defined recursively:
		//! <pre>
		//!     proxy_layer<T, D,...>
		//!         n[D]           = {n_1, n_2,...,n_{D - 1}, n_{D}}
		//!         ptr_{D}        = &data[0];
		//!         offset_{D}     = n_1 * n_2 * ... * n_{D - 1}
		//!
		//!     proxy_layer<T, D - 1,...>    <-  proxy_layer<T, D,...>::operator[](idx)
		//!         n[D - 1]       = {n_1, n_2,...,n_{D - 1}}
		//!         ptr_{D - 1}    = &ptr_{D - 1}[idx * offset_{D}]
		//!         offset_{D - 1} = n_1 * n_2 * ... * n_{D - 2}
		//!     ...
		//!     (ancher case definition) T&  <-  proxy_layer<T, 1,...>::operator[](idx)
		//! </pre>
		//!
		//! \tparam T type of the data stored in buffer<T,...>
		//! \tparam D recursion depth (and dimension the proxy is associated with)
		//! \tparam Layout any of SoA (structs of arrays) and AoS (array of structs)
		//! \tparam Enabled needed for partial specialization with T = vec<TT,...> and Layout = SoA
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		template <typename T, std::size_t D, data_layout Layout = AoS, typename Enabled = void>
		class proxy_layer
		{
			//! Base pointer
			T* ptr;
			//! Shape of the D-dimensional volume
			const sarray<std::size_t, D> n;

		public:

			//! \brief Constructor
			//!
			//! \param ptr base pointer
			//! \param n shape of the D-dimensional volume
			proxy_layer(T* ptr, const sarray<std::size_t, D>& n) : ptr(ptr), n(n) { ; }

			//! \brief Subscript operator
			//!
			//! Determine the base pointer of proxy_layer<T, D - 1,...> by multiplying the
			//! dimensions of the D-1-dimensional sub-volume and idx.
			//!
			//! \param idx index w.r.t. dimension D
			//! \return a proxy_layer<T, D - 1> object
			inline proxy_layer<T, D - 1> operator[](const std::size_t idx)
			{
				std::size_t offset = idx;
				for (std::size_t i = 0; i < (D - 1); ++i)
				{
					offset *= n[i];
				}
				return proxy_layer<T, D - 1>(&ptr[offset], n.template shrink<D - 1>());
			}
		};

		////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//! \brief Specialization with D = 1 (recursion ancher definition)
		//!
		//! This proxy implements a standard array subscript operator for accessing the data through the base pointer.
		//!
		//! \tparam T type of the data stored in buffer<T,...>
		//! \tparam Layout any of SoA (structs of arrays) and AoS (array of structs)
		//! \tparam Enabled needed for partial specialization with T = vec<TT, DD> and Layout = SoA
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		template <typename T, data_layout Layout, typename Enabled>
		class proxy_layer<T, 1, Layout, Enabled>
		{
			//! Base pointer
			T* ptr;
			//! Shape of the 1-dimensional volume
			const sarray<std::size_t, 1> n;

		public:

			//! \brief Constructor
			//!
			//! \param ptr base pointer
			//! \param n shape of the 1-dimensional volume
			proxy_layer(T* ptr, const sarray<std::size_t, 1>& n) : ptr(ptr), n(n) { ; }

			//! \brief Array subscript operator
			//!
			//! \param idx element to access
			//! \return reference to the actual data
			inline T& operator[](const std::size_t idx)
			{
				return ptr[idx];
			}
		};

		////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//! \brief Specialization with T = vec<TT,...> and Layout = SoA
		//!
		//! This proxy differs from the general case in that an additional next-to-the-innermost dimension is
		//! assumed implicitely.
		//! The extent of this dimension is given by the dimension of T, which is T::dim.
		//!
		//! \tparam T data type (vec<TT, DD> where TT and DD are recovered from T)
		//! \tparam D recursion depth (and dimension the proxy is associated with)
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		template <typename T, std::size_t D>
		class proxy_layer<T, D, SoA, typename std::enable_if<is_vec<T>::value>::type>
		{
			//! Recover the fundamental data type that is behind T
			using TT = typename T::fundamental_type;
			//! Recover the dimension of T
			static constexpr std::size_t DD = T::dim;

			//! Base pointer (it is of type TT, not T)
			TT* ptr;
			//! Shape of the D-dimensional volume
			const sarray<std::size_t, D> n;

		public:

			//! \brief Constructor
			//!
			//! \param ptr base pointer
			//! \param n shape of the D-dimensional volume
			proxy_layer(TT* ptr, const sarray<std::size_t, D>& n) : ptr(ptr), n(n) { ; }

			//! \brief Subscript operator
			//!
			//! Determine the base pointer of proxy_layer<T, D - 1,...> by multiplying the
			//! dimensions of the D-1-dimensional sub-volume and idx and the implicit dimension DD.
			//!
			//! \param idx index w.r.t. dimension D
			//! \return a proxy_layer<T, D - 1, SoA> object
			inline proxy_layer<T, D - 1, SoA> operator[](const std::size_t idx)
			{
				std::size_t offset = idx * DD;
				for (std::size_t i = 0; i < (D - 1); ++i)
				{
					offset *= n[i];
				}
				return proxy_layer<T, D - 1, SoA>(&ptr[offset], n.template shrink<D - 1>());
			}
		};

		////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//! \brief Specialization with T = vec<TT,...>, D = 1 (recursion ancher definition) and Layout = SoA
		//!
		//! This proxy differs from the general case in that an additional next-to-the-innermost dimension is
		//! assumed implicitely.
		//! The extent of this dimension is given by the dimension of T, which is T::dim.
		//! It is propagated to the vec_proxy<TT, DD> object when calling the array subscript operator.
		//!
		//! \tparam T data type (vec<TT, DD> where TT and DD are recovered from T)
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		template <typename T>
		class proxy_layer<T, 1, SoA, typename std::enable_if<is_vec<T>::value>::type>
		{
			//! Recover the fundamental data type that is behind T
			using TT = typename T::fundamental_type;
			//! Recover the dimension of T
			static constexpr std::size_t DD = T::dim;

			//! Base pointer (it is of type TT, not T)
			TT* ptr;
			//! Shape of the 1-dimensional volume
			const sarray<std::size_t, 1> n;

		public:

			//! \brief Constructor
			//!
			//! \param ptr base pointer
			//! \param n shape of the 1-dimensional volume
			proxy_layer(TT* ptr, const sarray<std::size_t, 1>& n) : ptr(ptr), n(n) { ; }

			//! \brief Array subscript operator
			//!
			//! For D = 1, the implicit next-to-innermost dimension DD is propagated to the proxy_vec<TT, DD> object
			//! which then sets up the references to x, y, and z component of the actual vec<TT, DD> object.
			//!
			//! \param idx element to access
			//! \return a proxy_vec<TT, DD> object
			inline proxy_vec<TT, DD> operator[](const std::size_t idx)
			{
				return proxy_vec<TT, DD>(&ptr[idx], n[0]);
			}
		};
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//! \brief Accessor data type implementing array subscript operator chaining
	//!
	//! This data type basically collects all indices and determines the final base pointer for data access.
	//!
	//! \tparam T data type
	//! \tparam D dimension
	//! \tparam Layout any of SoA (structs of arrays) and AoS (array of structs)
	//! \tparam Enabled needed for partial specialization with T = vec<TT, DD> and Layout = SoA
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	template <typename T, std::size_t D, data_layout Layout = AoS, typename Enabled = void>
	class accessor
	{
		//! Base pointer
		T* ptr;
		//! Extent of the D-dimensional array
		sarray<std::size_t, D> n;

	public:

		//! \brief Standard constructor
		//!
		//! \param ptr base pointer
		//! \param n extent of the D-dimensional array
		accessor(T* ptr, const sarray<std::size_t, D>& n) : ptr(ptr), n(n) { ; }

		//! \brief Array subscript operator
		//!
		//! Determine the base pointer of proxy_layer<T, D - 1> by multiplying the
		//! dimensions of the D-1-dimensional sub-volume and idx.
		//!
		//! \param idx index w.r.t. dimension D
		//! \return a proxy_layer<T, D - 1> object
		inline detail::proxy_layer<T, D - 1> operator[](const std::size_t idx)
		{
			std::size_t offset = idx;
			for (std::size_t i = 0; i < (D - 1); ++i)
			{
				offset *= n[i];
			}
			return detail::proxy_layer<T, D - 1>(&ptr[offset], n.template shrink<D - 1>());
		}
	};

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//! \brief Specialization with D = 1 (recursion ancher definition)
	//!
	//! \tparam T data type
	//! \tparam Layout any of SoA (structs of arrays) and AoS (array of structs)
	//! \tparam Enabled needed for partial specialization with T = vec<TT, DD> and Layout = SoA
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	template <typename T, data_layout Layout, typename Enabled>
	class accessor<T, 1, Layout, Enabled>
	{
		//! Base pointer
		T* ptr;
		//! Extent of the 1-dimensional array
		sarray<std::size_t, 1> n;

	public:

		//! \brief Standard constructor
		//!
		//! \param ptr base pointer
		//! \param n extent of the 1-dimensional array
		accessor(T* ptr, const sarray<std::size_t, 1>& n) : ptr(ptr), n(n) { ; }

		//! \brief Array subscript operator
		//!
		//! \param idx element to access
		//! \return the actual data to be accessed
		inline T& operator[](const std::size_t idx)
		{
			return ptr[idx];
		}

		//! \brief Array subscript operator
		//!
		//! \param idx element to access
		//! \return the actual data to be accessed
		inline const T& operator[](const std::size_t idx) const
		{
			return ptr[idx];
		}
	};

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//! \brief Specialization with T = vec<TT, DD> and Layout = SoA
	//!
	//! This accessor differs from the general case in that it internally accesses the data using the SoA
	//! layout.
	//! It is implemented for T = vec<TT, DD> only, where DD adds an implicit next-to-the-innermost dimension.
	//!
	//! \tparam T data type
	//! \tparam D dimension
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	template <typename T, std::size_t D>
	class accessor<T, D, SoA, typename std::enable_if<is_vec<T>::value>::type>
	{
		//! Recover the fundamental data type that is behind T
		using TT = typename T::fundamental_type;
		//! Recover the dimension of T
		static constexpr std::size_t DD = T::dim;

		//! Base pointer
		TT* ptr;
		//! Extent of the D-dimensional array
		sarray<std::size_t, D> n;

	public:

		//! \brief Standard constructor
		//!
		//! \param ptr base pointer
		//! \param n extent of the D-dimensional array
		accessor(TT* ptr, const sarray<std::size_t, D>& n) : ptr(ptr), n(n) { ; }

		//! \brief Array subscript operator
		//!
		//! Determine the base pointer of proxy_layer<T, D - 1> by multiplying the
		//! dimensions of the D-1-dimensional sub-volume and idx and DD.
		//!
		//! \param idx index w.r.t. dimension D
		//! \return a proxy_layer<T, D - 1> object
		inline detail::proxy_layer<T, D - 1, SoA> operator[](const std::size_t idx)
		{
			std::size_t offset = idx * DD;
			for (std::size_t i = 0; i < (D - 1); ++i)
			{
				offset *= n[i];
			}
			return detail::proxy_layer<T, D - 1, SoA>(&ptr[offset], n.template shrink<D - 1>());
		}
	};

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//! \brief Specialization with T = vec<TT, DD>, D = 1 and Layout = SoA
	//!
	//! This accessor differs from the general case in that it internally accesses the data using the SoA
	//! layout.
	//! It is implemented for T = vec<TT, DD> only, where DD adds an implicit next-to-the-innermost dimension.
	//!
	//! \tparam T data type
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	template <typename T>
	class accessor<T, 1, SoA, typename std::enable_if<is_vec<T>::value>::type>
	{
		//! Recover the fundamental data type that is behind T
		using TT = typename T::fundamental_type;
		//! Recover the dimension of T
		static constexpr std::size_t DD = T::dim;

		//! Base pointer
		TT* ptr;
		//! Extent of the 1-dimensional array
		sarray<std::size_t, 1> n;

	public:

		//! \brief Standard constructor
		//!
		//! \param ptr base pointer
		//! \param n extent of the D-dimensional array
		accessor(TT* ptr, const sarray<std::size_t, 1>& n) : ptr(ptr), n(n) { ; }

		//! \brief Array subscript operator
		//!
		//! \param idx element to access
		//! \return a proxy_vec<TT, DD> object
		inline detail::proxy_vec<TT, DD> operator[](const std::size_t idx)
		{
			return detail::proxy_vec<TT, DD>(&ptr[idx], n[0]);
		}

	};

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//! \brief Multi-dimensional buffer data type
	//!
	//! This buffer data type internally uses std::vector<T> to dynamically adapt to the needed memory requirement.
	//! In case of D > 1, the size of the std::vector<T> is determined as the product of all dimensions with the
	//! innermost dimension padded according to T and the value if the Alignment parameter.
	//! All memory thus is contiguous and can be moved as a whole (important e.g. for data transfers from/to GPGPU).
	//! The buffer, however, allows access to the data using array subscript operator chaining using proxy objects.
	//! GNU and Clang/LLVM can optimize the proxies away.
	//! \n\n
	//! In case of Layout = AoS, data is stored with all elements placed in main memory one after the other.
	//! \n
	//! In case of Layout = SoA (meaningful only if T is of type vec<TT, DD>) the individual components (x,y and z) are
	//! placed one after the other along the innermost dimension, e.g. for buffer<vec<double, 3>, 2, SoA, 32>({3, 2})
	//! the memory layout would be the following one:
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
	//!
	//! \tparam T data type
	//! \tparam D dimension
	//! \tparam Layout any of SoA (structs of arrays) and AoS (array of structs)
	//! \tparam Alignment data alignment (needs to be a power of 2)
	//! \tparam Enabled needed for partial specialization with T = vec<TT, DD> and Layout = SoA
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	template <typename T, std::size_t D, target Target = host, data_layout Layout = AoS, std::size_t Alignment = 32, typename Enabled = void>
	class buffer
	{
		static_assert(sizeof(T) < Alignment, "error: buffer alignment should not be smaller than the sizeof of T");

		//! Internal storage using boost's aligned_allocator for data alignment
		std::vector<T, boost::alignment::aligned_allocator<T, Alignment>> vdata;
		//! Base pointer (does not necessarily point to vdata)
		T* data;
		//! Shape of the buffer with innermost dimension padded
		sarray<std::size_t, D> size_internal;

	public:

		//! Shape of the buffer (as specified by the user)
		sarray<std::size_t, D> size;

		//! \brief Standard constructor
		buffer() : data(nullptr), size_internal{}, size{} { ; }

		//! \brief Constructor
		//!
		//! \param size shape of the buffer
		//! \param ptr external pointer (if specified, no internal storage will be allocated and no padding of the
		//! innermost dimension happens)
		buffer(const sarray<std::size_t, D>& size, T* ptr = nullptr)
		{
			resize(size, ptr);
		}

		//! \brief Set up the buffer
		//!
		//! \param size shape of the buffer
		//! \param ptr external pointer (if specified, no internal storage will be allocated and no padding of the
		//! innermost dimension happens)
		void resize(const sarray<std::size_t, D>& size, T* ptr = nullptr)
		{
			// assign default values
			size_internal = size;
			this->size = size;

			if (ptr == nullptr)
			{
				// padding according to data type T and the Alignment parameter
				constexpr std::size_t num_elements_align = Alignment / sizeof(T);
				size_internal[0] = ((size[0] + num_elements_align - 1) / num_elements_align) * num_elements_align;

				// rezise the internal buffer
				std::size_t num_elements_total = 1;
				for (std::size_t i = 0; i < D; ++i)
				{
					num_elements_total *= size_internal[i];
				}
				vdata.resize(num_elements_total);

				// base pointer points to the internal storage
				data = &vdata[0];
			}
			else
			{
				// a pointer to external memory is provided: clear internal storage
				vdata.clear();
				data = ptr;
			}
		}

		//! \brief Read accessor
		//!
		//! \return an accessor using const references internally
		inline const accessor<typename detail::make_const<T>::type, D, Layout> read() const
		{
			using const_T = typename detail::make_const<T>::type;
			return accessor<const_T, D, Layout>(reinterpret_cast<const_T*>(data), size_internal);
		}

		//! \brief Write accessor
		//!
		//! \return an accessor
		inline const accessor<T, D, Layout> write()
		{
			return accessor<T, D, Layout>(data, size_internal);
		}

		//! \brief Read-write accessor
		//!
		//! \return an accessor
		inline const accessor<T, D, Layout> read_write()
		{
			return write();
		}

		//! \brief Exchange the content of two buffers
		//!
		//! The buffers have to have the same size.
		//!
		//! \param b
		void swap(buffer& b)
		{
			if (size == b.size)
			{
				// swap internal storage (it does not matter whether any of the buffers uses external memory)
				vdata.swap(b.vdata);

				// swap the base pointer
				T* this_data = data;
				data = b.data;
				b.data = this_data;

				// re-assign base pointers only if internal storage is used
				if (vdata.size() > 0)
				{
					data = &vdata[0];
				}

				if (b.vdata.size() > 0)
				{
					b.data = &(b.vdata[0]);
				}
			}
			else
			{
				std::cerr << "error: buffer::swap -> you are trying to swap buffers of different size" << std::endl;
			}
		}
	};

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//! \brief Specialization with T = vec<TT, DD> and Layout = SoA
	//!
	//! This buffer differs from the general case in that it internally stores the data using the SoA
	//! layout.
	//! It is implemented for T = vec<TT, DD> only, where DD adds an implicit next-to-the-innermost dimension.
	//! Data access happens as usual through (recursive) array subscript operator chaining and
	//! proxy_vec<TT, DD> objects are returned when reaching the recursion ancher.
	//! \n\n
	//! For example you can access the individual components of buffer<vec<double, 3>, 2, SoA> b({3,2})
	//! as usual: b[1][0].x = ...
	//! \n
	//! GNU and Clang/LLVM can optimize the proxies away.
	//!
	//! \tparam T data type
	//! \tparam Alignment data alignment (needs to be a power of 2)
	template <typename T, std::size_t D, target Target, std::size_t Alignment>
	class buffer<T, D, Target, SoA, Alignment, typename std::enable_if<is_vec<T>::value>::type>
	{
		//! Recover the fundamental data type that is behind T
		using TT = typename T::fundamental_type;
		//! Recover the dimension of T
		static constexpr std::size_t DD = T::dim;

		//! Internal storage using boost's aligned_allocator for data alignment
		std::vector<TT, boost::alignment::aligned_allocator<TT, Alignment>> vdata;
		//! Base pointer (does not necessarily point to vdata)
		TT* data;
		//! Shape of the buffer with innermost dimension padded
		sarray<std::size_t, D> size_internal;

	public:

		//! Shape of the buffer (as specified by the user)
		sarray<std::size_t, D> size;

		//! \brief Standard constructor
		buffer() : data(nullptr), size_internal{}, size{} { ; }

		//! \brief Constructor
		//!
		//! \param size shape of the buffer
		//! \param ptr external pointer (if specified, no internal storage will be allocated and no padding of the
		//! innermost dimension happens)
		buffer(const sarray<std::size_t, D>& size, T* ptr = nullptr)
		{
			resize(size, ptr);
		}

		//! \brief Set up the buffer
		//!
		//! \param size shape of the buffer
		//! \param ptr external pointer (if specified, no internal storage will be allocated and no padding of the
		//! innermost dimension happens)
		void resize(const sarray<std::size_t, D>& size, T* ptr = nullptr)
		{
			// assign default values
			size_internal = size;
			this->size = size;

			if (ptr == nullptr)
			{
				// padding according to data type T and the Alignment parameter
				constexpr std::size_t num_elements_align = Alignment / sizeof(TT);
				size_internal[0] = ((size[0] + num_elements_align - 1) / num_elements_align) * num_elements_align;

				// rezise the internal buffer
				std::size_t num_elements_total = DD;
				for (std::size_t i = 0; i < D; ++i)
				{
					num_elements_total *= size_internal[i];
				}

				vdata.resize(num_elements_total);
				// base pointer points to the internal storage
				data = &vdata[0];
			}
			else
			{
				// a pointer to external memory is provided: clear internal storage
				vdata.clear();
				data = reinterpret_cast<TT*>(ptr);
			}
		}

		//! \brief Read accessor
		//!
		//! \return an accessor using const references internally
		inline const accessor<typename detail::make_const<T>::type, D, SoA> read() const
		{
			return accessor<typename detail::make_const<T>::type, D, SoA>(reinterpret_cast<const TT*>(data), size_internal);
		}

		//! \brief Write accessor
		//!
		//! \return an accessor
		inline const accessor<T, D, SoA> write()
		{
			return accessor<T, D, SoA>(data, size_internal);
		}

		//! \brief Read-write accessor
		//!
		//! \return an accessor
		inline const accessor<T, D, SoA> read_write()
		{
			return write();
		}

		//! \brief Exchange the content of two buffers
		//!
		//! The buffers have to have the same size.
		//!
		//! \param b
		void swap(buffer& b)
		{
			if (size == b.size)
			{
				// swap internal storage (it does not matter whether any of the buffers uses external memory)
				vdata.swap(b.vdata);

				// swap the base pointer
				TT* this_data = data;
				data = b.data;
				b.data = this_data;

				// re-assign base pointers only if internal storage is used
				if (vdata.size() > 0)
				{
					data = &vdata[0];
				}

				if (b.vdata.size() > 0)
				{
					b.data = &(b.vdata[0]);
				}
			}
			else
			{
				std::cerr << "error: buffer::swap -> you are trying to swap buffers of different size" << std::endl;
			}
		}
	};

	#if defined(HAVE_SYCL)
	namespace detail
	{
		template <typename T>
		void memcpy(T* dst, const T* src, const sarray<std::size_t, 1>& size, const std::size_t d_stride, const std::size_t s_stride)
		{
			for (std::size_t i = 0; i < size[0]; ++i)
			{
				dst[i] = src[i];
			}
		}

		template <typename T>
		void memcpy(T* dst, const T* src, const sarray<std::size_t, 2>& size, const std::size_t d_stride, const std::size_t s_stride)
		{
			for (std::size_t j = 0; j < size[1]; ++j)
			{
				for (std::size_t i = 0; i < size[0]; ++i)
				{
					dst[j * d_stride + i] = src[j * s_stride + i];
				}
			}
		}

		template <typename T>
		void memcpy(T* dst, const T* src, const sarray<std::size_t, 3>& size, const std::size_t d_stride, const std::size_t s_stride)
		{
			for (std::size_t k = 0; k < size[2]; ++k)
			{
				for (std::size_t j = 0; j < size[1]; ++j)
				{
					for (std::size_t i = 0; i < size[0]; ++i)
					{
						dst[(k * size[1] + j) * d_stride + i] = src[(k * size[1] + j) * s_stride + i];
					}
				}
			}
		}
	}

	template <typename T, std::size_t D, data_layout Layout, std::size_t Alignment, typename Enabled>
	class buffer<T, D, device, Layout, Alignment, Enabled>
	{
		static_assert(sizeof(T) < Alignment, "error: buffer alignment should not be smaller than the sizeof of T");

		//! SYCL device buffer
		cl::sycl::buffer<T, D>* d_data;
		//! External host pointer
		T* host_ptr;
		bool has_external_host_pointer;

	public:

		//! Shape of the buffer (as specified by the user)
		sarray<std::size_t, D> size;

		//! \brief Standard constructor
		buffer() : d_data(nullptr), host_ptr(nullptr), has_external_host_pointer(false), size{} { ; }

		//! \brief Constructor
		//!
		//! \param size shape of the buffer
		//! \param ptr external pointer (if specified, no internal storage will be allocated and no padding of the
		//! innermost dimension happens)
		buffer(const sarray<std::size_t, D>& size, T* ptr = nullptr) : d_data(nullptr), host_ptr(nullptr), has_external_host_pointer(false)
		{
			resize(size, ptr);
		}

		//! \brief Destructor
		~buffer()
		{
			if (d_data != nullptr)
			{
				delete d_data;
				d_data = nullptr;
			}
		}

		//! \brief Set up the buffer
		//!
		//! \param size shape of the buffer
		//! \param ptr external pointer (if specified, no internal storage will be allocated and no padding of the
		//! innermost dimension happens)
		void resize(const sarray<std::size_t, D>& size, T* ptr = nullptr)
		{
			// assign default values
			this->size = size;

			// SYCL range
			cl::sycl::range<D> size_internal;
			for (std::size_t i = 0; i < D; ++i)
			{
				size_internal[(D - 1) - i] = size[i];
			}

			// rezise the internal buffer
			if (d_data != nullptr)
			{
				delete d_data;
			}
			d_data = new cl::sycl::buffer<T, D>(size_internal);

			if (ptr != nullptr)
			{
				host_ptr = ptr;
				has_external_host_pointer = true;
			}
		}
		//! \brief Read accessor
		//!
		//! \return a read accessor
		inline auto read(cl::sycl::handler& h)
		{
			return d_data->template get_access<cl::sycl::access::mode::read, cl::sycl::access::target::global_buffer>(h);
		}

		//! \brief Write accessor
		//!
		//! \return a write accessor
		inline auto write(cl::sycl::handler& h)
		{
			return d_data->template get_access<cl::sycl::access::mode::write, cl::sycl::access::target::global_buffer>(h);
		}

		//! \brief Read-write accessor
		//!
		//! \return a read-write accessor
		inline auto read_write(cl::sycl::handler& h)
		{
			return d_data->template get_access<cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer>(h);
		}

		//! \brief Copy data from the host to the device
		inline void memcpy_h2d(const T* ptr, const sarray<std::size_t, D>& n, const std::size_t stride = 0)
		{
			const std::size_t s_stride = (stride == 0 ? n[0] : stride);
			const std::size_t d_stride = size[0];
			auto a_d_data = d_data->template get_access<cl::sycl::access::mode::write, cl::sycl::access::target::host_buffer>();
			detail::memcpy(a_d_data.get_pointer(), ptr, n, d_stride, s_stride);
		}

		//! \brief Copy data from the host to the device
		inline void memcpy_h2d(const T* ptr, const std::size_t stride = 0)
		{
			memcpy_h2d(ptr, size, stride);
		}

		//! \brief Copy data from the device to the host
		inline void memcpy_d2h(T* ptr, const sarray<std::size_t, D>& n, const std::size_t stride = 0)
		{
			const std::size_t s_stride = size[0];
			const std::size_t d_stride = (stride == 0 ? n[0] : stride);
			auto a_d_data = d_data->template get_access<cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>();
			detail::memcpy(ptr, a_d_data.get_pointer(), n, d_stride, s_stride);
		}

		//! \brief Copy data from the device to the host
		inline void memcpy_d2h(T* ptr, const std::size_t stride = 0)
		{
			memcpy_d2h(ptr, size, stride);
		}

		//! \brief Copy data from the device to the host
		inline void memcpy_d2h()
		{
			if (!has_external_host_pointer)
			{
				std::cerr << "warning: buffer::memcpy_d2h -> no external host pointer available" << std::endl;
			}
			else
			{
				auto a_d_data = d_data->template get_access<cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>();
				std::memcpy(host_ptr, a_d_data.get_pointer(), size.reduce([&] (const T a, const T b) { return a * b; }, 1) * sizeof(T));
			}
		}

		//! \brief Exchange the content of two buffers
		//!
		//! The buffers have to have the same size.
		//!
		//! \param b
		void swap(buffer& b)
		{
			if (size == b.size)
			{
				// swap internal storage
				cl::sycl::buffer<T, D>* this_d_data = d_data;
				d_data = b.d_data;
				b.d_data = this_d_data;
			}
			else
			{
				std::cerr << "error: buffer<>::swap -> you are trying to swap buffers of different size" << std::endl;
			}
		}
	};
	#endif
}

#include "buffer_math.hpp"

#undef XXX_NAMESPACE

#endif
