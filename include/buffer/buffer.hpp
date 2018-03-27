// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(DATA_TYPES_BUFFER_HPP)
#define DATA_TYPES_BUFFER_HPP

#if defined(HAVE_SYCL)
#include <CL/sycl.hpp>
#include <sched.h>
#include <sys/sysinfo.h>
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

namespace XXX_NAMESPACE
{
	enum class data_layout { AoS = 1, SoA = 2 };
}

#include "../vec/vec.hpp"
#include "../sarray/sarray.hpp"
#include "../misc/misc.hpp"
#include "../simd/simd.hpp"

namespace XXX_NAMESPACE
{
	constexpr std::size_t data_alignment = SIMD_NAMESPACE::simd::alignment;

	#if defined(HAVE_SYCL)
	enum class target { host = 1, device = 2 };
	enum class buffer_type { host = 1, device = 2, host_device = 3 };
	#else
	enum class target { host = 1 };
	enum class buffer_type { host = 1 };
	#endif

	namespace detail
	{
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//! \brief Accessor data type implementing array subscript operator chaining
		//!
		//! This data type basically collects all indices and determines the final base pointer for data access.
		//!
		//! \tparam T data type
		//! \tparam D dimension
		//! \tparam Data_layout any of SoA (structs of arrays) and AoS (array of structs)
		//! \tparam Enabled needed for partial specialization with T = vec<TT, DD> and Data_layout = SoA
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		template <typename T, std::size_t D, data_layout Data_layout = data_layout::AoS, typename Enabled = void>
		class accessor
		{
			//! Base pointer
			T* ptr;
			//! Extent of the D-dimensional array
			const sarray<std::size_t, D> n;

		public:

			//! \brief Standard constructor
			//!
			//! \param ptr base pointer
			//! \param n extent of the D-dimensional array
			accessor(T* ptr, const sarray<std::size_t, D>& n) : ptr(ptr), n(n) { ; }

			//! \brief Array subscript operator
			//!
			//! Determine the base pointer of accessor<T, D - 1> by multiplying the
			//! dimensions of the D-1-dimensional sub-volume and idx.
			//!
			//! \param idx index w.r.t. dimension D
			//! \return a accessor<T, D - 1> object
			inline accessor<T, D - 1> operator[](const std::size_t idx)
			{
				std::size_t offset = idx;
				for (std::size_t i = 0; i < (D - 1); ++i)
				{
					offset *= n[i];
				}
				return accessor<T, D - 1>(&ptr[offset], n);
			}

			//! \brief Array subscript operator
			//!
			//! Determine the base pointer of accessor<T, D - 1> by multiplying the
			//! dimensions of the D-1-dimensional sub-volume and idx.
			//!
			//! \param idx index w.r.t. dimension D
			//! \return a const accessor<T, D - 1> object
			inline accessor<T, D - 1> operator[](const std::size_t idx) const
			{
				std::size_t offset = idx;
				for (std::size_t i = 0; i < (D - 1); ++i)
				{
					offset *= n[i];
				}
				return accessor<T, D - 1>(&ptr[offset], n);
			}
		};

		////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//! \brief Specialization with D = 1 (recursion ancher definition)
		//!
		//! \tparam T data type
		//! \tparam Data_layout any of SoA (structs of arrays) and AoS (array of structs)
		//! \tparam Enabled needed for partial specialization with T = vec<TT, DD> and Data_layout = SoA
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		template <typename T, data_layout Data_layout, typename Enabled>
		class accessor<T, 1, Data_layout, Enabled>
		{
			//! Base pointer
			T* ptr;
			//! Extent of the 1-dimensional array
			const sarray<std::size_t, 1> n;

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

		////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//! \brief Specialization with T = vec<TT, DD> and Data_layout = SoA
		//!
		//! This accessor differs from the general case in that it internally accesses the data using the SoA
		//! layout.
		//! It is implemented for T = vec<TT, DD> only, where DD adds an implicit next-to-the-innermost dimension.
		//!
		//! \tparam T data type
		//! \tparam D dimension
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		template <typename T, std::size_t D>
		class accessor<T, D, data_layout::SoA, typename std::enable_if<is_vec<T>::value>::type>
		{
			//! Get the mapped data type that is behind T
			using TT = typename type_info<T, data_layout::SoA>::mapped_type;
			//! Recover the dimension of T
			static constexpr std::size_t DD = T::dim;

			//! Base pointer
			TT* ptr;
			//! Extent of the D-dimensional array
			const sarray<std::size_t, D> n;

		public:

			//! \brief Standard constructor
			//!
			//! \param ptr base pointer
			//! \param n extent of the D-dimensional array
			accessor(TT* ptr, const sarray<std::size_t, D>& n) : ptr(ptr), n(n) { ; }

			//! \brief Array subscript operator
			//!
			//! Determine the base pointer of accessor<T, D - 1> by multiplying the
			//! dimensions of the D-1-dimensional sub-volume and idx and DD.
			//!
			//! \param idx index w.r.t. dimension D
			//! \return a accessor<T, D - 1> object
			inline accessor<T, D - 1, data_layout::SoA> operator[](const std::size_t idx)
			{
				std::size_t offset = idx * DD;
				for (std::size_t i = 0; i < (D - 1); ++i)
				{
					offset *= n[i];
				}
				return accessor<T, D - 1, data_layout::SoA>(&ptr[offset], n);
			}

			//! \brief Array subscript operator
			//!
			//! Determine the base pointer of accessor<T, D - 1> by multiplying the
			//! dimensions of the D-1-dimensional sub-volume and idx and DD.
			//!
			//! \param idx index w.r.t. dimension D
			//! \return a const accessor<T, D - 1> object
			inline accessor<T, D - 1, data_layout::SoA> operator[](const std::size_t idx) const
			{
				std::size_t offset = idx * DD;
				for (std::size_t i = 0; i < (D - 1); ++i)
				{
					offset *= n[i];
				}
				return accessor<T, D - 1, data_layout::SoA>(&ptr[offset], n);
			}
		};

		////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//! \brief Specialization with T = vec<TT, DD>, D = 1 and Data_layout = SoA
		//!
		//! This accessor differs from the general case in that it internally accesses the data using the SoA
		//! layout.
		//! It is implemented for T = vec<TT, DD> only, where DD adds an implicit next-to-the-innermost dimension.
		//!
		//! \tparam T data type
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		template <typename T>
		class accessor<T, 1, data_layout::SoA, typename std::enable_if<is_vec<T>::value>::type>
		{
			//! Get the mapped data type that is behind T
			using TT = typename type_info<T, data_layout::SoA>::mapped_type;
			//! Recover the dimension of T
			static constexpr std::size_t DD = T::dim;

			//! Base pointer
			TT* ptr;
			//! Extent of the 1-dimensional array
			const sarray<std::size_t, 1> n;

		public:

			//! \brief Standard constructor
			//!
			//! \param ptr base pointer
			//! \param n extent of the D-dimensional array
			accessor(TT* ptr, const sarray<std::size_t, 1>& n) : ptr(ptr), n(n) { ; }

			//! \brief Array subscript operator
			//!
			//! \param idx element to access
			//! \return a vec_proxy<TT, DD> object
			inline detail::vec_proxy<TT, DD> operator[](const std::size_t idx)
			{
				return detail::vec_proxy<TT, DD>(&ptr[idx], n[0]);
			}

			//! \brief Array subscript operator
			//!
			//! \param idx element to access
			//! \return a const vec_proxy<TT, DD> object
			inline detail::vec_proxy<TT, DD> operator[](const std::size_t idx) const
			{
				return detail::vec_proxy<TT, DD>(&ptr[idx], n[0]);
			}
		};
	}
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//! \brief Multi-dimensional buffer data type
	//!
	//! This buffer data type internally uses std::vector<T> to dynamically adapt to the needed memory requirement.
	//! In case of D > 1, the size of the std::vector<T> is determined as the product of all dimensions with the
	//! innermost dimension padded according to T or the (default) data alignment.
	//! All memory thus is contiguous and can be moved as a whole (important e.g. for data transfers from/to GPGPU).
	//! The buffer, however, allows access to the data using array subscript operator chaining using proxy objects.
	//! GNU and Clang/LLVM can optimize the proxies away.
	//! \n\n
	//! In case of Data_layout = AoS, data is stored with all elements placed in main memory one after the other.
	//! \n
	//! In case of Data_layout = SoA (meaningful only if T is of type vec<TT, DD>) the individual components (x,y and z)
	//! are placed one after the other along the innermost dimension, e.g. for
	//! buffer<vec<double, 3>, 2, SoA, 32>({3, 2}) the memory layout would be the following one:
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
	//! For example you can access the individual components of buffer<vec<double, 3>, 2, SoA> b({3,2})
	//! as usual: b[1][0].x = ...
	//! \n
	//! GNU and Clang/LLVM seem to optimize the proxies away.
	//!
	//! \tparam T data type
	//! \tparam D dimension
	//! \tparam Data_layout any of SoA (structs of arrays) and AoS (array of structs)
	//! \tparam Enabled needed for partial specialization with T = vec<TT, DD> and Data_layout = SoA
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	template <typename T, std::size_t D, buffer_type Buffer_type = buffer_type::host, data_layout Data_layout = data_layout::AoS>
	class buffer
	{
		//! Mapped data type
		using TT = typename type_info<T, Data_layout>::mapped_type;
		//! Extra dimension: relevant for AoS data layout only
		static constexpr std::size_t DD = type_info<T, Data_layout>::extra_dim;

	protected:

		//! Internal storage using boost's aligned_allocator for data alignment
		std::vector<TT, boost::alignment::aligned_allocator<TT, data_alignment>> m_data;
		//! Base pointer (does not necessarily point to m_data)
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
		//! \param ptr external pointer (if specified, no internal storage will be allocated and no padding of
		//! the innermost dimension happens)
		buffer(const sarray<std::size_t, D>& size, T* ptr = nullptr)
		{
			resize(size, ptr);
		}

		//! \brief Set up the buffer
		//!
		//! \param size shape of the buffer
		//! \param ptr external pointer (if specified, no internal storage will be allocated and no padding of
		//! the innermost dimension happens)
		void resize(const sarray<std::size_t, D>& size, T* ptr = nullptr)
		{
			// assign default values
			size_internal = size;
			this->size = size;

			if (ptr == nullptr)
			{
				// padding according to data type T and the (default) data alignment
				constexpr std::size_t n_padd = type_info<T, Data_layout>::get_n_padd(data_alignment);
				size_internal[0] = ((size[0] + n_padd - 1) / n_padd) * n_padd;

				// resize internal buffer: in case of SoA layout, an internal dimension DD is implicit
				std::size_t num_elements_total = DD;
				for (std::size_t i = 0; i < D; ++i)
				{
					num_elements_total *= size_internal[i];
				}
				m_data.resize(num_elements_total);

				// base pointer points to the internal storage
				data = &m_data[0];
			}
			else
			{
				// a pointer to external memory is provided: clear internal storage
				m_data.clear();
				data = reinterpret_cast<TT*>(ptr);
			}
		}

		//! \brief Read accessor
		//!
		//! \return an accessor using const references internally
		inline const detail::accessor<const T, D, Data_layout> read() const
		{
			using const_TT = typename type_info<const T, Data_layout>::mapped_type;
			return detail::accessor<const T, D, Data_layout>(reinterpret_cast<const_TT*>(data), size_internal);
		}

		//! \brief Write accessor
		//!
		//! \return an accessor
		inline const detail::accessor<T, D, Data_layout> write()
		{
			return detail::accessor<T, D, Data_layout>(data, size_internal);
		}

		//! \brief Read-write accessor
		//!
		//! \return an accessor
		inline const detail::accessor<T, D, Data_layout> read_write()
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
				// swap internal storage (it does not matter whether any of the buffers uses
				// external memory)
				m_data.swap(b.m_data);

				// swap the base pointer
				TT* this_data = data;
				data = b.data;
				b.data = this_data;

				// re-assign base pointers only if internal storage is used
				if (m_data.size() > 0)
				{
					data = &m_data[0];
				}

				if (b.m_data.size() > 0)
				{
					b.data = &(b.m_data[0]);
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
		template <typename T, data_layout Data_layout_dst, data_layout Data_layout_src>
		void memcpy(const accessor<T, 1, Data_layout_dst>& dst, const accessor<typename make_const<T>::type, 1, Data_layout_src>& src, const XXX_NAMESPACE::sarray<std::size_t, 1>& size)
		{
			for (std::size_t i = 0; i < size[0]; ++i)
			{
				dst[i] = src[i];
			}
		}

		template <typename T, data_layout Data_layout_dst, data_layout Data_layout_src>
		void memcpy(const accessor<T, 2, Data_layout_dst>& dst, const accessor<typename make_const<T>::type, 2, Data_layout_src>& src, const XXX_NAMESPACE::sarray<std::size_t, 3>& size)
		{
			for (std::size_t j = 0; j < size[1]; ++j)
			{
				for (std::size_t i = 0; i < size[0]; ++i)
				{
					dst[j][i] = src[j][i];
				}
			}
		}

		template <typename T, data_layout Data_layout_dst, data_layout Data_layout_src>
		void memcpy(const accessor<T, 3, Data_layout_dst>& dst, const accessor<typename make_const<T>::type, 3, Data_layout_src>& src, const XXX_NAMESPACE::sarray<std::size_t, 3>& size)
		{
			for (std::size_t k = 0; k < size[2]; ++k)
			{
				for (std::size_t j = 0; j < size[1]; ++j)
				{
					for (std::size_t i = 0; i < size[0]; ++i)
					{
						dst[k][j][i] = src[k][j][i];
					}
				}
			}
		}
	}

	template <typename T, std::size_t D, data_layout Data_layout>
	class buffer<T, D, buffer_type::device, Data_layout>
	{
	protected:

		//! SYCL device buffer
		cl::sycl::buffer<T, D>* m_data;
		//! External host pointer
		T* external_host_ptr;
		//! Is it an external pointer
		bool has_external_host_ptr;

	public:

		//! Shape of the buffer (as specified by the user)
		sarray<std::size_t, D> size;

		//! \brief Standard constructor
		buffer() : m_data(nullptr), external_host_ptr(nullptr), has_external_host_ptr(false), size{} { ; }

		//! \brief Constructor
		//!
		//! \param size shape of the buffer
		//! \param ptr external pointer (if specified, no internal storage will be allocated and no padding of the
		//! innermost dimension happens)
		buffer(const sarray<std::size_t, D>& size, T* ptr = nullptr) : m_data(nullptr), external_host_ptr(nullptr), has_external_host_ptr(false)
		{
			resize(size, ptr);
		}

		//! \brief Destructor
		~buffer()
		{
			if (m_data != nullptr)
			{
				delete m_data;
				m_data = nullptr;
			}
			external_host_ptr = nullptr;
		}

		//! \brief Set up the buffer
		//!
		//! \param size shape of the buffer
		//! \param ptr external pointer (if specified, no internal storage will be allocated and no padding of the
		//! innermost dimension happens)
		void resize(const sarray<std::size_t, D>& size, T* ptr = nullptr)
		{
			static bool fix_thread_pinning = true;
			static cpu_set_t master_thread_cpu_mask;
			static cpu_set_t sycl_thread_cpu_mask;

			// fix thread affinity issue: SYCL internally spawns a scheduling thread
			// which inherits the master thread's cpu affinity -> place it to an exclusive CPU core
			if (fix_thread_pinning)
			{	
				// backup the master cpu mask
				sched_getaffinity(0, sizeof(master_thread_cpu_mask), &master_thread_cpu_mask);
				// get cpu core counts: phyical and logical
				std::size_t num_cpu_cores = get_nprocs();
				std::size_t num_cpu_cores_conf = get_nprocs_conf();
				// the cpu mask for the SYCL thread is given by 'num_cpu_cores' and 'num_cpu_cores - 1'
				CPU_ZERO(&sycl_thread_cpu_mask);
				CPU_SET(num_cpu_cores - 1, &sycl_thread_cpu_mask);
				CPU_SET(num_cpu_cores_conf - 1, &sycl_thread_cpu_mask);
				sched_setaffinity(0, sizeof(sycl_thread_cpu_mask), &sycl_thread_cpu_mask);
			}

			// assign default values
			this->size = size;

			// SYCL range: the order of the entries is inverse
			cl::sycl::range<D> size_internal;
			for (std::size_t i = 0; i < D; ++i)
			{
				size_internal[(D - 1) - i] = size[i];
			}

			// rezise the internal buffer
			if (m_data != nullptr)
			{
				delete m_data;
			}
			m_data = new cl::sycl::buffer<T, D>(size_internal);

			// is there an external host pointer?
			if (ptr != nullptr)
			{
				external_host_ptr = ptr;
				has_external_host_ptr = true;
			}

			// reset CPU affinity of the master thread
			if (fix_thread_pinning)
			{
				CPU_XOR(&master_thread_cpu_mask, &master_thread_cpu_mask, &sycl_thread_cpu_mask);
				sched_setaffinity(0, sizeof(master_thread_cpu_mask), &master_thread_cpu_mask);
				fix_thread_pinning = false;
			}
		}

		//! \brief Read accessor
		//!
		//! \return a read accessor
		inline auto read(cl::sycl::handler& h) const
		{
			return m_data->template get_access<cl::sycl::access::mode::read, cl::sycl::access::target::global_buffer>(h);
		}

		//! \brief Write accessor
		//!
		//! \return a write accessor
		inline auto write(cl::sycl::handler& h)
		{
			return m_data->template get_access<cl::sycl::access::mode::write, cl::sycl::access::target::global_buffer>(h);
		}

		//! \brief Read-write accessor
		//!
		//! \return a read-write accessor
		inline auto read_write(cl::sycl::handler& h)
		{
			return m_data->template get_access<cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer>(h);
		}

		//! \brief Copy data from the host to the device
		inline void memcpy_h2d(const T* ptr, const sarray<std::size_t, D>& n, const std::size_t stride = 0)
		{		
			// host accessor: inner dimension for data access is given by 'stride', or n[0]
			sarray<std::size_t, D> n_src = n;
			n_src[0] = (stride == 0 ? n[0] : stride);
			using TT = typename type_info<T, Data_layout>::mapped_type;
			using const_T = typename make_const<T>::type;
			using const_TT = typename make_const<TT>::type;
			const detail::accessor<const_T, D, Data_layout> src(reinterpret_cast<const_TT*>(ptr), n_src);
			
			// device accessor: data layout for the device is always AoS
			auto a_m_data = m_data->template get_access<cl::sycl::access::mode::write, cl::sycl::access::target::host_buffer>();
			const detail::accessor<T, D> dst(a_m_data.get_pointer(), size);

			detail::memcpy(dst, src, n);
		}

		//! \brief Copy data from the host to the device
		inline void memcpy_h2d(const T* ptr = nullptr, const std::size_t stride = 0)
		{
			if (ptr != nullptr)
			{
				memcpy_h2d(ptr, size, stride);
			}
			else
			{
				if (has_external_host_ptr)
				{
					memcpy_d2h(external_host_ptr, size, stride);
				}
				else
				{
					std::cerr << "warning: buffer::memcpy_h2d -> no external host pointer available" << std::endl;
				}
			}
		}

		//! \brief Copy data from the device to the host
		inline void memcpy_d2h(T* ptr, const sarray<std::size_t, D>& n, const std::size_t stride = 0)
		{
			// device accessor: data layout for the device is always AoS
			auto a_m_data = m_data->template get_access<cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>();
			using const_T = typename make_const<T>::type;
			const detail::accessor<const_T, D> src(reinterpret_cast<const_T*>(a_m_data.get_pointer()), size);

			// host accessor: inner dimension for data access is given by 'stride', or n[0]
			sarray<std::size_t, D> n_dst = n;
			n_dst[0] = (stride == 0 ? n[0] : stride);
			using TT = typename type_info<T, Data_layout>::mapped_type;
			const detail::accessor<T, D, Data_layout> dst(reinterpret_cast<TT*>(ptr), n_dst);

			detail::memcpy(dst, src, n);
		}

		//! \brief Copy data from the device to the host
		inline void memcpy_d2h(T* ptr = nullptr, const std::size_t stride = 0)
		{
			if (ptr != nullptr)
			{
				memcpy_d2h(ptr, size, stride);
			}
			else
			{
				if (has_external_host_ptr)
				{
					memcpy_d2h(external_host_ptr, size, stride);
				}
				else
				{
					std::cerr << "warning: buffer::memcpy_d2h -> no external host pointer available" << std::endl;
				}
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
				cl::sycl::buffer<T, D>* this_m_data = m_data;
				m_data = b.m_data;
				b.m_data = this_m_data;

				// swap external host pointer
				T* this_external_host_ptr = external_host_ptr;
				external_host_ptr = b.external_host_ptr;
				b.external_host_ptr = this_external_host_ptr;

				bool this_has_external_host_ptr = has_external_host_ptr;
				has_external_host_ptr = b.has_external_host_ptr;
				b.has_external_host_ptr = this_has_external_host_ptr;
			}
			else
			{
				std::cerr << "error: buffer::swap -> you are trying to swap buffers of different size" << std::endl;
			}
		}
	};

	template <typename T, std::size_t D, data_layout Data_layout>
	class buffer<T, D, buffer_type::host_device, Data_layout> : public buffer<T, D, buffer_type::host, Data_layout>, buffer<T, D, buffer_type::device, Data_layout>
	{
		//! Mapped data type
		using TT = typename type_info<T, Data_layout>::mapped_type;
		//! Extra dimension: relevant for AoS data layout only
		static constexpr std::size_t DD = type_info<T, Data_layout>::extra_dim;

		using host_buffer = buffer<T, D, buffer_type::host, Data_layout>;
		using device_buffer = buffer<T, D, buffer_type::device, Data_layout>;

	public:

		using host_buffer::size;

		buffer() : host_buffer(), device_buffer() { ; }

		buffer(const sarray<std::size_t, D>& size) : host_buffer(size), device_buffer(size, reinterpret_cast<T*>(host_buffer::data)) { ; }

		void resize(const sarray<std::size_t, D>& size)
		{
			host_buffer::resize(size);
			device_buffer::resize(size, reinterpret_cast<T*>(host_buffer::data));
		}

		template <target Target, typename X = typename std::enable_if<Target == target::host>::type>
		inline auto read(X* dummy = nullptr) const
		{
			return host_buffer::read();
		}

		template <target Target, typename X = typename std::enable_if<Target == target::host>::type>
		inline auto write(X* dummy = nullptr)
		{
			return host_buffer::write();
		}

		template <target Target, typename X = typename std::enable_if<Target == target::host>::type>
		inline auto read_write(X* dummy = nullptr)
		{
			return host_buffer::read_write();
		}

		template <target Target>
		inline auto read(cl::sycl::handler& h) const
		{
			return device_buffer::read(h);
		}

		template <target Target>
		inline auto write(cl::sycl::handler& h)
		{
			return device_buffer::write(h);
		}

		template <target Target>
		inline auto read_write(cl::sycl::handler& h)
		{
			return device_buffer::read_write(h);
		}

		inline void memcpy_h2d()
		{
			device_buffer::memcpy_h2d(reinterpret_cast<T*>(host_buffer::data), size, host_buffer::size_internal[0]);
		}

		inline void memcpy_d2h()
		{
			device_buffer::memcpy_d2h(reinterpret_cast<T*>(host_buffer::data), size, host_buffer::size_internal[0]);
		}

		void swap(buffer& b)
		{
			if (size == b.size)
			{
				// host buffer:

				// swap internal storage (it does not matter whether any of the buffers uses
				// external memory)
				host_buffer::m_data.swap(b.host_buffer::m_data);

				// swap the base pointer
				TT* this_data = host_buffer::data;
				host_buffer::data = b.host_buffer::data;
				b.host_buffer::data = this_data;

				// re-assign base pointers only if internal storage is used
				if (host_buffer::m_data.size() > 0)
				{
					host_buffer::data = &host_buffer::m_data[0];
				}

				if (b.host_buffer::m_data.size() > 0)
				{
					b.host_buffer::data = &(b.host_buffer::m_data[0]);
				}

				// SYCL buffer:

				// swap internal storage
				cl::sycl::buffer<T, D>* this_m_data = device_buffer::m_data;
				device_buffer::m_data = b.device_buffer::m_data;
				b.device_buffer::m_data = this_m_data;

				// swap external host pointer
				T* this_external_host_ptr = device_buffer::external_host_ptr;
				device_buffer::external_host_ptr = b.device_buffer::external_host_ptr;
				b.device_buffer::external_host_ptr = this_external_host_ptr;

				bool this_has_external_host_ptr = device_buffer::has_external_host_ptr;
				device_buffer::has_external_host_ptr = b.device_buffer::has_external_host_ptr;
				b.device_buffer::has_external_host_ptr = this_has_external_host_ptr;
			}
			else
			{
				std::cerr << "error: buffer::swap -> you are trying to swap buffers of different size" << std::endl;
			}
		}
	};
	
	#endif
}

#endif