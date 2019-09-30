// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(DATA_TYPES_VEC_VEC_HPP)
#define DATA_TYPES_VEC_VEC_HPP

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#if !defined(VEC_NAMESPACE)
#define VEC_NAMESPACE XXX_NAMESPACE
#endif

#include <common/Math.hpp>
#include <data_types/DataTypes.hpp>
#include <platform/Target.hpp>

namespace VEC_NAMESPACE
{
    // some forward declarations
    namespace internal
    {
        template <typename T, SizeType D>
        class vec_proxy;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //! \brief A simple vector with D components of the same type
    //!
    //! \tparam T data type
    //! \tparam D dimension
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <typename T, SizeType D>
    class vec;

    //! \brief D = 1 specialization with component x
    //!
    //! \tparam T data type
    template <typename T>
    class vec<T, 1>
    {
        static_assert(std::is_fundamental<T>::value, "error: T is not a fundamental data type");
        static_assert(!std::is_void<T>::value, "error: T is void -> not allowed");
        static_assert(!std::is_volatile<T>::value, "error: T is volatile -> not allowed");
        static_assert(!std::is_const<T>::value, "error: T is const -> not allowed");

    public:

        using type = vec<T, 1>;
        using ProxyType = typename internal::vec_proxy<T, 1>;
        //! Remember the template type parameter T
        using value_type = T;
        //! Remember the template parameter D (=1)
        static constexpr SizeType d = 1;

        T x;

        //! Constructors
        HOST_VERSION
        CUDA_DEVICE_VERSION
        vec(const T x = 0)
            :
            x(x) {}

        template <typename X>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        vec(const vec<X, 1>& v)
            :
            x(v.x) {}

        template <typename X>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        vec(const internal::vec_proxy<X, 1>& vp)
            :
            x(vp.x) {}
        
        //! Some operators
        inline vec operator-() const
        {
            return vec(-x);
        }

    #define MACRO(OP, IN_T)                             \
        template <typename X>                           \
        inline vec& operator OP (const IN_T<X, 1>& v)   \
        {                                               \
            x OP v.x;                                   \
            return *this;                               \
        }                                               \
        
        MACRO(=, vec)
        MACRO(+=, vec)
        MACRO(-=, vec)
        MACRO(*=, vec)
        MACRO(/=, vec)

        MACRO(=, internal::vec_proxy)
        MACRO(+=, internal::vec_proxy)
        MACRO(-=, internal::vec_proxy)
        MACRO(*=, internal::vec_proxy)
        MACRO(/=, internal::vec_proxy)

    #undef MACRO

    #define MACRO(OP)                                   \
        template <typename X>                           \
        inline vec& operator OP (const X x)             \
        {                                               \
            x OP x;                                     \
            return *this;                               \
        }                                               \

        MACRO(=)
        MACRO(+=)
        MACRO(-=)
        MACRO(*=)
        MACRO(/=)

    #undef MACRO

        //! \brief Return the Euclidean norm of the vector
        //!
        //! \return Euclidean norm
        inline T length() const
        {
            return std::abs(x);
        }
    };

    //! \brief D = 2 specialization with component x
    //!
    //! \tparam T data type
    template <typename T>
    class vec<T, 2>
    {
        static_assert(std::is_fundamental<T>::value, "error: T is not a fundamental data type");
        static_assert(!std::is_void<T>::value, "error: T is void -> not allowed");
        static_assert(!std::is_volatile<T>::value, "error: T is volatile -> not allowed");
        static_assert(!std::is_const<T>::value, "error: T is const -> not allowed");

    public:

        using type = vec<T, 2>;
        using ProxyType = typename internal::vec_proxy<T, 2>;
        //! Remember the template type parameter T
        using value_type = T;
        //! Remember the template parameter D (=2)
        static constexpr SizeType d = 2;

        T x;
        T y;

        //! Constructors
        HOST_VERSION
        CUDA_DEVICE_VERSION
        vec(const T xy = 0) 
            :
            x(xy),
            y(xy) {}

        HOST_VERSION
        CUDA_DEVICE_VERSION
        vec(const T x, const T y)
            :
            x(x),
            y(y) {}

        template <typename X>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        vec(const vec<X, 2>& v)
            :
            x(v.x),
            y(v.y) {}

        template <typename X>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        vec(const internal::vec_proxy<X, 2>& vp)
            :
            x(vp.x),
            y(vp.y) {}
        
        //! Some operators
        inline vec operator-() const
        {
            return vec(-x, -y);
        }

    #define MACRO(OP, IN_T)                             \
        template <typename X>                           \
        inline vec& operator OP (const IN_T<X, 2>& v)   \
        {                                               \
            x OP v.x;                                   \
            y OP v.y;                                   \
            return *this;                               \
        }                                               \

        MACRO(=, vec)
        MACRO(+=, vec)
        MACRO(-=, vec)
        MACRO(*=, vec)
        MACRO(/=, vec)

        MACRO(=, internal::vec_proxy)
        MACRO(+=, internal::vec_proxy)
        MACRO(-=, internal::vec_proxy)
        MACRO(*=, internal::vec_proxy)
        MACRO(/=, internal::vec_proxy)

    #undef MACRO

    #define MACRO(OP)                                   \
        template <typename X>                           \
        inline vec& operator OP (const X xy)            \
        {                                               \
            x OP xy;                                    \
            y OP xy;                                    \
            return *this;                               \
        }                                               \

        MACRO(=)
        MACRO(+=)
        MACRO(-=)
        MACRO(*=)
        MACRO(/=)

    #undef MACRO

        //! \brief Return the Euclidean norm of the vector
        //!
        //! \return Euclidean norm
        inline T length() const
        {
            return MATH_NAMESPACE::math<T>::sqrt(x * x + y * y);
        }
    };
    
    //! \brief D = 3 specialization with component x
    //!
    //! \tparam T data type
    template <typename T>
    class vec<T, 3>
    {
        static_assert(std::is_fundamental<T>::value, "error: T is not a fundamental data type");
        static_assert(!std::is_void<T>::value, "error: T is void -> not allowed");
        static_assert(!std::is_volatile<T>::value, "error: T is volatile -> not allowed");
        static_assert(!std::is_const<T>::value, "error: T is const -> not allowed");

    public:

        using type = vec<T, 3>;
        using ProxyType = typename internal::vec_proxy<T, 3>;
        //! Remember the template type parameter T
        using value_type = T;
        //! Remember the template parameter D (=3)
        static constexpr SizeType d = 3;

        T x;
        T y;
        T z;

        //! Constructors
        HOST_VERSION
        CUDA_DEVICE_VERSION
        vec(const T xyz = 0)
            :
            x(xyz),
            y(xyz),
            z(xyz) {}

        HOST_VERSION
        CUDA_DEVICE_VERSION
        vec(const T x, const T y, const T z)
            :
            x(x),
            y(y),
            z(z) {}

        template <typename X>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        vec(const vec<X, 3>& v)
            :
            x(v.x),
            y(v.y),
            z(v.z) {}

        template <typename X>
        HOST_VERSION
        CUDA_DEVICE_VERSION
        vec(const internal::vec_proxy<X, 3>& vp)
            :
            x(vp.x),
            y(vp.y),
            z(vp.z) {}
      
        //! Some operators
        inline vec operator-() const
        {
            return vec(-x, -y, -z);
        }

    #define MACRO(OP, IN_T)                             \
        template <typename X>                           \
        inline vec& operator OP (const IN_T<X, 3>& v)   \
        {                                               \
            x OP v.x;                                   \
            y OP v.y;                                   \
            z OP v.z;                                   \
            return *this;                               \
        }                                               \
        
        MACRO(=, vec)
        MACRO(+=, vec)
        MACRO(-=, vec)
        MACRO(*=, vec)
        MACRO(/=, vec)

        MACRO(=, internal::vec_proxy)
        MACRO(+=, internal::vec_proxy)
        MACRO(-=, internal::vec_proxy)
        MACRO(*=, internal::vec_proxy)
        MACRO(/=, internal::vec_proxy)

    #undef MACRO

    #define MACRO(OP)                                   \
        template <typename X>                           \
        inline vec& operator OP (const X xyz)           \
        {                                               \
            x OP xyz;                                   \
            y OP xyz;                                   \
            z OP xyz;                                   \
            return *this;                               \
        }                                               \

        MACRO(=)
        MACRO(+=)
        MACRO(-=)
        MACRO(*=)
        MACRO(/=)

    #undef MACRO

        //! \brief Return the Euclidean norm of the vector
        //!
        //! \return Euclidean norm
        inline T length() const
        {
            return MATH_NAMESPACE::math<T>::sqrt(x * x + y * y + z * z);
        }
    };

    template <typename T>
    std::ostream& operator<<(std::ostream& os, const vec<T, 1>& v)
    {
        os << "(" << v.x << ")";
        return os;
    }

    template <typename T>
    std::ostream& operator<<(std::ostream& os, const vec<T, 2>& v)
    {
        os << "(" << v.x << "," << v.y << ")";
        return os;
    }

    template <typename T>
    std::ostream& operator<<(std::ostream& os, const vec<T, 3>& v)
    {
        os << "(" << v.x << "," << v.y << "," << v.z << ")";
        return os;
    }
}

#include <data_types/vec/VecProxy.hpp>
#include <data_types/vec/VecMath.hpp>

#include <common/Traits.hpp>

namespace XXX_NAMESPACE
{
    namespace internal
    {
        template <typename T, SizeType D>
        struct provides_proxy_type<VEC_NAMESPACE::vec<T, D>>
        {
            static constexpr bool value = true;
        };

        template <typename T, SizeType D>
        struct provides_proxy_type<const VEC_NAMESPACE::vec<T, D>>
        {
            static constexpr bool value = true;
        };
    }
}

namespace MATH_NAMESPACE
{
    template <typename T, SizeType D>
    struct math<VEC_NAMESPACE::vec<T, D>>
    {
        using type = VEC_NAMESPACE::vec<T, D>;
        using value_type = typename std::remove_cv<typename type::value_type>::type;

        static constexpr value_type one = math<value_type>::one;
        static constexpr value_type minus_one = math<value_type>::minus_one;

        template <typename X>
        static type sqrt(X x)
        {
            return MATH_NAMESPACE::sqrt(x);
        }

        template <typename X>
        static type log(X x)
        {
            return MATH_NAMESPACE::log(x);
        }

        template <typename X>
        static type exp(X x)
        {
            return MATH_NAMESPACE::exp(x);
        }
    };
}

#endif
