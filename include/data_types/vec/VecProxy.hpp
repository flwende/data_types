// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(DATA_TYPES_VEC_VEC_PROXY_HPP)
#define DATA_TYPES_VEC_VEC_PROXY_HPP

#if defined(MEGA_OLD)

#if defined(OLD)

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#include <common/Memory.hpp>
#include <common/DataLayout.hpp>
#include <data_types/DataTypes.hpp>

// some forward declarations
namespace XXX_NAMESPACE
{
    namespace internal 
    {
        template <typename P, typename R>
        class iterator;
    }
}

namespace XXX_NAMESPACE
{
    namespace dataTypes
    {
        namespace internal 
        {
            template <typename X, SizeT N, SizeT D, ::XXX_NAMESPACE::memory::DataLayout L>
            class Accessor;
        }
    }
}


namespace XXX_NAMESPACE
{
    namespace dataTypes
    {
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //! \brief A proxy data type for Vec<T, D>
        //!
        //! This data type is returned by accessor<T, D, Layout>::operator[] if D = 1 and SoA data layout.
        //! It holds references to component(s) x [,y [and z]], so that data access via,
        //! e.g. obj[ ]..[ ].x, is possible.
        //!
        //! \tparam T data type
        //! \tparam D dimension
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        namespace internal
        {
            //! \brief D = 1 specialization with component x
            //!
            //! \tparam T data type
            template <typename T>
            class VecProxy<T, 1>
            {
                static_assert(std::is_fundamental<T>::value, "error: T is not a fundamental data type");
                static_assert(!std::is_void<T>::value, "error: T is void -> not allowed");
                static_assert(!std::is_volatile<T>::value, "error: T is volatile -> not allowed");

                template <typename, SizeT, SizeT, ::XXX_NAMESPACE::memory::DataLayout>
                friend class XXX_NAMESPACE::dataTypes::internal::Accessor;

            public:

                using type = VecProxy<T, 1>;
                using ConstT = const VecProxy<const T, 1>;
                using T_unqualified = typename std::remove_cv<T>::type;
                //! Remember the template type parameter T
                using value_type = T;
                //! Remember the template parameter D (=1)
                static constexpr SizeT d = 1;
                using BasePointer = ::XXX_NAMESPACE::memory::PointerN<T, 1>;
                using original_type = typename std::conditional<std::is_const<T>::value, 
                    const Vec<typename std::remove_cv<T>::type, 1>,
                    Vec<typename std::remove_cv<T>::type, 1>>::type;

                T& x;

            private:

                HOST_VERSION
                CUDA_DEVICE_VERSION
                VecProxy(std::tuple<T&> v)
                    :
                    x(std::get<0>(v)) {}    

            public:

                inline ::XXX_NAMESPACE::dataTypes::Vec<T_unqualified, 1>  operator-() const
                {
                    return ::XXX_NAMESPACE::dataTypes::Vec<T_unqualified, 1>(-x);
                }

            #define MACRO(OP, IN_T)                                                 \
                inline VecProxy& operator OP (IN_T<T, 1>& v)        \
                {                                                                   \
                    x OP v.x;                                                       \
                    return *this;                                                   \
                }                                                                   \
                                                                                    \
                inline VecProxy& operator OP (const IN_T<T, 1>& v)  \
                {                                                                   \
                    x OP v.x;                                                       \
                    return *this;                                                   \
                }                                                                   \
                                                                                    \
                template <typename X>                                               \
                inline VecProxy& operator OP (IN_T<X, 1>& v)        \
                {                                                                   \
                    x OP v.x;                                                       \
                    return *this;                                                   \
                }                                                                   \
                                                                                    \
                template <typename X>                                               \
                inline VecProxy& operator OP (const IN_T<X, 1>& v)  \
                {                                                                   \
                    x OP v.x;                                                       \
                    return *this;                                                   \
                }                                                                   \

                MACRO(=, ::XXX_NAMESPACE::dataTypes::Vec)
                MACRO(+=, ::XXX_NAMESPACE::dataTypes::Vec)
                MACRO(-=, ::XXX_NAMESPACE::dataTypes::Vec)
                MACRO(*=, ::XXX_NAMESPACE::dataTypes::Vec)
                MACRO(/=, ::XXX_NAMESPACE::dataTypes::Vec)

                MACRO(=, ::XXX_NAMESPACE::dataTypes::internal::VecProxy)
                MACRO(+=, ::XXX_NAMESPACE::dataTypes::internal::VecProxy)
                MACRO(-=, ::XXX_NAMESPACE::dataTypes::internal::VecProxy)
                MACRO(*=, ::XXX_NAMESPACE::dataTypes::internal::VecProxy)
                MACRO(/=, ::XXX_NAMESPACE::dataTypes::internal::VecProxy)

            #undef MACRO

            #define MACRO(OP)                                                       \
                template <typename X>                                               \
                inline VecProxy& operator OP (const X x)                           \
                {                                                                   \
                    x OP x;                                                         \
                    return *this;                                                   \
                }                                                                   \

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
            class VecProxy<T, 2>
            {
                static_assert(std::is_fundamental<T>::value, "error: T is not a fundamental data type");
                static_assert(!std::is_void<T>::value, "error: T is void -> not allowed");
                static_assert(!std::is_volatile<T>::value, "error: T is volatile -> not allowed");

                template <typename, SizeT, SizeT, ::XXX_NAMESPACE::memory::DataLayout>
                friend class XXX_NAMESPACE::dataTypes::internal::Accessor;

            public:

                using type = VecProxy<T, 2>;
                using ConstT = const VecProxy<const T, 2>;
                using T_unqualified = typename std::remove_cv<T>::type;
                //! Remember the template type parameter T
                using value_type = T;
                //! Remember the template parameter D (=2)
                static constexpr SizeT d = 2;
                using BasePointer = ::XXX_NAMESPACE::memory::PointerN<T, 2>;
                using original_type = typename std::conditional<std::is_const<T>::value, 
                    const Vec<typename std::remove_cv<T>::type, 2>,
                    Vec<typename std::remove_cv<T>::type, 2>>::type;

                T& x;
                T& y;

            private:

                HOST_VERSION
                CUDA_DEVICE_VERSION
                VecProxy(std::tuple<T&, T&> v)
                    :
                    x(std::get<0>(v)),
                    y(std::get<1>(v)) {}

            public:

                inline ::XXX_NAMESPACE::dataTypes::Vec<T_unqualified, 2>  operator-() const
                {
                    return ::XXX_NAMESPACE::dataTypes::Vec<T_unqualified, 2>(-x, -y);
                }

            #define MACRO(OP, IN_T)                                                 \
                inline VecProxy& operator OP (IN_T<T, 2>& v)        \
                {                                                                   \
                    x OP v.x;                                                       \
                    y OP v.y;                                                       \
                    return *this;                                                   \
                }                                                                   \
                                                                                    \
                inline VecProxy& operator OP (const IN_T<T, 2>& v)  \
                {                                                                   \
                    x OP v.x;                                                       \
                    y OP v.y;                                                       \
                    return *this;                                                   \
                }                                                                   \
                                                                                    \
                template <typename X>                                               \
                inline VecProxy& operator OP (IN_T<X, 2>& v)        \
                {                                                                   \
                    x OP v.x;                                                       \
                    y OP v.y;                                                       \
                    return *this;                                                   \
                }                                                                   \
                                                                                    \
                template <typename X>                                               \
                inline VecProxy& operator OP (const IN_T<X, 2>& v)  \
                {                                                                   \
                    x OP v.x;                                                       \
                    y OP v.y;                                                       \
                    return *this;                                                   \
                }                                                                   \

                MACRO(=, ::XXX_NAMESPACE::dataTypes::Vec)
                MACRO(+=, ::XXX_NAMESPACE::dataTypes::Vec)
                MACRO(-=, ::XXX_NAMESPACE::dataTypes::Vec)
                MACRO(*=, ::XXX_NAMESPACE::dataTypes::Vec)
                MACRO(/=, ::XXX_NAMESPACE::dataTypes::Vec)

                MACRO(=, ::XXX_NAMESPACE::dataTypes::internal::VecProxy)
                MACRO(+=, ::XXX_NAMESPACE::dataTypes::internal::VecProxy)
                MACRO(-=, ::XXX_NAMESPACE::dataTypes::internal::VecProxy)
                MACRO(*=, ::XXX_NAMESPACE::dataTypes::internal::VecProxy)
                MACRO(/=, ::XXX_NAMESPACE::dataTypes::internal::VecProxy)

            #undef MACRO

            #define MACRO(OP)                                                       \
                template <typename X>                                               \
                inline VecProxy& operator OP (const X xy)                          \
                {                                                                   \
                    x OP xy;                                                        \
                    y OP xy;                                                        \
                    return *this;                                                   \
                }                                                                   \

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
                    return ::XXX_NAMESPACE::math::internal::Func<T>::sqrt(x * x + y * y);
                }
            };

            //! \brief D = 3 specialization with component x
            //!
            //! \tparam T data type
            template <typename T>
            class VecProxy<T, 3>
            {
                static_assert(std::is_fundamental<T>::value, "error: T is not a fundamental data type");
                static_assert(!std::is_void<T>::value, "error: T is void -> not allowed");
                static_assert(!std::is_volatile<T>::value, "error: T is volatile -> not allowed");

                template <typename, SizeT, SizeT, ::XXX_NAMESPACE::memory::DataLayout>
                friend class XXX_NAMESPACE::dataTypes::internal::Accessor;

            public:

                using type = VecProxy<T, 3>;
                using ConstT = const VecProxy<const T, 3>;
                using T_unqualified = typename std::remove_cv<T>::type;
                //! Remember the template type parameter T
                using value_type = T;
                //! Remember the template parameter D (=3)
                static constexpr SizeT d = 3;
                using BasePointer = ::XXX_NAMESPACE::memory::PointerN<T, 3>;
                using original_type = typename std::conditional<std::is_const<T>::value, 
                    const Vec<typename std::remove_cv<T>::type, 3>,
                    Vec<typename std::remove_cv<T>::type, 3>>::type;

                T& x;
                T& y;
                T& z;

            private:

                HOST_VERSION
                CUDA_DEVICE_VERSION
                VecProxy(std::tuple<T&, T&, T&> v)
                    :
                    x(std::get<0>(v)),
                    y(std::get<1>(v)),
                    z(std::get<2>(v)) {}

            public:

                inline ::XXX_NAMESPACE::dataTypes::Vec<T_unqualified, 3>  operator-() const
                {
                    return ::XXX_NAMESPACE::dataTypes::Vec<T_unqualified, 3>(-x, -y, -z);
                }
                
            #define MACRO(OP, IN_T)                                                 \
                inline VecProxy& operator OP (IN_T<T, 3>& v)        \
                {                                                                   \
                    x OP v.x;                                                       \
                    y OP v.y;                                                       \
                    z OP v.z;                                                       \
                    return *this;                                                   \
                }                                                                   \
                                                                                    \
                inline VecProxy& operator OP (const IN_T<T, 3>& v)  \
                {                                                                   \
                    x OP v.x;                                                       \
                    y OP v.y;                                                       \
                    z OP v.z;                                                       \
                    return *this;                                                   \
                }                                                                   \
                                                                                    \
                template <typename X>                                               \
                inline VecProxy& operator OP (IN_T<X, 3>& v)        \
                {                                                                   \
                    x OP v.x;                                                       \
                    y OP v.y;                                                       \
                    z OP v.z;                                                       \
                    return *this;                                                   \
                }                                                                   \
                                                                                    \
                template <typename X>                                               \
                inline VecProxy& operator OP (const IN_T<X, 3>& v)  \
                {                                                                   \
                    x OP v.x;                                                       \
                    y OP v.y;                                                       \
                    z OP v.z;                                                       \
                    return *this;                                                   \
                }                                                                   \

                MACRO(=, ::XXX_NAMESPACE::dataTypes::Vec)
                MACRO(+=, ::XXX_NAMESPACE::dataTypes::Vec)
                MACRO(-=, ::XXX_NAMESPACE::dataTypes::Vec)
                MACRO(*=, ::XXX_NAMESPACE::dataTypes::Vec)
                MACRO(/=, ::XXX_NAMESPACE::dataTypes::Vec)

                MACRO(=, ::XXX_NAMESPACE::dataTypes::internal::VecProxy)
                MACRO(+=, ::XXX_NAMESPACE::dataTypes::internal::VecProxy)
                MACRO(-=, ::XXX_NAMESPACE::dataTypes::internal::VecProxy)
                MACRO(*=, ::XXX_NAMESPACE::dataTypes::internal::VecProxy)
                MACRO(/=, ::XXX_NAMESPACE::dataTypes::internal::VecProxy)

            #undef MACRO

            #define MACRO(OP)                                                       \
                template <typename X>                                               \
                inline VecProxy& operator OP (const X xyz)                         \
                {                                                                   \
                    x OP xyz;                                                       \
                    y OP xyz;                                                       \
                    z OP xyz;                                                       \
                    return *this;                                                   \
                }                                                                   \

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
                    return ::XXX_NAMESPACE::math::internal::Func<T>::sqrt(x * x + y * y + z * z);
                }
            };

            template <typename T>
            std::ostream& operator<<(std::ostream& os, const VecProxy<T, 1>& vp)
            {
                os << "(" << vp.x << ")";
                return os;
            }

            template <typename T>
            std::ostream& operator<<(std::ostream& os, const VecProxy<T, 2>& vp)
            {
                os << "(" << vp.x << "," << vp.y << ")";
                return os;
            }

            template <typename T>
            std::ostream& operator<<(std::ostream& os, const VecProxy<T, 3>& vp)
            {
                os << "(" << vp.x << "," << vp.y << "," << vp.z << ")";
                return os;
            }
        }
    }
}

#elif defined(New)

#include <type_traits>
#include <utility>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#include <auxiliary/Template.hpp>
#include <common/DataLayout.hpp>
#include <common/Memory.hpp>
#include <data_types/DataTypes.hpp>
#include <data_types/integer_sequence/IntegerSequence.hpp>
#include <data_types/vec/Vec.hpp>

namespace XXX_NAMESPACE
{
    namespace dataTypes
    {
        namespace internal
        {
            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            //
            // Forward declarations.
            //
            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            template <typename, SizeT, SizeT, ::XXX_NAMESPACE::memory::DataLayout>
            class Accessor;

            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            //! \brief A proxy data type for Vec<T, D>
            //!
            //! This data type is returned by accessor<T, D, Layout>::operator[] if D = 1 and SoA data layout.
            //! It holds references to component(s) x [,y [and z]], so that data access via,
            //! e.g. obj[ ]..[ ].x, is possible.
            //!
            //! \tparam T data type
            //! \tparam D dimension
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            template <typename ValueT, SizeT D>
            class VecProxy : public ::XXX_NAMESPACE::dataTypes::Vec<ValueT&, D>
            {
                using Base = ::XXX_NAMESPACE::dataTypes::Vec<ValueT&, D>;

                // Member types must be fundamental and non-void.
                static_assert(std::is_fundamental<ValueT>::value, "error: fundamental parameter types assumed.");
                static_assert(!std::is_void<ValueT>::value, "error: non-void parameter types assumed.");

                using TupleT = ::XXX_NAMESPACE::dataTypes::Builder<std::tuple, ValueT&, D>;

                // Friend declarations.
                template <typename, SizeT, SizeT, ::XXX_NAMESPACE::memory::DataLayout>
                friend class ::XXX_NAMESPACE::dataTypes::internal::Accessor;

            //private:
              public:
                //!
                //! \brief Constructor.
                //!
                //! This constructor unpacks the references in the tuple argument and forwards them to the base class constructor.
                //!
                //! \tparam I an `IndexSequence` used for unpacking the tuple argument
                //! \param tuple a tuple holding references to memory that is associated with the members of the base class
                //! \param unnamed used for template parameter deduction
                //!
                template <SizeT... I>
                HOST_VERSION CUDA_DEVICE_VERSION VecProxy(TupleT&& vec, ::XXX_NAMESPACE::dataTypes::IndexSequence<I...>) : Base(std::get<I>(vec)...)
                {
                }

                //!
                //! \brief Constructor.
                //!
                //! Create a `TupleProxy` from a tuple of reference to memory.
                //!
                //! \param tuple a tuple holding references to memory that is associated with the members of the base class
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                VecProxy(TupleT vec) : VecProxy(std::move(vec), ::XXX_NAMESPACE::dataTypes::MakeIndexSequence<D>()) {}


            public:
                using ConstT = const VecProxy<const ValueT, D>;
                using BasePointer = ::XXX_NAMESPACE::memory::PointerN<ValueT, D>;
                //!
                //! \brief Assignment operator.
                //!
                //! The assignment operators defined in the base class (if there are any) are non-virtual.
                //! We thus need to add a version that returns a reference to this `TupleProxy` type.
                //!
                //! Note: the base class assignment operators can handle `TupleProxy` types as arguments due to inheritence.
                //! Note: we use the base class type as argument type as it covers both the `Tuple` and the `TupleProxy` case.
                //!
                //! \tparam T a variadic list of type parameters
                //! \param tuple a `Tuple` (or `TupleProxy`) instance
                //! \return a reference to this `TupleProxy` instance
                //!
                template <typename T>
                HOST_VERSION CUDA_DEVICE_VERSION inline auto operator=(const ::XXX_NAMESPACE::dataTypes::Vec<T, D>& vec) -> VecProxy&
                {
                    static_assert(std::is_convertible<T, ValueT>::value, "error: types are not convertible.");

                    ::XXX_NAMESPACE::compileTime::Loop<D>::Execute([&vec, this](const auto I) { Get<I>(*this) = Get<I>(vec); });

                    return *this;
                }

                //!
                //! \brief Assignment operator.
                //!
                //! The assignment operators defined in the base class (if there are any) are non-virtual.
                //! We thus need to add a version that returns a reference to this `TupleProxy` type.
                //!
                //! Note: the base class assignment operators can handle `TupleProxy` types as arguments due to inheritence.
                //!
                //! \tparam T the type of the value to be assigned
                //! \param value the value to be assigned
                //! \return a reference to this `TupleProxy` instance
                //!
                template <typename T, typename EnableType = std::enable_if_t<std::is_fundamental<T>::value>>
                HOST_VERSION CUDA_DEVICE_VERSION inline auto operator=(const T value) -> VecProxy&
                {
                    static_assert(std::is_convertible<T, ValueT>::value, "error: types are not convertible.");

                    ::XXX_NAMESPACE::compileTime::Loop<D>::Execute([value, this](const auto I) { Get<I>(*this) = value; });

                    return *this;
                }
            };
        }
    }
}

#else

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#include <common/Memory.hpp>
#include <common/DataLayout.hpp>
#include <data_types/DataTypes.hpp>

// some forward declarations
namespace XXX_NAMESPACE
{
    namespace internal 
    {
        template <typename P, typename R>
        class iterator;
    }
}

namespace XXX_NAMESPACE
{
    namespace dataTypes
    {
        namespace internal 
        {
            template <typename X, SizeT N, SizeT D, ::XXX_NAMESPACE::memory::DataLayout L>
            class Accessor;
        }
    }
}


namespace XXX_NAMESPACE
{
    namespace dataTypes
    {
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //! \brief A proxy data type for Vec<T, D>
        //!
        //! This data type is returned by accessor<T, D, Layout>::operator[] if D = 1 and SoA data layout.
        //! It holds references to component(s) x [,y [and z]], so that data access via,
        //! e.g. obj[ ]..[ ].x, is possible.
        //!
        //! \tparam T data type
        //! \tparam D dimension
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        namespace internal
        {
            //! \brief D = 1 specialization with component x
            //!
            //! \tparam T data type
            template <typename T>
            class VecProxy<T, 1>
            {
                static_assert(std::is_fundamental<T>::value, "error: T is not a fundamental data type");
                static_assert(!std::is_void<T>::value, "error: T is void -> not allowed");
                static_assert(!std::is_volatile<T>::value, "error: T is volatile -> not allowed");

                template <typename, SizeT, SizeT, ::XXX_NAMESPACE::memory::DataLayout>
                friend class XXX_NAMESPACE::dataTypes::internal::Accessor;

            public:

                using type = VecProxy<T, 1>;
                using ConstT = const VecProxy<const T, 1>;
                using T_unqualified = typename std::remove_cv<T>::type;
                //! Remember the template type parameter T
                using value_type = T;
                //! Remember the template parameter D (=1)
                static constexpr SizeT d = 1;
                using BasePointer = ::XXX_NAMESPACE::memory::PointerN<T, 1>;
                using original_type = typename std::conditional<std::is_const<T>::value, 
                    const Vec<typename std::remove_cv<T>::type, 1>,
                    Vec<typename std::remove_cv<T>::type, 1>>::type;

                T& x;

            private:

                HOST_VERSION
                CUDA_DEVICE_VERSION
                VecProxy(std::tuple<T&> v)
                    :
                    x(std::get<0>(v)) {}    

            public:

                inline ::XXX_NAMESPACE::dataTypes::Vec<T_unqualified, 1>  operator-() const
                {
                    return ::XXX_NAMESPACE::dataTypes::Vec<T_unqualified, 1>(-x);
                }

            #define MACRO(OP, IN_T)                                                 \
                inline VecProxy& operator OP (IN_T<T, 1>& v)        \
                {                                                                   \
                    x OP v.x;                                                       \
                    return *this;                                                   \
                }                                                                   \
                                                                                    \
                inline VecProxy& operator OP (const IN_T<T, 1>& v)  \
                {                                                                   \
                    x OP v.x;                                                       \
                    return *this;                                                   \
                }                                                                   \
                                                                                    \
                template <typename X>                                               \
                inline VecProxy& operator OP (IN_T<X, 1>& v)        \
                {                                                                   \
                    x OP v.x;                                                       \
                    return *this;                                                   \
                }                                                                   \
                                                                                    \
                template <typename X>                                               \
                inline VecProxy& operator OP (const IN_T<X, 1>& v)  \
                {                                                                   \
                    x OP v.x;                                                       \
                    return *this;                                                   \
                }                                                                   \

                MACRO(=, ::XXX_NAMESPACE::dataTypes::Vec)
                MACRO(+=, ::XXX_NAMESPACE::dataTypes::Vec)
                MACRO(-=, ::XXX_NAMESPACE::dataTypes::Vec)
                MACRO(*=, ::XXX_NAMESPACE::dataTypes::Vec)
                MACRO(/=, ::XXX_NAMESPACE::dataTypes::Vec)

                MACRO(=, ::XXX_NAMESPACE::dataTypes::internal::VecProxy)
                MACRO(+=, ::XXX_NAMESPACE::dataTypes::internal::VecProxy)
                MACRO(-=, ::XXX_NAMESPACE::dataTypes::internal::VecProxy)
                MACRO(*=, ::XXX_NAMESPACE::dataTypes::internal::VecProxy)
                MACRO(/=, ::XXX_NAMESPACE::dataTypes::internal::VecProxy)

            #undef MACRO

            #define MACRO(OP)                                                       \
                template <typename X>                                               \
                inline VecProxy& operator OP (const X x)                           \
                {                                                                   \
                    x OP x;                                                         \
                    return *this;                                                   \
                }                                                                   \

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
            class VecProxy<T, 2>
            {
                static_assert(std::is_fundamental<T>::value, "error: T is not a fundamental data type");
                static_assert(!std::is_void<T>::value, "error: T is void -> not allowed");
                static_assert(!std::is_volatile<T>::value, "error: T is volatile -> not allowed");

                template <typename, SizeT, SizeT, ::XXX_NAMESPACE::memory::DataLayout>
                friend class XXX_NAMESPACE::dataTypes::internal::Accessor;

            public:

                using type = VecProxy<T, 2>;
                using ConstT = const VecProxy<const T, 2>;
                using T_unqualified = typename std::remove_cv<T>::type;
                //! Remember the template type parameter T
                using value_type = T;
                //! Remember the template parameter D (=2)
                static constexpr SizeT d = 2;
                using BasePointer = ::XXX_NAMESPACE::memory::PointerN<T, 2>;
                using original_type = typename std::conditional<std::is_const<T>::value, 
                    const Vec<typename std::remove_cv<T>::type, 2>,
                    Vec<typename std::remove_cv<T>::type, 2>>::type;

                T& x;
                T& y;

            private:

                HOST_VERSION
                CUDA_DEVICE_VERSION
                VecProxy(std::tuple<T&, T&> v)
                    :
                    x(std::get<0>(v)),
                    y(std::get<1>(v)) {}

            public:

                inline ::XXX_NAMESPACE::dataTypes::Vec<T_unqualified, 2>  operator-() const
                {
                    return ::XXX_NAMESPACE::dataTypes::Vec<T_unqualified, 2>(-x, -y);
                }

            #define MACRO(OP, IN_T)                                                 \
                inline VecProxy& operator OP (IN_T<T, 2>& v)        \
                {                                                                   \
                    x OP v.x;                                                       \
                    y OP v.y;                                                       \
                    return *this;                                                   \
                }                                                                   \
                                                                                    \
                inline VecProxy& operator OP (const IN_T<T, 2>& v)  \
                {                                                                   \
                    x OP v.x;                                                       \
                    y OP v.y;                                                       \
                    return *this;                                                   \
                }                                                                   \
                                                                                    \
                template <typename X>                                               \
                inline VecProxy& operator OP (IN_T<X, 2>& v)        \
                {                                                                   \
                    x OP v.x;                                                       \
                    y OP v.y;                                                       \
                    return *this;                                                   \
                }                                                                   \
                                                                                    \
                template <typename X>                                               \
                inline VecProxy& operator OP (const IN_T<X, 2>& v)  \
                {                                                                   \
                    x OP v.x;                                                       \
                    y OP v.y;                                                       \
                    return *this;                                                   \
                }                                                                   \

                MACRO(=, ::XXX_NAMESPACE::dataTypes::Vec)
                MACRO(+=, ::XXX_NAMESPACE::dataTypes::Vec)
                MACRO(-=, ::XXX_NAMESPACE::dataTypes::Vec)
                MACRO(*=, ::XXX_NAMESPACE::dataTypes::Vec)
                MACRO(/=, ::XXX_NAMESPACE::dataTypes::Vec)

                MACRO(=, ::XXX_NAMESPACE::dataTypes::internal::VecProxy)
                MACRO(+=, ::XXX_NAMESPACE::dataTypes::internal::VecProxy)
                MACRO(-=, ::XXX_NAMESPACE::dataTypes::internal::VecProxy)
                MACRO(*=, ::XXX_NAMESPACE::dataTypes::internal::VecProxy)
                MACRO(/=, ::XXX_NAMESPACE::dataTypes::internal::VecProxy)

            #undef MACRO

            #define MACRO(OP)                                                       \
                template <typename X>                                               \
                inline VecProxy& operator OP (const X xy)                          \
                {                                                                   \
                    x OP xy;                                                        \
                    y OP xy;                                                        \
                    return *this;                                                   \
                }                                                                   \

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
                    return ::XXX_NAMESPACE::math::internal::Func<T>::sqrt(x * x + y * y);
                }
            };

            //! \brief D = 3 specialization with component x
            //!
            //! \tparam T data type
            template <typename T>
            class VecProxy<T, 3> : public ::XXX_NAMESPACE::dataTypes::Vec<T&, 3>
            {
                using Base = ::XXX_NAMESPACE::dataTypes::Vec<T&, 3>;

                /*
                static_assert(std::is_fundamental<T>::value, "error: T is not a fundamental data type");
                static_assert(!std::is_void<T>::value, "error: T is void -> not allowed");
                static_assert(!std::is_volatile<T>::value, "error: T is volatile -> not allowed");
*/
                template <typename, SizeT, SizeT, ::XXX_NAMESPACE::memory::DataLayout>
                friend class XXX_NAMESPACE::dataTypes::internal::Accessor;

            public:

                using type = VecProxy<T, 3>;
                using ConstT = const VecProxy<const T, 3>;
                using T_unqualified = typename std::remove_cv<T>::type;
                //! Remember the template type parameter T
                using value_type = T;
                //! Remember the template parameter D (=3)
                static constexpr SizeT d = 3;
                using BasePointer = ::XXX_NAMESPACE::memory::PointerN<T, 3>;
                using original_type = typename std::conditional<std::is_const<T>::value, 
                    const Vec<typename std::remove_cv<T>::type, 3>,
                    Vec<typename std::remove_cv<T>::type, 3>>::type;
/*
                T& x;
                T& y;
                T& z;
*/
                using Base::x;
                using Base::y;
                using Base::z;

            private:

                HOST_VERSION
                CUDA_DEVICE_VERSION
                VecProxy(std::tuple<T&, T&, T&> v)
                    :
                    /*
                    x(std::get<0>(v)),
                    y(std::get<1>(v)),
                    z(std::get<2>(v)) {}
                    */
                    Base(std::get<0>(v), std::get<1>(v), std::get<2>(v)) {}

            public:

                inline ::XXX_NAMESPACE::dataTypes::Vec<T_unqualified, 3>  operator-() const
                {
                    return ::XXX_NAMESPACE::dataTypes::Vec<T_unqualified, 3>(-x, -y, -z);
                }
                
            #define MACRO(OP, IN_T)                                                 \
                inline VecProxy& operator OP (IN_T<T, 3>& v)        \
                {                                                                   \
                    x OP v.x;                                                       \
                    y OP v.y;                                                       \
                    z OP v.z;                                                       \
                    return *this;                                                   \
                }                                                                   \
                                                                                    \
                inline VecProxy& operator OP (const IN_T<T, 3>& v)  \
                {                                                                   \
                    x OP v.x;                                                       \
                    y OP v.y;                                                       \
                    z OP v.z;                                                       \
                    return *this;                                                   \
                }                                                                   \
                                                                                    \
                template <typename X>                                               \
                inline VecProxy& operator OP (IN_T<X, 3>& v)        \
                {                                                                   \
                    x OP v.x;                                                       \
                    y OP v.y;                                                       \
                    z OP v.z;                                                       \
                    return *this;                                                   \
                }                                                                   \
                                                                                    \
                template <typename X>                                               \
                inline VecProxy& operator OP (const IN_T<X, 3>& v)  \
                {                                                                   \
                    x OP v.x;                                                       \
                    y OP v.y;                                                       \
                    z OP v.z;                                                       \
                    return *this;                                                   \
                }                                                                   \

                MACRO(=, ::XXX_NAMESPACE::dataTypes::Vec)
                MACRO(+=, ::XXX_NAMESPACE::dataTypes::Vec)
                MACRO(-=, ::XXX_NAMESPACE::dataTypes::Vec)
                MACRO(*=, ::XXX_NAMESPACE::dataTypes::Vec)
                MACRO(/=, ::XXX_NAMESPACE::dataTypes::Vec)

                MACRO(=, ::XXX_NAMESPACE::dataTypes::internal::VecProxy)
                MACRO(+=, ::XXX_NAMESPACE::dataTypes::internal::VecProxy)
                MACRO(-=, ::XXX_NAMESPACE::dataTypes::internal::VecProxy)
                MACRO(*=, ::XXX_NAMESPACE::dataTypes::internal::VecProxy)
                MACRO(/=, ::XXX_NAMESPACE::dataTypes::internal::VecProxy)

            #undef MACRO

            #define MACRO(OP)                                                       \
                template <typename X>                                               \
                inline VecProxy& operator OP (const X xyz)                         \
                {                                                                   \
                    x OP xyz;                                                       \
                    y OP xyz;                                                       \
                    z OP xyz;                                                       \
                    return *this;                                                   \
                }                                                                   \

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
                    return ::XXX_NAMESPACE::math::internal::Func<T>::sqrt(x * x + y * y + z * z);
                }
            };

            template <typename T>
            std::ostream& operator<<(std::ostream& os, const VecProxy<T, 1>& vp)
            {
                os << "(" << vp.x << ")";
                return os;
            }

            template <typename T>
            std::ostream& operator<<(std::ostream& os, const VecProxy<T, 2>& vp)
            {
                os << "(" << vp.x << "," << vp.y << ")";
                return os;
            }

            template <typename T>
            std::ostream& operator<<(std::ostream& os, const VecProxy<T, 3>& vp)
            {
                os << "(" << vp.x << "," << vp.y << "," << vp.z << ")";
                return os;
            }
        }
    }
}

#endif

#endif

#endif