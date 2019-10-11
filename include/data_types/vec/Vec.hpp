// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(DATA_TYPES_VEC_VEC_HPP)
#define DATA_TYPES_VEC_VEC_HPP

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#include <data_types/DataTypes.hpp>
#include <data_types/tuple/Tuple.hpp>

namespace XXX_NAMESPACE
{
    namespace dataTypes
    {
        template <typename ValueT, SizeT N>
        using Vec = Builder<Tuple, ValueT, N>;
    }
}

#if defined(OLD)

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#include <auxiliary/Template.hpp>
#include <common/Math.hpp>
#include <data_types/DataTypes.hpp>
#include <data_types/tuple/Get.hpp>
#include <data_types/tuple/Record.hpp>
#include <data_types/integer_sequence/IntegerSequence.hpp>
#include <platform/Target.hpp>

namespace XXX_NAMESPACE
{
    namespace dataTypes
    {
        namespace internal
        {
            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            //!
            //! \brief A vec type (base class).
            //!
            //! Note: This definition has at least five parameters in the list.
            //!
            //! \tparam ValueT the type of the members
            //!
            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            template <typename ValueT, SizeT D>
            class VecBase
            {
                static constexpr SizeT Size = D;
                using Record = ::XXX_NAMESPACE::dataTypes::Builder<::XXX_NAMESPACE::dataTypes::internal::Record, ValueT, D>;

              protected:
                //!
                //! \brief Standard constructor.
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr VecBase() : data{} {};

                //!
                //! \brief Constructor.
                //!
                //! Assign the same `value` to all members.
                //!
                //! \tparam T the type of the value to be assigned (can be different from the member type, but must be convertible)
                //! \param value the value to be assigned to all members
                //!
                template <typename T, typename EnableType = std::enable_if_t<std::is_fundamental<T>::value>>
                HOST_VERSION CUDA_DEVICE_VERSION constexpr VecBase(T value) : data(value) {}
                
                //!
                //! \brief Constructor.
                //!
                //! Assign some `values` to the members.
                //!
                //! \tparam T the types of the values to be assigned
                //! \param values the values to be assigned to the members
                //!
                template <typename ...T>
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr VecBase(T... values) : data(static_cast<ValueT>(values)...)
                {
                    static_assert(sizeof...(T) < D, "error: parameter list does not match the size of this array.");
                    static_assert(::XXX_NAMESPACE::variadic::Pack<T...>::template IsConvertibleTo<ValueT>(), "error: types are not convertible.");
                }

                //!
                //! \brief Copy constructor.
                //!
                //! \param vec another `VecBase` instance
                //!
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr VecBase(const VecBase& vec) : data(vec.data) {}

              public:
                Record data;
            };

            //!
            //! \brief A vec type (base class).
            //!
            //! Note: This definition has three parameters in the list.
            //! The `Record` member overlaps with members `x,y,z`.
            //!
            //! \tparam ValueT type of the members
            //!
            template <typename ValueT>
            class VecBase<ValueT, 3>
            {
                static constexpr SizeT Size = 3;
                using Record = ::XXX_NAMESPACE::dataTypes::Builder<::XXX_NAMESPACE::dataTypes::internal::Record, ValueT, 3>;

              protected:
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr VecBase() : data{} {}

                template <typename T, typename EnableType = std::enable_if_t<std::is_fundamental<T>::value>>
                HOST_VERSION CUDA_DEVICE_VERSION constexpr VecBase(T value) : data(static_cast<ValueT>(value))
                {
                    static_assert(std::is_convertible<T, ValueT>::value, "error: types are not convertible.");
                }

                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr VecBase(ValueT x, ValueT y, ValueT z) : data(x, y, z) {}

                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr VecBase(const VecBase& vec) : data(vec.data) {}

              public:
                union {
                    struct
                    {
                        ValueT x;
                        ValueT y;
                        ValueT z;
                    };
                    Record data;
                };
            };

            //!
            //! \brief Definition of a vec type with no members (base class).
            //!
            template <typename ValueT>
            class VecBase<ValueT, 0>
            {
                HOST_VERSION
                CUDA_DEVICE_VERSION
                constexpr VecBase() {}
            };
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        //
        // Forward declarations.
        //
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        namespace internal
        {
            template <typename, SizeT>
            class VecProxy;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //!
        //! \brief A vec type with (implementation).
        //!
        //! The implementation comprises simple arithmetic functions.
        //!
        //! \tparam ValueT a type parameter list defining the member types
        //!
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        template <typename ValueT, SizeT D>
        class Vec : public internal::VecBase<ValueT, D>
        {
            using Base = internal::VecBase<ValueT, D>;

            template <typename ...T>
            using Pack = ::XXX_NAMESPACE::variadic::Pack<T...>;
            template <SizeT N>
            using CompileTimeLoop = ::XXX_NAMESPACE::compileTime::Loop<N>;
            
        public:
            using Type = Vec<ValueT, D>;
            using Proxy = typename ::XXX_NAMESPACE::dataTypes::internal::VecProxy<ValueT, D>;

            //!
            //! \brief Standard constructor.
            //!
            HOST_VERSION
            CUDA_DEVICE_VERSION
            constexpr Vec() = default;

            //!
            //! \brief Constructor.
            //!
            //! Assign the same `value` to all members.
            //!
            //! \tparam T the type of the value to be assigned (can be different from the member type, but must be convertible)
            //! \param value the value to be assigned to all members
            //!
            template <typename T, typename EnableType = std::enable_if_t<std::is_fundamental<T>::value>>
            HOST_VERSION CUDA_DEVICE_VERSION constexpr Vec(T value) : Base(value)
            {
            }

            //!
            //! \brief Constructor.
            //!
            //! Assign some `values` to the members.
            //!
            //! \tparam T the types of the values to be assigned 
            //! \param values the values to be assigned to the members
            //!
            /*
            template <typename ...T>
            HOST_VERSION
            CUDA_DEVICE_VERSION
            constexpr Vec(T... values) : Base(values...)
            {
            }
            */
            HOST_VERSION
            CUDA_DEVICE_VERSION
            constexpr Vec(T... values) : Base(values...)
            //!
            //! \brief Copy constructor.
            //!
            //! \param tuple another `Tuple` instance
            //!
            HOST_VERSION
            CUDA_DEVICE_VERSION
            constexpr Vec(const Vec& vec) : Base(vec) {}
        
        private:
            //!
            //! \brief Constructor.
            //!
            //! \tparam T the decay type of the proxy members (must be convertible to `ValueT`)
            //! \tparam I a list of indices used for the parameter access
            //! \param proxy a `VecProxy` instance
            //! \param unnamed used for template parameter deduction
            //!
            template <typename T, SizeT... I>
            HOST_VERSION CUDA_DEVICE_VERSION Vec(const internal::VecProxy<T, D>& proxy, ::XXX_NAMESPACE::dataTypes::IndexSequence<I...>) : Vec(Get<I>(proxy)...)
            {
            }

          public:
            //!
            //! \brief Constructor.
            //!
            //! Construct a `Vec` instance from a `VecProxy` with a different member type (must be convertible).
            //!
            //! \tparam T the decay type of proxy members (must be convertible to `ValueT`)
            //! \param proxy a `VecProxy` instance
            //!
            template <typename T, bool IsReference = std::is_reference<ValueT>::value, typename Enable = std::enable_if_t<!IsReference>>
            HOST_VERSION CUDA_DEVICE_VERSION constexpr Vec(const internal::VecProxy<T, D>& proxy) : Vec(proxy, ::XXX_NAMESPACE::dataTypes::MakeIndexSequence<D>())
            {
            }

            //!
            //! \brief Constructor (Invalid object construction).
            //!
            //! This constructor is enabled in case of a base class conversion of a `VecProxy` where in the calling context a non-reference
            //! instance of the base class is needed, e.g.
            //! ```
            //! template <typename T, SizeT D>
            //! void foo(Vec<T, D> vec) {..}
            //! ..
            //! VecProxy<float, 4> proxy = ..;
            //! foo(proxy); // ERROR
            //! ```
            //! In this case the `VecProxy` is converted to its base class `Vec<float&, 4>`, and references to the members are given to the `foo`.
            //! This is errorneous behavior. 
            //! Instead the programmer must make an explicit copy of the `VecProxy` type to `Vec`, which would happen anyway when calling `foo`,
            //! or use a static cast to `Vec<float, 4>`
            //!
            //! \tparam T the decay type of proxy members (must be convertible to `ValueT`)
            //! \param proxy a `VecProxy` instance
            //!
            template <typename T, bool IsReference = std::is_reference<ValueT>::value, typename Enable = std::enable_if_t<IsReference>>
            HOST_VERSION CUDA_DEVICE_VERSION Vec(internal::VecProxy<T, D> proxy) : Vec(proxy, ::XXX_NAMESPACE::dataTypes::MakeIndexSequence<D>())
            {
                static_assert(!IsReference, "error: you're trying to copy a proxy where an explicit copy to the original type is needed -> make an explicit copy or use a static cast!");
            }

            //!
            //! \brief Sign operator.
            //!
            //! Change the sign of all members.
            //!
            //! \return a new `Vec` with sign-changed members
            //!
            HOST_VERSION
            CUDA_DEVICE_VERSION
            inline auto operator-() const
            {
                Vec vec;

                CompileTimeLoop<D>::Execute([&vec, this](const auto I) { Get<I>(vec) = -Get<I>(*this); });

                return vec;
            }

            //!
            //! \brief Some operator definitions: assignment and arithmetic.
            //!
            //! \tparam T the member (decay) type of the `Vec` or `VecProxy` type (must be convertible to `ValueT`)
            //! \param tuple a `Vec` or `VecProxy` instance with the same number of members
            //! \return a reference to this `Tuple` after having applied the operation
            //!
#define MACRO(OP, IN_T) \
    template <typename T> \
    HOST_VERSION CUDA_DEVICE_VERSION inline auto operator OP(const IN_T<T, D>& vec)->Vec& \
    { \
        static_assert(std::is_convertible<T, ValueT>::value, "error: types are not convertible."); \
    \
        CompileTimeLoop<D>::Execute([&vec, this](const auto I) { Get<I>(*this) OP Get<I>(vec); }); \
    \
        return *this; \
    } \
            
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

            //!
            //! \brief Some operator definitions: assignment and arithmetic.
            //!
            //! \tparam T the type of the value
            //! \param value the value used for the operation
            //! \return a reference to this `Vec` after having applied the operation
            //!
#define MACRO(OP)                                                                                                                                                                                                          \
    template <typename T, typename EnableType = std::enable_if_t<std::is_fundamental<T>::value>>                                                                                                                           \
    HOST_VERSION CUDA_DEVICE_VERSION inline auto operator OP(T value)->Vec&                                                                                                                                              \
    {                                                                                                                                                                                                                      \
        CompileTimeLoop<D>::Execute([&value, this](const auto I) { Get<I>(*this) OP value; });                                                                                                             \
                                                                                                                                                                                                                           \
        return *this;                                                                                                                                                                                                      \
    }

            MACRO(=)
            MACRO(+=)
            MACRO(-=)
            MACRO(*=)
            MACRO(/=)

        #undef MACRO

            //!
            //! \brief Return the Euclidean norm of the vector.
            //!
            //! \return Euclidean norm
            //!
            inline auto Length() const
            {
                ValueT length{};

                CompileTimeLoop<D>::Execute([&length, this](const auto I) { length += Get<I>(*this) * Get<I>(*this); });
                
                return length;
            }
        };

        //!
        //! \brief Definition of a `Vec` type with no members (implementation).
        //!
        template <typename ValueT>
        class Vec<ValueT, 0>
        {
          public:
            HOST_VERSION
            CUDA_DEVICE_VERSION
            constexpr Vec(){};
        };

        template <typename ValueT, SizeT D>
        std::ostream& operator<<(std::ostream& os, const Vec<ValueT, D>& vec)
        {
            os << "( ";

            ::XXX_NAMESPACE::compileTime::Loop<D>::Execute([&vec, &os](const auto I) { os << Get<I>(vec) << " "; });

            os << ")";

            return os;
        }
    }
}

#include <data_types/vec/VecProxy.hpp>
#include <data_types/vec/VecMath.hpp>
#include <common/Traits.hpp>

namespace XXX_NAMESPACE
{
    namespace internal
    {
        //!
        //! \brief Specialization of the `ProvidesProxy` type for `Vec`s.
        //!
        template <typename T, SizeT D>
        struct ProvidesProxy<::XXX_NAMESPACE::dataTypes::Vec<T, D>>
        {
            static constexpr bool value = true;
        };

        template <typename T, SizeT D>
        struct ProvidesProxy<const ::XXX_NAMESPACE::dataTypes::Vec<T, D>>
        {
            static constexpr bool value = true;
        };
    }
}

#endif

#endif
