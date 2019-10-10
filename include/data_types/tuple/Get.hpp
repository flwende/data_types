// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(DATA_TYPES_TUPLE_GET_HPP)
#define DATA_TYPES_TUPLE_GET_HPP

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#include <data_types/DataTypes.hpp>
#include <platform/Target.hpp>

namespace XXX_NAMESPACE
{
    namespace dataTypes
    {
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        //
        // Forward declarations.
        //
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        namespace internal
        {
            template <typename...>
            class ReverseRecord;

            template <typename...>
            class Record;
        } // namespace internal

        template <typename... ValueT>
        class Tuple;

        namespace internal
        {
            namespace
            {
                ////////////////////////////////////////////////////////////////////////////////////////////////////////
                //!
                //! \brief Definition of a getter function to access the members of a `ReverseRecord` type.
                //!
                //! Access to the members happens through type casting to the base class of the
                //! `ReverseRecord` type recursively.
                //!
                //! \tparam Index the index of the member
                //! \tparam ValueT a variadic parameter list
                //!
                ////////////////////////////////////////////////////////////////////////////////////////////////////////
                template <SizeT Index, typename... ValueT>
                struct GetImplementation;

                //!
                //! \brief Definition of a getter function (recursive).
                //!
                //! \tparam Index the index of the data element
                //! \tparam ValueT the type of the front most (current) member
                //! \tparam Tail a variadic parameter list corresponding to the parameters of the base class of the current `ReverseRecord` type
                //!
                template <SizeT Index, typename ValueT, typename... Tail>
                struct GetImplementation<Index, ValueT, Tail...>
                {
                    static_assert(Index > 0 && (Index < (1 + sizeof...(Tail))), "error: out of bounds.");

                    //!
                    //! \brief Get the value of the requested member (according to `Index`).
                    //!
                    //! \param tuple a reference to the considered `ReverseRecord` type
                    //! \return a reference to the requested member
                    //!
                    HOST_VERSION
                    CUDA_DEVICE_VERSION
                    static inline constexpr auto& Value(::XXX_NAMESPACE::dataTypes::internal::ReverseRecord<ValueT, Tail...>& tuple) { return GetImplementation<Index - 1, Tail...>::Value(tuple); }

                    //!
                    //! \brief Get the value of the requested member (according to `Index`).
                    //!
                    //! \param tuple a const reference to the considered `ReverseRecord` type
                    //! \return a const reference to the requested member
                    //!
                    HOST_VERSION
                    CUDA_DEVICE_VERSION
                    static inline constexpr const auto& Value(const ::XXX_NAMESPACE::dataTypes::internal::ReverseRecord<ValueT, Tail...>& tuple) { return GetImplementation<Index - 1, Tail...>::Value(tuple); }
                };

                //!
                //! \brief Definition of a getter function (recursion anchor).
                //!
                //! \tparam ValueT the type of the front most (requested) member
                //! \tparam Tail a variadic parameter list corresponding to the parameters of the base class of the current `ReverseRecord` type
                //!
                template <typename ValueT, typename... Tail>
                struct GetImplementation<0, ValueT, Tail...>
                {
                    //!
                    //! \brief Get the value of the requested member (according to `Index`).
                    //!
                    //! \param tuple a reference to the considered `ReverseRecord` type
                    //! \return a reference to the requested member
                    //!
                    HOST_VERSION
                    CUDA_DEVICE_VERSION
                    static inline constexpr auto& Value(::XXX_NAMESPACE::dataTypes::internal::ReverseRecord<ValueT, Tail...>& tuple) { return tuple.Get(); }

                    //!
                    //! \brief Get the value of the requested member (according to `Index`).
                    //!
                    //! \param tuple a reference to the considered `ReverseRecord` type
                    //! \return a reference to the requested member
                    //!
                    HOST_VERSION
                    CUDA_DEVICE_VERSION
                    static inline constexpr const auto& Value(const ::XXX_NAMESPACE::dataTypes::internal::ReverseRecord<ValueT, Tail...>& tuple) { return tuple.Get(); }
                };
            } // namespace

            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            //!
            //! \brief A Getter function to access the members of a `ReverseRecord` type.
            //!
            //! \tparam Index the index of the member
            //! \tparam ValueT a variadic parameter list
            //! \param tuple a reference to the considered `ReverseRecord` type
            //! \return a reference to the requested member
            //!
            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            template <SizeT Index, typename... ValueT>
            HOST_VERSION CUDA_DEVICE_VERSION constexpr inline auto& Get(internal::ReverseRecord<ValueT...>& tuple)
            {
                constexpr SizeT N = sizeof...(ValueT);

                static_assert(Index < N, "error: out of bounds.");

                return GetImplementation<Index, ValueT...>::Value(tuple);
            }

            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            //!
            //! \brief A Getter function to access the members of a `ReverseRecord` type.
            //!
            //! \tparam Index the index of the member
            //! \tparam ValueT a variadic parameter list
            //! \param tuple a const reference to the considered `ReverseRecord` type
            //! \return a const reference to the requested member
            //!
            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            template <SizeT Index, typename... ValueT>
            HOST_VERSION CUDA_DEVICE_VERSION constexpr inline const auto& Get(const internal::ReverseRecord<ValueT...>& tuple)
            {
                constexpr SizeT N = sizeof...(ValueT);

                static_assert(Index < N, "error: out of bounds.");

                return GetImplementation<Index, ValueT...>::Value(tuple);
            }

            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            //!
            //! \brief A Getter function to access the members of a `Record` type.
            //!
            //! \tparam Index the index of the member
            //! \tparam ValueT a variadic parameter list
            //! \param tuple a reference to the considered `Record` type
            //! \return a reference to the requested member
            //!
            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            template <SizeT Index, typename... ValueT>
            HOST_VERSION CUDA_DEVICE_VERSION constexpr inline auto& Get(internal::Record<ValueT...>& tuple)
            {
                constexpr SizeT N = sizeof...(ValueT);

                static_assert(N > 0 && Index < N, "error: out of bounds.");

                return Get<(N - 1) - Index>(static_cast<typename internal::Record<ValueT...>::Base&>(tuple));
            }

            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            //!
            //! \brief A Getter function to access the members of a `Record` type.
            //!
            //! \tparam Index the index of the member
            //! \tparam ValueT a variadic parameter list
            //! \param tuple a const reference to the considered `Record` type
            //! \return a const reference to the requested member
            //!
            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            template <SizeT Index, typename... ValueT>
            HOST_VERSION CUDA_DEVICE_VERSION constexpr inline const auto& Get(const internal::Record<ValueT...>& tuple)
            {
                constexpr SizeT N = sizeof...(ValueT);

                static_assert(N > 0 && Index < N, "error: out of bounds.");

                return Get<(N - 1) - Index>(static_cast<const typename internal::Record<ValueT...>::Base&>(tuple));
            }
        } // namespace internal

        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        //!
        //! \brief A Getter function to access the members of a `Tuple` type.
        //!
        //! \tparam Index the index of the member
        //! \tparam ValueT a variadic parameter list
        //! \param tuple a reference to the considered `Tuple` type
        //! \return a reference to the requested member
        //!
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        template <SizeT Index, typename... ValueT>
        static inline constexpr auto& Get(Tuple<ValueT...>& tuple)
        {
            return internal::Get<Index>(tuple.data);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        //!
        //! \brief A Getter function to access the members of a `Tuple` type.
        //!
        //! \tparam Index the index of the member
        //! \tparam ValueT a variadic parameter list
        //! \param tuple a const reference to the considered `Tuple` type
        //! \return a const reference to the requested member
        //!
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        template <SizeT Index, typename... ValueT>
        static inline constexpr const auto& Get(const Tuple<ValueT...>& tuple)
        {
            return internal::Get<Index>(tuple.data);
        }        
    } // namespace dataTypes
} // namespace XXX_NAMESPACE

////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Overload of `std::get` for compatibility.
//
////////////////////////////////////////////////////////////////////////////////////////////////////////
namespace std
{
    template <std::size_t Index, typename ...ValueT>
    static inline constexpr auto& get(::XXX_NAMESPACE::dataTypes::Tuple<ValueT...>& tuple)
    {
        return ::XXX_NAMESPACE::dataTypes::Get<Index>(tuple);
    }

    template <std::size_t Index, typename ...ValueT>
    static inline constexpr const auto& get(const ::XXX_NAMESPACE::dataTypes::Tuple<ValueT...>& tuple)
    {
        return ::XXX_NAMESPACE::dataTypes::Get<Index>(tuple);
    }
}

#endif