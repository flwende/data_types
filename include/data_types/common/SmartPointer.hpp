// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(COMMON_SMARTPOINTER_HPP)
#define COMMON_SMARTPOINTER_HPP

#include <type_traits>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

#include <common/Memory.hpp>
#include <DataTypes.hpp>

namespace XXX_NAMESPACE
{
    namespace memory
    {
        using namespace dataTypes;

        template <typename T, typename Deleter = void>
        class SmartPointer;

        template <typename T, typename Deleter>
        SmartPointer<T, Deleter> operator+(SmartPointer<T, Deleter>& other, const SizeT offset)
        {
            SmartPointer<T, Deleter> pointer(other);
            pointer.offset += offset;

            return pointer;
        }

        template <typename T, typename Deleter>
        class SmartPointer
        {
            template <typename OtherT, typename OtherDeleter>
            friend SmartPointer<OtherT, OtherDeleter> operator+(SmartPointer<OtherT, OtherDeleter>&, const SizeT);
            
          public:
            SmartPointer() : pointer{}, counter{} {}

            SmartPointer(T* pointer, const SizeT offset = 0) : pointer(pointer), counter(new SizeT), offset(offset)
            {
                *counter = 1;
            }

            SmartPointer(const SmartPointer& other) : pointer(other.pointer), counter(other.counter), offset(other.offset)
            {
                if (counter)
                {
                    *counter += 1;
                }
            }

            SmartPointer(SmartPointer&& other) : pointer(other.pointer), counter(other.counter), offset(other.offset)
            {
                other.pointer = nullptr;
                other.counter = nullptr;
            }

            SmartPointer& operator=(const SmartPointer& other)
            {
                if (this != &other)
                {
                    pointer = other.pointer;
                    counter = other.counter;
                    offset = other.offset;

                    if (counter)
                    {
                        *counter += 1;
                    }
                }

                return *this;
            }

            SmartPointer& operator=(SmartPointer&& other)
            {
                if (this != &other)
                {
                    pointer = other.pointer;
                    counter = other.counter;
                    offset = other.offset;

                    other.pointer = nullptr;
                    other.counter = nullptr;
                }

                return *this;
            }

            ~SmartPointer()
            {
                if (counter)
                {
                    if (*counter == 1)
                    {
                        if (pointer)
                        {
                            DeletePointer(pointer);
                        }

                        delete counter;
                    }
                    else
                    {
                        *counter -= 1;
                    }
                }
            }

            auto* Get() { return pointer + offset; }

            const auto* Get() const { return pointer + offset; }

            auto& operator[](const SizeT index) 
            { 
                assert(pointer != nullptr);

                return pointer[offset + index]; 
            }

            const auto& operator[](const SizeT index) const
            {
                assert(pointer != nullptr);

                return pointer[offset + index]; 
            }

            auto operator->() { return pointer; }

            const auto operator->() const { return pointer; }

            auto& operator*() { return pointer[offset]; }

            const auto& operator*() const { return pointer[offset]; }

          protected:
            template <typename Enable = Deleter>
            auto DeletePointer(T* pointer) -> std::enable_if_t<std::is_void<Enable>::value, void>
            {
                delete pointer;
            }

            template <typename Enable = Deleter>
            auto DeletePointer(T* pointer) -> std::enable_if_t<!std::is_void<Enable>::value, void>
            {
                Deleter{}(pointer);
            }

            T* pointer;
            SizeT* counter;
            SizeT offset;
        };
    } // namespace memory
} // namespace XXX_NAMESPACE

#endif