
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_CORE_BYTE_SWAP_H_INCLUDED
#define SCM_CORE_BYTE_SWAP_H_INCLUDED

#include <cassert>

#include <boost/static_assert.hpp>

#include <scm/core/numeric_types.h>
#include <scm/core/platform/platform.h>

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS
#   if _MSC_VER >= 1400
#   include <stdlib.h>
#   endif // 
#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS

namespace scm {

inline
void
swap_bytes_1(void* d, void* s)
{
    assert(s != 0);
    assert(d != 0);

    uint8*const s8 = reinterpret_cast<uint8*>(s);
    uint8*const d8 = reinterpret_cast<uint8*>(d);

    *d8 = *s8;
}

inline
void
swap_bytes_2(void* d, void* s)
{
    assert(s != 0);
    assert(d != 0);

    uint16*const s16 = reinterpret_cast<uint16*>(s);
    uint16*const d16 = reinterpret_cast<uint16*>(d);

#if    (SCM_PLATFORM == SCM_PLATFORM_WINDOWS) \
    && (_MSC_VER >= 1400)
    *d16 = _byteswap_ushort(*s16);
#else
    *d16 =   (*s16 >> 8)
           | (*s16 << 8);
#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS
}

inline
void
swap_bytes_4(void* d, void* s)
{
    assert(s != 0);
    assert(d != 0);

    uint32*const s32 = reinterpret_cast<uint32*>(s);
    uint32*const d32 = reinterpret_cast<uint32*>(d);

#if    (SCM_PLATFORM == SCM_PLATFORM_WINDOWS) \
    && (_MSC_VER >= 1400)
    *d32 = _byteswap_ulong(*s32);
#else
    *d32 =   ((*s32 & 0xff000000) >> 24)
           | ((*s32 & 0x00ff0000) >> 8)
           | ((*s32 & 0x0000ff00) << 8)
           | ((*s32 & 0x000000ff) << 24);
#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS
}

inline
void
swap_bytes_8(void* d, void* s)
{
    assert(s != 0);
    assert(d != 0);

    uint64*const s64 = reinterpret_cast<uint64*>(s);
    uint64*const d64 = reinterpret_cast<uint64*>(d);

#if    (SCM_PLATFORM == SCM_PLATFORM_WINDOWS) \
    && (_MSC_VER >= 1400)
    *d64 = _byteswap_uint64(*s64);
#else
    *d64 =   ((*s64 & 0xff00000000000000ull) >> 56)
           | ((*s64 & 0x00ff000000000000ull) >> 40)
           | ((*s64 & 0x0000ff0000000000ull) >> 24)
           | ((*s64 & 0x000000ff00000000ull) >> 8)
           | ((*s64 & 0x00000000ff000000ull) << 8)
           | ((*s64 & 0x0000000000ff0000ull) << 24)
           | ((*s64 & 0x000000000000ff00ull) << 40)
           | ((*s64 & 0x00000000000000ffull) << 56);
#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS
}

template<typename T, size_t st>
struct do_swap_bytes
{
    inline void operator()(T*) {
        throw std::out_of_range("do_swap_bytes(): there be dragons...");
    }
};

template<typename T>
struct do_swap_bytes<T, 1>
{
    inline void operator()(T* d, T* s) {
        swap_bytes_1(d, s);
    }
};

template<typename T>
struct do_swap_bytes<T, 2>
{
    inline void operator()(T* d, T* s) {
        swap_bytes_2(d, s);
    }
};

template<typename T>
struct do_swap_bytes<T, 4>
{
    inline void operator()(T* d, T* s) {
        swap_bytes_4(d, s);
    }
};

template<typename T>
struct do_swap_bytes<T, 8>
{
    inline void operator()(T* d, T* s) {
        swap_bytes_8(d, s);
    }
};

template<typename T>
inline void
swap_bytes(T* d)
{
    BOOST_STATIC_ASSERT(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8);
    
    do_swap_bytes<T, sizeof(T)>()(d, d);
}

template<typename T>
inline void
swap_bytes_array(T* d, scm::size_t c)
{
    BOOST_STATIC_ASSERT(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8);
    
    for (scm::size_t i = 0; i < c; ++i) {
        do_swap_bytes<T, sizeof(T)>()(d + i, d + i);
    }
}

template<typename T>
inline void
swap_bytes_array(T* d, T* s, scm::size_t c)
{
    BOOST_STATIC_ASSERT(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8);
    
    for (scm::size_t i = 0; i < c; ++i) {
        do_swap_bytes<T, sizeof(T)>()(d + i, s + i);
    }
}

} // namespace scm

#endif // SCM_CORE_BYTE_SWAP_H_INCLUDED
