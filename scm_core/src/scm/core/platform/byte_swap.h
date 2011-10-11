
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
swap_bytes_1(void* d)
{
    assert(d != 0);
}

inline
void
swap_bytes_2(void* d)
{
    assert(d != 0);

    uint16*const d16 = reinterpret_cast<uint16*>(d);

#if    (SCM_PLATFORM == SCM_PLATFORM_WINDOWS) \
    && (_MSC_VER >= 1400)
    *d16 = _byteswap_ushort(*d16);
#else
    *d16 =   (*d16 >> 8)
           | (*d16 << 8);
#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS
}

inline
void
swap_bytes_4(void* d)
{
    assert(d != 0);

    uint32*const d32 = reinterpret_cast<uint32*>(d);

#if    (SCM_PLATFORM == SCM_PLATFORM_WINDOWS) \
    && (_MSC_VER >= 1400)
    *d32 = _byteswap_ulong(*d32);
#else
    *d32 =   ((*d32 & 0xff000000) >> 24)
           | ((*d32 & 0x00ff0000) >> 8)
           | ((*d32 & 0x0000ff00) << 8)
           | ((*d32 & 0x000000ff) << 24);
#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS
}

inline
void
swap_bytes_8(void* d)
{
    assert(d != 0);

    uint64*const d64 = reinterpret_cast<uint64*>(d);

#if    (SCM_PLATFORM == SCM_PLATFORM_WINDOWS) \
    && (_MSC_VER >= 1400)
    *d64 = _byteswap_uint64(*d64);
#else
    *d64 =   ((*d64 & 0xff00000000000000ull) >> 56)
           | ((*d64 & 0x00ff000000000000ull) >> 40)
           | ((*d64 & 0x0000ff0000000000ull) >> 24)
           | ((*d64 & 0x000000ff00000000ull) >> 8);
           | ((*d64 & 0x00000000ff000000ull) << 8);
           | ((*d64 & 0x0000000000ff0000ull) << 24);
           | ((*d64 & 0x000000000000ff00ull) << 40);
           | ((*d64 & 0x00000000000000ffull) << 56);
#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS
}

template<typename T, size_t st>
struct do_swap_bytes
{
    inline void operator()(T*) {
        throw std::out_of_range("there be dragons...");
    }
};

template<typename T>
struct do_swap_bytes<T, 1>
{
    inline void operator()(T* d) {
        swap_bytes_1(d);
    }
};

template<typename T>
struct do_swap_bytes<T, 2>
{
    inline void operator()(T* d) {
        swap_bytes_2(d);
    }
};

template<typename T>
struct do_swap_bytes<T, 4>
{
    inline void operator()(T* d) {
        swap_bytes_4(d);
    }
};

template<typename T>
struct do_swap_bytes<T, 8>
{
    inline void operator()(T* d) {
        swap_bytes_8(d);
    }
};

template<typename T>
inline void
swap_bytes(T* d)
{
    BOOST_STATIC_ASSERT(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8);
    
    do_swap_bytes<T, sizeof(T)>()(d);
}

} // namespace scm

#endif // SCM_CORE_BYTE_SWAP_H_INCLUDED
