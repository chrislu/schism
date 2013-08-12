
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef INT_TYPES_H_INCLUDED
#define INT_TYPES_H_INCLUDED

#include <cstddef>
#include <boost/cstdint.hpp>

namespace scm {

// 8bit integer types
typedef boost::int8_t       int8;
typedef boost::uint8_t      uint8;            
                     
// 16bit integer types
typedef boost::int16_t      int16;
typedef boost::uint16_t     uint16;

// 32bit integer types
typedef boost::int32_t      int32;
typedef boost::uint32_t     uint32;

// 64bit integer types
typedef boost::int64_t      int64;
typedef boost::uint64_t     uint64;

typedef boost::intmax_t     intmax;
typedef boost::uintmax_t    uintmax;

typedef std::size_t         size_t;

// floating point types
typedef float               float32;
typedef double              float64;

} // namespace scm


namespace scm {

inline
size_t
round_to_multiple(const size_t p, const size_t a) {
    const size_t r  = p % a;
    //return ((p8 + a - 1) / a) * a;
    return p + (r == 0 ? 0 : a - r);
}

} // namespace scm

#endif // INT_TYPES_H_INCLUDED
