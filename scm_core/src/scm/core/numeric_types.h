
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

} // namespace scm

#endif // INT_TYPES_H_INCLUDED
