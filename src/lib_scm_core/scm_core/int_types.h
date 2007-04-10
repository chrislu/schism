
#ifndef INT_TYPES_H_INCLUDED
#define INT_TYPES_H_INCLUDED

#include <boost/cstdint.hpp>

namespace scm
{
    namespace core
    {
        // 8bit integer types
        using boost::int8_t;             
        using boost::uint8_t;            
                             
        // 16bit integer types
        using boost::int16_t;            
        using boost::uint16_t;           
                             
        // 32bit integer types
        using boost::int32_t;            
        using boost::uint32_t;           

        // 64bit integer types
        using boost::int64_t;
        using boost::uint64_t;

    } // core

} // scm

#endif // INT_TYPES_H_INCLUDED
