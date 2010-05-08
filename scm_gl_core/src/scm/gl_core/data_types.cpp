
#include "data_types.h"

#include <cassert>

namespace  {

const char* type_strings[] = {
    "UNKNOWN",

    "FLOAT", "VEC2F", "VEC3F", "VEC4F",
    "MAT2F", "MAT3F", "MAT4F",
    "MAT2X3F", "MAT2X4F", "MAT3X2F", "MAT3X4F", "MAT4X2F", "MAT4X3F",

    "INT", "VEC2I", "VEC3I", "VEC4I",

    "UINT", "VEC2UI", "VEC3UI", "VEC4UI",

    "BOOL", "VEC2B", "VEC3B", "VEC4B",

    "SHORT",
    "USHORT",
    "BYTE",
    "UBYTE"
};

} // namespace 

namespace scm {
namespace gl {

const char*
type_string(data_type d)
{
    assert(TYPE_UNKNOWN <= d && d < TYPE_COUNT);
    return (type_strings[d]);
}

} // namespace gl
} // namespace scm
