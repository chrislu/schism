
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "data_types.h"

#include <cassert>

#include <boost/static_assert.hpp>

namespace  {

const char* type_strings[] = {
    "UNKNOWN",

    "FLOAT", "VEC2F", "VEC3F", "VEC4F",
    "MAT2F", "MAT3F", "MAT4F",
    "MAT2X3F", "MAT2X4F", "MAT3X2F", "MAT3X4F", "MAT4X2F", "MAT4X3F",

#if SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_400
    "DOUBLE", "VEC2D", "VEC3D", "VEC4D",
    "MAT2D", "MAT3D", "MAT4D",
    "MAT2X3D", "MAT2X4D", "MAT3X2D", "MAT3X4D", "MAT4X2D", "MAT4X3D",
#endif // SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_400

    "INT", "VEC2I", "VEC3I", "VEC4I",

    "UINT", "VEC2UI", "VEC3UI", "VEC4UI",

    "BOOL", "VEC2B", "VEC3B", "VEC4B",

    "SHORT",
    "USHORT",
    "BYTE",
    "UBYTE",

    "SAMPLER",
    "IMAGE"
};

} // namespace 

namespace scm {
namespace gl {

const char*
type_string(data_type d)
{
    BOOST_STATIC_ASSERT((sizeof(type_strings) / sizeof(const char*)) == TYPE_COUNT);
    assert(TYPE_UNKNOWN <= d && d < TYPE_COUNT);
    return (type_strings[d]);
}

} // namespace gl
} // namespace scm
