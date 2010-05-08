
#ifndef SCM_GL_CORE_DATA_TYPES_H_INCLUDED
#define SCM_GL_CORE_DATA_TYPES_H_INCLUDED

#include <scm/core/platform/platform.h>

namespace scm {
namespace gl {

enum data_type {
    TYPE_UNKNOWN            = 0x00u,

    TYPE_FLOAT,
    TYPE_VEC2F,
    TYPE_VEC3F,
    TYPE_VEC4F,

    TYPE_MAT2F,
    TYPE_MAT3F,
    TYPE_MAT4F,

    TYPE_MAT2X3F,
    TYPE_MAT2X4F,
    TYPE_MAT3X2F,
    TYPE_MAT3X4F,
    TYPE_MAT4X2F,
    TYPE_MAT4X3F,

    TYPE_INT,
    TYPE_VEC2I,
    TYPE_VEC3I,
    TYPE_VEC4I,

    TYPE_UINT,
    TYPE_VEC2UI,
    TYPE_VEC3UI,
    TYPE_VEC4UI,

    TYPE_BOOL,
    TYPE_VEC2B,
    TYPE_VEC3B,
    TYPE_VEC4B,

    TYPE_SHORT,
    TYPE_USHORT,
    TYPE_BYTE,
    TYPE_UBYTE,

    TYPE_COUNT
}; // enum data_type

int size_of_type(data_type d);
int components(data_type d);

bool is_integer_type(data_type d);
bool is_float_type(data_type d);

__scm_export(gl_core) const char* type_string(data_type d);

} // namespace gl
} // namespace scm

#include "data_types.inl"

#endif // SCM_GL_CORE_DATA_TYPES_H_INCLUDED
