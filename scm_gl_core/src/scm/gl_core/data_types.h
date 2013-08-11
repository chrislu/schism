
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_DATA_TYPES_H_INCLUDED
#define SCM_GL_CORE_DATA_TYPES_H_INCLUDED

#include <scm/core/math.h>

#include <scm/gl_core/config.h>

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

#if SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_400
    TYPE_DOUBLE,
    TYPE_VEC2D,
    TYPE_VEC3D,
    TYPE_VEC4D,

    TYPE_MAT2D,
    TYPE_MAT3D,
    TYPE_MAT4D,

    TYPE_MAT2X3D,
    TYPE_MAT2X4D,
    TYPE_MAT3X2D,
    TYPE_MAT3X4D,
    TYPE_MAT4X2D,
    TYPE_MAT4X3D,
#endif // SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_400

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

    TYPE_SAMPLER,
    TYPE_IMAGE,

    TYPE_COUNT
}; // enum data_type

int size_of_type(data_type d);
int components(data_type d);

bool is_integer_type(data_type d);
bool is_float_type(data_type d);

bool is_sampler_type(data_type d);
bool is_image_type(data_type d);

__scm_export(gl_core) const char* type_string(data_type d);

struct texture_region
{
    texture_region()
    {}
    texture_region(const math::vec3ui& o,
                   const math::vec3ui& d)
      : _origin(o), _dimensions(d)
    {}

    math::vec3ui    _origin;
    math::vec3ui    _dimensions;
}; // struct texture_region

} // namespace gl
} // namespace scm

#include "data_types.inl"

#endif // SCM_GL_CORE_DATA_TYPES_H_INCLUDED
