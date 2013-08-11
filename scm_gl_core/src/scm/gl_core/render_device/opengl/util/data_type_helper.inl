
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include <cassert>

#include <boost/static_assert.hpp>

#include <boost/unordered_set.hpp>
#include <boost/assign/list_of.hpp>

#include <scm/gl_core/config.h>
#include <scm/gl_core/render_device/opengl/gl_core.h>

namespace scm {
namespace gl {
namespace util {

inline
data_type
from_gl_data_type(unsigned gl_type)
{
    if (is_sampler_type(gl_type)) {
        return TYPE_SAMPLER;
    }
    
    if (is_image_type(gl_type)) {
        return TYPE_IMAGE;
    }

    switch (gl_type) {
        case GL_FLOAT:                  return TYPE_FLOAT; break;
        case GL_FLOAT_VEC2:             return TYPE_VEC2F; break;
        case GL_FLOAT_VEC3:             return TYPE_VEC3F; break;
        case GL_FLOAT_VEC4:             return TYPE_VEC4F; break;

        case GL_FLOAT_MAT2:             return TYPE_MAT2F; break;
        case GL_FLOAT_MAT3:             return TYPE_MAT3F; break;
        case GL_FLOAT_MAT4:             return TYPE_MAT4F; break;

        case GL_FLOAT_MAT2x3:           return TYPE_MAT2X3F; break;
        case GL_FLOAT_MAT2x4:           return TYPE_MAT2X4F; break;
        case GL_FLOAT_MAT3x2:           return TYPE_MAT3X2F; break;
        case GL_FLOAT_MAT3x4:           return TYPE_MAT3X4F; break;
        case GL_FLOAT_MAT4x2:           return TYPE_MAT4X2F; break;
        case GL_FLOAT_MAT4x3:           return TYPE_MAT4X3F; break;

#if SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_400
        case GL_DOUBLE:                  return TYPE_DOUBLE; break;
        case GL_DOUBLE_VEC2:             return TYPE_VEC2D; break;
        case GL_DOUBLE_VEC3:             return TYPE_VEC3D; break;
        case GL_DOUBLE_VEC4:             return TYPE_VEC4D; break;

        case GL_DOUBLE_MAT2:             return TYPE_MAT2D; break;
        case GL_DOUBLE_MAT3:             return TYPE_MAT3D; break;
        case GL_DOUBLE_MAT4:             return TYPE_MAT4D; break;

        case GL_DOUBLE_MAT2x3:           return TYPE_MAT2X3D; break;
        case GL_DOUBLE_MAT2x4:           return TYPE_MAT2X4D; break;
        case GL_DOUBLE_MAT3x2:           return TYPE_MAT3X2D; break;
        case GL_DOUBLE_MAT3x4:           return TYPE_MAT3X4D; break;
        case GL_DOUBLE_MAT4x2:           return TYPE_MAT4X2D; break;
        case GL_DOUBLE_MAT4x3:           return TYPE_MAT4X3D; break;
#endif // SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_400

        case GL_INT:                    return TYPE_INT; break;
        case GL_INT_VEC2:               return TYPE_VEC2I; break;
        case GL_INT_VEC3:               return TYPE_VEC3I; break;
        case GL_INT_VEC4:               return TYPE_VEC4I; break;

        case GL_UNSIGNED_INT:           return TYPE_UINT; break;
        case GL_UNSIGNED_INT_VEC2:      return TYPE_VEC2UI; break;
        case GL_UNSIGNED_INT_VEC3:      return TYPE_VEC3UI; break;
        case GL_UNSIGNED_INT_VEC4:      return TYPE_VEC4UI; break;

        case GL_BOOL:                   return TYPE_BOOL; break;
        case GL_BOOL_VEC2:              return TYPE_VEC2B; break;
        case GL_BOOL_VEC3:              return TYPE_VEC3B; break;
        case GL_BOOL_VEC4:              return TYPE_VEC4B; break;

        case GL_SHORT:                  return TYPE_SHORT; break;
        case GL_UNSIGNED_SHORT:         return TYPE_USHORT; break;
        case GL_BYTE:                   return TYPE_BYTE; break;
        case GL_UNSIGNED_BYTE:          return TYPE_UBYTE; break;

        default:                        return TYPE_UNKNOWN; break;
    }
}

inline
bool
is_sampler_type(unsigned gl_type)
{
    switch(gl_type) {
        case GL_SAMPLER_1D:
        case GL_SAMPLER_2D:
        case GL_SAMPLER_3D:
        case GL_SAMPLER_CUBE:
        case GL_SAMPLER_1D_SHADOW:
        case GL_SAMPLER_2D_SHADOW:
        case GL_SAMPLER_1D_ARRAY:
        case GL_SAMPLER_2D_ARRAY:
        case GL_SAMPLER_1D_ARRAY_SHADOW:
        case GL_SAMPLER_2D_ARRAY_SHADOW:
        case GL_SAMPLER_2D_MULTISAMPLE:
        case GL_SAMPLER_2D_MULTISAMPLE_ARRAY:
        case GL_SAMPLER_CUBE_SHADOW:
        case GL_SAMPLER_BUFFER:
        case GL_SAMPLER_2D_RECT:
        case GL_SAMPLER_2D_RECT_SHADOW:
        case GL_INT_SAMPLER_1D:
        case GL_INT_SAMPLER_2D:
        case GL_INT_SAMPLER_3D:
        case GL_INT_SAMPLER_CUBE:
        case GL_INT_SAMPLER_1D_ARRAY:
        case GL_INT_SAMPLER_2D_ARRAY:
        case GL_INT_SAMPLER_2D_MULTISAMPLE:
        case GL_INT_SAMPLER_2D_MULTISAMPLE_ARRAY:
        case GL_INT_SAMPLER_BUFFER:
        case GL_INT_SAMPLER_2D_RECT:
        case GL_UNSIGNED_INT_SAMPLER_1D:
        case GL_UNSIGNED_INT_SAMPLER_2D:
        case GL_UNSIGNED_INT_SAMPLER_3D:
        case GL_UNSIGNED_INT_SAMPLER_CUBE:
        case GL_UNSIGNED_INT_SAMPLER_1D_ARRAY:
        case GL_UNSIGNED_INT_SAMPLER_2D_ARRAY:
        case GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE:
        case GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE_ARRAY:
        case GL_UNSIGNED_INT_SAMPLER_BUFFER:
        case GL_UNSIGNED_INT_SAMPLER_2D_RECT:
#if SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_400
        case GL_SAMPLER_CUBE_MAP_ARRAY:
        case GL_SAMPLER_CUBE_MAP_ARRAY_SHADOW:
        case GL_INT_SAMPLER_CUBE_MAP_ARRAY:
        case GL_UNSIGNED_INT_SAMPLER_CUBE_MAP_ARRAY:
#endif // SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_400
            return true;
        default:
            return false;
    };
}

inline
bool
is_image_type(unsigned gl_type)
{
    if (SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_420) {
        switch(gl_type) {
            case GL_IMAGE_1D:
            case GL_IMAGE_2D:
            case GL_IMAGE_3D:
            case GL_IMAGE_2D_RECT:
            case GL_IMAGE_CUBE:
            case GL_IMAGE_BUFFER:
            case GL_IMAGE_1D_ARRAY:
            case GL_IMAGE_2D_ARRAY:
            case GL_IMAGE_CUBE_MAP_ARRAY:
            case GL_IMAGE_2D_MULTISAMPLE:
            case GL_IMAGE_2D_MULTISAMPLE_ARRAY:
            case GL_INT_IMAGE_1D:
            case GL_INT_IMAGE_2D:
            case GL_INT_IMAGE_3D:
            case GL_INT_IMAGE_2D_RECT:
            case GL_INT_IMAGE_CUBE:
            case GL_INT_IMAGE_BUFFER:
            case GL_INT_IMAGE_1D_ARRAY:
            case GL_INT_IMAGE_2D_ARRAY:
            case GL_INT_IMAGE_CUBE_MAP_ARRAY:
            case GL_INT_IMAGE_2D_MULTISAMPLE:
            case GL_INT_IMAGE_2D_MULTISAMPLE_ARRAY:
            case GL_UNSIGNED_INT_IMAGE_1D:
            case GL_UNSIGNED_INT_IMAGE_2D:
            case GL_UNSIGNED_INT_IMAGE_3D:
            case GL_UNSIGNED_INT_IMAGE_2D_RECT:
            case GL_UNSIGNED_INT_IMAGE_CUBE:
            case GL_UNSIGNED_INT_IMAGE_BUFFER:
            case GL_UNSIGNED_INT_IMAGE_1D_ARRAY:
            case GL_UNSIGNED_INT_IMAGE_2D_ARRAY:
            case GL_UNSIGNED_INT_IMAGE_CUBE_MAP_ARRAY:
            case GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE:
            case GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE_ARRAY:
                return true;
            default:
                return false;
        };
    }
    else { // fall back to EXT_shader_image_load_store
        switch(gl_type) {
            case GL_IMAGE_1D_EXT:
            case GL_IMAGE_2D_EXT:
            case GL_IMAGE_3D_EXT:
            case GL_IMAGE_2D_RECT_EXT:
            case GL_IMAGE_CUBE_EXT:
            case GL_IMAGE_BUFFER_EXT:
            case GL_IMAGE_1D_ARRAY_EXT:
            case GL_IMAGE_2D_ARRAY_EXT:
            case GL_IMAGE_CUBE_MAP_ARRAY_EXT:
            case GL_IMAGE_2D_MULTISAMPLE_EXT:
            case GL_IMAGE_2D_MULTISAMPLE_ARRAY_EXT:
            case GL_INT_IMAGE_1D_EXT:
            case GL_INT_IMAGE_2D_EXT:
            case GL_INT_IMAGE_3D_EXT:
            case GL_INT_IMAGE_2D_RECT_EXT:
            case GL_INT_IMAGE_CUBE_EXT:
            case GL_INT_IMAGE_BUFFER_EXT:
            case GL_INT_IMAGE_1D_ARRAY_EXT:
            case GL_INT_IMAGE_2D_ARRAY_EXT:
            case GL_INT_IMAGE_CUBE_MAP_ARRAY_EXT:
            case GL_INT_IMAGE_2D_MULTISAMPLE_EXT:
            case GL_INT_IMAGE_2D_MULTISAMPLE_ARRAY_EXT:
            case GL_UNSIGNED_INT_IMAGE_1D_EXT:
            case GL_UNSIGNED_INT_IMAGE_2D_EXT:
            case GL_UNSIGNED_INT_IMAGE_3D_EXT:
            case GL_UNSIGNED_INT_IMAGE_2D_RECT_EXT:
            case GL_UNSIGNED_INT_IMAGE_CUBE_EXT:
            case GL_UNSIGNED_INT_IMAGE_BUFFER_EXT:
            case GL_UNSIGNED_INT_IMAGE_1D_ARRAY_EXT:
            case GL_UNSIGNED_INT_IMAGE_2D_ARRAY_EXT:
            case GL_UNSIGNED_INT_IMAGE_CUBE_MAP_ARRAY_EXT:
            case GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE_EXT:
            case GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE_ARRAY_EXT:
                return true;
            default:
                return false;
        };
    }
}


inline
bool is_vaild_index_type(const data_type d)
{
    return (   (TYPE_USHORT == d)
            || (TYPE_UINT == d));
}

inline
unsigned
gl_base_type(const data_type d)
{
    static unsigned gl_base_types[] = {
        GL_NONE,
        // float
        GL_FLOAT, GL_FLOAT, GL_FLOAT, GL_FLOAT,
        // matrices
        GL_FLOAT, GL_FLOAT, GL_FLOAT,
        GL_FLOAT, GL_FLOAT, GL_FLOAT,
        GL_FLOAT, GL_FLOAT, GL_FLOAT,

#if SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_400
        // double
        GL_DOUBLE, GL_DOUBLE, GL_DOUBLE, GL_DOUBLE,
        // matrices
        GL_DOUBLE, GL_DOUBLE, GL_DOUBLE,
        GL_DOUBLE, GL_DOUBLE, GL_DOUBLE,
        GL_DOUBLE, GL_DOUBLE, GL_DOUBLE,
#endif // SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_400

        // int
        GL_INT, GL_INT, GL_INT, GL_INT,
        // unsigned int
        GL_UNSIGNED_INT, GL_UNSIGNED_INT, GL_UNSIGNED_INT, GL_UNSIGNED_INT,
        // bool
        GL_UNSIGNED_BYTE, GL_UNSIGNED_BYTE, GL_UNSIGNED_BYTE, GL_UNSIGNED_BYTE,
        // small int types
        GL_SHORT,
        GL_UNSIGNED_SHORT,
        GL_BYTE,
        GL_UNSIGNED_BYTE,

        // samplers and images
        GL_UNSIGNED_INT64_ARB,
        GL_UNSIGNED_INT64_ARB
    };

    BOOST_STATIC_ASSERT((sizeof(gl_base_types) / sizeof(unsigned)) == TYPE_COUNT);

    assert((sizeof(gl_base_types) / sizeof(unsigned)) == TYPE_COUNT);
    assert(TYPE_UNKNOWN <= d && d < TYPE_COUNT);

    return (gl_base_types[d]);
}

} // namespace util
} // namespace gl
} // namespace scm
