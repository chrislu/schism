
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include <cassert>

#include <boost/static_assert.hpp>

#include <scm/gl_core/config.h>

namespace scm {
namespace gl {
namespace util {

inline
unsigned
gl_buffer_targets(const buffer_binding b)
{
    switch (b) {
        case BIND_VERTEX_BUFFER:                return GL_ARRAY_BUFFER;
        case BIND_INDEX_BUFFER:                 return GL_ELEMENT_ARRAY_BUFFER;
        //case BIND_PIXEL_BUFFER:
        case BIND_PIXEL_PACK_BUFFER:            return GL_PIXEL_PACK_BUFFER;
        case BIND_PIXEL_UNPACK_BUFFER:          return GL_PIXEL_UNPACK_BUFFER;
        case BIND_UNIFORM_BUFFER:               return GL_UNIFORM_BUFFER;
        case BIND_TEXTURE_BUFFER:               return GL_TEXTURE_BUFFER;
        case BIND_TRANSFORM_FEEDBACK_BUFFER:    return GL_TRANSFORM_FEEDBACK_BUFFER;
        case BIND_ATOMIC_COUNTER_BUFFER:        return GL_ATOMIC_COUNTER_BUFFER;
        case BIND_STORAGE_BUFFER:               return GL_SHADER_STORAGE_BUFFER;
        default:                                return 0;
    }
}

inline
unsigned
gl_buffer_bindings(const buffer_binding b)
{
    switch (b) {
        case BIND_VERTEX_BUFFER:                return GL_ARRAY_BUFFER_BINDING;
        case BIND_INDEX_BUFFER:                 return GL_ELEMENT_ARRAY_BUFFER_BINDING;
        //case BIND_PIXEL_BUFFER:
        case BIND_PIXEL_PACK_BUFFER:            return GL_PIXEL_PACK_BUFFER_BINDING;
        case BIND_PIXEL_UNPACK_BUFFER:          return GL_PIXEL_UNPACK_BUFFER_BINDING;
        case BIND_UNIFORM_BUFFER:               return GL_UNIFORM_BUFFER_BINDING;
        case BIND_TEXTURE_BUFFER:               return GL_TEXTURE_BINDING_BUFFER;
        case BIND_TRANSFORM_FEEDBACK_BUFFER:    return GL_TRANSFORM_FEEDBACK_BUFFER_BINDING;
        case BIND_ATOMIC_COUNTER_BUFFER:        return GL_ATOMIC_COUNTER_BUFFER_BINDING;
        case BIND_STORAGE_BUFFER:               return GL_SHADER_STORAGE_BUFFER_BINDING;
        default:                                return 0;
    }
}

inline
int
gl_usage_flags(const buffer_usage b)
{
    static int glbufu[] = {
        // write once
        GL_STATIC_DRAW,     // GPU r,  CPU
        GL_STATIC_READ,     // GPU     CPU r
        GL_STATIC_COPY,     // GPU rw, CPU
        // low write frequency
        GL_STREAM_DRAW,     // GPU r,  CPU w
        GL_STREAM_READ,     // GPU w,  CPU r
        GL_STREAM_COPY,     // GPU rw, CPU
        // high write frequency
        GL_DYNAMIC_DRAW,    // GPU r,  CPU w
        GL_DYNAMIC_READ,    // GPU w,  CPU r
        GL_DYNAMIC_COPY     // GPU rw, CPU
    };

    BOOST_STATIC_ASSERT((sizeof(glbufu) / sizeof(int)) == USAGE_COUNT);

    assert((sizeof(glbufu) / sizeof(int)) == USAGE_COUNT);
    assert(USAGE_STATIC_DRAW <= b && b < USAGE_COUNT);

    return glbufu[b];
}

inline
unsigned
gl_buffer_access_mode(const access_mode a)
{
    assert(ACCESS_READ_ONLY <= a && a < ACCESS_COUNT);

    switch (a) {
        case ACCESS_READ_ONLY:                  return GL_MAP_READ_BIT;
        case ACCESS_WRITE_ONLY:                 return GL_MAP_WRITE_BIT;
        case ACCESS_READ_WRITE:                 return GL_MAP_READ_BIT | GL_MAP_WRITE_BIT;
        case ACCESS_WRITE_INVALIDATE_RANGE:     return GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_RANGE_BIT;
        case ACCESS_WRITE_INVALIDATE_BUFFER:    return GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT;
        case ACCESS_WRITE_UNSYNCHRONIZED:       return GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_RANGE_BIT | GL_MAP_UNSYNCHRONIZED_BIT;
        default:                                return 0;                       
    }
}

inline
unsigned
gl_image_access_mode(const access_mode a)
{
    assert(ACCESS_READ_ONLY <= a && a < ACCESS_COUNT);

    switch (a) {
        case ACCESS_READ_ONLY:                  return GL_READ_ONLY;
        case ACCESS_WRITE_ONLY:                 return GL_WRITE_ONLY;
        case ACCESS_READ_WRITE:                 return GL_READ_WRITE;
        default:                                return 0;                       
    }
}

inline
unsigned
gl_primitive_type(const primitive_type p)
{
    static unsigned types[] = {
        GL_POINTS,                  // PRIMITIVE_POINTS                = 0x00,
        GL_LINES,                   // PRIMITIVE_LINES,
        GL_TRIANGLES                // PRIMITIVE_TRIANGLES
    };

    BOOST_STATIC_ASSERT((sizeof(types) / sizeof(unsigned)) == PRIMITIVE_TYPE_COUNT);

    assert((sizeof(types) / sizeof(unsigned)) == PRIMITIVE_TYPE_COUNT);
    assert(PRIMITIVE_POINTS <= p && p < PRIMITIVE_TYPE_COUNT);

    return types[p];
}

inline
unsigned
gl_primitive_topology(const primitive_topology p)
{
    static unsigned types[] = {
        GL_POINTS,                  // PRIMITIVE_POINT_LIST                = 0x00,
        GL_LINES,                   // PRIMITIVE_LINE_LIST,
        GL_LINE_STRIP,              // PRIMITIVE_LINE_STRIP,
        GL_LINE_LOOP,               // PRIMITIVE_LINE_LOOP,
        GL_LINES_ADJACENCY,         // PRIMITIVE_LINE_LIST_ADJACENCY,
        GL_LINE_STRIP_ADJACENCY,    // PRIMITIVE_LINE_STRIP_ADJACENCY,
        GL_TRIANGLES,               // PRIMITIVE_TRIANGLE_LIST,
        GL_TRIANGLE_STRIP,          // PRIMITIVE_TRIANGLE_STRIP,
        GL_TRIANGLES_ADJACENCY,     // PRIMITIVE_TRIANGLE_LIST_ADJACENCY,
        GL_TRIANGLE_STRIP_ADJACENCY // PRIMITIVE_TRIANGLE_STRIP_ADJACENCY
#if SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_400
        ,
        GL_PATCHES,                 // PRIMITIVE_PATCH_LIST_1_CONTROL_POINTS,
        GL_PATCHES,                 // PRIMITIVE_PATCH_LIST_2_CONTROL_POINTS,
        GL_PATCHES,                 // PRIMITIVE_PATCH_LIST_3_CONTROL_POINTS,
        GL_PATCHES,                 // PRIMITIVE_PATCH_LIST_4_CONTROL_POINTS,
        GL_PATCHES,                 // PRIMITIVE_PATCH_LIST_5_CONTROL_POINTS,
        GL_PATCHES,                 // PRIMITIVE_PATCH_LIST_6_CONTROL_POINTS,
        GL_PATCHES,                 // PRIMITIVE_PATCH_LIST_7_CONTROL_POINTS,
        GL_PATCHES,                 // PRIMITIVE_PATCH_LIST_8_CONTROL_POINTS,
        GL_PATCHES,                 // PRIMITIVE_PATCH_LIST_9_CONTROL_POINTS,
        GL_PATCHES,                 // PRIMITIVE_PATCH_LIST_10_CONTROL_POINTS,
        GL_PATCHES,                 // PRIMITIVE_PATCH_LIST_11_CONTROL_POINTS,
        GL_PATCHES,                 // PRIMITIVE_PATCH_LIST_12_CONTROL_POINTS,
        GL_PATCHES,                 // PRIMITIVE_PATCH_LIST_13_CONTROL_POINTS,
        GL_PATCHES,                 // PRIMITIVE_PATCH_LIST_14_CONTROL_POINTS,
        GL_PATCHES,                 // PRIMITIVE_PATCH_LIST_15_CONTROL_POINTS,
        GL_PATCHES,                 // PRIMITIVE_PATCH_LIST_16_CONTROL_POINTS,
        GL_PATCHES,                 // PRIMITIVE_PATCH_LIST_17_CONTROL_POINTS,
        GL_PATCHES,                 // PRIMITIVE_PATCH_LIST_18_CONTROL_POINTS,
        GL_PATCHES,                 // PRIMITIVE_PATCH_LIST_19_CONTROL_POINTS,
        GL_PATCHES,                 // PRIMITIVE_PATCH_LIST_20_CONTROL_POINTS,
        GL_PATCHES,                 // PRIMITIVE_PATCH_LIST_21_CONTROL_POINTS,
        GL_PATCHES,                 // PRIMITIVE_PATCH_LIST_22_CONTROL_POINTS,
        GL_PATCHES,                 // PRIMITIVE_PATCH_LIST_23_CONTROL_POINTS,
        GL_PATCHES,                 // PRIMITIVE_PATCH_LIST_24_CONTROL_POINTS,
        GL_PATCHES,                 // PRIMITIVE_PATCH_LIST_25_CONTROL_POINTS,
        GL_PATCHES,                 // PRIMITIVE_PATCH_LIST_26_CONTROL_POINTS,
        GL_PATCHES,                 // PRIMITIVE_PATCH_LIST_27_CONTROL_POINTS,
        GL_PATCHES,                 // PRIMITIVE_PATCH_LIST_28_CONTROL_POINTS,
        GL_PATCHES,                 // PRIMITIVE_PATCH_LIST_29_CONTROL_POINTS,
        GL_PATCHES,                 // PRIMITIVE_PATCH_LIST_30_CONTROL_POINTS,
        GL_PATCHES,                 // PRIMITIVE_PATCH_LIST_31_CONTROL_POINTS,
        GL_PATCHES                  // PRIMITIVE_PATCH_LIST_32_CONTROL_POINTS,
#endif // SCM_GL_CORE_OPENGL_CORE_VERSION < SCM_GL_CORE_OPENGL_CORE_VERSION_400
    };

    BOOST_STATIC_ASSERT((sizeof(types) / sizeof(unsigned)) == PRIMITIVE_TOPOLOGY_COUNT);

    assert((sizeof(types) / sizeof(unsigned)) == PRIMITIVE_TOPOLOGY_COUNT);
    assert(PRIMITIVE_POINT_LIST <= p && p < PRIMITIVE_TOPOLOGY_COUNT);

    return types[p];
}

inline
int
gl_shader_types(const shader_stage s)
{
    static int shader_types[] = {
        GL_VERTEX_SHADER,           // STAGE_VERTEX_SHADER      = 0x00,
        GL_GEOMETRY_SHADER,         // STAGE_GEOMETRY_SHADER
        GL_FRAGMENT_SHADER,         // STAGE_FRAGMENT_SHADER
        GL_COMPUTE_SHADER           // STAGE_COMPUTE_SHADER
#if SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_400
        ,
        GL_TESS_EVALUATION_SHADER,  // STAGE_TESS_EVALUATION_SHADER,
        GL_TESS_CONTROL_SHADER      // STAGE_TESS_CONTROL_SHADER
#endif // SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_400
    };

    BOOST_STATIC_ASSERT((sizeof(shader_types) / sizeof(int)) == SHADER_STAGE_COUNT);

    assert((sizeof(shader_types) / sizeof(int)) == SHADER_STAGE_COUNT);
    assert(STAGE_VERTEX_SHADER <= s && s < SHADER_STAGE_COUNT);

    return shader_types[s];
}

inline
unsigned
gl_compare_func(const compare_func c)
{
    static unsigned compare_funcs[] = {
        GL_NEVER,                   // COMPARISON_NEVER = 0x00,
        GL_ALWAYS,                  // COMPARISON_ALWAYS,
        GL_LESS,                    // COMPARISON_LESS,
        GL_LEQUAL,                  // COMPARISON_LESS_EQUAL,
        GL_EQUAL,                   // COMPARISON_EQUAL,
        GL_GREATER,                 // COMPARISON_GREATER,
        GL_GEQUAL,                  // COMPARISON_GREATER_EQUAL,
        GL_NOTEQUAL                 // COMPARISON_NOT_EQUAL
    };

    BOOST_STATIC_ASSERT((sizeof(compare_funcs) / sizeof(unsigned)) == COMPARISON_FUNC_COUNT);

    assert((sizeof(compare_funcs) / sizeof(unsigned)) == COMPARISON_FUNC_COUNT);
    assert(COMPARISON_NEVER <= c && c < COMPARISON_FUNC_COUNT);

    return compare_funcs[c];
}

inline
unsigned
gl_stencil_op(const stencil_op s)
{
    static unsigned stencil_ops[] = {
        GL_KEEP,                    // STENCIL_KEEP = 0x00,
        GL_ZERO,                    // STENCIL_ZERO,
        GL_REPLACE,                 // STENCIL_REPLACE,
        GL_INCR,                    // STENCIL_INCR_SAT,
        GL_DECR,                    // STENCIL_DECR_SAT,
        GL_INVERT,                  // STENCIL_INVERT,
        GL_INCR_WRAP,               // STENCIL_INCR_WRAP,
        GL_DECR_WRAP                // STENCIL_DECR_WRAP
    };

    BOOST_STATIC_ASSERT((sizeof(stencil_ops) / sizeof(unsigned)) == STENCIL_OP_COUNT);

    assert((sizeof(stencil_ops) / sizeof(unsigned)) == STENCIL_OP_COUNT);
    assert(STENCIL_KEEP <= s && s < STENCIL_OP_COUNT);

    return stencil_ops[s];
}

inline
unsigned
gl_fill_mode(const fill_mode s)
{
    static unsigned fill_modes[] = {
        GL_FILL,    // FILL_SOLID = 0x00,
        GL_LINE,    // FILL_WIREFRAME
        GL_POINT    // FILL_POINT
    };

    BOOST_STATIC_ASSERT((sizeof(fill_modes) / sizeof(unsigned)) == FILL_MODE_COUNT);

    assert((sizeof(fill_modes) / sizeof(unsigned)) == FILL_MODE_COUNT);
    assert(FILL_SOLID <= s && s < FILL_MODE_COUNT);

    return fill_modes[s];
}

inline
unsigned
gl_cull_mode(const cull_mode s)
{
    static unsigned cull_modes[] = {
        GL_NONE,    // CULL_NONE = 0x00,
        GL_FRONT,   // CULL_FRONT,
        GL_BACK     // CULL_BACK
    };

    BOOST_STATIC_ASSERT((sizeof(cull_modes) / sizeof(unsigned)) == CULL_MODE_COUNT);

    assert((sizeof(cull_modes) / sizeof(unsigned)) == CULL_MODE_COUNT);
    assert(CULL_NONE <= s && s < CULL_MODE_COUNT);

    return cull_modes[s];
}

inline
unsigned
gl_polygon_orientation(const polygon_orientation s)
{
    static unsigned polygon_orientations[] = {
        GL_CW,      // ORIENT_CW = 0x00,
        GL_CCW      // ORIENT_CCW
    };

    BOOST_STATIC_ASSERT((sizeof(polygon_orientations) / sizeof(unsigned)) == POLY_ORIENT_COUNT);

    assert((sizeof(polygon_orientations) / sizeof(unsigned)) == POLY_ORIENT_COUNT);
    assert(ORIENT_CW <= s && s < POLY_ORIENT_COUNT);

    return polygon_orientations[s];
}

inline
unsigned
gl_origin_mode(const enum origin_mode o)
{
    switch (o) {
    case ORIGIN_UPPER_LEFT: return GL_UPPER_LEFT; break;
    case ORIGIN_LOWER_LEFT: return GL_LOWER_LEFT; break;
    default:                assert(0 && (o < ORIGIN_MODE_COUNT));return 0;
    }
}

inline
unsigned
gl_blend_func(const blend_func s)
{
    static unsigned blend_funcs[] = {
        GL_ZERO,                        // FUNC_ZERO = 0x00,
        GL_ONE,                         // FUNC_ONE,
        GL_SRC_COLOR,                   // FUNC_SRC_COLOR,
        GL_ONE_MINUS_SRC_COLOR,         // FUNC_ONE_MINUS_SRC_COLOR,
        GL_DST_COLOR,                   // FUNC_DST_COLOR,
        GL_ONE_MINUS_DST_COLOR,         // FUNC_ONE_MINUS_DST_COLOR,
        GL_SRC_ALPHA,                   // FUNC_SRC_ALPHA,
        GL_ONE_MINUS_SRC_ALPHA,         // FUNC_ONE_MINUS_SRC_ALPHA,
        GL_DST_ALPHA,                   // FUNC_DST_ALPHA,
        GL_ONE_MINUS_DST_ALPHA,         // FUNC_ONE_MINUS_DST_ALPHA,
        GL_CONSTANT_COLOR,              // FUNC_CONSTANT_COLOR,
        GL_ONE_MINUS_CONSTANT_COLOR,    // FUNC_ONE_MINUS_CONSTANT_COLOR,
        GL_CONSTANT_ALPHA,              // FUNC_CONSTANT_ALPHA,
        GL_ONE_MINUS_CONSTANT_ALPHA,    // FUNC_ONE_MINUS_CONSTANT_ALPHA,
        GL_SRC_ALPHA_SATURATE,          // FUNC_SRC_ALPHA_SATURATE,
        GL_SRC1_COLOR,                  // FUNC_SRC1_COLOR,
        GL_ONE_MINUS_SRC1_COLOR,        // FUNC_ONE_MINUS_SRC1_COLOR,
        GL_SRC1_ALPHA,                  // FUNC_SRC1_ALPHA,
        GL_ONE_MINUS_SRC1_ALPHA         // FUNC_ONE_MINUS_SRC1_ALPHA
    };

    BOOST_STATIC_ASSERT((sizeof(blend_funcs) / sizeof(unsigned)) == BLEND_FUNC_COUNT);

    assert((sizeof(blend_funcs) / sizeof(unsigned)) == BLEND_FUNC_COUNT);
    assert(FUNC_ZERO <= s && s < BLEND_FUNC_COUNT);

    return blend_funcs[s];
}

inline
unsigned
gl_blend_equation(const blend_equation s)
{
    static unsigned blend_equations[] = {
        GL_FUNC_ADD,                    // EQ_FUNC_ADD = 0x00,
        GL_FUNC_SUBTRACT,               // EQ_FUNC_SUBTRACT,
        GL_FUNC_REVERSE_SUBTRACT,       // EQ_FUNC_REVERSE_SUBTRACT,
        GL_MIN,                         // EQ_MIN,
        GL_MAX                          // EQ_MAX
    };

    BOOST_STATIC_ASSERT((sizeof(blend_equations) / sizeof(unsigned)) == BLEND_EQ_COUNT);

    assert((sizeof(blend_equations) / sizeof(unsigned)) == BLEND_EQ_COUNT);
    assert(EQ_FUNC_ADD <= s && s < BLEND_EQ_COUNT);

    return blend_equations[s];
}

inline
bool
masked(unsigned in_color_mask, const color_mask in_color) {
    return (0 != (in_color_mask & in_color));
}

inline
unsigned
gl_texture_min_filter_mode(const texture_filter_mode s)
{
    static unsigned texture_min_filter_modes[] = {
        GL_NEAREST,                 // FILTER_MIN_MAG_NEAREST = 0x00,
        GL_NEAREST,                 // FILTER_MIN_NEAREST_MAG_LINEAR,
        GL_LINEAR,                  // FILTER_MIN_LINEAR_MAG_NEAREST,
        GL_LINEAR,                  // FILTER_MIN_MAG_LINEAR,
        GL_NEAREST_MIPMAP_NEAREST,  // FILTER_MIN_MAG_MIP_NEAREST,
        GL_NEAREST_MIPMAP_LINEAR,   // FILTER_MIN_MAG_NEAREST_MIP_LINEAR,
        GL_NEAREST_MIPMAP_NEAREST,  // FILTER_MIN_NEAREST_MAG_LINEAR_MIP_NEAREST,
        GL_NEAREST_MIPMAP_LINEAR,   // FILTER_MIN_NEAREST_MAG_MIP_LINEAR,
        GL_LINEAR_MIPMAP_NEAREST,   // FILTER_MIN_LINEAR_MAG_MIP_NEAREST,
        GL_LINEAR_MIPMAP_LINEAR,    // FILTER_MIN_LINEAR_MAG_NEAREST_MIP_LINEAR,
        GL_LINEAR_MIPMAP_NEAREST,   // FILTER_MIN_MAG_LINEAR_MIP_NEAREST,
        GL_LINEAR_MIPMAP_LINEAR,    // FILTER_MIN_MAG_MIP_LINEAR,
        GL_LINEAR_MIPMAP_LINEAR     // FILTER_ANISOTROPIC
    };

    BOOST_STATIC_ASSERT((sizeof(texture_min_filter_modes) / sizeof(unsigned)) == TEXTURE_FILTER_COUNT);

    assert((sizeof(texture_min_filter_modes) / sizeof(unsigned)) == TEXTURE_FILTER_COUNT);
    assert(FILTER_MIN_MAG_NEAREST <= s && s < TEXTURE_FILTER_COUNT);

    return texture_min_filter_modes[s];
}

inline
unsigned
gl_texture_mag_filter_mode(const texture_filter_mode s)
{
    static unsigned texture_mag_filter_modes[] = {
        GL_NEAREST,  // FILTER_MIN_MAG_NEAREST = 0x00,
        GL_LINEAR,   // FILTER_MIN_NEAREST_MAG_LINEAR,
        GL_NEAREST,  // FILTER_MIN_LINEAR_MAG_NEAREST,
        GL_LINEAR,   // FILTER_MIN_MAG_LINEAR,
        GL_NEAREST,  // FILTER_MIN_MAG_MIP_NEAREST,
        GL_NEAREST,  // FILTER_MIN_MAG_NEAREST_MIP_LINEAR,
        GL_LINEAR,   // FILTER_MIN_NEAREST_MAG_LINEAR_MIP_NEAREST,
        GL_LINEAR,   // FILTER_MIN_NEAREST_MAG_MIP_LINEAR,
        GL_NEAREST,  // FILTER_MIN_LINEAR_MAG_MIP_NEAREST,
        GL_NEAREST,  // FILTER_MIN_LINEAR_MAG_NEAREST_MIP_LINEAR,
        GL_LINEAR,   // FILTER_MIN_MAG_LINEAR_MIP_NEAREST,
        GL_LINEAR,   // FILTER_MIN_MAG_MIP_LINEAR,
        GL_LINEAR    // FILTER_ANISOTROPIC
    };

    BOOST_STATIC_ASSERT((sizeof(texture_mag_filter_modes) / sizeof(unsigned)) == TEXTURE_FILTER_COUNT);

    assert((sizeof(texture_mag_filter_modes) / sizeof(unsigned)) == TEXTURE_FILTER_COUNT);
    assert(FILTER_MIN_MAG_NEAREST <= s && s < TEXTURE_FILTER_COUNT);

    return texture_mag_filter_modes[s];
}

inline
unsigned
gl_wrap_mode(const texture_wrap_mode s)
{
    static unsigned wrap_modes[] = {
        GL_CLAMP_TO_EDGE,   // WRAP_CLAMP_TO_EDGE  = 0x00,
        GL_REPEAT,          // WRAP_REPEAT,
        GL_MIRRORED_REPEAT  // WRAP_MIRRORED_REPEAT
    };

    BOOST_STATIC_ASSERT((sizeof(wrap_modes) / sizeof(unsigned)) == WRAP_MODE_COUNT);

    assert((sizeof(wrap_modes) / sizeof(unsigned)) == WRAP_MODE_COUNT);
    assert(WRAP_CLAMP_TO_EDGE <= s && s < WRAP_MODE_COUNT);

    return wrap_modes[s];
}

inline
unsigned
gl_texture_compare_mode(const texture_compare_mode s)
{
    static unsigned texture_compare_modes[] = {
        GL_NONE,                    // TEXCOMPARE_NONE = 0x00,
        GL_COMPARE_REF_TO_TEXTURE   // TEXCOMPARE_COMPARE_REF_TO_TEXTURE
    };

    BOOST_STATIC_ASSERT((sizeof(texture_compare_modes) / sizeof(unsigned)) == TEXCOMARE_COUNT);

    assert((sizeof(texture_compare_modes) / sizeof(unsigned)) == TEXCOMARE_COUNT);
    assert(TEXCOMPARE_NONE <= s && s < TEXCOMARE_COUNT);

    return texture_compare_modes[s];
}

inline
unsigned
gl_framebuffer_binding(const frame_buffer_binding s)
{
    static unsigned framebuffer_bindings[] = {
        GL_DRAW_FRAMEBUFFER,  // FRAMEBUFFER_DRAW = 0x00,
        GL_READ_FRAMEBUFFER   // FRAMEBUFFER_READ
    };

    BOOST_STATIC_ASSERT((sizeof(framebuffer_bindings) / sizeof(unsigned)) == FRAMEBUFFER_BINDING_COUNT);

    assert((sizeof(framebuffer_bindings) / sizeof(unsigned)) == FRAMEBUFFER_BINDING_COUNT);
    assert(FRAMEBUFFER_DRAW <= s && s < FRAMEBUFFER_BINDING_COUNT);

    return framebuffer_bindings[s];
}

inline
unsigned
gl_framebuffer_binding_point(const frame_buffer_binding s)
{
    static unsigned framebuffer_binding_points[] = {
        GL_DRAW_FRAMEBUFFER_BINDING,  // FRAMEBUFFER_DRAW = 0x00,
        GL_READ_FRAMEBUFFER_BINDING   // FRAMEBUFFER_READ
    };

    BOOST_STATIC_ASSERT((sizeof(framebuffer_binding_points) / sizeof(unsigned)) == FRAMEBUFFER_BINDING_COUNT);

    assert((sizeof(framebuffer_binding_points) / sizeof(unsigned)) == FRAMEBUFFER_BINDING_COUNT);
    assert(FRAMEBUFFER_DRAW <= s && s < FRAMEBUFFER_BINDING_COUNT);

    return framebuffer_binding_points[s];
}

inline
unsigned
gl_frame_buffer_target(const frame_buffer_target s)
{
    static unsigned framebuffer_targets[] = {
        GL_FRONT_LEFT,                // FRAMEBUFFER_FRONT_LEFT      = 0x00,
        GL_FRONT_RIGHT,               // FRAMEBUFFER_FRONT_RIGHT,
        GL_BACK_LEFT,                 // FRAMEBUFFER_BACK_LEFT,
        GL_BACK_RIGHT,                // FRAMEBUFFER_BACK_RIGHT,
        GL_FRONT,                     // FRAMEBUFFER_FRONT,
        GL_BACK,                      // FRAMEBUFFER_BACK,
        GL_LEFT,                      // FRAMEBUFFER_LEFT,
        GL_RIGHT,                     // FRAMEBUFFER_RIGHT,
        GL_FRONT_AND_BACK             // FRAMEBUFFER_FRONT_AND_BACK,
    };

    BOOST_STATIC_ASSERT((sizeof(framebuffer_targets) / sizeof(unsigned)) == FRAMEBUFFER_TARGET_COUNT);

    assert((sizeof(framebuffer_targets) / sizeof(unsigned)) == FRAMEBUFFER_TARGET_COUNT);
    assert(GL_FRONT_LEFT <= s && s < FRAMEBUFFER_TARGET_COUNT);

    return framebuffer_targets[s];
}

inline
debug_source
gl_to_debug_source(unsigned s)
{
    switch (s) {
    case GL_DEBUG_SOURCE_API_ARB:               return DEBUG_SOURCE_API; break;
    case GL_DEBUG_SOURCE_WINDOW_SYSTEM_ARB:     return DEBUG_SOURCE_WINDOW_SYSTEM; break;
    case GL_DEBUG_SOURCE_SHADER_COMPILER_ARB:   return DEBUG_SOURCE_SHADER_COMPILER; break;
    case GL_DEBUG_SOURCE_THIRD_PARTY_ARB:       return DEBUG_SOURCE_THIRD_PARTY; break;
    case GL_DEBUG_SOURCE_APPLICATION_ARB:       return DEBUG_SOURCE_APPLICATION; break;
    case GL_DEBUG_SOURCE_OTHER_ARB:             return DEBUG_SOURCE_OTHER; break;
    default:                                    return DEBUG_SOURCE_OTHER; break;
    }
}

inline
debug_type
gl_to_debug_type(unsigned t)
{
    switch (t) {
    case GL_DEBUG_TYPE_ERROR_ARB:               return DEBUG_TYPE_ERROR; break;
    case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR_ARB: return DEBUG_TYPE_DEPRECATED_BEHAVIOR; break;
    case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR_ARB:  return DEBUG_TYPE_UNDEFINED_BEHAVIOR; break;
    case GL_DEBUG_TYPE_PORTABILITY_ARB:         return DEBUG_TYPE_PORTABILITY; break;
    case GL_DEBUG_TYPE_PERFORMANCE_ARB:         return DEBUG_TYPE_PERFORMANCE; break;
    case GL_DEBUG_TYPE_OTHER_ARB:               return DEBUG_TYPE_OTHER; break;
    default:                                    return DEBUG_TYPE_OTHER; break;
    }
}

inline
debug_severity
gl_to_debug_severity(unsigned s)
{
    switch (s) {
    case GL_DEBUG_SEVERITY_HIGH_ARB:            return DEBUG_SEVERITY_HIGH; break;
    case GL_DEBUG_SEVERITY_MEDIUM_ARB:          return DEBUG_SEVERITY_MEDIUM; break;
    case GL_DEBUG_SEVERITY_LOW_ARB:             return DEBUG_SEVERITY_LOW; break;
    default:                                    return DEBUG_SEVERITY_LOW; break;
    }
}

} // namespace util
} // namespace gl
} // namespace scm
