
#ifndef SCM_GL_CORE_CONSTANTS_H_INCLUDED
#define SCM_GL_CORE_CONSTANTS_H_INCLUDED

#include <scm/gl_core/config.h>

namespace scm {
namespace gl {

// buffer /////////////////////////////////////////////////////////////////////////////////////////

enum buffer_binding
{
    BIND_UNKNOWN                     = 0x00,
    BIND_VERTEX_BUFFER               = 0x01,
    BIND_INDEX_BUFFER                = 0x02,
    BIND_PIXEL_BUFFER                = 0x04,
    BIND_PIXEL_UNPACK_BUFFER         = 0x08,
    BIND_UNIFORM_BUFFER              = 0x10,
    BIND_TRANSFORM_FEEDBACK_BUFFER   = 0x20
}; // enum buffer_binding

enum buffer_usage
{
    // write once
    USAGE_STATIC_DRAW = 0x00, // GPU r,  CPU
    USAGE_STATIC_READ,        // GPU     CPU r
    USAGE_STATIC_COPY,        // GPU rw, CPU
    // low write frequency
    USAGE_STREAM_DRAW,        // GPU r,  CPU w
    USAGE_STREAM_READ,        // GPU w,  CPU r
    USAGE_STREAM_COPY,        // GPU rw, CPU
    // high write frequency
    USAGE_DYNAMIC_DRAW,       // GPU r,  CPU w
    USAGE_DYNAMIC_READ,       // GPU w,  CPU r
    USAGE_DYNAMIC_COPY,       // GPU rw, CPU

    USAGE_COUNT
}; // enum buffer_usage

enum buffer_access
{
    ACCESS_READ_ONLY = 0x00,
    ACCESS_WRITE_ONLY,
    ACCESS_READ_WRITE,
    ACCESS_WRITE_INVALIDATE_RANGE,
    ACCESS_WRITE_INVALIDATE_BUFFER,
    ACCESS_WRITE_UNSYNCHRONIZED,

    ACCESS_COUNT
}; // enum buffer_access

enum primitive_topology {
    PRIMITIVE_POINT_LIST = 0x00,
    PRIMITIVE_LINE_LIST,
    PRIMITIVE_LINE_STRIP,
    PRIMITIVE_LINE_LOOP,
    PRIMITIVE_LINE_LIST_ADJACENCY,
    PRIMITIVE_LINE_STRIP_ADJACENCY,
    PRIMITIVE_TRIANGLE_LIST,
    PRIMITIVE_TRIANGLE_STRIP,
    PRIMITIVE_TRIANGLE_LIST_ADJACENCY,
    PRIMITIVE_TRIANGLE_STRIP_ADJACENCY,

    PRIMITIVE_TOPOLOGY_COUNT
}; // enum primitive_topology

enum interger_handling
{
    INT_PURE,
    INT_FLOAT,
    INT_FLOAT_NORMALIZE
};

// shader /////////////////////////////////////////////////////////////////////////////////////////

enum shader_stage
{
    STAGE_VERTEX_SHADER      = 0x00,
    STAGE_GEOMETRY_SHADER,
    STAGE_FRAGMENT_SHADER,
#if SCM_GL_CORE_OPENGL_40
    STAGE_TESS_EVALUATION_SHADER,
    STAGE_TESS_CONTROL_SHADER,
#endif
    SHADER_STAGE_COUNT
}; // enum shader_stage

// state objects //////////////////////////////////////////////////////////////////////////////////

enum compare_func
{
    COMPARISON_NEVER = 0x00,
    COMPARISON_ALWAYS,
    COMPARISON_LESS,
    COMPARISON_LESS_EQUAL,
    COMPARISON_EQUAL,
    COMPARISON_GREATER,
    COMPARISON_GREATER_EQUAL,
    COMPARISON_NOT_EQUAL,

    COMPARISON_FUNC_COUNT
}; // enum depth_func

enum stencil_op
{
    STENCIL_KEEP = 0x00,
    STENCIL_ZERO,
    STENCIL_REPLACE,
    STENCIL_INCR_SAT,
    STENCIL_DECR_SAT,
    STENCIL_INVERT,
    STENCIL_INCR_WRAP,
    STENCIL_DECR_WRAP,

    STENCIL_OP_COUNT
}; // enum stencil_op

enum fill_mode
{
    FILL_SOLID = 0x00,
    FILL_WIREFRAME,
    FILL_POINT,

    FILL_MODE_COUNT
}; // enum fill_mode

enum cull_mode
{
    CULL_NONE = 0x00,
    CULL_FRONT,
    CULL_BACK,

    CULL_MODE_COUNT
}; // enum cull_mode

enum polygon_orientation
{
    ORIENT_CW = 0x00,
    ORIENT_CCW,

    POLY_ORIENT_COUNT
};

enum blend_func
{
    FUNC_ZERO = 0x00,
    FUNC_ONE,
    FUNC_SRC_COLOR,
    FUNC_ONE_MINUS_SRC_COLOR,
    FUNC_DST_COLOR,
    FUNC_ONE_MINUS_DST_COLOR,
    FUNC_SRC_ALPHA,
    FUNC_ONE_MINUS_SRC_ALPHA,
    FUNC_DST_ALPHA,
    FUNC_ONE_MINUS_DST_ALPHA,
    FUNC_CONSTANT_COLOR,
    FUNC_ONE_MINUS_CONSTANT_COLOR,
    FUNC_CONSTANT_ALPHA,
    FUNC_ONE_MINUS_CONSTANT_ALPHA,
    FUNC_SRC_ALPHA_SATURATE,
    FUNC_SRC1_COLOR,
    FUNC_ONE_MINUS_SRC1_COLOR,
    FUNC_SRC1_ALPHA,
    FUNC_ONE_MINUS_SRC1_ALPHA,

    BLEND_FUNC_COUNT
}; // enum blend_func

enum blend_equation
{
    EQ_FUNC_ADD = 0x00,
    EQ_FUNC_SUBTRACT,
    EQ_FUNC_REVERSE_SUBTRACT,
    EQ_MIN,
    EQ_MAX,

    BLEND_EQ_COUNT
}; // enum blend_equation

enum color_mask
{
    COLOR_RED   = 0x01,
    COLOR_GREEN = 0x02,
    COLOR_BLUE  = 0x04,
    COLOR_ALPHA = 0x08,
    COLOR_ALL   = COLOR_RED | COLOR_GREEN | COLOR_BLUE | COLOR_ALPHA
}; // enum color_mask


enum texture_filter_mode
{
    FILTER_MIN_MAG_NEAREST = 0x00,
    FILTER_MIN_NEAREST_MAG_LINEAR,
    FILTER_MIN_LINEAR_MAG_NEAREST,
    FILTER_MIN_MAG_LINEAR,
    FILTER_MIN_MAG_MIP_NEAREST,
    FILTER_MIN_MAG_NEAREST_MIP_LINEAR,
    FILTER_MIN_NEAREST_MAG_LINEAR_MIP_NEAREST,
    FILTER_MIN_NEAREST_MAG_MIP_LINEAR,
    FILTER_MIN_LINEAR_MAG_MIP_NEAREST,
    FILTER_MIN_LINEAR_MAG_NEAREST_MIP_LINEAR,
    FILTER_MIN_MAG_LINEAR_MIP_NEAREST,
    FILTER_MIN_MAG_MIP_LINEAR,
    FILTER_ANISOTROPIC,

    TEXTURE_FILTER_COUNT
}; // enum texture_filter

enum texture_wrap_mode
{
    WRAP_CLAMP_TO_EDGE  = 0x00,
    WRAP_REPEAT,
    WRAP_MIRRORED_REPEAT,

    WRAP_MODE_COUNT
}; // enum texture_wrap_mode

enum texture_compare_mode
{
    TEXCOMPARE_NONE = 0x00,
    TEXCOMPARE_COMPARE_REF_TO_TEXTURE,

    TEXCOMARE_COUNT
}; // texture_compare_mode

enum frame_buffer_binding {
    FRAMEBUFFER_DRAW    = 0x00,
    FRAMEBUFFER_READ,

    FRAMEBUFFER_BINDING_COUNT
}; // enum frame_buffer_binding

enum frame_buffer_target {
    FRAMEBUFFER_FRONT_LEFT      = 0x00,
    FRAMEBUFFER_FRONT_RIGHT,
    FRAMEBUFFER_BACK_LEFT,
    FRAMEBUFFER_BACK_RIGHT,
    FRAMEBUFFER_FRONT,
    FRAMEBUFFER_BACK,
    FRAMEBUFFER_LEFT,
    FRAMEBUFFER_RIGHT,
    FRAMEBUFFER_FRONT_AND_BACK,
    
    FRAMEBUFFER_TARGET_COUNT
}; // enum frame_buffer_target

} // namespace gl
} // namespace scm

#endif // SCM_GL_CORE_CONSTANTS_H_INCLUDED
