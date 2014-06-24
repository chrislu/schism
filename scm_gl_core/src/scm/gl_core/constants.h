
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_CONSTANTS_H_INCLUDED
#define SCM_GL_CORE_CONSTANTS_H_INCLUDED

#include <scm/core/numeric_types.h>
#include <scm/core/platform/platform.h>

#include <scm/gl_core/config.h>

namespace scm {
namespace gl {

// buffer /////////////////////////////////////////////////////////////////////////////////////////

enum buffer_binding
{
    BIND_UNKNOWN                     = 0x0000,
    BIND_VERTEX_BUFFER               = 0x0001,
    BIND_INDEX_BUFFER                = 0x0002,
    BIND_PIXEL_PACK_BUFFER           = 0x0004,
    BIND_PIXEL_UNPACK_BUFFER         = 0x0008,
    BIND_PIXEL_BUFFER                = BIND_PIXEL_PACK_BUFFER,
    BIND_UNIFORM_BUFFER              = 0x0010,
    BIND_TEXTURE_BUFFER              = 0x0020,
    BIND_TRANSFORM_FEEDBACK_BUFFER   = 0x0040,
    BIND_ATOMIC_COUNTER_BUFFER       = 0x0080,
    BIND_STORAGE_BUFFER              = 0x0100,

    BUFFER_BINDING_COUNT
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

enum access_mode
{
    ACCESS_READ_ONLY = 0x00,
    ACCESS_WRITE_ONLY,
    ACCESS_READ_WRITE,
    ACCESS_WRITE_INVALIDATE_RANGE,
    ACCESS_WRITE_INVALIDATE_BUFFER,
    ACCESS_WRITE_UNSYNCHRONIZED,

    ACCESS_COUNT
}; // enum access_mode

enum primitive_type
{
    PRIMITIVE_POINTS                    = 0x00,
    PRIMITIVE_LINES,
    PRIMITIVE_TRIANGLES,

    PRIMITIVE_TYPE_COUNT
}; // enum primitive_type

enum primitive_topology {
    PRIMITIVE_POINT_LIST                = 0x00,
    PRIMITIVE_LINE_LIST,
    PRIMITIVE_LINE_STRIP,
    PRIMITIVE_LINE_LOOP,
    PRIMITIVE_LINE_LIST_ADJACENCY,
    PRIMITIVE_LINE_STRIP_ADJACENCY,
    PRIMITIVE_TRIANGLE_LIST,
    PRIMITIVE_TRIANGLE_STRIP,
    PRIMITIVE_TRIANGLE_LIST_ADJACENCY,
    PRIMITIVE_TRIANGLE_STRIP_ADJACENCY,

#if SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_400
    PRIMITIVE_PATCH_LIST_1_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_2_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_3_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_4_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_5_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_6_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_7_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_8_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_9_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_10_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_11_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_12_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_13_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_14_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_15_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_16_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_17_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_18_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_19_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_20_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_21_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_22_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_23_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_24_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_25_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_26_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_27_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_28_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_29_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_30_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_31_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_32_CONTROL_POINTS,
#endif // SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_400

    PRIMITIVE_TOPOLOGY_COUNT
}; // enum primitive_topology

int primitive_patch_control_points(primitive_topology t);

enum interger_handling
{
    INT_PURE,
    INT_FLOAT,
    INT_FLOAT_NORMALIZE
};

// debugging //////////////////////////////////////////////////////////////////////////////////////
enum debug_source
{
    DEBUG_SOURCE_API = 0x00,
    DEBUG_SOURCE_WINDOW_SYSTEM,
    DEBUG_SOURCE_SHADER_COMPILER,
    DEBUG_SOURCE_THIRD_PARTY,
    DEBUG_SOURCE_APPLICATION,
    DEBUG_SOURCE_OTHER,

    DEBUG_SOURCE_COUNT
}; // enum debug_source

enum debug_type
{
    DEBUG_TYPE_ERROR =0x00,
    DEBUG_TYPE_DEPRECATED_BEHAVIOR,
    DEBUG_TYPE_UNDEFINED_BEHAVIOR,
    DEBUG_TYPE_PORTABILITY,
    DEBUG_TYPE_PERFORMANCE,
    DEBUG_TYPE_OTHER,

    DEBUG_TYPE_COUNT
}; // enum debug_type
            
enum debug_severity
{
    DEBUG_SEVERITY_HIGH = 0x00,
    DEBUG_SEVERITY_MEDIUM,
    DEBUG_SEVERITY_LOW,

    DEBUG_SEVERITY_COUNT
}; // enum debug_severity

__scm_export(gl_core) const char* debug_source_string(debug_source s);
__scm_export(gl_core) const char* debug_type_string(debug_type s);
__scm_export(gl_core) const char* debug_severity_string(debug_severity s);

// shader /////////////////////////////////////////////////////////////////////////////////////////

enum shader_stage
{
    STAGE_VERTEX_SHADER      = 0x00,
    STAGE_GEOMETRY_SHADER,
    STAGE_FRAGMENT_SHADER,
    STAGE_COMPUTE_SHADER,
#if SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_400
    STAGE_TESS_EVALUATION_SHADER,
    STAGE_TESS_CONTROL_SHADER,
#endif // SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_400
    SHADER_STAGE_COUNT
}; // enum shader_stage

__scm_export(gl_core) const char* shader_stage_string(shader_stage s);

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
}; // enum polygon_orientation

enum origin_mode
{
    ORIGIN_UPPER_LEFT = 0x00,
    ORIGIN_LOWER_LEFT,

    ORIGIN_MODE_COUNT
}; // enum origin_mode

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

enum sync_status {
    SYNC_UNSIGNALED = 0x00,
    SYNC_SIGNALED,

    SYNC_STATUS_COUNT
}; // enum sync_status

enum sync_wait_result {
    SYNC_WAIT_FAILED = 0x00,
    SYNC_WAIT_CONDITION_SATISFIED,
    SYNC_WAIT_ALREADY_SIGNALED,
    SYNC_WAIT_TIMEOUT_EXPIRED,

    SYNC_WAIT_RESULT_COUNT
}; // enum sync_wait_result

enum occlusion_query_mode {
    OQMODE_SAMPLES_PASSED = 0x00,
    OQMODE_ANY_SAMPLES_PASSED,
    OQMODE_ANY_SAMPLES_PASSED_CONSERVATIVE,

    OCCLUSION_QUERY_MODE_COUNT
}; // occlusion_query_mode

const scm::uint64 sync_timeout_ignored = 0xffffffffffffffffull;

} // namespace gl
} // namespace scm

#include "constants.inl"

#endif // SCM_GL_CORE_CONSTANTS_H_INCLUDED
