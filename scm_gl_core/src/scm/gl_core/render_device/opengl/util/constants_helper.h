
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_CONSTANTS_HELPER_H_INCLUDED
#define SCM_GL_CORE_CONSTANTS_HELPER_H_INCLUDED

#include <cassert>

namespace scm {
namespace gl {
namespace util {

unsigned gl_buffer_targets(const buffer_binding b);
unsigned gl_buffer_bindings(const buffer_binding b);
int      gl_usage_flags(const buffer_usage b);
unsigned gl_buffer_access_mode(const access_mode a);
unsigned gl_image_access_mode(const access_mode a);
unsigned gl_primitive_type(const primitive_type p);
unsigned gl_primitive_topology(const primitive_topology p);
int      gl_shader_types(const shader_stage s);
unsigned gl_compare_func(const compare_func c);
unsigned gl_stencil_op(const stencil_op s);
unsigned gl_fill_mode(const fill_mode s);
unsigned gl_cull_mode(const cull_mode s);
unsigned gl_polygon_orientation(const polygon_orientation s);
unsigned gl_origin_mode(const enum origin_mode o);
unsigned gl_blend_func(const blend_func s);
unsigned gl_blend_equation(const blend_equation s);
bool     masked(unsigned in_color_mask, const color_mask in_color);
unsigned gl_texture_min_filter_mode(const texture_filter_mode s);
unsigned gl_texture_mag_filter_mode(const texture_filter_mode s); 
unsigned gl_wrap_mode(const texture_wrap_mode s);
unsigned gl_texture_compare_mode(const texture_compare_mode s);
unsigned gl_framebuffer_binding(const frame_buffer_binding s);
unsigned gl_framebuffer_binding_point(const frame_buffer_binding s);
unsigned gl_frame_buffer_target(const frame_buffer_target s);

debug_source    gl_to_debug_source(unsigned s);
debug_type      gl_to_debug_type(unsigned t);
debug_severity  gl_to_debug_severity(unsigned s);

} // namespace util
} // namespace gl
} // namespace scm

#include "constants_helper.inl"

#endif // SCM_GL_CORE_CONSTANTS_HELPER_H_INCLUDED
