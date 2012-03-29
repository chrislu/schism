
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_UTIL_GEOMETRY_HIGHLIGHT_H_INCLUDED
#define SCM_GL_UTIL_GEOMETRY_HIGHLIGHT_H_INCLUDED

#include <scm/core/math.h>
#include <scm/core/numeric_types.h>
#include <scm/core/memory.h>

#include <scm/gl_core/render_device/render_device_fwd.h>
#include <scm/gl_core/shader_objects/shader_objects_fwd.h>
#include <scm/gl_core/state_objects/state_objects_fwd.h>

#include <scm/gl_util/primitives/primitives_fwd.h>
#include <scm/gl_util/primitives/geometry.h>
#include <scm/gl_util/utilities/utilities_fwd.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(gl_util) geometry_highlight
{
public:
    geometry_highlight(const gl::render_device_ptr& device);
    virtual ~geometry_highlight();

    void            draw(const gl::render_context_ptr& context,
                         const gl::geometry_ptr&       geom,
                         const math::mat4f&            proj_matrix,
                         const math::mat4f&            view_matrix,
                         const gl::geometry::draw_mode dm,
                         const math::vec4f&            color      = math::vec4f(1.0f),
                         const float                   line_width = 1.0f);
    void            draw_overlay(const gl::render_context_ptr& context,
                                 const gl::geometry_ptr&       geom,
                                 const math::mat4f&            proj_matrix,
                                 const math::mat4f&            view_matrix,
                                 const gl::geometry::draw_mode dm,
                                 const math::vec4f&            color      = math::vec4f(1.0f),
                                 const float                   line_width = 1.0f);

private:
    gl::program_ptr                 _wire_program;
    gl::program_ptr                 _solid_program;
    gl::depth_stencil_state_ptr     _dstate_less;
    gl::depth_stencil_state_ptr     _dstate_noz;
    gl::rasterizer_state_ptr        _raster_no_cull;
    gl::blend_state_ptr             _no_blend;
    gl::blend_state_ptr             _blend;
}; // geometry_highlight

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif //SCM_GL_UTIL_GEOMETRY_HIGHLIGHT_H_INCLUDED
