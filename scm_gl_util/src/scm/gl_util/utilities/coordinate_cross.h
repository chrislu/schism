
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_UTIL_COORD_CROSS_H_INCLUDED
#define SCM_GL_UTIL_COORD_CROSS_H_INCLUDED

#include <scm/core/math.h>
#include <scm/core/numeric_types.h>
#include <scm/core/memory.h>

#include <scm/gl_core/gl_core_fwd.h>
#include <scm/gl_core/constants.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

#include <scm/gl_util/utilities/utilities_fwd.h>

namespace scm {
namespace gl {

class __scm_export(gl_util) coordinate_cross
{
public:
    coordinate_cross(const gl::render_device_ptr& device,
                     const float                  line_length);
    virtual ~coordinate_cross();

    void            draw(const gl::render_context_ptr& context,
                         const math::mat4f&            proj_matrix,
                         const math::mat4f&            view_matrix,
                         const float                   line_width = 1.0f);
    void            draw_overlay(
                         const gl::render_context_ptr& context,
                         const math::mat4f&            proj_matrix,
                         const math::mat4f&            view_matrix,
                         const float                   line_width = 1.0f);

private:
    gl::program_ptr                 _coord_program;
    gl::depth_stencil_state_ptr     _dstate_less;
    gl::depth_stencil_state_ptr     _dstate_overlay;
    gl::rasterizer_state_ptr        _raster_no_cull;
    gl::blend_state_ptr             _no_blend;

    gl::buffer_ptr                  _vertices;
    gl::vertex_array_ptr            _vertex_array;
    gl::primitive_topology          _prim_topology;
    int                             _vertex_count;

}; // coordinate_cross

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif //SCM_GL_UTIL_COORD_CROSS_H_INCLUDED
