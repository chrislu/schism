
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_UTIL_TEXT_RENDERER_H_INCLUDED
#define SCM_GL_UTIL_TEXT_RENDERER_H_INCLUDED

#include <scm/core/math.h>

#include <scm/gl_core/gl_core_fwd.h>

#include <scm/gl_util/font/font_fwd.h>
#include <scm/gl_util/primitives/primitives_fwd.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(gl_util) text_renderer
{
public:
    text_renderer(const render_device_ptr& device);
    virtual ~text_renderer();

    void            draw(const render_context_ptr& context,
                         const math::vec2i&        pos,
                         const text_ptr&           txt) const;
    void            draw_shadowed(const render_context_ptr& context,
                                  const math::vec2i&        pos,
                                  const text_ptr&           txt) const;
    void            draw_outlined(const render_context_ptr& context,
                                  const math::vec2i&        pos,
                                  const text_ptr&           txt) const;

    void            projection_matrix(const math::mat4f& m);

protected:
    program_ptr                 _font_program_gray;
    program_ptr                 _font_program_lcd;
    program_ptr                 _font_program_outline_gray;
    program_ptr                 _font_program_outline_lcd;
    sampler_state_ptr           _font_sampler_state;
    depth_stencil_state_ptr     _font_dstate;
    rasterizer_state_ptr        _font_raster_state;
    blend_state_ptr             _font_blend_gray;
    blend_state_ptr             _font_blend_lcd;

    math::mat4f                 _projection_matrix;

    //// temporary
    quad_geometry_ptr           _quad;
}; // class text_renderer

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_UTIL_TEXT_RENDERER_H_INCLUDED
