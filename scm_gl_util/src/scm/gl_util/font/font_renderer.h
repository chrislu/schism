
#ifndef SCM_GL_UTIL_FONT_RENDERER_H_INCLUDED
#define SCM_GL_UTIL_FONT_RENDERER_H_INCLUDED

#include <scm/gl_core/gl_core_fwd.h>

#include <scm/gl_util/font/font_fwd.h>
#include <scm/gl_util/primitives/primitives_fwd.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(gl_util) font_renderer
{
public:
    font_renderer(const render_device_ptr& device);
    virtual ~font_renderer();

    void            draw(const render_context_ptr& context,
                         const font_face_cptr&     font) const;

protected:
    program_ptr                 _font_shader_program;
    sampler_state_ptr           _font_sampler_state;
    depth_stencil_state_ptr     _font_dstate;
    rasterizer_state_ptr        _font_raster_state;
    blend_state_ptr             _font_blend_state;

    // temporary
    quad_geometry_ptr           _quad;
}; // class font_renderer

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_UTIL_FONT_RENDERER_H_INCLUDED
