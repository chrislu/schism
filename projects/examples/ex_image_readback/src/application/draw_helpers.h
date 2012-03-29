
#ifndef IMAGE_RENDERING_DRAW_HELPERS_H_INCLUDED
#define IMAGE_RENDERING_DRAW_HELPERS_H_INCLUDED

#include <scm/core/math.h>
#include <scm/core/numeric_types.h>
#include <scm/core/pointer_types.h>

#include <scm/gl_core/render_device/render_device_fwd.h>
#include <scm/gl_core/shader_objects/shader_objects_fwd.h>
#include <scm/gl_core/state_objects/state_objects_fwd.h>

#include <scm/gl_util/primitives/primitives_fwd.h>

namespace scm {
namespace data {

class quad_highlight
{
public:
    quad_highlight(const gl::render_device_ptr& device);
    virtual ~quad_highlight();

    void            draw(const gl::render_context_ptr& context,
                         const gl::quad_geometry&      quad,
                         const math::mat4f&            mvp,
                         const math::vec4f&            color      = math::vec4f(1.0f),
                         const float                   line_width = 1.0f);

private:
    gl::program_ptr                 _program;
    gl::depth_stencil_state_ptr     _dstate_less;
    gl::rasterizer_state_ptr        _raster_no_cull;
    gl::blend_state_ptr             _no_blend;
}; // quad_highlight

} // namespace data
} // namespace scm

#endif //IMAGE_RENDERING_DRAW_HELPERS_H_INCLUDED
