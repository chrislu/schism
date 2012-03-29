
#ifndef SCM_LARGE_DATA_VOLUME_RENDERER_H_INCLUDED
#define SCM_LARGE_DATA_VOLUME_RENDERER_H_INCLUDED

#include <scm/core/math.h>
#include <scm/gl_core/gl_core_fwd.h>

#include <scm/gl_util/primitives/primitives_fwd.h>
#include <scm/gl_util/viewer/viewer_fwd.h>

#include <renderer/renderer_fwd.h>

namespace scm {
namespace data {

class volume_renderer
{
public:
    enum vr_mode {
        volume_raw          = 0x00,
        volume_color_map
    };
public:
    volume_renderer(const gl::render_device_ptr& device, const math::vec2ui& vp_size);
    virtual ~volume_renderer();

    void                            draw(const gl::render_context_ptr& context,
                                         const volume_data_ptr&        vdata,
                                         const vr_mode                 mode) const;
    void                            update(const gl::render_context_ptr& context,
                                           const gl::camera&             cam);

    bool                            reload_shaders(const gl::render_device_ptr& device);

protected:
    math::vec2ui                    _viewport_size;

    gl::program_ptr                 _program;

    gl::depth_stencil_state_ptr     _dstate;
    gl::blend_state_ptr             _bstate;
    gl::rasterizer_state_ptr        _rstate;
    gl::sampler_state_ptr           _sstate_lin;
    gl::sampler_state_ptr           _sstate_lin_mip;

    gl::camera_uniform_block_ptr    _camera_block;

}; // class volume_renderer

} // namespace data
} // namespace scm

#endif // SCM_LARGE_DATA_VOLUME_RENDERER_H_INCLUDED
