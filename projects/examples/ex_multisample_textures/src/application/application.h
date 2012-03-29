
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef IMAGE_APPLICATION_H_INCLUDED
#define IMAGE_APPLICATION_H_INCLUDED

class QAction;

#include <list>
#include <map>
#include <string>
#include <vector>

#include <QtGui/QDialog>

#include <scm/core/math.h>

#include <scm/gl_core/gl_core_fwd.h>
#include <scm/gl_util/primitives/primitives_fwd.h>

#include <gui_support/viewer_window.h>

namespace scm {
namespace gl {
namespace gui {

class quad_highlight;

class application_window : public viewer_window
{
    Q_OBJECT

public:
    application_window(const math::vec2ui&                      vp_size,
                       const gl::viewer::viewer_attributes&     view_attrib = gl::viewer::viewer_attributes(),
                       const gl::wm::context::attribute_desc&   ctx_attrib = gl::wm::context::default_attributes(),
                       const gl::wm::surface::format_desc&      win_fmt = gl::wm::surface::default_format());
    virtual ~application_window();

protected:
    void                            shutdown();
    bool                            init_renderer();

    bool                            reload_shaders(const gl::render_device_ptr& device);

    void                            update(const gl::render_device_ptr& device,
                                           const gl::render_context_ptr& context);
    void                            display(const gl::render_context_ptr& context);
    void                            reshape(const gl::render_device_ptr& device,
                                            const gl::render_context_ptr& context,
                                            int w, int h);

    void                            keyboard_input(int k, bool state, scm::uint32 mod);
    void                            mouse_double_click(gl::viewer::mouse_button b, int x, int y);
    void                            mouse_press(gl::viewer::mouse_button b, int x, int y);
    void                            mouse_release(gl::viewer::mouse_button b, int x, int y);
    void                            mouse_move(gl::viewer::mouse_button b, int x, int y);

private Q_SLOTS:

protected:
    coordinate_cross_ptr            _coord_cross;

    int                             _ms_samples;
    gl::texture_2d_ptr              _ms_color;
    gl::texture_2d_ptr              _ms_depth;
    gl::frame_buffer_ptr            _ms_fbo;

    gl::program_ptr                 _resolve_prog;
    gl::rasterizer_state_ptr        _resolve_rstate;
    gl::blend_state_ptr             _resolve_bstate;
    gl::depth_stencil_state_ptr     _resolve_dstate;
    gl::sampler_state_ptr           _resolve_sstate;
    gl::quad_geometry_ptr           _resolve_quad;
    gl::fullscreen_triangle_ptr     _resolve_tri;

    gl::program_ptr                 _model_prog;
    gl::rasterizer_state_ptr        _model_rstate;
    gl::blend_state_ptr             _model_bstate;
    gl::depth_stencil_state_ptr     _model_dstate;
    gl::sampler_state_ptr           _model_sstate;

    gl::wavefront_obj_geometry_ptr  _model;
    gl::texture_2d_ptr              _model_normal_map;
    gl::texture_2d_ptr              _model_diffuse_map;
    gl::texture_2d_ptr              _env_map;
    gl::texture_2d_ptr              _env_map_diff;

    bool                            _animation_enabled;
    float                           _rotation_angle;


}; // class application_window

} // namespace gui
} // namespace gl
} // namespace scm

#endif // IMAGE_APPLICATION_H_INCLUDED
