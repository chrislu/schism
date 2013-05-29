
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef IMAGE_APPLICATION_H_INCLUDED
#define IMAGE_APPLICATION_H_INCLUDED

#include <list>
#include <map>
#include <string>
#include <vector>

#include <scm/core/math.h>
#include <scm/core/time/accum_timer.h>
#include <scm/core/time/high_res_timer.h>

#include <scm/gl_core/frame_buffer_objects/frame_buffer_objects_fwd.h>
#include <scm/gl_core/render_device/render_device_fwd.h>
#include <scm/gl_core/shader_objects/shader_objects_fwd.h>
#include <scm/gl_core/state_objects/state_objects_fwd.h>
#include <scm/gl_core/texture_objects/texture_objects_fwd.h>

#include <scm/gl_util/primitives/primitives_fwd.h>
#include <scm/gl_util/utilities/utilities_fwd.h>
#include <scm/gl_util/viewer/viewer_fwd.h>

#include <gui_support/viewer_window.h>

class QHBoxLayout;
class QMenuBar;

namespace scm {

namespace gl {
namespace gui {
} // namespace gui
} // namespace gl

namespace data {

class quad_highlight;

class application_window : public gl::gui::viewer_window
{
    Q_OBJECT

public:
    typedef scm::time::accum_timer<scm::time::high_res_timer>  accum_timer_type;

public:
    application_window(const math::vec2ui&                      vp_size,
                       const gl::viewer::viewer_attributes&     view_attrib = gl::viewer::viewer_attributes(),
                       const gl::wm::context::attribute_desc&   ctx_attrib = gl::wm::context::default_attributes(),
                       const gl::wm::surface::format_desc&      win_fmt = gl::wm::surface::default_format());
    virtual ~application_window();

protected:
    bool                            initialize();
    bool                            init_renderer();
    void                            shutdown();

    void                            update(const gl::render_device_ptr& device,
                                           const gl::render_context_ptr& context);
    void                            display(const gl::render_context_ptr& context);
    void                            reshape(const gl::render_device_ptr& device,
                                            const gl::render_context_ptr& context,
                                            int w, int h);

    void                            keyboard_input(int k, bool state);
    void                            mouse_double_click(gl::viewer::mouse_button b, int x, int y);
    void                            mouse_press(gl::viewer::mouse_button b, int x, int y);
    void                            mouse_release(gl::viewer::mouse_button b, int x, int y);
    void                            mouse_move(gl::viewer::mouse_button b, int x, int y);

protected:
    // GUI

private:

private:
    //
    math::vec2ui                    _viewport_size;

    // state objects
    gl::blend_state_ptr             _bstate_off;
    gl::depth_stencil_state_ptr     _dstate_less;
    gl::rasterizer_state_ptr        _rstate_cback;
    gl::sampler_state_ptr           _sstate_linear;
    gl::sampler_state_ptr           _sstate_nearest;

    gl::camera_uniform_block_ptr    _camera_block;

    // texturing
    gl::texture_2d_ptr              _texture;
    gl::texture_2d_ptr              _texture_uint_view;

    // geometry
    gl::wavefront_obj_geometry_ptr  _model_geometry;

    // shader programs
    gl::program_ptr                 _shader_prog;

}; // class application_window


} // namespace data
} // namespace scm

#endif // IMAGE_APPLICATION_H_INCLUDED
