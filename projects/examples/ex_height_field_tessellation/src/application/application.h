
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef HEIGHT_FIELD_TESSELLATION_APPLICATION_H_INCLUDED
#define HEIGHT_FIELD_TESSELLATION_APPLICATION_H_INCLUDED

#include <string>
#include <vector>

#include <scm/core/math.h>

#include <scm/gl_core/render_device/render_device_fwd.h>

#include <gui_support/viewer_window.h>
#include <renderers/renderers_fwd.h>

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

protected:
    typedef std::vector<height_field_data_ptr>  hf_ptr_container;

public:
    application_window(const std::string&                       input_file,
                       const math::vec2ui&                      vp_size,
                       const gl::viewer::viewer_attributes&     view_attrib = gl::viewer::viewer_attributes(),
                       const gl::wm::context::attribute_desc&   ctx_attrib = gl::wm::context::default_attributes(),
                       const gl::wm::surface::format_desc&      win_fmt = gl::wm::surface::default_format());
    virtual ~application_window();

protected:
    void                            shutdown();
    bool                            init_renderer();

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
    // 
    std::string                     _input_file;

    // height field renderer
    hf_ptr_container                _height_fields;
    height_field_tessellator_ptr    _height_field_renderer;

    // interaction
    height_field_data_ptr           _hf_mouse_over;
    gl::geometry_highlight_ptr      _hf_mouse_over_highlight;

    bool                            _hf_draw_wireframe;
    bool                            _hf_draw_quad_mesh;

    const height_field_data_ptr     pick_box_instance(int x, int y) const;
    const height_field_data_ptr     pick_box_instance(int x, int y, math::vec3f& out_hit) const;

}; // class application_window

} // namespace data
} // namespace scm

#endif // HEIGHT_FIELD_TESSELLATION_APPLICATION_H_INCLUDED
