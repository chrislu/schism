
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef APPLICATION_H_INCLUDED
#define APPLICATION_H_INCLUDED

class QAction;

#include <list>
#include <map>
#include <string>
#include <vector>

#include <QtGui/QDialog>

#include <cuda_runtime.h>

#include <scm/core/math.h>

#include <scm/gl_util/font/font_fwd.h>
#include <scm/gl_util/primitives/primitives_fwd.h>
#include <scm/gl_util/utilities/utilities_fwd.h>

#include <gui_support/viewer_window.h>

#include <renderer/renderer_fwd.h>

namespace scm {
namespace gl {
namespace gui {
} // namespace gui
} // namespace gl

namespace gui {

class volume_data_dialog;

} // namespace gui

namespace data {

class application_window : public gl::gui::viewer_window
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
    void                            open_volume_data_dialog();
    void                            open_volume();

protected:
    bool                            _show_raw;

    math::vec2ui                    _viewport_size;
    gl::coordinate_cross_ptr        _coord_cross;

    volume_data_ptr                 _volume_data;
    volume_renderer_ptr             _volume_renderer;
    cuda_volume_data_ptr            _volume_data_cuda;
    cuda_volume_renderer_ptr        _volume_renderer_cuda;

    bool                            _use_opencl_renderer;

    gl::geometry_highlight_ptr      _volume_highlight;

    // text rendering
    gl::text_renderer_ptr           _text_renderer;
    gl::text_ptr                    _output_text;

    gui::volume_data_dialog*        _volume_data_dialog;

    // cuda
    cudaStream_t                    _cuda_stream;

}; // class application_window

} // namespace data
} // namespace scm

#endif // APPLICATION_H_INCLUDED
