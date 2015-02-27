
#ifndef SCM_GL_GUI_VIEWER_WINDOW_H_INCLUDED
#define SCM_GL_GUI_VIEWER_WINDOW_H_INCLUDED

class QAction;

#include <list>
#include <map>
#include <string>
#include <vector>

#include <QtWidgets/QDialog>

#include <scm/core/math.h>

#include <scm/gl_core/render_device/render_device_fwd.h>

#include <scm/gl_util/viewer/viewer.h>

class QHBoxLayout;
class QMenuBar;

namespace scm {
namespace gl {
namespace gui {

class viewer_widget;

class viewer_window : public QDialog
{
    Q_OBJECT

public:
public:
    viewer_window(const math::vec2ui&                      vp_size,
                  const gl::viewer::viewer_attributes&     view_attrib = gl::viewer::viewer_attributes(),
                  const gl::wm::context::attribute_desc&   ctx_attrib = gl::wm::context::default_attributes(),
                  const gl::wm::surface::format_desc&      win_fmt = gl::wm::surface::default_format());
    virtual ~viewer_window();

protected:
    virtual void                    pre_frame_update(const gl::render_device_ptr& device,
                                                     const gl::render_context_ptr& context);
    virtual void                    post_frame_update(const gl::render_device_ptr& device,
                                                      const gl::render_context_ptr& context);
    virtual void                    display_scene(const gl::render_context_ptr& context) = 0;
    virtual void                    display_gui(const gl::render_context_ptr& context);
    virtual void                    reshape(const gl::render_device_ptr& device,
                                            const gl::render_context_ptr& context,
                                            int w, int h);

    virtual void                    keyboard_input(int k, bool state, scm::uint32 mod);
    virtual void                    mouse_double_click(gl::viewer::mouse_button b, int x, int y);
    virtual void                    mouse_press(gl::viewer::mouse_button b, int x, int y);
    virtual void                    mouse_release(gl::viewer::mouse_button b, int x, int y);
    virtual void                    mouse_move(gl::viewer::mouse_button b, int x, int y);

    void                            closeEvent(QCloseEvent* e);

protected Q_SLOTS:
    void                            close_program();
    void                            switch_vsync_mode(bool c);
    void                            switch_full_screen_mode(bool f);
    void                            switch_frame_time_display(bool c);
    void                            switch_auto_update_display(bool a);

protected:
    // GUI
    QMenuBar*                       _main_menubar;
    QHBoxLayout*                    _main_layout;
    gl::gui::viewer_widget*         _viewer_widget;
    shared_ptr<gl::viewer>          _viewer;

protected:
    bool                            init_viewer(const gl::viewer::viewer_attributes&     view_attrib,
                                                const gl::wm::context::attribute_desc&   ctx_attrib,
                                                const gl::wm::surface::format_desc&      win_fmt);
    bool                            init_gui();

protected:
    // rendering
    math::vec2ui                    _viewport_size;

}; // class viewer_window

} // namespace gui
} // namespace gl
} // namespace scm

#endif // SCM_GL_GUI_VIEWER_WINDOW_H_INCLUDED
