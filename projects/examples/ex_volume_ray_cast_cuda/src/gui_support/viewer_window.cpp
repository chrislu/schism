
#include "viewer_window.h"

#include <exception>
#include <stdexcept>
#include <iostream>
#include <limits>
#include <string>
#include <sstream>

#include <boost/bind.hpp>

#include <QtGui/QCloseEvent>
#include <QtGui/QMouseEvent>

#include <QtWidgets/QApplication>
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QVBoxLayout>

#include <scm/log.h>
#include <scm/core/math.h>

#include <scm/gl_core/math.h>
#include <scm/gl_core/render_device.h>

#include <gui_support/viewer_widget.h>

namespace {

const scm::math::vec3f diffuse(0.7f, 0.7f, 0.7f);
const scm::math::vec3f specular(0.2f, 0.7f, 0.9f);
const scm::math::vec3f ambient(0.1f, 0.1f, 0.1f);
const scm::math::vec3f position(1, 1, 1);

} // namespace 

namespace scm {
namespace gl {
namespace gui {

viewer_window::viewer_window(const math::vec2ui&                    vp_size,
                             const gl::viewer::viewer_attributes&   view_attrib,
                             const gl::wm::context::attribute_desc& ctx_attrib,
                             const gl::wm::surface::format_desc&    win_fmt)
  : QDialog(0, Qt::Dialog | Qt::MSWindowsFixedSizeDialogHint)
  , _viewport_size(vp_size)
{
    // because we have no parent
    setAttribute(Qt::WA_DeleteOnClose);

    if (!init_viewer(view_attrib, ctx_attrib, win_fmt)) {
        std::stringstream msg;
        msg << "viewer_window::viewer_window(): error initializing application.";
        err() << msg.str() << log::end;
        throw (std::runtime_error(msg.str()));
    }

    if (!init_gui()) {
        std::stringstream msg;
        msg << "viewer_window::viewer_window(): error initializing GUI.";
        err() << msg.str() << log::end;
        throw (std::runtime_error(msg.str()));
    }

}

viewer_window::~viewer_window()
{
    std::cout << "viewer_window::~viewer_window(): bye, bye..." << std::endl;
}

bool
viewer_window::init_viewer(const gl::viewer::viewer_attributes&     view_attrib,
                           const gl::wm::context::attribute_desc&   ctx_attrib,
                           const gl::wm::surface::format_desc&      win_fmt)
{
    //_viewer_widget      = new gl::gui::viewer_widget(this, format);
    _viewer_widget      = new gl::gui::viewer_widget(this, view_attrib, ctx_attrib, win_fmt);
    _viewer             = _viewer_widget->base_viewer();

    _viewer->render_pre_frame_update_func(boost::bind(&viewer_window::pre_frame_update, this, _1, _2));
    _viewer->render_post_frame_update_func(boost::bind(&viewer_window::post_frame_update, this, _1, _2));
    _viewer->render_resize_func(boost::bind(&viewer_window::reshape, this, _1, _2, _3, _4));
    _viewer->render_display_scene_func(boost::bind(&viewer_window::display_scene, this, _1));
    _viewer->render_display_gui_func(boost::bind(&viewer_window::display_gui, this, _1));

    _viewer->keyboard_input_func(boost::bind(&viewer_window::keyboard_input, this, _1, _2, _3));
    _viewer->mouse_double_click_func(boost::bind(&viewer_window::mouse_double_click, this, _1, _2, _3));
    _viewer->mouse_press_func(boost::bind(&viewer_window::mouse_press, this, _1, _2, _3));
    _viewer->mouse_release_func(boost::bind(&viewer_window::mouse_release, this, _1, _2, _3));
    _viewer->mouse_move_func(boost::bind(&viewer_window::mouse_move, this, _1, _2, _3));

    _viewer_widget->setFixedSize(_viewport_size.x, _viewport_size.y);

    return (true);
}

bool
viewer_window::init_gui()
{
    _main_layout        = new QHBoxLayout(this);
    _main_menubar       = new QMenuBar(this);
    _main_layout->setContentsMargins(0, 0, 0, 0);
    _main_layout->setMenuBar(_main_menubar);
    _main_layout->addWidget(_viewer_widget);

    // initialize menus ///////////////////////////////////////////////////////////////////////////
    // viewer menu
    QMenu*          viewer_menu    = new QMenu("Viewer", this);

    QAction*        toggle_vsync    = viewer_menu->addAction("vsync");
    QAction*        toggle_frmtme   = viewer_menu->addAction("show frame time");
    QAction*        toggle_fllscn   = viewer_menu->addAction("full screen");
    QAction*        toggle_aupdte   = viewer_menu->addAction("screen auto update");
                                      viewer_menu->addSeparator();
    QAction*        close_app       = viewer_menu->addAction("Exit");

    toggle_vsync->setCheckable(true);
    toggle_vsync->setChecked(_viewer->settings()._vsync);
    toggle_frmtme->setCheckable(true);
    toggle_frmtme->setChecked(_viewer->settings()._show_frame_times);
    toggle_fllscn->setCheckable(true);
    toggle_fllscn->setChecked(_viewer->settings()._full_screen);
    toggle_aupdte->setCheckable(true);
    toggle_aupdte->setChecked(_viewer_widget->auto_update());

    connect(toggle_vsync,  SIGNAL(toggled(bool)), this, SLOT(switch_vsync_mode(bool)));
    connect(toggle_frmtme, SIGNAL(toggled(bool)), this, SLOT(switch_frame_time_display(bool)));
    connect(toggle_fllscn, SIGNAL(toggled(bool)), this, SLOT(switch_full_screen_mode(bool)));
    connect(toggle_aupdte, SIGNAL(toggled(bool)), this, SLOT(switch_auto_update_display(bool)));
    connect(close_app, SIGNAL(triggered()), this, SLOT(close_program()));

    _main_menubar->addMenu(viewer_menu);

    return (true);
}

void
viewer_window::pre_frame_update(const gl::render_device_ptr& device,
                                const gl::render_context_ptr& context)
{
}

void
viewer_window::post_frame_update(const gl::render_device_ptr& device,
                                 const gl::render_context_ptr& context)
{
}

//void
//viewer_window::display_scene(const gl::render_context_ptr& context)
//{
//}

void
viewer_window::display_gui(const gl::render_context_ptr& context)
{
}

void
viewer_window::reshape(const gl::render_device_ptr& device,
                            const gl::render_context_ptr& context,
                            int w, int h)
{
    // set the new viewport into which now will be rendered
    using namespace scm::gl;
    using namespace scm::math;

    _viewport_size = vec2ui(w, h);
    context->set_viewport(viewport(vec2ui(0, 0), _viewport_size));
    _viewer->main_camera().projection_perspective(60.f, float(w)/float(h), 0.01f, 10.0f);
}


void
viewer_window::keyboard_input(int k, bool state, scm::uint32 mod)
{
    if (state) { // only fire on key down
        switch(k) {
            case Qt::Key_M:
                if (mod & gl::viewer::km_control_modifier) {
                    if (_main_menubar->isVisible()) {
                        _main_menubar->hide();
                    }
                    else {
                        _main_menubar->show();
                    }
                }
                break;
            case Qt::Key_F:
                if (mod & gl::viewer::km_control_modifier) {
                    switch_full_screen_mode(!_viewer->settings()._full_screen);
                }
                break;
            case Qt::Key_Escape:    close_program();break;
            default:;
        }
    }
    //switch(k) { // key toggles
    //    default:;
    //}
}

void
viewer_window::mouse_double_click(gl::viewer::mouse_button b, int x, int y)
{
    using namespace scm::gl;
    using namespace scm::math;
}

void
viewer_window::mouse_press(gl::viewer::mouse_button b, int x, int y)
{
}

void
viewer_window::mouse_release(gl::viewer::mouse_button b, int x, int y)
{
}

void
viewer_window::mouse_move(gl::viewer::mouse_button b, int x, int y)
{
}

void
viewer_window::closeEvent(QCloseEvent* e)
{
    out() << "viewer_window::closeEvent()" << log::end;
}

void
viewer_window::close_program()
{
    if ( _viewer->settings()._full_screen) {
        switch_full_screen_mode(false);
    }

    hide();
    close();
}

void
viewer_window::switch_vsync_mode(bool c)
{
    _viewer->settings()._vsync = c;
}

void
viewer_window::switch_full_screen_mode(bool f)
{
    if (f) {
        _main_menubar->hide();
        this->showFullScreen();
    }
    else {
        _main_menubar->show();
        this->showNormal();
    }

    _viewer->settings()._full_screen = f;
}

void
viewer_window::switch_frame_time_display(bool c)
{
    _viewer->settings()._show_frame_times = c;
}

void
viewer_window::switch_auto_update_display(bool a)
{
    _viewer_widget->auto_update(a);
}

} // namespace gui
} // namespace gl
} // namespace scm
