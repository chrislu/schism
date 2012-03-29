
#include "viewer_widget.h"

#include <exception>
#include <stdexcept>
#include <string>
#include <sstream>

#include <QtGui/QCloseEvent>
#include <QtGui/QKeyEvent>
#include <QtGui/QMouseEvent>
#include <QtGui/QResizeEvent>
#include <QtCore/QTimer>

#include <scm/log.h>
#include <scm/core/math/math.h>

#include <scm/gl_util/viewer/viewer.h>

#include <scm/gl_core/window_management/window.h>


namespace scm {
namespace gl {
namespace gui {

namespace detail {

viewer::mouse_button
mouse_button(Qt::MouseButton button)
{
    viewer::mouse_button ret = viewer::no_button;
    switch (button) {
        case Qt::LeftButton:    ret = viewer::left_button;    break;
        case Qt::MidButton:     ret = viewer::middle_button;  break;
        case Qt::RightButton:   ret = viewer::right_button;   break;
    }
    return (ret);
}

scm::uint32
qt_to_key_modifier(Qt::KeyboardModifiers m)
{
    scm::uint32 r = 0u;

    if (m & Qt::ShiftModifier)   r |= viewer::km_shift_modifier;
    if (m & Qt::ControlModifier) r |= viewer::km_control_modifier;
    if (m & Qt::AltModifier)     r |= viewer::km_alt_modifier;
    if (m & Qt::MetaModifier)    r |= viewer::km_meta_modifer;

    return r;
}

} // namespace detail

viewer_widget::viewer_widget(QWidget* parent,
                             const viewer::viewer_attributes&     view_attrib,
                             const wm::context::attribute_desc&   ctx_attrib,
                             const wm::surface::format_desc&      win_fmt)
  : QWidget(parent, Qt::MSWindowsOwnDC)
{
    setMouseTracking(true);
    setFocusPolicy(Qt::StrongFocus);

    setAttribute(Qt::WA_NativeWindow, true);
    //setAttribute(Qt::WA_DontCreateNativeAncestors, false);
    setAttribute(Qt::WA_PaintOnScreen, true); // disables qt double buffering (seems X11 only since qt4.5, ...)
    //setAttribute(Qt::WA_OpaquePaintEvent, true);
    setAttribute(Qt::WA_NoSystemBackground, true);
    //setAttribute(Qt::WA_ForceUpdatesDisabled, true);
    //setAttribute(Qt::WA_PaintUnclipped, true);
    setAutoFillBackground(false);

    try {
        _viewer = make_shared<viewer>(math::vec2ui(100, 100), parent->winId(), view_attrib, ctx_attrib, win_fmt);

        // ok set the native window as this widgets window...and hold thumbs
        this->create(_viewer->window()->window_handle(), true, true);
    }
    catch(std::exception& e) {
        std::stringstream msg;
        msg << "viewer_widget::viewer_widget(): "
            << "unable to create viewer (evoking error: " << e.what() << ").";
        err() << msg.str();
        throw (std::runtime_error(msg.str()));
    }
}

viewer_widget::~viewer_widget()
{
    _viewer.reset();
}

QPaintEngine*
viewer_widget::paintEngine() const
{
    return (0);//(QWidget::paintEngine());
}

void
viewer_widget::keyPressEvent(QKeyEvent* key_event)
{
    _viewer->send_keyboard_input(key_event->key(), true, detail::qt_to_key_modifier(key_event->modifiers()));
    //this->update();
}

void
viewer_widget::keyReleaseEvent(QKeyEvent* key_event)
{
    _viewer->send_keyboard_input(key_event->key(), false, detail::qt_to_key_modifier(key_event->modifiers()));
    //this->update();
}

void
viewer_widget::mouseDoubleClickEvent(QMouseEvent* mouse_event)
{
    //out() << "mouseDoubleClickEvent" << log::end;
    _viewer->send_mouse_double_click(detail::mouse_button(mouse_event->button()),
                                     mouse_event->x(),
                                     mouse_event->y());
    //this->update();
}

void
viewer_widget::mouseMoveEvent(QMouseEvent* mouse_event)
{
    //out() << "mouseMoveEvent" << log::end;
    _viewer->send_mouse_move(detail::mouse_button(mouse_event->button()),
                             mouse_event->x(),
                             mouse_event->y());
    //this->update();
}

void
viewer_widget::mousePressEvent(QMouseEvent* mouse_event)
{
    //out() << "mousePressEvent" << log::end;
    _viewer->send_mouse_press(detail::mouse_button(mouse_event->button()),
                              mouse_event->x(),
                              mouse_event->y());
    //this->update();
}

void
viewer_widget::mouseReleaseEvent(QMouseEvent* mouse_event)
{
    //out() << "mouseReleaseEvent" << log::end;
    _viewer->send_mouse_release(detail::mouse_button(mouse_event->button()),
                                mouse_event->x(),
                                mouse_event->y());
    //this->update();
}

bool
viewer_widget::event(QEvent* ev)
{
    //if (ev->type() == QEvent::Paint) {
    //    //force_display();
    //    ev->accept();
    //    return (true);
    //}
    //else
    {
        return (QWidget::event(ev));
    }
}

void
viewer_widget::paintEvent(QPaintEvent* paint_event)
{
    paint_event->accept();

    _viewer->window_context()->make_current(_viewer->window());
    
    _viewer->send_render_update();
    _viewer->send_render_display();

    this->update();
}

void
viewer_widget::resizeEvent(QResizeEvent* resize_event)
{
    _viewer->send_render_reshape(resize_event->size().width(),
                                 resize_event->size().height());
}

const shared_ptr<viewer>&
viewer_widget::base_viewer() const
{
    return (_viewer);
}

} // namespace gui
} // namespace gl_classic
} // namespace scm
