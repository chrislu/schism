
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_IMAGE_GUI_SUPPORT_VIEWER_WIDGET_H_INCLUDED
#define SCM_IMAGE_GUI_SUPPORT_VIEWER_WIDGET_H_INCLUDED

#include <QtGui/QWidget>

#include <scm/core/pointer_types.h>

//#include <scm/gl_util/render_context/context_format.h>
#include <scm/gl_util/viewer/viewer.h>
#include <scm/gl_core/window_management/wm_fwd.h>
#include <scm/gl_core/window_management/surface.h>
#include <scm/gl_core/window_management/context.h>

class QPaintEngine;

namespace scm {
namespace gl {
namespace gui {

class viewer_widget : public QWidget
{
    Q_OBJECT

public:
    viewer_widget(QWidget*                             parent = 0,
                  const viewer::viewer_attributes&     view_attrib = viewer::viewer_attributes(),
                  const wm::context::attribute_desc&   ctx_attrib = wm::context::default_attributes(),
                  const wm::surface::format_desc&      win_fmt = wm::surface::default_format());
    virtual ~viewer_widget();

    QPaintEngine*               paintEngine() const;

    const shared_ptr<viewer>&   base_viewer() const;

protected:
    void                        mouseDoubleClickEvent(QMouseEvent* mouse_event);
    void                        mousePressEvent(QMouseEvent* mouse_event);
    void                        mouseMoveEvent(QMouseEvent* mouse_event);
    void                        mouseReleaseEvent(QMouseEvent* mouse_event);
    void                        keyPressEvent(QKeyEvent* key_event);
    void                        keyReleaseEvent(QKeyEvent* key_event);

    bool                        event(QEvent* ev);
    void                        paintEvent(QPaintEvent* paint_event);
    void                        resizeEvent(QResizeEvent* resize_event);

protected:
    shared_ptr<viewer>          _viewer;

}; // class viewer_widget

} // namespace gui
} // namespace gl
} // namespace scm

#endif // SCM_IMAGE_GUI_SUPPORT_VIEWER_WIDGET_H_INCLUDED
