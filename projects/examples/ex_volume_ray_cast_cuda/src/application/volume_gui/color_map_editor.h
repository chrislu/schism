
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef TRANSFERFUNCTION_EDITOR_1D_COLOR_H_INCLUDED
#define TRANSFERFUNCTION_EDITOR_1D_COLOR_H_INCLUDED

#include <vector>

#include <QtCore/QPointF>
#include <QtGui/QFrame>

#include <QtGui/QBrush>
#include <QtGui/QGradient>
#include <QtGui/QPen>

#include <scm/core/memory.h>
#include <scm/core/math.h>
#include <scm/gl_util/data/analysis/transfer_function/piecewise_function_1d.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gui {

struct color_map
{
    typedef scm::data::piecewise_function_1d<float, scm::math::vec3f>   type;
    typedef type::stop_type                                             stop_type;
}; // struct color_map

class color_map_anchor
{
public:
    color_map_anchor(const QPointF&                /*pnt*/,
                     const color_map::stop_type&   /*stop*/);

    const QPointF&                  point() const;
    void                            update(const QPointF&               /*pnt*/,
                                           const color_map::stop_type&  /*stop*/);
    const color_map::stop_type&     stop() const;

protected:
    QPointF                                 _point;
    color_map::stop_type    _stop;
}; // class color_map_anchor

class color_map_editor : public QFrame
{
    Q_OBJECT

public:
    enum orientation_t{
        horizontal,
        vertical
    };

public:
    typedef color_map::type                function_type;
    typedef std::vector<color_map_anchor>  anchor_container;

public:
    color_map_editor(QWidget*         /*parent*/ = 0,
                     Qt::WindowFlags  /*flags*/  = 0);
    virtual ~color_map_editor();

    void                            set_function(const shared_ptr<function_type>& fun);

    float                           anchor_point_size() const;
    void                            anchor_point_size(float /*psize*/);
    orientation_t                   orientation() const;
    void                            orientation(orientation_t /*ori*/);

protected:
    void                            paintEvent(QPaintEvent*         /*paint_event*/);
    void                            resizeEvent(QResizeEvent*       /*resize_event*/);
    void                            mouseMoveEvent(QMouseEvent*     /*mouse_event*/);
    void                            mousePressEvent(QMouseEvent*    /*mouse_event*/);
    void                            mouseReleaseEvent(QMouseEvent*  /*mouse_event*/);

    void                            initialize_anchors();
    void                            update_representation();

private:
    void                            set_selected_anchor(anchor_container::iterator /*anchor*/);

private:
    shared_ptr<function_type>       _transfer_function;

    anchor_container                _anchors;
    QGradientStops                  _gradient_stops;

    anchor_container::iterator      _active_anchor;
    anchor_container::iterator      _selected_anchor;

    orientation_t                   _orientation;

    float                           _anchor_point_size;

    QBrush                          _anchor_brush;
    QPen                            _anchor_pen;
    QBrush                          _anchor_highlight_brush;
    QPen                            _anchor_highlight_pen;

    QPen                            _line_pen;

public Q_SLOTS:
    void                            stop_values_changed(const color_map::stop_type& /*stop*/);
    void                            insert_stop(const color_map::stop_type& /*stop*/);
    void                            remove_current_stop();
    void                            reinitialize_function();

Q_SIGNALS:
     void                           selected_anchor_changed(const color_map::stop_type& /*stop*/);
     void                           anchor_deselected();
     void                           function_changed();

}; // class color_map_editor

} // namespace gui
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // TRANSFERFUNCTION_EDITOR_1D_COLOR_H_INCLUDED
