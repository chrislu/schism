
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "color_map_editor.h"

#include <algorithm>
#include <iostream>

#include <scm/core/math/math.h>
#include <scm/core/utilities/foreach.h>

#include <QtGui/QBrush>
#include <QtGui/QLinearGradient>
#include <QtGui/QMouseEvent>
#include <QtGui/QLinearGradient>
#include <QtGui/QPainter>
#include <QtGui/QPainterPath>
#include <QtGui/QPaintEvent>
#include <QtGui/QResizeEvent>

namespace {

inline QRectF bounding_rect(const QPointF& pnt, float anchor_size)
{
    return (QRectF(pnt.x() - anchor_size * 0.5f,
                   pnt.y() - anchor_size * 0.5f,
                   anchor_size,
                   anchor_size));
}

inline QPainterPath anchor_shape(float value, float anchor_size, const QRect& area)
{
    QPainterPath    path;

    path.moveTo(area.width() * value,                       area.y() + anchor_size);
    path.lineTo(area.width() * value + anchor_size * 0.5f,  area.y());
    path.lineTo(area.width() * value - anchor_size * 0.5f,  area.y());
    path.lineTo(area.width() * value,                       area.y() + anchor_size);

    path.moveTo(area.width() * value,                       area.y() + area.height() - anchor_size);
    path.lineTo(area.width() * value + anchor_size * 0.5f,  area.y() + area.height());
    path.lineTo(area.width() * value - anchor_size * 0.5f,  area.y() + area.height());
    path.lineTo(area.width() * value,                       area.y() + area.height() - anchor_size);

    return (path);
}

struct less_x
{
    bool operator()(const scm::gui::color_map_anchor& lhs,
                    const scm::gui::color_map_anchor& rhs) const
    {
        return (lhs.point().x() < rhs.point().x());
    }
}; // struct less_x

struct less_y
{
    bool operator()(const scm::gui::color_map_anchor& lhs,
                    const scm::gui::color_map_anchor& rhs) const
    {
        return (lhs.point().y() < rhs.point().y());
    }
}; // struct less_y



inline scm::gui::color_map::stop_type horizontal_stop(const QPointF& pnt,
                                                                             const QRectF&  range)
{
    return (scm::gui::color_map::stop_type(static_cast<float>((pnt.x() - range.x()) / range.width()),
                                                                   scm::gui::color_map::type::result_type(0)));
}

inline scm::gui::color_map::stop_type vertical_stop(const QPointF& pnt,
                                                                            const QRectF&  range)
{
    return (scm::gui::color_map::stop_type(static_cast<float>(1.0f - (pnt.y() - range.y()) / range.height()),
                                                                   scm::gui::color_map::type::result_type(0)));
}

} // namespace

namespace scm {
namespace gui {

color_map_anchor::color_map_anchor(const QPointF&                               pnt,
                                                             const color_map::stop_type&  stop)
 : _point(pnt),
   _stop(stop)
{
}

const QPointF& color_map_anchor::point() const
{
    return (_point);
}

const color_map::stop_type& color_map_anchor::stop() const
{
    return (_stop);
}

void color_map_anchor::update(const QPointF&                               pnt,
                                           const color_map::stop_type&  stop)
{
    _point = pnt;
    _stop  = stop;
}


color_map_editor::color_map_editor(QWidget* parent, Qt::WindowFlags  flags)
  : QFrame(parent, flags),
    _anchor_point_size(10.0f),
    _orientation(horizontal),
    _active_anchor(_anchors.end()),
    _selected_anchor(_anchors.end())
{
    _anchor_brush = QBrush(QColor(64, 64, 64, 170));
    _anchor_pen.setColor(Qt::black);
    _anchor_pen.setStyle(Qt::SolidLine);

    _anchor_highlight_brush = QBrush(QColor(190, 190, 190, 170));
    _anchor_highlight_pen.setColor(Qt::red);
    _anchor_highlight_pen.setStyle(Qt::SolidLine);

    _line_pen.setColor(Qt::black);
    _line_pen.setStyle(Qt::SolidLine);
}

color_map_editor::~color_map_editor()
{
}

void
color_map_editor::set_function(const shared_ptr<function_type>& fun)
{
    _transfer_function = fun;
    reinitialize_function();
}

float color_map_editor::anchor_point_size() const
{
    return (_anchor_point_size);
}

void color_map_editor::anchor_point_size(float psize)
{
    _anchor_point_size = psize;
}

color_map_editor::orientation_t color_map_editor::orientation() const
{
    return (_orientation);
}

void color_map_editor::orientation(orientation_t ori)
{
    _orientation = ori;
    reinitialize_function();
}

void color_map_editor::paintEvent(QPaintEvent* paint_event)
{
    // paint the frame
    QFrame::paintEvent(paint_event);

    QPainter    painter(this);

    // fill background with gradient
    QLinearGradient gradient(0, 0, 0, 0);

    switch (_orientation) {
        case horizontal:
            gradient.setFinalStop(this->contentsRect().width(), 0);
            break;
        case vertical:
            gradient.setFinalStop(0, this->contentsRect().height());
            break;
    }

    QBrush          brush(Qt::black);
    brush.setStyle(Qt::SolidPattern);

    painter.fillRect(this->contentsRect(), brush);

    painter.setRenderHint(QPainter::Antialiasing);

    // draw gradient
    QLinearGradient color_grad(0, 0, 0, 0);

    if (_transfer_function) {
        if (_transfer_function->num_stops() > 1) {

            QRectF  fill_rect(this->contentsRect());

            float start     = _transfer_function->stops_begin()->first;
            float end       = boost::prior(_transfer_function->stops_end())->first;

            switch (_orientation) {
                case horizontal:
                    fill_rect.setX(this->contentsRect().x() + start * this->contentsRect().width());
                    fill_rect.setWidth((end - start) * this->contentsRect().width());

                    color_grad.setFinalStop(this->contentsRect().width(), 0);
                    break;
                case vertical:
                    fill_rect.setY(this->contentsRect().y() + (1.0f - end) * this->contentsRect().height());
                    fill_rect.setHeight((end - start) * this->contentsRect().height());

                    color_grad.setFinalStop(0, 0);
                    color_grad.setStart(0, this->contentsRect().height());
                    break;
            }
            color_grad.setStops(_gradient_stops);

            QBrush          grad_brush(color_grad);
            grad_brush.setStyle(Qt::LinearGradientPattern);

            painter.fillRect(fill_rect, grad_brush);
        }

        // draw edit anchors
        for (anchor_container::iterator anchor = _anchors.begin(); anchor != _anchors.end(); ++anchor) {
            if (anchor != _selected_anchor) {
                painter.setPen(_anchor_pen);
                painter.setBrush(_anchor_brush);
                //QColor          stop_col;
                //stop_col.setRgbF(anchor->stop().second.x,
                //                 anchor->stop().second.y,
                //                 anchor->stop().second.z);
                //painter.setBrush(stop_col);

                painter.drawPath(anchor_shape(anchor->stop().first, _anchor_point_size, this->contentsRect()));
            }
        }
        if (_selected_anchor != _anchors.end()) {
            painter.setPen(_anchor_highlight_pen);
            painter.setBrush(_anchor_highlight_brush);
            //QColor          stop_col;
            //stop_col.setRgbF(_selected_anchor->stop().second.x,
            //                 _selected_anchor->stop().second.y,
            //                 _selected_anchor->stop().second.z);
            //painter.setBrush(stop_col);

            painter.drawPath(anchor_shape(_selected_anchor->stop().first, _anchor_point_size, this->contentsRect()));
        }
    }
}

void color_map_editor::resizeEvent(QResizeEvent* resize_event)
{
    reinitialize_function();
}

void color_map_editor::mouseMoveEvent(QMouseEvent* mouse_event)
{
    if (_transfer_function) {
        if (_active_anchor != _anchors.end()) {
            QPointF                                         new_point;
            color_map::stop_type            new_stop;
            color_map::type::result_type    old_color = _active_anchor->stop().second;
            
            new_point.setX(scm::math::clamp(mouse_event->pos().x(), this->contentsRect().x(), this->contentsRect().x() + this->contentsRect().width()));
            new_point.setY(scm::math::clamp(mouse_event->pos().y(), this->contentsRect().y(), this->contentsRect().y() + this->contentsRect().height()));

            switch (_orientation) {
                case horizontal:
                    new_stop = horizontal_stop(new_point, this->contentsRect());
                    break;
                case vertical:
                    new_stop = vertical_stop(new_point, this->contentsRect());
                    break;
            }

            new_stop.second = old_color;
            color_map::type::insert_return_type  ret = _transfer_function->add_stop(new_stop);

            if (   ret.second == true
                || ret.first->first == _active_anchor->stop().first) {
                
                _transfer_function->del_stop(_active_anchor->stop());
                _transfer_function->add_stop(new_stop);
                _active_anchor->update(new_point, new_stop);

                /*emit*/ function_changed();
                /*emit*/ selected_anchor_changed(_active_anchor->stop());

                update_representation();
                update();
            }
        }
    }
}

void color_map_editor::mousePressEvent(QMouseEvent* mouse_event)
{
    if (_transfer_function) {

        anchor_container::iterator hit_existing_anchor = _anchors.end();

        for (anchor_container::iterator i = _anchors.begin(); i != _anchors.end() && hit_existing_anchor == _anchors.end(); ++i) {
            if (anchor_shape(i->stop().first, _anchor_point_size, this->contentsRect()).contains(mouse_event->pos())) {
                hit_existing_anchor      = i;
            }
        }

        if (mouse_event->button() == Qt::LeftButton) {
            if (hit_existing_anchor != _anchors.end()) {
                _active_anchor = hit_existing_anchor;
            }
            else {
                QPointF                                 new_point;
                color_map::stop_type    new_stop;
                
                new_point.setX(scm::math::clamp(mouse_event->pos().x(), this->contentsRect().x(), this->contentsRect().x() + this->contentsRect().width()));
                new_point.setY(scm::math::clamp(mouse_event->pos().y(), this->contentsRect().y(), this->contentsRect().y() + this->contentsRect().height()));
                
                switch (_orientation) {
                    case horizontal:
                        new_stop = horizontal_stop(new_point, this->contentsRect());
                        break;
                    case vertical:
                        new_stop = vertical_stop(new_point, this->contentsRect());
                        break;
                }

                new_stop.second = (*_transfer_function)[new_stop.first];

                color_map::type::insert_return_type  ret = _transfer_function->add_stop(new_stop);

                if (ret.second == true) {

                    _anchors.push_back(color_map_anchor(new_point, new_stop));
                    _transfer_function->add_stop(new_stop);

                    //std::cout << mouse_event->pos().x() << "x" << mouse_event->pos().y() << " "
                    //          << _anchors.back().stop().first << "x" << _anchors.back().stop().second << std::endl;

                    _active_anchor = boost::prior(_anchors.end());

                    /*emit*/ function_changed();

                    update_representation();
                }

            }
            set_selected_anchor(_active_anchor);
            update();
        }
        else if(mouse_event->button() == Qt::RightButton) {
            if (hit_existing_anchor != _anchors.end()) {
                _transfer_function->del_stop(hit_existing_anchor->stop());
                _anchors.erase(hit_existing_anchor);

                _active_anchor      = _anchors.end();
                set_selected_anchor(_anchors.end());

                /*emit*/ function_changed();

                update_representation();
                update();
            }
        }
    }
}

void color_map_editor::mouseReleaseEvent(QMouseEvent* mouse_event)
{
    _active_anchor = _anchors.end();
}

void color_map_editor::initialize_anchors()
{
    if (_transfer_function) {

        function_type::const_stop_iterator pnts_iter;
        function_type::const_stop_iterator pnts_begin = _transfer_function->stops_begin();
        function_type::const_stop_iterator pnts_end   = _transfer_function->stops_end();

        bool find_selection = _selected_anchor != _anchors.end();
        float sel_stop_val  = find_selection ? _selected_anchor->stop().first : 0.0f;
        
        _anchors.clear();

        for (pnts_iter  = pnts_begin;
             pnts_iter != pnts_end;
             ++pnts_iter) {
            QPointF  new_point;

            switch (_orientation) {
                case horizontal:
                    new_point.setX(this->contentsRect().x() + pnts_iter->first * this->contentsRect().width());
                    new_point.setY(this->contentsRect().height() / 2);
                    break;
                case vertical:
                    new_point.setX(this->contentsRect().width() / 2);
                    new_point.setY(this->contentsRect().y() + (1.0f - pnts_iter->first) * this->contentsRect().height());
                    break;
            }

            _anchors.push_back(color_map_anchor(QPointF(new_point), *pnts_iter));
        }

        set_selected_anchor(_anchors.end());
        _active_anchor   = _anchors.end();

        if (find_selection) {
            for (anchor_container::iterator i = _anchors.begin(); i != _anchors.end(); ++i) {
                if (i->stop().first == sel_stop_val) {
                    _selected_anchor = i;
                }
            }
        }
    }
}

void color_map_editor::update_representation()
{
    anchor_container sort_vec(_anchors);

    switch (_orientation) {
        case horizontal:
            std::sort(sort_vec.begin(), sort_vec.end(), less_x());
            break;
        case vertical:
            std::sort(sort_vec.begin(), sort_vec.end(), less_y());
            break;
    }

    _gradient_stops.clear();
    for (std::size_t i = 0; i < sort_vec.size(); ++i) {
        QColor              grad_col;
        scm::math::vec3f    col = sort_vec.at(i).stop().second;

        grad_col.setRgbF(col.x, col.y, col.z);
        _gradient_stops.push_back(QGradientStop(sort_vec.at(i).stop().first, grad_col));
    }
}

void color_map_editor::set_selected_anchor(anchor_container::iterator anchor)
{
    _selected_anchor = anchor;

    if (_selected_anchor != _anchors.end()) {
        /*emit*/ selected_anchor_changed(_selected_anchor->stop());
    }
    else {
        /*emit*/ anchor_deselected();
    }
}

void color_map_editor::stop_values_changed(const color_map::stop_type& stop)
{
    if (_transfer_function) {
        if (_selected_anchor != _anchors.end()) {

            QPointF  new_point;

            switch (_orientation) {
                case horizontal:
                    new_point.setX(this->contentsRect().x() +         stop.first   * this->contentsRect().width());
                    new_point.setY(this->contentsRect().height() / 2);
                    break;
                case vertical:
                    new_point.setX(this->contentsRect().width() / 2);
                    new_point.setY(this->contentsRect().y() + (1.0f - stop.first) * this->contentsRect().height());
                    break;
            }

            color_map::type::insert_return_type  ret = _transfer_function->add_stop(stop);

            if (   ret.second == true
                || ret.first->first == _selected_anchor->stop().first) {
                
                _transfer_function->del_stop(_selected_anchor->stop());
                _transfer_function->add_stop(stop);
                _selected_anchor->update(new_point, stop);

                /*emit*/ function_changed();

                update_representation();
                update();
            }
        }
    }
}

void color_map_editor::insert_stop(const color_map::stop_type& stop)
{
    if (_transfer_function) {
        color_map::type::insert_return_type  ret = _transfer_function->add_stop(stop);

        set_selected_anchor(_anchors.end());

        /*emit*/ function_changed();

        reinitialize_function();
    }
}

void color_map_editor::remove_current_stop()
{
    if (_transfer_function) {
        if (_selected_anchor != _anchors.end()) {
            _transfer_function->del_stop(_selected_anchor->stop());
            _anchors.erase(_selected_anchor);

            set_selected_anchor(_anchors.end());

            /*emit*/ function_changed();

            update_representation();
            update();
        }
    }
}

void color_map_editor::reinitialize_function()
{
    initialize_anchors();
    update_representation();
    update();
}

} // namespace gui
} // namespace scm
