
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "color_picker_button.h"

#include <QtGui/QColorDialog>
#include <QtGui/QPainter>

namespace scm {
namespace gui {

color_button::color_button(QWidget* parent)
  : QPushButton(parent),
    _current_color(Qt::black)
{
    QObject::connect(this,              SIGNAL(clicked()),
                     this,              SLOT(open_color_dialog()));
}

color_button::~color_button()
{
}

const QColor& color_button::current_color() const
{
    return (_current_color);
}

void color_button::current_color(const QColor& color)
{
    _current_color = color;

    /*emit*/ current_color_changed(_current_color);

    update();
}

void color_button::open_color_dialog()
{
    QColor      new_color = QColorDialog::getColor(_current_color, this);

    if (new_color.isValid()) {
        current_color(new_color);
    }
}

void color_button::paintEvent(QPaintEvent* paint_event)
{
    QPushButton::paintEvent(paint_event);

    QPainter        painter(this);
    QBrush          fill_brush(_current_color);

    fill_brush.setStyle(Qt::SolidPattern);

    painter.fillRect(this->contentsRect(), fill_brush);
}

} // namespace gui
} // namespace scm
