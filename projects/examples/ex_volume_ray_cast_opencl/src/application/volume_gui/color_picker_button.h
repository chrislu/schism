
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef COLOR_PICKER_BUTTON_H_INCLUDED
#define COLOR_PICKER_BUTTON_H_INCLUDED

#include <QtGui/QColor>
#include <QtGui/QPushButton>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gui {

class color_button : public QPushButton
{
    Q_OBJECT

public:
    color_button(QWidget* /*parent*/);
    virtual ~color_button();

    const QColor&       current_color() const;

protected:
     void               paintEvent(QPaintEvent*         /*paint_event*/);

private:
    QColor              _current_color;

public Q_SLOTS:
    void                current_color(const QColor& color);

private Q_SLOTS:
    void                open_color_dialog();

Q_SIGNALS:
    void                current_color_changed(const QColor& color);

}; // class color_button

} // namespace gui
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // COLOR_PICKER_BUTTON_H_INCLUDED
