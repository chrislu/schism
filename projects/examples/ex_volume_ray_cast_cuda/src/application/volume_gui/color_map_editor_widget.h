
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef TRANSFERFUNCTION_EDITOR_1D_COLOR_WIDGET_H_INCLUDED
#define TRANSFERFUNCTION_EDITOR_1D_COLOR_WIDGET_H_INCLUDED

#include <vector>

#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QWidget>

#include <scm/core/memory.h>

#include <application/volume_gui/color_map_editor.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gui {

class color_button;

class color_map_editor_widget : public QWidget
{
    Q_OBJECT

public:
    color_map_editor_widget(QWidget*                            /*parent*/ = 0,
                            Qt::WindowFlags                     /*flags*/  = 0);
    virtual ~color_map_editor_widget();

    void                                set_function(const shared_ptr<color_map::type>& fun);

protected:
    color_map_editor*                   _editor;

    QDoubleSpinBox*                     _value;
    color_button*                       _mapping_button;

    QPushButton*                        _insert_button;
    QPushButton*                        _remove_button;

    color_map::type::stop_type          retrieve_stop() const;

public Q_SLOTS:
    void                                reinitialize_function();

protected Q_SLOTS:
    void                                set_current_stop_values(const color_map::stop_type& /*stop*/);
    void                                stop_values_changed();
    void                                stop_value_changed(double /*value*/);
    void                                stop_value_changed(const QColor& /*value*/);
    void                                mapping_button_pressed();
    void                                insert_button_pressed();
    void                                remove_button_pressed();
    void                                editor_changed_function();

Q_SIGNALS:
    void                                stop_values_changed(const color_map::stop_type& /*stop*/);
    void                                insert_stop(const color_map::stop_type& /*stop*/);
    void                                remove_current_stop();
    void                                function_changed();

}; // class color_map_editor_widget

} // namespace gui
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // TRANSFERFUNCTION_EDITOR_1D_COLOR_WIDGET_H_INCLUDED
