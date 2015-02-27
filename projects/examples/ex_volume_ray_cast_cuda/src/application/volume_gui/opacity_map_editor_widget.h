
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef opacity_map_editor_WIDGET_H_INCLUDED
#define opacity_map_editor_WIDGET_H_INCLUDED

#include <vector>

#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QWidget>

#include <application/volume_gui/opacity_map_editor.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gui {

class opacity_map_editor_widget : public QWidget
{
    Q_OBJECT

public:
    opacity_map_editor_widget(QWidget* /*parent*/ = 0, Qt::WindowFlags /*flags*/  = 0);
    virtual ~opacity_map_editor_widget();

    void                                set_function(const shared_ptr<opacity_map::type>& fun);

protected:
    opacity_map_editor*                 _editor;

    QDoubleSpinBox*                     _value;
    QDoubleSpinBox*                     _mapping;

    QPushButton*                        _insert_button;
    QPushButton*                        _remove_button;

    opacity_map::type::stop_type retrieve_stop() const;

public Q_SLOTS:
    void                                reinitialize_function();

protected Q_SLOTS:
    void                                set_current_stop_values(const opacity_map::stop_type& /*stop*/);
    void                                stop_values_changed();
    void                                stop_value_changed(double /*value*/);
    void                                insert_button_pressed();
    void                                remove_button_pressed();
    void                                editor_changed_function();

Q_SIGNALS:
    void                                stop_values_changed(const opacity_map::stop_type& /*stop*/);
    void                                insert_stop(const opacity_map::stop_type& /*stop*/);
    void                                remove_current_stop();
    void                                function_changed();

private:

}; // class opacity_map_editor_widget

} // namespace gui
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // opacity_map_editor_WIDGET_H_INCLUDED
