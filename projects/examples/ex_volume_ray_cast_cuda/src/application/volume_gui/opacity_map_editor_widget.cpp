
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "opacity_map_editor_widget.h"

#include <iostream>

#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>

#include <gui_support/signal_block_sentry.h>

namespace scm {
namespace gui {

opacity_map_editor_widget::opacity_map_editor_widget(QWidget* parent, Qt::WindowFlags flags)
  : QWidget(parent, flags)
{
    QHBoxLayout*     main_layout = new QHBoxLayout(this);
    main_layout->setContentsMargins(0, 0, 0, 0);

    _editor = new opacity_map_editor(this);
    _editor->setFrameShadow(QFrame::Plain);
    _editor->setFrameShape(QFrame::Panel);
    _editor->setMinimumSize(300, 100);
    _editor->setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::Minimum);
    _editor->anchor_point_size(9.0f);

    _value          = new QDoubleSpinBox(this);
    _mapping        = new QDoubleSpinBox(this);

    _value->setMinimum(0.0);
    _value->setMaximum(1.0);
    _value->setDecimals(3);
    _value->setSingleStep(0.005);

    _mapping->setMinimum(0.0);
    _mapping->setMaximum(1.0);
    _mapping->setDecimals(3);
    _mapping->setSingleStep(0.005);

    _insert_button  = new QPushButton("insert", this);
    _remove_button  = new QPushButton("remove", this);

    _remove_button->setDisabled(true);

    QGridLayout*    controls_layout = new QGridLayout();

    QLabel*         value_label     = new QLabel("Value:", this);
    QLabel*         mapping_label   = new QLabel("Mapping:", this);

    QFrame*         separator_line  = new QFrame(this);
    separator_line->setFrameStyle(QFrame::HLine | QFrame::Sunken);
    separator_line->setLineWidth(1);

    controls_layout->addWidget(value_label,     1, 0);
    controls_layout->addWidget(mapping_label,   1, 1);
    controls_layout->addWidget(_value,          2, 0);
    controls_layout->addWidget(_mapping,        2, 1);
    controls_layout->addWidget(separator_line,  3, 0, 1, 2);
    controls_layout->addWidget(_insert_button,  4, 0);
    controls_layout->addWidget(_remove_button,  4, 1);
    controls_layout->setRowStretch(0, 1);

    main_layout->addWidget(_editor);
    main_layout->addLayout(controls_layout);

    QObject::connect(_insert_button,    SIGNAL(clicked()),
                     this,              SLOT(insert_button_pressed()));
    QObject::connect(_remove_button,    SIGNAL(clicked()),
                     this,              SLOT(remove_button_pressed()));
    QObject::connect(_remove_button,    SIGNAL(clicked()),
                     _editor,           SLOT(remove_current_stop()));
    QObject::connect(_editor,           SIGNAL(selected_anchor_changed(const opacity_map::stop_type&)),
                     this,              SLOT(set_current_stop_values(const opacity_map::stop_type&)));
    QObject::connect(_editor,           SIGNAL(anchor_deselected()),
                     this,              SLOT(remove_button_pressed()));
    //QObject::connect(_value,            SIGNAL(editingFinished()),
    //                 this,              SLOT(stop_values_changed()));
    QObject::connect(_value,            SIGNAL(valueChanged(double)),
                     this,              SLOT(stop_value_changed(double)));
    //QObject::connect(_mapping,          SIGNAL(editingFinished()),
    //                 this,              SLOT(stop_values_changed()));
    QObject::connect(_mapping,          SIGNAL(valueChanged(double)),
                     this,              SLOT(stop_value_changed(double)));
    QObject::connect(this,              SIGNAL(stop_values_changed(const opacity_map::stop_type&)),
                     _editor,           SLOT(stop_values_changed(const opacity_map::stop_type&)));
    QObject::connect(this,              SIGNAL(insert_stop(const opacity_map::stop_type&)),
                     _editor,           SLOT(insert_stop(const opacity_map::stop_type&)));
    QObject::connect(_editor,           SIGNAL(function_changed()),
                     this,              SLOT(editor_changed_function()));
}

opacity_map_editor_widget::~opacity_map_editor_widget()
{
}

void
opacity_map_editor_widget::set_function(const shared_ptr<opacity_map::type>& fun)
{
    _editor->set_function(fun);
}

opacity_map::type::stop_type opacity_map_editor_widget::retrieve_stop() const
{
    opacity_map::stop_type      new_stop;

    new_stop.first  = _value->value();
    new_stop.second = _mapping->value();

    return (new_stop);
}

void opacity_map_editor_widget::set_current_stop_values(const opacity_map::stop_type& stop)
{
    signal_block_sentry     value_sentry(_value);
    signal_block_sentry     mapping_sentry(_mapping);

    _value->setValue(stop.first);
    _mapping->setValue(stop.second);

    // enable remove button
    _remove_button->setDisabled(false);
}

void opacity_map_editor_widget::stop_values_changed()
{
    /*emit*/ stop_values_changed(retrieve_stop());
}

void opacity_map_editor_widget::stop_value_changed(double value)
{
    stop_values_changed();
}

void opacity_map_editor_widget::insert_button_pressed()
{
    /*emit*/ insert_stop(retrieve_stop());
}

void opacity_map_editor_widget::remove_button_pressed()
{
    _remove_button->setDisabled(true);
}

void opacity_map_editor_widget::editor_changed_function()
{
    /*emit*/ function_changed();
}

void opacity_map_editor_widget::reinitialize_function()
{
    _editor->reinitialize_function();
    /*emit*/ function_changed();
}

} // namespace gui
} // namespace scm
