
#include "color_map_editor_widget.h"

#include <QtGui/QFrame>
#include <QtGui/QGridLayout>
#include <QtGui/QHBoxLayout>
#include <QtGui/QLabel>

#include <application/volume_gui/color_map_editor.h>
#include <application/volume_gui/color_picker_button.h>
#include <gui_support/signal_block_sentry.h>

namespace scm {
namespace gui {

color_map_editor_widget::color_map_editor_widget(QWidget* parent, Qt::WindowFlags flags)
  : QWidget(parent, flags)
{
    QHBoxLayout*     main_layout = new QHBoxLayout(this);
    main_layout->setContentsMargins(0, 0, 0, 0);

    _editor = new color_map_editor(this);
    _editor->setFrameShadow(QFrame::Plain);
    _editor->setFrameShape(QFrame::Panel);
    _editor->setMinimumSize(300, 100);
    _editor->setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::Minimum);

    _value          = new QDoubleSpinBox(this);
    _mapping_button = new color_button(this);

    _value->setMinimum(0.0);
    _value->setMaximum(1.0);
    _value->setDecimals(3);
    _value->setSingleStep(0.005);

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
    controls_layout->addWidget(_mapping_button, 2, 1);
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
    QObject::connect(_editor,           SIGNAL(selected_anchor_changed(const color_map::stop_type&)),
                     this,              SLOT(set_current_stop_values(const color_map::stop_type&)));
    QObject::connect(_editor,           SIGNAL(anchor_deselected()),
                     this,              SLOT(remove_button_pressed()));
    QObject::connect(_value,            SIGNAL(editingFinished()),
                     this,              SLOT(stop_values_changed()));
    QObject::connect(_value,            SIGNAL(valueChanged(double)),
                     this,              SLOT(stop_value_changed(double)));
    QObject::connect(_mapping_button,   SIGNAL(current_color_changed(const QColor&)),
                     this,              SLOT(stop_value_changed(const QColor&)));
    QObject::connect(this,              SIGNAL(stop_values_changed(const color_map::stop_type&)),
                     _editor,           SLOT(stop_values_changed(const color_map::stop_type&)));
    QObject::connect(this,              SIGNAL(insert_stop(const color_map::stop_type&)),
                     _editor,           SLOT(insert_stop(const color_map::stop_type&)));
    QObject::connect(_editor,           SIGNAL(function_changed()),
                     this,              SLOT(editor_changed_function()));
}

color_map_editor_widget::~color_map_editor_widget()
{
}

void
color_map_editor_widget::set_function(const shared_ptr<color_map::type>& fun)
{
    _editor->set_function(fun);
}

color_map::type::stop_type color_map_editor_widget::retrieve_stop() const
{
    color_map::stop_type      new_stop;

    new_stop.first  = _value->value();

    scm::math::vec3f new_color(_mapping_button->current_color().redF(),
                               _mapping_button->current_color().greenF(),
                               _mapping_button->current_color().blueF());

    new_stop.second = new_color;

    return (new_stop);
}

void color_map_editor_widget::set_current_stop_values(const color_map::stop_type& stop)
{
    signal_block_sentry     value_sentry(_value);
    signal_block_sentry     mapping_sentry(_mapping_button);

    _value->setValue(stop.first);

    QColor          stop_col;
    stop_col.setRgbF(stop.second.x, stop.second.y, stop.second.z);

    _mapping_button->current_color(stop_col);

    // enable remove button
    _remove_button->setDisabled(false);
}

void color_map_editor_widget::stop_values_changed()
{
    /*emit*/ stop_values_changed(retrieve_stop());
}

void color_map_editor_widget::stop_value_changed(double value)
{
    stop_values_changed();
}

void color_map_editor_widget::stop_value_changed(const QColor& value)
{
    stop_values_changed();
}

void color_map_editor_widget::insert_button_pressed()
{
    /*emit*/ insert_stop(retrieve_stop());
}

void color_map_editor_widget::remove_button_pressed()
{
    _remove_button->setDisabled(true);
}

void color_map_editor_widget::mapping_button_pressed()
{
}

void color_map_editor_widget::editor_changed_function()
{
    /*emit*/ function_changed();
}

void color_map_editor_widget::reinitialize_function()
{
    _editor->reinitialize_function();
    /*emit*/ function_changed();
}

} // namespace gui
} // namespace scm
