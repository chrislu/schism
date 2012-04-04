
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "volume_data_dialog.h"

#include <cassert>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>

#include <QtGui/QFileDialog>
#include <QtGui/QCheckBox>
#include <QtGui/QGroupBox>
#include <QtGui/QLabel>
#include <QtGui/QLineEdit>
#include <QtGui/QMenu>
#include <QtGui/QMenuBar>
#include <QtGui/QMessageBox>
#include <QtGui/QPushButton>
#include <QtGui/QRadioButton>
#include <QtGui/QSlider>

#include <QtGui/QGridLayout>
#include <QtGui/QHBoxLayout>
#include <QtGui/QVBoxLayout>

#include <scm/core/math.h>

#include <scm/gl_core/primitives/box.h>

#include <scm/gl_util/data/analysis/transfer_function/piecewise_function_1d_serialization.h>

#include <application/volume_gui/opacity_map_editor_widget.h>
#include <application/volume_gui/color_map_editor_widget.h>

#include <renderer/volume_data.h>

#include <gui_support/signal_block_sentry.h>

namespace  {

const float min_sampling_value = 0.01f;
const float max_sampling_value = 4.0f;

const float min_ess_threshold  = 0.01f;
const float max_ess_threshold  = 1.0f;

const float min_volume_scale   = -2.0f;
const float max_volume_scale   = 2.0f;

} // namespace 

namespace scm {
namespace gui {

volume_data_dialog::volume_data_dialog(const data::volume_data_ptr   vol,
                                       QWidget*                      parent)
  : QDialog(parent),
    _volume(vol)
{
    // set dialog standard layout
    _dialog_layout = new QVBoxLayout(this);
    _dialog_layout->setContentsMargins(5, 5, 5, 5);

    setSizeGripEnabled(true);

    // create menu
    QMenuBar*       menu            = new QMenuBar(this);
    QMenu*          file            = new QMenu("File", this);

    QAction*        add_lopfunc     = file->addAction("Load opacity function...");
    QAction*        add_lcolfunc    = file->addAction("Load color function...");
                                      file->addSeparator();
    QAction*        add_sopfunc     = file->addAction("Save opacity function...");
    QAction*        add_scolfunc    = file->addAction("Save color function...");
                                      file->addSeparator();
    QAction*        close_dial      = file->addAction("Close");

    menu->addMenu(file);

    // generate widgets
    _trafu_alpha_editor     = new opacity_map_editor_widget(this);
    _trafu_color_editor     = new color_map_editor_widget(this);

    _trafu_alpha_editor->set_function(_volume->alpha_map());
    _trafu_color_editor->set_function(_volume->color_map());

    layout()->setMenuBar(menu);
    layout()->addWidget(_trafu_alpha_editor);
    layout()->addWidget(_trafu_color_editor);

    QObject::connect(add_lopfunc,           SIGNAL(triggered()),
                     this,                  SLOT(load_alpha_transfer_function()));
    QObject::connect(add_lcolfunc,          SIGNAL(triggered()),
                     this,                  SLOT(load_color_transfer_function()));
    QObject::connect(add_sopfunc,           SIGNAL(triggered()),
                     this,                  SLOT(save_alpha_transfer_function()));
    QObject::connect(add_scolfunc,          SIGNAL(triggered()),
                     this,                  SLOT(save_color_transfer_function()));
    QObject::connect(close_dial,            SIGNAL(triggered()),
                     this,                  SLOT(close()));

    create_step_controls();
    update_controls();

    QObject::connect(close_dial,            SIGNAL(triggered()),
                     this,                  SLOT(close()));

    QObject::connect(_trafu_alpha_editor,   SIGNAL(function_changed()),
                     this,                  SLOT(functions_changed()));
    QObject::connect(_trafu_color_editor,   SIGNAL(function_changed()),
                     this,                  SLOT(functions_changed()));
}

volume_data_dialog::~volume_data_dialog()
{
}

const data::volume_data_ptr&
volume_data_dialog::volume() const
{
    return _volume;
}

void
volume_data_dialog::volume(const data::volume_data_ptr& vol)
{
    _volume = vol;
    update_controls();
}

void
volume_data_dialog::create_step_controls()
{
    QGroupBox       *step_control_box   = new QGroupBox("sampling controls", this);

    step_control_box->setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::Fixed);

    QLabel*         step_label          = new QLabel("samples",         step_control_box);
    QLabel*         opacity_label       = new QLabel("opacity base",    step_control_box);

    _step_line_ed                       = new QLineEdit(this);
    _opacity_line_ed                    = new QLineEdit(this);

    _step_line_ed->setReadOnly(true);
    _step_line_ed->setFixedWidth(30);
    _opacity_line_ed->setReadOnly(true);
    _opacity_line_ed->setFixedWidth(30);

    _step_size      = new QSlider(Qt::Horizontal, step_control_box);
    _opacity_base   = new QSlider(Qt::Horizontal, step_control_box);

    _step_size->setRange(10, 5000);
    _step_size->setTickInterval(150);
    _step_size->setTickPosition(QSlider::TicksBelow);
    _opacity_base->setRange(10, 5000);
    _opacity_base->setTickInterval(150);
    _opacity_base->setTickPosition(QSlider::TicksBelow);

    QGridLayout     *box_layout = new QGridLayout();
    box_layout->setContentsMargins(5, 2, 5, 2);

    box_layout->addWidget(_step_size,       0, 0);
    box_layout->addWidget(_step_line_ed,    0, 1);
    box_layout->addWidget(step_label,       0, 2);
    box_layout->addWidget(_opacity_base,    1, 0);
    box_layout->addWidget(_opacity_line_ed, 1, 1);
    box_layout->addWidget(opacity_label,    1, 2);

    step_control_box->setLayout(box_layout);
    _dialog_layout->addWidget(step_control_box);

    QObject::connect(_step_size,            SIGNAL(valueChanged(int)),
                     this,                  SLOT(step_size_manipulated(int)));
    QObject::connect(_opacity_base,         SIGNAL(valueChanged(int)),
                     this,                  SLOT(opacity_base_manipulated(int)));
}

namespace  {
QSlider* create_slider(QWidget* p) {
    QSlider* n = new QSlider(Qt::Horizontal, p);
    n->setRange(0, 1000);
    n->setTickInterval(100);
    n->setTickPosition(QSlider::TicksBelow);
    return (n);
}
} // namespace 

void
volume_data_dialog::update_controls()
{
    assert(_volume);

    using namespace scm::gui;

    _trafu_alpha_editor->set_function(_volume->alpha_map());
    _trafu_color_editor->set_function(_volume->color_map());

    signal_block_sentry     step_size_sentry(_step_size);
    signal_block_sentry     opacity_base_sentry(_opacity_base);

    scm::math::clamp(_volume->sample_distance_factor(), min_sampling_value, max_sampling_value);
    float rel_step_size = (_volume->sample_distance_factor() - min_sampling_value) / (max_sampling_value - min_sampling_value);
    int   new_step_size_slider_value = static_cast<int>(rel_step_size * (_step_size->maximum() - _step_size->minimum())) + _step_size->minimum();

    _step_size->setValue(new_step_size_slider_value);
    step_size_manipulated(_step_size->value());

    scm::math::clamp(_volume->sample_distance_ref_factor(), min_sampling_value, max_sampling_value);
    float rel_op_size = (_volume->sample_distance_ref_factor() - min_sampling_value) / (max_sampling_value - min_sampling_value);
    int   new_op_size_slider_value = static_cast<int>(rel_step_size * (_opacity_base->maximum() - _opacity_base->minimum())) + _opacity_base->minimum();

    _opacity_base->setValue(new_op_size_slider_value);
    opacity_base_manipulated(_opacity_base->value());
}

void
volume_data_dialog::save_open_dir(const std::string& file)
{
    using namespace boost::filesystem;

    path                    ifile_path(file);
    path                    ifile_directory  = ifile_path.parent_path();

    _open_dir_save = ifile_directory.string();
}

void
volume_data_dialog::save_save_dir(const std::string& file)
{
    using namespace boost::filesystem;

    path                    ifile_path(file);
    path                    ifile_directory  = ifile_path.parent_path();

    _save_dir_save = ifile_directory.string();
}

void
volume_data_dialog::load_alpha_transfer_function()
{
    std::string     trafu_file_name = QFileDialog::getOpenFileName(this,
                                                                   "Choose Opacity Function File",
                                                                   _open_dir_save.c_str(),
                                                                   ".otrf (*.otrf)").toStdString();

    if (trafu_file_name != "") {
        std::ifstream   trafu_file;

        trafu_file.open(trafu_file_name.c_str(), std::ios_base::in);

        if (trafu_file) {
            assert(_volume);

            save_open_dir(trafu_file_name);

            trafu_file >> *(_volume->alpha_map());

            if (!trafu_file) {
                std::string message = std::string("unable to load file ('") + trafu_file_name + std::string("')");
                QMessageBox::warning(this,
                                     "Error loading file",
                                     message.c_str(),
                                     QMessageBox::Ok,
                                     QMessageBox::Ok);
            }
            _trafu_alpha_editor->reinitialize_function();
        }
    }
}

void
volume_data_dialog::save_alpha_transfer_function()
{
    std::string     trafu_file_name = QFileDialog::getSaveFileName(this,
                                                                   "Choose Opacity Function File",
                                                                   _save_dir_save.c_str(),
                                                                   ".otrf (*.otrf)").toStdString();

    if (trafu_file_name != "") {
        std::ofstream   trafu_file;

        trafu_file.open(trafu_file_name.c_str(), std::ios_base::out);

        if (trafu_file) {
            assert(_volume);

            save_save_dir(trafu_file_name);

            trafu_file << *(_volume->alpha_map());

            if (!trafu_file) {
                std::string message = std::string("unable to save file ('") + trafu_file_name + std::string("')");
                QMessageBox::warning(this,
                                     "Error saving file",
                                     message.c_str(),
                                     QMessageBox::Ok,
                                     QMessageBox::Ok);
            }
        }
    }
}

void
volume_data_dialog::load_color_transfer_function()
{
    std::string     trafu_file_name = QFileDialog::getOpenFileName(this,
                                                                   "Choose Color Function File",
                                                                   _open_dir_save.c_str(),
                                                                   ".ctrf (*.ctrf)").toStdString();

    if (trafu_file_name != "") {
        std::ifstream   trafu_file;

        trafu_file.open(trafu_file_name.c_str(), std::ios_base::in);

        if (trafu_file) {
            assert(_volume);

            save_open_dir(trafu_file_name);

            trafu_file >> *(_volume->color_map());

            if (!trafu_file) {
                std::string message = std::string("unable to load file ('") + trafu_file_name + std::string("')");
                QMessageBox::warning(this,
                                     "Error loading file",
                                     message.c_str(),
                                     QMessageBox::Ok,
                                     QMessageBox::Ok);
            }
            _trafu_color_editor->reinitialize_function();
        }
    }
}

void
volume_data_dialog::save_color_transfer_function()
{
    std::string     trafu_file_name = QFileDialog::getSaveFileName(this,
                                                                   "Choose Color Function File",
                                                                   _save_dir_save.c_str(),
                                                                   ".ctrf (*.ctrf)").toStdString();

    if (trafu_file_name != "") {
        std::ofstream   trafu_file;

        trafu_file.open(trafu_file_name.c_str(), std::ios_base::out);

        if (trafu_file) {
            assert(_volume);

            save_save_dir(trafu_file_name);

            trafu_file << *(_volume->color_map());

            if (!trafu_file) {
                std::string message = std::string("unable to save file ('") + trafu_file_name + std::string("')");
                QMessageBox::warning(this,
                                     "Error saving file",
                                     message.c_str(),
                                     QMessageBox::Ok,
                                     QMessageBox::Ok);
            }
        }
    }
}

void
volume_data_dialog::step_size_manipulated(int value)
{
    float step_size_value =   static_cast<float>(value - _step_size->minimum())
                            / static_cast<float>(_step_size->maximum() - _step_size->minimum());


    float new_step_size_value = step_size_value * (max_sampling_value - min_sampling_value) + min_sampling_value;

    std::ostringstream out;
    out.precision(2);
    out << std::fixed << new_step_size_value;

    _step_line_ed->setText(out.str().c_str());

    assert(_volume);
    _volume->sample_distance_factor(new_step_size_value);

    ///*emit*/ sampling_changed();
}

void
volume_data_dialog::opacity_base_manipulated(int value)
{
    float op_size_value =   static_cast<float>(value - _opacity_base->minimum())
                          / static_cast<float>(_opacity_base->maximum() - _opacity_base->minimum());


    float new_op_size_value = op_size_value * (max_sampling_value - min_sampling_value) + min_sampling_value;

    std::ostringstream out;
    out.precision(2);
    out << std::fixed << new_op_size_value;

    _opacity_line_ed->setText(out.str().c_str());

    assert(_volume);
    _volume->sample_distance_ref_factor(new_op_size_value);

    ///*emit*/ sampling_changed();
}

void
volume_data_dialog::functions_changed()
{
    _volume->update_color_alpha_maps();
}

} // namespace gui
} // namespace scm
