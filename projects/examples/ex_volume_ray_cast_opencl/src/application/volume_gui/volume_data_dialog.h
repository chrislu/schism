
#ifndef SCM_VOLREN_APP_VOLUME_PROPERTIES_DIALOG_H_INCLUDED
#define SCM_VOLREN_APP_VOLUME_PROPERTIES_DIALOG_H_INCLUDED

#include <QtGui/QDialog>

class QCheckBox;
class QLineEdit;
class QRadioButton;
class QSlider;
class QVBoxLayout;

#include <renderer/renderer_fwd.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gui {

class opacity_map_editor_widget;
class color_map_editor_widget;

class volume_data_dialog : public QDialog
{
    Q_OBJECT

public:
    volume_data_dialog(const data::volume_data_ptr  /*vol*/,
                       QWidget*                     /*parent*/);
    virtual ~volume_data_dialog();

    const data::volume_data_ptr&    volume() const;
    void                            volume(const data::volume_data_ptr& vol);

protected:
    void                            save_open_dir(const std::string& /*file*/);
    void                            save_save_dir(const std::string& /*file*/);

    //void                            create_controls();

    void                            create_step_controls();

protected:
    // the volume in operation
    data::volume_data_ptr           _volume;

    std::string                     _open_dir_save;
    std::string                     _save_dir_save;

    // gui elements
    QVBoxLayout*                    _dialog_layout;

    opacity_map_editor_widget*      _trafu_alpha_editor;
    color_map_editor_widget*        _trafu_color_editor;

    // sampling control
    QSlider*                        _step_size;
    QLineEdit*                      _step_line_ed;
    QSlider*                        _opacity_base;
    QLineEdit*                      _opacity_line_ed;

public Q_SLOTS:
    void                            update_controls();

    void                            load_alpha_transfer_function();
    void                            save_alpha_transfer_function();
    void                            load_color_transfer_function();
    void                            save_color_transfer_function();

    // sampling control
    void                            step_size_manipulated(int /*value*/);
    void                            opacity_base_manipulated(int /*value*/);

    void                            functions_changed();

protected Q_SLOTS:

}; // class volume_properties_dialog

} // namespace gui
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_VOLREN_APP_VOLUME_PROPERTIES_DIALOG_H_INCLUDED
