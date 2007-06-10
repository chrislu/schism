
#ifndef SCM_GUI_TEXT_BOX_H_INCLUDED
#define SCM_GUI_TEXT_BOX_H_INCLUDED

#include <string>

#include <scm/core/gui/gui.h>
#include <scm/core/gui/frame.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gui {

class __scm_export(core) text_box : public frame
{
public:
    text_box();
    virtual ~text_box();

    using frame::draw;

    void                        text_orientation(gui::text_orientation   /*ori*/);
    void                        text_flow(gui::text_flow                 /*flow*/);
    void                        text_alignment(gui::text_alignment       /*align*/);

    gui::text_orientation       text_orientation() const;
    gui::text_flow              text_flow() const;
    gui::text_alignment         text_alignment() const;

protected:
    gui::text_orientation       _text_orientation;
    gui::text_flow              _text_flow;
    gui::text_alignment         _text_alignment;

private:

}; // class text_box

} // namespace gui
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GUI_TEXT_BOX_H_INCLUDED
