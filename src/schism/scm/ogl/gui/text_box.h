
#ifndef SCM_GL_GUI_TEXT_BOX_H_INCLUDED
#define SCM_GL_GUI_TEXT_BOX_H_INCLUDED

#include <scm/core/gui/text_box.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {
namespace gui {

class __scm_export(ogl) text_box : public scm::gui::text_box
{
public:
    text_box();
    virtual ~text_box();

    virtual void                draw();

protected:

private:

}; // class text_box

} // namespace gui
} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_GUI_TEXT_BOX_H_INCLUDED
