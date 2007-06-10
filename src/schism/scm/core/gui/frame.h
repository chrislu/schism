
#ifndef SCM_GUI_FRAME_H_INCLUDED
#define SCM_GUI_FRAME_H_INCLUDED

#include <scm/core/gui/draw_area.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gui {

class __scm_export(core) frame : public draw_area
{
public:
    frame();
    virtual ~frame();

    using draw_area::draw;

    void                        content_margins(const math::vec4i_t& /*margins*/);
    const math::vec4i_t&        content_margins() const;

protected:
    math::vec4i_t               _content_margins; // l, r, b, t

private:

}; // class frame

} // namespace gui
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GUI_FRAME_H_INCLUDED
