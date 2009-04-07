
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
    typedef enum {
        background_mode_none,
        background_mode_color,
        background_mode_image
    } background_mode_t;
public:
    frame();
    virtual ~frame();

    void                        background_mode(background_mode_t /*mode*/);
    background_mode_t           background_mode() const;

    void                        background_color(const scm::math::vec4f& /*col*/);
    const scm::math::vec4f&     background_color() const;

    void                        content_margins(const scm::math::vec4i& /*margins*/);
    const scm::math::vec4i&     content_margins() const;

protected:
    scm::math::vec4i            _content_margins; // l, r, b, t
    background_mode_t           _background_mode;
    scm::math::vec4f            _background_color;

private:

}; // class frame

} // namespace gui
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GUI_FRAME_H_INCLUDED
