
#ifndef SCM_GUI_DRAW_AREA_H_INCLUDED
#define SCM_GUI_DRAW_AREA_H_INCLUDED

#include <scm/core/math/math.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gui {

class __scm_export(core) draw_area
{
public:
    draw_area();
    virtual ~draw_area();

    virtual void            draw() = 0;
    virtual void            update();

    void                    position(const scm::math::vec2i& /*pos*/);
    const scm::math::vec2i& position() const;

    void                    size(const scm::math::vec2i& /*s*/);
    const scm::math::vec2i& size() const;

    void                    opacity(float /*op*/);
    float                   opacity() const;

protected:
    scm::math::vec2i        _position;
    scm::math::vec2i        _size;

    float                   _opacity;

private:

}; // class draw_area

} // namespace gui
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GUI_DRAW_AREA_H_INCLUDED
