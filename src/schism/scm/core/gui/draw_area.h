
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

    void                    position(const math::vec2i_t& /*pos*/);
    const math::vec2i_t&    position() const;

    void                    size(const math::vec2i_t& /*s*/);
    const math::vec2i_t&    size() const;

    void                    opacity(float /*op*/);
    float                   opacity() const;

protected:
    math::vec2i_t           _position;
    math::vec2i_t           _size;

    float                   _opacity;

private:

}; // class draw_area

} // namespace gui
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GUI_DRAW_AREA_H_INCLUDED
