
#ifndef TEXTURE_2D_RECT_H_INCLUDED
#define TEXTURE_2D_RECT_H_INCLUDED

#include <scm/ogl/textures/texture_2d.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/luabind_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(ogl) texture_2d_rect : public texture_2d
{
public:
    texture_2d_rect();
    virtual ~texture_2d_rect();

protected:

private:

}; // class texture_2d_rect

} // namespace scm
} // namespace gl

#include <scm/core/utilities/luabind_warning_enable.h>

#endif // TEXTURE_2D_RECT_H_INCLUDED
