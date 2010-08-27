
#ifndef SCM_GL_RENDER_CONTEXT_SURFACE_H_INCLUDED
#define SCM_GL_RENDER_CONTEXT_SURFACE_H_INCLUDED

#include <scm/core/pointer_types.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {
namespace wm {

class window;

class __scm_export(gl_util) surface
{
public:
    surface(const window& in_window);
    virtual ~surface();

private:
    // non_copyable
    surface(const surface&);
    surface& operator=(const surface&);
}; // class surface

} // namespace wm
} // namepspace gl
} // namepspace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_RENDER_CONTEXT_SURFACE_H_INCLUDED
