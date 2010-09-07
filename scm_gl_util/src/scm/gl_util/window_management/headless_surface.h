
#ifndef SCM_GL_UTIL_WM_HEADLESS_SURFACE_H_INCLUDED
#define SCM_GL_UTIL_WM_HEADLESS_SURFACE_H_INCLUDED

#include <scm/core/pointer_types.h>

#include <scm/gl_util/window_management/surface.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {
namespace wm {

class window;

class __scm_export(gl_util) headless_surface : public surface
{
public:
    headless_surface(const window& in_parent_wnd);
    virtual ~headless_surface();

private:
    struct headless_surface_impl;
    shared_ptr<headless_surface_impl>     _impl;

private:
    // non_copyable
    headless_surface(const headless_surface&);
    headless_surface& operator=(const headless_surface&);

}; // class headless_surface

} // namespace wm
} // namepspace gl
} // namepspace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_UTIL_WM_HEADLESS_SURFACE_H_INCLUDED
