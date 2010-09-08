
#ifndef SCM_GL_UTIL_WM_X_HEADLESS_SURFACE_IMPL_X_H_INCLUDED
#define SCM_GL_UTIL_WM_X_HEADLESS_SURFACE_IMPL_X_H_INCLUDED

#include <scm/core/platform/platform.h>

#if SCM_PLATFORM == SCM_PLATFORM_LINUX

#include <scm/core/pointer_types.h>

#include <scm/gl_util/window_management/headless_surface.h>
#include <scm/gl_util/window_management/window.h>
#include <scm/gl_util/window_management/wm_x2/surface_impl_x.h>

namespace scm {
namespace gl {
namespace wm {

namespace util {

} // namespace util

struct headless_surface::headless_surface_impl : public surface::surface_impl
{
    headless_surface_impl(const window_ptr& in_parent_wnd);
    virtual ~headless_surface_impl();

    void            cleanup();

}; // class headless_surface_impl

} // namespace wm
} // namepspace gl
} // namepspace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS
#endif // SCM_GL_UTIL_WM_X_HEADLESS_SURFACE_IMPL_X_H_INCLUDED
