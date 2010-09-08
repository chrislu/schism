
#ifndef SCM_GL_UTIL_WM_X_SURFACE_IMPL_X_H_INCLUDED
#define SCM_GL_UTIL_WM_X_SURFACE_IMPL_X_H_INCLUDED

#include <scm/core/platform/platform.h>

#if SCM_PLATFORM == SCM_PLATFORM_LINUX

#include <scm/core/platform/windows.h>

#include <scm/gl_util/window_management/surface.h>

namespace scm {
namespace gl {
namespace wm {

struct surface::surface_impl
{
    surface_impl() {}
}; // class surface_impl

} // namespace wm
} // namepspace gl
} // namepspace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_LINUX
#endif // SCM_GL_UTIL_WM_X_SURFACE_IMPL_X_H_INCLUDED
