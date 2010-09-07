
#ifndef SCM_GL_UTIL_WM_WIN32_SURFACE_IMPL_WIN32_H_INCLUDED
#define SCM_GL_UTIL_WM_WIN32_SURFACE_IMPL_WIN32_H_INCLUDED

#include <scm/core/platform/platform.h>

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS

#include <scm/core/platform/windows.h>

#include <scm/gl_util/window_management/surface.h>

namespace scm {
namespace gl {
namespace wm {

struct surface::surface_impl
{
    HDC             _device_handle;

    surface_impl() : _device_handle(0) {}
}; // class surface_impl

} // namespace wm
} // namepspace gl
} // namepspace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS
#endif // SCM_GL_UTIL_WM_WIN32_SURFACE_IMPL_WIN32_H_INCLUDED
