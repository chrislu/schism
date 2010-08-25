
#ifndef SCM_GL_UTIL_WM_WIN32_DISPLAY_IMPL_H_INCLUDED
#define SCM_GL_UTIL_WM_WIN32_DISPLAY_IMPL_H_INCLUDED

#include <scm/core/platform/platform.h>

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS

#include <scm/core/platform/windows.h>

#include <scm/core/pointer_types.h>

#include <scm/gl_util/window_management/display.h>

namespace scm {
namespace gl {
namespace wm {

struct display::display_impl
{
    display_impl(const std::string& name);
    virtual ~display_impl();

    HINSTANCE       _hinstance;
    ATOM            _window_class;

}; // class display_impl

} // namespace wm
} // namepspace gl
} // namepspace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS
#endif // SCM_GL_UTIL_WM_WIN32_DISPLAY_IMPL_H_INCLUDED
