
#ifndef SCM_GL_UTIL_WM_WIN32_WINDOW_IMPL_WIN32_H_INCLUDED
#define SCM_GL_UTIL_WM_WIN32_WINDOW_IMPL_WIN32_H_INCLUDED

#include <scm/core/platform/platform.h>

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS

#include <scm/core/platform/windows.h>

#include <scm/core/pointer_types.h>

#include <scm/gl_util/window_management/window.h>

namespace scm {
namespace gl {
namespace wm {

namespace util {

class wgl_extensions;

} // namespace util

struct window::window_impl
{
    window_impl(const display&           in_display,
                const std::string&       in_title,
                const math::vec2i&       in_position,
                const math::vec2ui&      in_size,
                const pixel_format_desc& in_pf);
    virtual ~window_impl();

    void            cleanup();

    void            show();
    void            hide();

    HWND            _window_handle;
    HDC             _device_handle;

    shared_ptr<util::wgl_extensions>  _wgl_extensions;

}; // class window_impl

} // namespace wm
} // namepspace gl
} // namepspace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS
#endif // SCM_GL_UTIL_WM_WIN32_WINDOW_IMPL_WIN32_H_INCLUDED
