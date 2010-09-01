
#include "window_impl_win32.h"

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS

#include <exception>
#include <stdexcept>
#include <sstream>

#include <scm/log.h>

#include <scm/gl_util/window_management/display.h>
#include <scm/gl_util/window_management/pixel_format.h>
#include <scm/gl_util/window_management/wm_win32/display_impl_win32.h>
#include <scm/gl_util/window_management/wm_win32/error_win32.h>
#include <scm/gl_util/window_management/wm_x/display_impl_x.h>

namespace scm {
namespace gl {
namespace wm {

window::window_impl::window_impl(const display&           in_display,
                                 const std::string&       in_title,
                                 const math::vec2i&       in_position,
                                 const math::vec2ui&      in_size,
                                 const pixel_format_desc& in_pf)
  : _window_handle(0)
{
    bool fullscreen_window = false;

    if (in_size == in_display._impl->_info->_screen_size) {
        fullscreen_window = true;
    }

    DWORD wnd_style     = 0;
    DWORD wnd_style_ex  = 0;

    if (fullscreen_window) {
        wnd_style    = WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_POPUP;
        wnd_style_ex = WS_EX_TOPMOST; // WS_EX_APPWINDOW
    }
    else {
        wnd_style    = WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_OVERLAPPED;//WS_OVERLAPPEDWINDOW & ~WS_THICKFRAME;//; //WS_OVERLAPPEDWINDOW;//WS_POPUP | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX; //
        wnd_style_ex = 0;//WS_EX_OVERLAPPEDWINDOW; //WS_EX_STATICEDGE; //
    }

    math::vec2i wnd_position = in_display._impl->_info->_screen_origin + in_position;
    RECT        wnd_rect;

    ::SetRect(&wnd_rect, 0, 0, in_size.x, in_size.y);

    if (0 == ::AdjustWindowRectEx(&wnd_rect, wnd_style, false, wnd_style_ex)) {
        std::ostringstream s;
        s << "window::window_impl::window_impl() <win32>: "
          << "AdjustWindowRectEx failed" << std::endl
          << " - system message: " << std::endl
          << util::win32_error_message();
        err() << log::fatal << s.str() << log::end;
        throw(std::runtime_error(s.str()));
    }

    _window_handle = ::CreateWindowEx(wnd_style_ex, reinterpret_cast<LPCSTR>(in_display._impl->_window_class),
                                      in_title.c_str(),
                                      wnd_style,
                                      wnd_position.x, wnd_position.y,
                                      wnd_rect.right - wnd_rect.left, wnd_rect.bottom - wnd_rect.top,
                                      0, 0, in_display._impl->_hinstance,
                                      0);

    if (0 == _window_handle) {
        std::ostringstream s;
        s << "window::window_impl::window_impl() <win32>: "
          << "unable to create window" << std::endl
          << " - system message: " << std::endl
          << util::win32_error_message();
        err() << log::fatal << s.str() << log::end;
        throw(std::runtime_error(s.str()));
    }
}

window::window_impl::~window_impl()
{
    if (_window_handle) {
        if (0 == ::DestroyWindow(_window_handle)) {
            err() << log::error
                  << "window::window_impl::~window_impl() <win32>: " 
                  << "unable to destroy window"
                  << " - system message: " << log::nline
                  << util::win32_error_message()
                  << log::end;
        }
    }
}

void
window::window_impl::show()
{
    ::ShowWindow(_window_handle, SW_SHOWNORMAL);
}

void
window::window_impl::hide()
{
    ::ShowWindow(_window_handle, SW_HIDE);
}

} // namespace wm
} // namepspace gl
} // namepspace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS
