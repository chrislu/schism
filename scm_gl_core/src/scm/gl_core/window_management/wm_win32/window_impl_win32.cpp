
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "window_impl_win32.h"

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS

#include <exception>
#include <stdexcept>
#include <sstream>
#include <vector>

#include <scm/log.h>
#include <scm/gl_core/constants.h>

#include <scm/gl_core/window_management/display.h>
#include <scm/gl_core/window_management/wm_win32/display_impl_win32.h>
#include <scm/gl_core/window_management/wm_win32/util/error_win32.h>
#include <scm/gl_core/window_management/wm_win32/util/pixel_format_selection_win32.h>
#include <scm/gl_core/window_management/wm_win32/util/wgl_extensions.h>

namespace scm {
namespace gl {
namespace wm {

window::window_impl::window_impl(const display_cptr&      in_display,
                                 const HWND               in_parent,
                                 const std::string&       in_title,
                                 const math::vec2i&       in_position,
                                 const math::vec2ui&      in_size,
                                 const format_desc&       in_sf)
  : surface::surface_impl(),
    _window_handle(0),
    _wgl_extensions(in_display->_impl->_wgl_extensions)
{
    try {
        bool fullscreen_window = false;

        if (in_size == in_display->_impl->_info->_screen_size) {
            fullscreen_window = true;
        }

        DWORD wnd_style     = 0;
        DWORD wnd_style_ex  = 0;
        HWND parent_wnd = ::GetDesktopWindow();

        if (fullscreen_window) {
            wnd_style    = WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_POPUP;
            wnd_style_ex = WS_EX_TOPMOST;
        }
        else {
            if (0 != in_parent) {
                parent_wnd = in_parent;
                wnd_style  = WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_CHILD;
            }
            else {
                wnd_style  = WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_POPUP;
            }
            wnd_style_ex = 0;//WS_EX_OVERLAPPEDWINDOW; //0;//WS_EX_STATICEDGE; //
        }

        math::vec2i wnd_position = in_display->_impl->_info->_screen_origin + in_position;
        RECT        wnd_rect;

        ::SetRect(&wnd_rect, 0, 0, in_size.x, in_size.y);

        if (0 == ::AdjustWindowRectEx(&wnd_rect, wnd_style, false, wnd_style_ex)) {
            std::ostringstream s;
            s << "window::window_impl::window_impl() <win32>: "
              << "AdjustWindowRectEx failed "
              << "(system message: " << util::win32_error_message() << ")";
            //err() << log::fatal << s.str() << log::end;
            throw(std::runtime_error(s.str()));
        }


        _window_handle = ::CreateWindowEx(wnd_style_ex, reinterpret_cast<LPCSTR>(in_display->_impl->_window_class),
                                            in_title.c_str(),
                                            wnd_style,
                                            wnd_position.x, wnd_position.y,
                                            wnd_rect.right - wnd_rect.left, wnd_rect.bottom - wnd_rect.top,
                                            parent_wnd, 0, in_display->_impl->_hinstance,
                                            0);

        if (0 == _window_handle) {
            std::ostringstream s;
            s << "window::window_impl::window_impl() <win32>: "
              << "unable to create window "
              << "(system message: " << util::win32_error_message() << ")";
            //err() << log::fatal << s.str() << log::end;
            throw(std::runtime_error(s.str()));
        }

        _device_handle = ::GetDC(_window_handle);

        if (0 == _device_handle) {
            std::ostringstream s;
            s << "window::window_impl::window_impl() <win32>: "
              << "unable to optain window device handle"
              << "(system message: " << util::win32_error_message() << ")";
            //err() << log::fatal << s.str() << log::end;
            throw(std::runtime_error(s.str()));
        }

        if (channel_count(in_sf._color_format) < 3) {
            std::ostringstream s;
            s << "window::window_impl::window_impl() <win32>: "
              << "window pixel format must contain RGB or RGBA color format "
              << "(requested: " << format_string(in_sf._color_format) << ").";
            //err() << log::fatal << s.str() << log::end;
            throw(std::runtime_error(s.str()));
        }

        std::stringstream pfd_err;
        int pfd_num = util::pixel_format_selector::choose(in_display->_impl->_device_handle,
                                                 in_sf, util::pixel_format_selector::window_surface,
                                                 _wgl_extensions, pfd_err);
        if (0 == pfd_num) {
            std::ostringstream s;
            s << "window::window_impl::window_impl() <win32>: "
              << "unable select pixel format: "
              << pfd_err.str();
            //err() << log::fatal << s.str() << log::end;
            throw(std::runtime_error(s.str()));
        }

        if (TRUE != ::SetPixelFormat(_device_handle, pfd_num, NULL)) {
            std::ostringstream s;
            s << "window::window_impl::window_impl() <win32>: "
              << "SetPixelFormat failed for format number: " << pfd_num << " "
              << "(system message: " << util::win32_error_message() << ")";
            //err() << log::fatal << s.str() << log::end;
            throw(std::runtime_error(s.str()));
        }
    }
    catch(...) {
        cleanup();
        throw;
    }
}

window::window_impl::~window_impl()
{
    cleanup();
}

void
window::window_impl::cleanup()
{
    if (_device_handle) {
        if (0 == ::ReleaseDC(_window_handle, _device_handle)) {
            err() << log::error
                  << "window::window_impl::~window_impl() <win32>: " 
                  << "unable to release window device context"
                  << "(system message: " << util::win32_error_message() << ")" << log::end;
        }
    }
    if (_window_handle) {
        if (FALSE == ::DestroyWindow(_window_handle)) {
            err() << log::error
                  << "window::window_impl::~window_impl() <win32>: " 
                  << "unable to destroy window "
                  << "(system message: " << util::win32_error_message() << ")" << log::end;
        }
    }
}

void
window::window_impl::swap_buffers(int interval) const
{
    if (_wgl_extensions->_swap_control_supported) {
        _wgl_extensions->wglSwapIntervalEXT(interval);
    }
    ::SwapBuffers(_device_handle);
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
