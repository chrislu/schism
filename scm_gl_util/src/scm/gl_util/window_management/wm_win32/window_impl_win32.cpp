
#include "window_impl_win32.h"

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS

#include <exception>
#include <stdexcept>
#include <sstream>
#include <vector>

#include <scm/log.h>
#include <scm/gl_core/constants.h>

#include <scm/gl_util/window_management/display.h>
#include <scm/gl_util/window_management/pixel_format.h>
#include <scm/gl_util/window_management/wm_win32/display_impl_win32.h>
#include <scm/gl_util/window_management/wm_win32/error_win32.h>
#include <scm/gl_util/window_management/wm_win32/wgl_extensions.h>
#include <scm/gl_util/window_management/wm_x/display_impl_x.h>

namespace scm {
namespace gl {
namespace wm {

window::window_impl::window_impl(const display&           in_display,
                                 const std::string&       in_title,
                                 const math::vec2i&       in_position,
                                 const math::vec2ui&      in_size,
                                 const pixel_format_desc& in_pf)
  : _window_handle(0),
    _device_handle(0),
    _wgl_extensions(new util::wgl_extensions())
{
    try {
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
            wnd_style    = WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX;//WS_OVERLAPPEDWINDOW & ~WS_THICKFRAME;//; //WS_OVERLAPPEDWINDOW;//WS_POPUP | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX; //
            wnd_style_ex = 0;//WS_EX_OVERLAPPEDWINDOW; //WS_EX_STATICEDGE; //
        }

        math::vec2i wnd_position = in_display._impl->_info->_screen_origin + in_position;
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

        if (!_wgl_extensions->initialize()) {
            std::ostringstream s;
            s << "window::window_impl::window_impl() <win32>: "
              << "unable to initialize WGL ARB extensions, WGL init failed.";
            //err() << log::fatal << s.str() << log::end;
            throw(std::runtime_error(s.str()));
        }

        if (channel_count(in_pf._color_format) < 3) {
            std::ostringstream s;
            s << "window::window_impl::window_impl() <win32>: "
              << "window pixel format must contain RGB or RGBA color format "
              << "(requested: " << format_string(in_pf._color_format) << ").";
            //err() << log::fatal << s.str() << log::end;
            throw(std::runtime_error(s.str()));
        }

        std::vector<int>  pixel_desc;

        pixel_desc.push_back(WGL_ACCELERATION_ARB);
        pixel_desc.push_back(WGL_FULL_ACCELERATION_ARB);
        
        pixel_desc.push_back(WGL_DRAW_TO_WINDOW_ARB);
        pixel_desc.push_back(GL_TRUE);
        
        pixel_desc.push_back(WGL_SUPPORT_OPENGL_ARB);
        pixel_desc.push_back(GL_TRUE);
        
        pixel_desc.push_back(WGL_SWAP_METHOD_ARB);
        pixel_desc.push_back(WGL_SWAP_EXCHANGE_ARB);

        // color buffer
        pixel_desc.push_back(WGL_COLOR_BITS_ARB);
        pixel_desc.push_back(size_of_format(in_pf._color_format) * 8);
        
        pixel_desc.push_back(WGL_ALPHA_BITS_ARB);
        if ((channel_count(in_pf._color_format) > 3)) {
            pixel_desc.push_back(size_of_channel(in_pf._color_format) * 8);
        }
        else {
            pixel_desc.push_back(0);
        }

        pixel_desc.push_back(WGL_PIXEL_TYPE_ARB);
        if (is_integer_type(in_pf._color_format)) {
            pixel_desc.push_back(WGL_TYPE_RGBA_ARB);
        }
        else if (is_packed_format(in_pf._color_format)) {
            pixel_desc.push_back(WGL_TYPE_RGBA_UNSIGNED_FLOAT_EXT);
        }
        else if (is_float_type(in_pf._color_format)) {
            pixel_desc.push_back(WGL_TYPE_RGBA_FLOAT_ARB);
        }
        else {
            std::ostringstream s;
            s << "window::window_impl::window_impl() <win32>: "
              << "unable to determine pixel type of requested color format "
              << "(requested: " << format_string(in_pf._color_format) << ").";
            //err() << log::fatal << s.str() << log::end;
            throw(std::runtime_error(s.str()));
        }
        
        // depth buffer
        pixel_desc.push_back(WGL_DEPTH_BITS_ARB);
        pixel_desc.push_back(size_of_depth_component(in_pf._depth_stencil_format) * 8);
        
        pixel_desc.push_back(WGL_STENCIL_BITS_ARB);
        pixel_desc.push_back(size_of_stencil_component(in_pf._depth_stencil_format) * 8);
        
        // double, quad buffer
        pixel_desc.push_back(WGL_DOUBLE_BUFFER_ARB);
        pixel_desc.push_back(in_pf._double_buffer);
        
        pixel_desc.push_back(WGL_STEREO_ARB);
        pixel_desc.push_back(in_pf._quad_buffer_stereo);

        pixel_desc.push_back(WGL_SAMPLE_BUFFERS_ARB);   pixel_desc.push_back(GL_FALSE);//desc.max_samples() > 0 ? GL_TRUE : GL_FALSE);
        pixel_desc.push_back(WGL_SAMPLES_ARB);          pixel_desc.push_back(0);//desc.max_samples());
        pixel_desc.push_back(WGL_AUX_BUFFERS_ARB);      pixel_desc.push_back(0);

        pixel_desc.push_back(0);                        pixel_desc.push_back(0); // terminate list

        const int         query_max_formats = 20;
        int               result_pixel_fmts[query_max_formats];
        unsigned int      result_num_pixel_fmts = 0;

        if (_wgl_extensions->wglChoosePixelFormatARB(_device_handle,
                                          static_cast<const int*>(&(pixel_desc[0])),
                                          NULL,
                                          query_max_formats,
                                          result_pixel_fmts,
                                          &result_num_pixel_fmts) != TRUE)
        {
            std::ostringstream s;
            s << "window::window_impl::window_impl() <win32>: "
              << "wglChoosePixelFormat failed - requested pixel format: " << std::endl
              << in_pf << std::endl;
            //err() << log::fatal << s.str() << log::end;
            throw(std::runtime_error(s.str()));
        }

        if (result_num_pixel_fmts < 1) {
            std::ostringstream s;
            s << "window::window_impl::window_impl() <win32>: "
              << "wglChoosePixelFormat returned 0 matching pixel formats - requested pixel format: " << std::endl
              << in_pf << std::endl;
            //err() << log::fatal << s.str() << log::end;
            throw(std::runtime_error(s.str()));
        }

        if (TRUE != ::SetPixelFormat(_device_handle, result_pixel_fmts[0], NULL)) {
            std::ostringstream s;
            s << "window::window_impl::window_impl() <win32>: "
              << "SetPixelFormat failed for format number: " << result_pixel_fmts[0] << " "
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
        if (0 == ::DestroyWindow(_window_handle)) {
            err() << log::error
                  << "window::window_impl::~window_impl() <win32>: " 
                  << "unable to destroy window "
                  << "(system message: " << util::win32_error_message() << ")" << log::end;
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
