
#include "display_impl_win32.h"

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS

#include <scm/log.h>

#include <scm/gl_util/window_management/wm_win32/error_win32.h>

namespace scm {
namespace gl {
namespace wm {

display::display_impl::display_impl(const std::string& name)
{
    _hinstance = ::GetModuleHandle(0);

    if (!_hinstance) {
        err() << log::error
              << "display::display_impl::display_impl() <win32>: " 
              << "unable to get module handle"
              << " - system message: " << log::nline
              << util::win32_error_message()
              << log::end;
    }

    WNDCLASSEX wnd_class;
    ZeroMemory(&wnd_class, sizeof(WNDCLASSEX));
    wnd_class.cbSize        = sizeof(WNDCLASSEX);
    wnd_class.style         = CS_OWNDC | CS_HREDRAW | CS_VREDRAW;
    wnd_class.hInstance     = _hinstance;
    wnd_class.lpszClassName = name.c_str();

    _window_class = ::RegisterClassEx(&wnd_class);

    if (0 == _window_class) {
        err() << log::error
              << "display::display_impl::display_impl() <win32>: " 
              << "unable to register window class (" << name << ")"
              << " - system message: " << log::nline
              << util::win32_error_message()
              << log::end;
    }
}

display::display_impl::~display_impl()
{
    if (0 == ::UnregisterClass(reinterpret_cast<LPCSTR>(_window_class), _hinstance)) {
        err() << log::error
              << "display::display_impl::~display_impl() <win32>: " 
              << "unable to unregister window class"
              << " - system message: " << log::nline
              << util::win32_error_message()
              << log::end;
    }
}

} // namespace wm
} // namepspace gl
} // namepspace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS
