
#ifndef SCM_GL_UTIL_WM_WIN32_DISPLAY_IMPL_WIN32_H_INCLUDED
#define SCM_GL_UTIL_WM_WIN32_DISPLAY_IMPL_WIN32_H_INCLUDED

#include <scm/core/platform/platform.h>

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS

#include <scm/core/platform/windows.h>

#include <scm/core/math.h>
#include <scm/core/pointer_types.h>

#include <scm/gl_util/window_management/display.h>

namespace scm {
namespace gl {
namespace wm {

struct display_info {
    std::string         _dev_name;
    std::string         _dev_string;

    math::vec2i         _screen_origin;
    math::vec2ui        _screen_size;
    unsigned            _screen_refresh_rate;
    unsigned            _screen_bpp;
}; // struct display_info

struct display::display_impl
{
    display_impl(const std::string& name);
    virtual ~display_impl();

    void                        cleanup();

    HINSTANCE                   _hinstance;

    ATOM                        _window_class;
    HDC                         _device_handle;

    shared_ptr<display_info>    _info;

}; // class display_impl

} // namespace wm
} // namepspace gl
} // namepspace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS
#endif // SCM_GL_UTIL_WM_WIN32_DISPLAY_IMPL_WIN32_H_INCLUDED
